#include <array>
#include <type_traits>

#include <ATen/Dispatch.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <cub/device/device_select.cuh>
#include <glm/common.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <torch/types.h>

#include "marching_cubes_table.h"
#include <torchhull/cuda_utils.h>
#include <torchhull/image_utils.h>
#include <torchhull/marching_cubes_utils.h>
#include <torchhull/math.h>
#include <torchhull/ravelled_sparse_tensor.h>
#include <torchhull/stdgpu_allocator.h>

namespace torchhull
{

template <typename ImageT, typename IntegralT>
__global__ void
check_integral_image(const torch::PackedTensorAccessor64<ImageT, 4, torch::RestrictPtrTraits> image,
                     const torch::PackedTensorAccessor64<IntegralT, 4, torch::RestrictPtrTraits> integral_image,
                     torch::PackedTensorAccessor64<bool, 4, torch::RestrictPtrTraits> valid)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto sizes = glm::i64vec3{ image.size(2), image.size(1), image.size(0) };
    const auto channels = image.size(3);
    const auto N = numel(sizes);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto p = unravel_index(tid, sizes);

        for (auto c = int64_t{ 0 }; c < channels; ++c)
        {
            IntegralT integral_00 = sample_zeros_padding(integral_image, p.y - 1, p.x - 1, p.z, c);
            IntegralT integral_10 = sample_zeros_padding(integral_image, p.y, p.x - 1, p.z, c);
            IntegralT integral_01 = sample_zeros_padding(integral_image, p.y - 1, p.x, p.z, c);
            IntegralT integral_11 = sample_zeros_padding(integral_image, p.y, p.x, p.z, c);

            auto image_value = static_cast<IntegralT>(image[p.z][p.y][p.x][c]);
            auto image_value_integral = integral_11 + integral_00 - integral_10 - integral_01;

            if constexpr (std::is_floating_point_v<IntegralT>)
            {
                // NOTE: Due to the large range of sizes, numerical errors may quickly build up
                const auto epsilon = IntegralT{ 1e-1 };

                valid[p.z][p.y][p.x][c] =
                        glm::epsilonEqual(static_cast<IntegralT>(image_value), image_value_integral, epsilon);
            }
            else
            {
                valid[p.z][p.y][p.x][c] = (image_value == image_value_integral);
            }
        }
    }
}

torch::Tensor
integral_image(const torch::Tensor& self, c10::ScalarType dtype)
{
    TORCH_CHECK_EQ(self.dim(), 4); // N, H, W, C

    // NOTE: torch::cumsum is limited to 32-bit indices in its internal kernel.
    //       Thus, tensors with more than 2^32 elements will have broken values at indices beyond this limit.
    auto result = self;
    result = torch::cumsum(result, 1, dtype);
    result = torch::cumsum(result, 2, dtype);

    TORCH_CHECK_EQ(result.dim(), 4); // N, H, W, C

    return result;
}

bool
is_integral_image_valid(const torch::Tensor& image, const torch::Tensor& integral_image)
{
    TORCH_CHECK_EQ(image.device(), integral_image.device());
    TORCH_CHECK_EQ(image.sizes(), integral_image.sizes());

    at::cuda::CUDAGuard device_guard{ image.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_bool = torch::TensorOptions{}.dtype(torch::kBool).device(image.device());

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(numel({ image.size(2), image.size(1), image.size(0) }),
                           grid,
                           image.device().index(),
                           threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    auto valid = torch::empty(image.sizes(), dtype_bool);

    AT_DISPATCH_ALL_TYPES(
            image.scalar_type(),
            "is_integral_image_valid",
            [&]()
            {
                auto image_ = image.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

                AT_DISPATCH_ALL_TYPES(
                        integral_image.scalar_type(),
                        "is_integral_image_valid",
                        [&]()
                        {
                            auto integral_image_ =
                                    integral_image.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

                            check_integral_image<<<grid, threads, 0, stream>>>(
                                    image_,
                                    integral_image_,
                                    valid.packed_accessor64<bool, 4, torch::RestrictPtrTraits>());
                            AT_CUDA_CHECK(cudaGetLastError());
                            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                        });
            });

    auto valid_count = valid.sum(torch::kInt64).cpu().item<int64_t>();

    return valid_count == valid.numel();
}

template <typename ValueT>
inline C10_DEVICE glm::vec4
bmm_4x4_transforms(const glm::vec3& v,
                   const torch::PackedTensorAccessor64<ValueT, 3, torch::RestrictPtrTraits>& transforms,
                   const int batch)
{
#define A(batch, i, j) transforms[batch][i][j]

    return glm::vec4{ fmaf(A(batch, 0, 0), v.x, fmaf(A(batch, 0, 1), v.y, fmaf(A(batch, 0, 2), v.z, A(batch, 0, 3)))),
                      fmaf(A(batch, 1, 0), v.x, fmaf(A(batch, 1, 1), v.y, fmaf(A(batch, 1, 2), v.z, A(batch, 1, 3)))),
                      fmaf(A(batch, 2, 0), v.x, fmaf(A(batch, 2, 1), v.y, fmaf(A(batch, 2, 2), v.z, A(batch, 2, 3)))),
                      fmaf(A(batch, 3, 0), v.x, fmaf(A(batch, 3, 1), v.y, fmaf(A(batch, 3, 2), v.z, A(batch, 3, 3)))) };

#undef A
}

template <typename Pair>
struct select1st
{
    inline C10_DEVICE typename Pair::first_type
    operator()(const Pair& pair) const
    {
        return pair.first;
    }
};

template <typename TransformT>
__global__ void
classify_children_full(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
                       const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_masks,
                       const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                       const glm::i64vec3 resolution,
                       const glm::i64vec3 resolution_children,
                       const glm::vec3 cube_corner_bfl,
                       const float cube_length,
                       torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> occupied_voxel)
{
    const auto N = occupied_voxel.size(0);

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = integral_masks.size(1);
    const auto W = integral_masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto candidate_id = tid / 8;
        auto child_id = tid % 8;

        auto g = unravel_index(candidates[candidate_id], resolution);
        auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

        auto is_empty = false;
        [[maybe_unused]] auto is_object = false;
        auto should_refine = false;
        for (auto batch = int64_t{ 0 }; batch < integral_masks.size(0); ++batch)
        {
            auto bb_min = glm::vec2{ FLT_MAX, FLT_MAX };
            auto bb_max = glm::vec2{ -FLT_MAX, -FLT_MAX };
            for (auto i = 0; i < 8; ++i)
            {
                auto v = cube_vertex(g_child, i);
                auto v_world =
                        glm::vec3{ cube_corner_bfl.x + static_cast<float>(v.x) /
                                                               static_cast<float>(resolution_children.x) * cube_length,
                                   cube_corner_bfl.y + static_cast<float>(v.y) /
                                                               static_cast<float>(resolution_children.y) * cube_length,
                                   cube_corner_bfl.z + static_cast<float>(v.z) /
                                                               static_cast<float>(resolution_children.z) *
                                                               cube_length };

                auto v_camera = bmm_4x4_transforms(v_world, transforms, batch);

                auto v_camera_ndc = glm::vec2{ v_camera.x / v_camera.w, v_camera.y / v_camera.w };

                bb_min.x = fminf(bb_min.x, v_camera_ndc.x);
                bb_min.y = fminf(bb_min.y, v_camera_ndc.y);
                bb_max.x = fmaxf(bb_max.x, v_camera_ndc.x);
                bb_max.y = fmaxf(bb_max.y, v_camera_ndc.y);
            }

            const auto ROUND_DOWN = -0.5f;
            const auto ROUND_UP = 0.5f;

            auto bb_min_unnormalized =
                    glm::vec2{ unnormalize_ndc_false(bb_min.x, W), unnormalize_ndc_false(bb_min.y, H) };
            auto bb_max_unnormalized =
                    glm::vec2{ unnormalize_ndc_false(bb_max.x, W), unnormalize_ndc_false(bb_max.y, H) };

            auto bb_min_border =
                    glm::i64vec2{ glm::clamp<int64_t>(roundf(bb_min_unnormalized.x + ROUND_DOWN), 0, W - 1),
                                  glm::clamp<int64_t>(roundf(bb_min_unnormalized.y + ROUND_DOWN), 0, H - 1) };
            auto bb_max_border =
                    glm::i64vec2{ glm::clamp<int64_t>(roundf(bb_max_unnormalized.x + ROUND_UP), 0, W - 1),
                                  glm::clamp<int64_t>(roundf(bb_max_unnormalized.y + ROUND_UP), 0, H - 1) };

            auto area_bb = (bb_max_border.y - bb_min_border.y + 1) * (bb_max_border.x - bb_min_border.x + 1);

            auto integral_mask_00 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_min_border.x - 1, batch, 0);
            auto integral_mask_10 =
                    sample_zeros_padding(integral_masks, bb_max_border.y, bb_min_border.x - 1, batch, 0);
            auto integral_mask_01 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_max_border.x, batch, 0);
            auto integral_mask_11 = sample_zeros_padding(integral_masks, bb_max_border.y, bb_max_border.x, batch, 0);

            auto integral_bb = integral_mask_11 + integral_mask_00 - integral_mask_10 - integral_mask_01;

            // NOTE: Due to the large range of sizes, numerical errors may quickly build up
            const auto epsilon = 1e-1f;

            CUDA_DEVICE_CHECK(integral_bb >= 0.f - epsilon);
            CUDA_DEVICE_CHECK(integral_bb <= static_cast<float>(area_bb) + epsilon);

            // Take the (image) isolevel into account when evaluating the accumulated mask values
            const auto isolevel = 0.5f;
            const float margin_isosurface = isolevel - epsilon;

            if (integral_bb <= 0.f + margin_isosurface ||
                (bb_min.x >= 1.f || bb_max.x <= -1.f || bb_min.y >= 1.f || bb_max.y <= -1.f))
            {
                is_empty = true;
            }
            else if (integral_bb >= static_cast<float>(area_bb) - margin_isosurface)
            {
                is_object = true;
            }
            else
            {
                should_refine = true;
            }
        }

        occupied_voxel[tid] = (should_refine && !is_empty);
    }
}

template <typename TransformT>
__global__ void
classify_children_partial(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
                          const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_masks,
                          const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                          const glm::i64vec3 resolution,
                          const glm::i64vec3 resolution_children,
                          const glm::vec3 cube_corner_bfl,
                          const float cube_length,
                          const bool last_children,
                          torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> occupied_voxel)
{
    const auto N = occupied_voxel.size(0);

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = integral_masks.size(1);
    const auto W = integral_masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto candidate_id = tid / 8;
        auto child_id = tid % 8;

        auto g = unravel_index(candidates[candidate_id], resolution);
        auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

        auto is_empty = false;
        [[maybe_unused]] auto is_object = false;
        auto should_refine = false;
        bool corner_inside[8] = { false, false, false, false, false, false, false, false };
        for (auto batch = int64_t{ 0 }; batch < integral_masks.size(0); ++batch)
        {
            auto bb_min = glm::vec2{ FLT_MAX, FLT_MAX };
            auto bb_max = glm::vec2{ -FLT_MAX, -FLT_MAX };
            for (auto i = 0; i < 8; ++i)
            {
                auto v = cube_vertex(g_child, i);
                auto v_world =
                        glm::vec3{ cube_corner_bfl.x + static_cast<float>(v.x) /
                                                               static_cast<float>(resolution_children.x) * cube_length,
                                   cube_corner_bfl.y + static_cast<float>(v.y) /
                                                               static_cast<float>(resolution_children.y) * cube_length,
                                   cube_corner_bfl.z + static_cast<float>(v.z) /
                                                               static_cast<float>(resolution_children.z) *
                                                               cube_length };

                auto v_camera = bmm_4x4_transforms(v_world, transforms, batch);

                auto v_camera_ndc = glm::vec2{ v_camera.x / v_camera.w, v_camera.y / v_camera.w };

                if (v_camera_ndc.x >= -1.f && v_camera_ndc.x <= 1.f && v_camera_ndc.y >= -1.f && v_camera_ndc.y <= 1.f)
                {
                    corner_inside[i] = true;
                }

                bb_min.x = fminf(bb_min.x, v_camera_ndc.x);
                bb_min.y = fminf(bb_min.y, v_camera_ndc.y);
                bb_max.x = fmaxf(bb_max.x, v_camera_ndc.x);
                bb_max.y = fmaxf(bb_max.y, v_camera_ndc.y);
            }

            const auto ROUND_DOWN = -0.5f;
            const auto ROUND_UP = 0.5f;

            auto bb_min_unnormalized =
                    glm::vec2{ unnormalize_ndc_false(bb_min.x, W), unnormalize_ndc_false(bb_min.y, H) };
            auto bb_max_unnormalized =
                    glm::vec2{ unnormalize_ndc_false(bb_max.x, W), unnormalize_ndc_false(bb_max.y, H) };

            auto bb_min_rounded = glm::i64vec2{ roundf(bb_min_unnormalized.x + ROUND_DOWN),
                                                roundf(bb_min_unnormalized.y + ROUND_DOWN) };
            auto bb_max_rounded =
                    glm::i64vec2{ roundf(bb_max_unnormalized.x + ROUND_UP), roundf(bb_max_unnormalized.y + ROUND_UP) };

            auto bb_min_border = glm::i64vec2{ glm::clamp<int64_t>(bb_min_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_min_rounded.y, 0, H - 1) };
            auto bb_max_border = glm::i64vec2{ glm::clamp<int64_t>(bb_max_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_max_rounded.y, 0, H - 1) };

            auto area_bb = (bb_max_border.y - bb_min_border.y + 1) * (bb_max_border.x - bb_min_border.x + 1);
            auto full_area_bb = (bb_max_rounded.y - bb_min_rounded.y + 1) * (bb_max_rounded.x - bb_min_rounded.x + 1);

            auto integral_mask_00 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_min_border.x - 1, batch, 0);
            auto integral_mask_10 =
                    sample_zeros_padding(integral_masks, bb_max_border.y, bb_min_border.x - 1, batch, 0);
            auto integral_mask_01 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_max_border.x, batch, 0);
            auto integral_mask_11 = sample_zeros_padding(integral_masks, bb_max_border.y, bb_max_border.x, batch, 0);

            auto integral_bb = integral_mask_11 + integral_mask_00 - integral_mask_10 - integral_mask_01;
            auto full_integral_bb = full_area_bb - area_bb + integral_bb;

            // NOTE: Due to the large range of sizes, numerical errors may quickly build up
            const auto epsilon = 1e-1f;

            CUDA_DEVICE_CHECK(integral_bb >= 0.f - epsilon);
            CUDA_DEVICE_CHECK(integral_bb <= static_cast<float>(area_bb) + epsilon);

            CUDA_DEVICE_CHECK(full_integral_bb >= 0.f - epsilon);
            CUDA_DEVICE_CHECK(full_integral_bb <= static_cast<float>(full_area_bb) + epsilon);

            // Take the (image) isolevel into account when evaluating the accumulated mask values
            const auto isolevel = 0.5f;
            const float margin_isosurface = isolevel - epsilon;

            if (full_integral_bb <= 0.f + margin_isosurface)
            {
                is_empty = true;
            }
            else if (integral_bb >= static_cast<float>(area_bb) - margin_isosurface && area_bb == full_area_bb)
            {
                is_object = true;
            }
            else
            {
                should_refine = true;
            }
        }

        bool voxel_inside = true;
        for (auto i = 0; i < 8; ++i)
        {
            voxel_inside &= corner_inside[i];
        }

        occupied_voxel[tid] = (should_refine && !is_empty && (!last_children || voxel_inside));
    }
}

template <typename MaskT, typename TransformT>
__global__ void
accumulate_hull_counts_full(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                            const int64_t N,
                            const torch::PackedTensorAccessor64<MaskT, 4, torch::RestrictPtrTraits> masks,
                            const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                            const glm::i64vec3 resolution_cells,
                            const glm::vec3 cube_corner_bfl,
                            const float cube_length,
                            const int64_t batch,
                            torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_values)
{
    const auto resolution_grid = glm::i64vec3{ resolution_cells.x + 1, resolution_cells.y + 1, resolution_cells.z + 1 };

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = masks.size(1);
    const auto W = masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(sparse_indices[tid], resolution_grid);

        auto g_world = glm::vec3{
            cube_corner_bfl.x + static_cast<float>(g.x) / static_cast<float>(resolution_cells.x) * cube_length,
            cube_corner_bfl.y + static_cast<float>(g.y) / static_cast<float>(resolution_cells.y) * cube_length,
            cube_corner_bfl.z + static_cast<float>(g.z) / static_cast<float>(resolution_cells.z) * cube_length
        };

        auto g_camera = bmm_4x4_transforms(g_world, transforms, batch);

        auto g_camera_ndc = glm::vec2{ g_camera.x / g_camera.w, g_camera.y / g_camera.w };

        auto g_pixel = glm::vec2{ unnormalize_ndc_false(g_camera_ndc.x, W), unnormalize_ndc_false(g_camera_ndc.y, H) };

        sparse_values[tid] *= sample_bilinear_mode_zeros_padding(masks, g_pixel.y, g_pixel.x, batch, 0);
    }
}

template <typename MaskT, typename TransformT>
__global__ void
accumulate_hull_counts_partial(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                               const int64_t N,
                               const torch::PackedTensorAccessor64<MaskT, 4, torch::RestrictPtrTraits> masks,
                               const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                               const glm::i64vec3 resolution_cells,
                               const glm::vec3 cube_corner_bfl,
                               const float cube_length,
                               const int64_t batch,
                               torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_values)
{
    const auto resolution_grid = glm::i64vec3{ resolution_cells.x + 1, resolution_cells.y + 1, resolution_cells.z + 1 };

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = masks.size(1);
    const auto W = masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(sparse_indices[tid], resolution_grid);

        auto g_world = glm::vec3{
            cube_corner_bfl.x + static_cast<float>(g.x) / static_cast<float>(resolution_cells.x) * cube_length,
            cube_corner_bfl.y + static_cast<float>(g.y) / static_cast<float>(resolution_cells.y) * cube_length,
            cube_corner_bfl.z + static_cast<float>(g.z) / static_cast<float>(resolution_cells.z) * cube_length
        };

        auto g_camera = bmm_4x4_transforms(g_world, transforms, batch);

        auto g_camera_ndc = glm::vec2{ g_camera.x / g_camera.w, g_camera.y / g_camera.w };

        auto g_pixel = glm::vec2{ unnormalize_ndc_false(g_camera_ndc.x, W), unnormalize_ndc_false(g_camera_ndc.y, H) };

        sparse_values[tid] *= sample_bilinear_mode_ones_padding(masks, g_pixel.y, g_pixel.x, batch, 0);
    }
}

__global__ void
extract_sparse_indices(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                       const glm::i64vec3 resolution_cells,
                       torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> sparse_indices_unraveled)
{
    const auto N = sparse_indices.size(0);
    const auto resolution_grid = resolution_cells + int64_t{ 1 };

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(sparse_indices[tid], resolution_grid);

        // Permute indices to z-y-x order
        sparse_indices_unraveled[0][tid] = 0;
        sparse_indices_unraveled[1][tid] = g.z;
        sparse_indices_unraveled[2][tid] = g.y;
        sparse_indices_unraveled[3][tid] = g.x;
    }
}

std::tuple<RavelledSparseTensor, std::vector<torch::Tensor>>
sparse_visual_hull_field_cuda_ravelled(const torch::Tensor& masks,
                                       const torch::Tensor& transforms,
                                       const int level,
                                       const std::array<float, 3>& cube_corner_bfl,
                                       const float cube_length,
                                       const bool masks_partial)
{
    TORCH_CHECK_EQ(masks.device(), transforms.device());
    TORCH_CHECK_EQ(masks.dim(), 4);
    TORCH_CHECK_EQ(transforms.dim(), 3);
    TORCH_CHECK_EQ(transforms.size(1), 4);
    TORCH_CHECK_EQ(transforms.size(2), 4);
    TORCH_CHECK_EQ(masks.size(0), transforms.size(0));
    TORCH_CHECK_EQ(masks.size(3), 1);
    TORCH_CHECK_GE(level, 0);
    TORCH_CHECK_GT(cube_length, 0.f);

    at::cuda::CUDAGuard device_guard{ masks.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(masks.device());
    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(masks.device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(masks.device());

    auto integral_masks = integral_image(masks, torch::kFloat32);
    auto candidates = torch::tensor({ 0 }, dtype_int64);
    auto candidates_octree = std::vector<torch::Tensor>{};
    candidates_octree.push_back(candidates);

    auto cube_corner_bfl_cuda = glm::vec3{ cube_corner_bfl[0], cube_corner_bfl[1], cube_corner_bfl[2] };

    // 1. Hierarchically compute sparse cells
    for (int i = 0; i < level; ++i)
    {
        const auto N = 8 * candidates.numel();
        const auto resolution = glm::i64vec3{ 1 << i };
        const auto resolution_children = glm::i64vec3{ 1 << (i + 1) };

        auto occupied_voxel = torch::empty({ N }, dtype_uint8).contiguous();

        auto candidates_ = candidates.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
        auto integral_masks_ = integral_masks.packed_accessor64<float, 4, torch::RestrictPtrTraits>();
        auto occupied_voxel_ = occupied_voxel.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                transforms.scalar_type(),
                "classify_children",
                [&]()
                {
                    auto transforms_ = transforms.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                    const int threads_per_block = 128;
                    dim3 grid_volume;
                    at::cuda::getApplyGrid(N, grid_volume, masks.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    if (masks_partial)
                    {
                        classify_children_partial<<<grid_volume, threads, 0, stream>>>(candidates_,
                                                                                       integral_masks_,
                                                                                       transforms_,
                                                                                       resolution,
                                                                                       resolution_children,
                                                                                       cube_corner_bfl_cuda,
                                                                                       cube_length,
                                                                                       i == level - 1,
                                                                                       occupied_voxel_);
                        AT_CUDA_CHECK(cudaGetLastError());
                        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                    else
                    {
                        classify_children_full<<<grid_volume, threads, 0, stream>>>(candidates_,
                                                                                    integral_masks_,
                                                                                    transforms_,
                                                                                    resolution,
                                                                                    resolution_children,
                                                                                    cube_corner_bfl_cuda,
                                                                                    cube_length,
                                                                                    occupied_voxel_);
                        AT_CUDA_CHECK(cudaGetLastError());
                        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                });

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

        // cuda::proclaim_return_type may require a higher CUDA version on Windows, so use this hacky version instead
        auto f = [candidates_, resolution, resolution_children] C10_HOST_DEVICE(const int64_t tid) -> int64_t
        {
#if defined(__CUDA_ARCH__)
            auto candidate_id = tid / 8;
            auto child_id = tid % 8;

            auto g = unravel_index(candidates_[candidate_id], resolution);
            auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

            return ravel_multi_index(g_child, resolution_children);
#else
            (void)tid;
            TORCH_CHECK(false,
                        "Host version of extended lambda is only defined to workaround NVCC limitations."
                        "Do not call this function on the host side.");
            return int64_t{ 0 };
#endif
        };

        auto new_candidates = torch::empty({ N }, dtype_int64).contiguous();

        // Flagged is limited to 32-bit indices at least up to cub 2.6
        TORCH_CHECK_LT(N, (static_cast<int64_t>(1) << 31));

        AT_CUDA_CHECK(
                cub::DeviceSelect::Flagged(d_temp_storage,
                                           temp_storage_bytes,
                                           thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                                           occupied_voxel.data_ptr<uint8_t>(),
                                           new_candidates.data_ptr<int64_t>(),
                                           num_selected_out.data_ptr<int64_t>(),
                                           N,
                                           stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

        AT_CUDA_CHECK(
                cub::DeviceSelect::Flagged(temp_storage.data_ptr(),
                                           temp_storage_bytes,
                                           thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                                           occupied_voxel.data_ptr<uint8_t>(),
                                           new_candidates.data_ptr<int64_t>(),
                                           num_selected_out.data_ptr<int64_t>(),
                                           N,
                                           stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        new_candidates.resize_({ num_selected_out.cpu().item<int64_t>() });

        candidates = new_candidates;
        candidates_octree.push_back(candidates);

        if (candidates.numel() == 0)
        {
            break;
        }
    }

    if (candidates.numel() == 0)
    {
        auto sparse_indices = torch::empty({ 0 }, dtype_int64);
        auto sparse_values = torch::empty({ 0 }, dtype_float);

        return { RavelledSparseTensor{ sparse_indices,
                                       sparse_values,
                                       { 1, (1 << level) + 1, (1 << level) + 1, (1 << level) + 1 } },
                 candidates_octree };
    }

    // Release no longer needed tensors early to reduce memory pressure
    integral_masks = torch::Tensor{};

    // 2. Convert sparse cells to sparse grid indices
    const auto resolution_cells = glm::i64vec3{ 1 << level };

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

    auto all_corners = torch::empty({ 8 * candidates.numel() }, dtype_int64).contiguous();
    auto sparse_indices = torch::empty({ 8 * candidates.numel() }, dtype_int64).contiguous();

    auto candidates_ = candidates.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    const auto resolution_grid = resolution_cells + int64_t{ 1 };

    thrust::transform(policy,
                      thrust::counting_iterator<int64_t>(0),
                      thrust::counting_iterator<int64_t>(all_corners.numel()),
                      all_corners.data_ptr<int64_t>(),
                      [candidates_, resolution_cells, resolution_grid] C10_DEVICE(const int64_t tid) -> int64_t
                      {
                          auto cell_id = tid / 8;
                          auto corner_id = tid % 8;

                          auto g = unravel_index(candidates_[cell_id], resolution_cells);
                          auto v = cube_vertex(g, corner_id);

                          return ravel_multi_index(v, resolution_grid);
                      });

    thrust::sort(policy, all_corners.data_ptr<int64_t>(), all_corners.data_ptr<int64_t>() + all_corners.numel());

    // Unique is limited to 32-bit indices, at least up to cub 2.6
    TORCH_CHECK_LT(8 * candidates.numel(), (static_cast<int64_t>(1) << 31));

    AT_CUDA_CHECK(cub::DeviceSelect::Unique(d_temp_storage,
                                            temp_storage_bytes,
                                            all_corners.data_ptr<int64_t>(),
                                            sparse_indices.data_ptr<int64_t>(),
                                            num_selected_out.data_ptr<int64_t>(),
                                            all_corners.numel(),
                                            stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

    AT_CUDA_CHECK(cub::DeviceSelect::Unique(temp_storage.data_ptr(),
                                            temp_storage_bytes,
                                            all_corners.data_ptr<int64_t>(),
                                            sparse_indices.data_ptr<int64_t>(),
                                            num_selected_out.data_ptr<int64_t>(),
                                            all_corners.numel(),
                                            stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto N = num_selected_out.cpu().item<int64_t>();

    sparse_indices.resize_({ N });

    // Release no longer needed tensors early to reduce memory pressure
    all_corners = torch::Tensor{};

    // 3. Compute sparse hull counts
    auto sparse_values = torch::ones({ N }, dtype_float);

    auto sparse_indices_ = sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    auto sparse_values_ = sparse_values.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
    AT_DISPATCH_ALL_TYPES_AND(
            torch::ScalarType::Half,
            masks.scalar_type(),
            "accumulate_hull_counts",
            [&]()
            {
                auto masks_ = masks.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        transforms.scalar_type(),
                        "accumulate_hull_counts",
                        [&]()
                        {
                            auto transforms_ = transforms.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                            for (auto batch = int64_t{ 0 }; batch < masks.size(0); ++batch)
                            {
                                const int threads_per_block = 128;
                                dim3 grid_corners;
                                at::cuda::getApplyGrid(N, grid_corners, masks.device().index(), threads_per_block);
                                dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                                if (masks_partial)
                                {
                                    accumulate_hull_counts_partial<<<grid_corners, threads, 0, stream>>>(
                                            sparse_indices_,
                                            N,
                                            masks_,
                                            transforms_,
                                            resolution_cells,
                                            cube_corner_bfl_cuda,
                                            cube_length,
                                            batch,
                                            sparse_values_);
                                    AT_CUDA_CHECK(cudaGetLastError());
                                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                }
                                else
                                {
                                    accumulate_hull_counts_full<<<grid_corners, threads, 0, stream>>>(
                                            sparse_indices_,
                                            N,
                                            masks_,
                                            transforms_,
                                            resolution_cells,
                                            cube_corner_bfl_cuda,
                                            cube_length,
                                            batch,
                                            sparse_values_);
                                    AT_CUDA_CHECK(cudaGetLastError());
                                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                }
                            }
                        });
            });

    return { RavelledSparseTensor{ sparse_indices,
                                   sparse_values,
                                   { 1, (1 << level) + 1, (1 << level) + 1, (1 << level) + 1 } },
             candidates_octree };
}

torch::Tensor
sparse_visual_hull_field_cuda(const torch::Tensor& masks,
                              const torch::Tensor& transforms,
                              const int level,
                              const std::array<float, 3>& cube_corner_bfl,
                              const float cube_length,
                              const bool masks_partial)
{
    auto [sparse_volume, _] = sparse_visual_hull_field_cuda_ravelled(masks,
                                                                     transforms,
                                                                     level,
                                                                     cube_corner_bfl,
                                                                     cube_length,
                                                                     masks_partial);

    at::cuda::CUDAGuard device_guard{ masks.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(masks.device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(masks.device());

    const auto resolution_cells = glm::i64vec3{ 1 << level };

    // 4. Convert to sparse tensor
    const auto N = sparse_volume.indices().numel();

    auto sparse_indices_unraveled = torch::empty({ 4, N }, dtype_int64);
    auto sparse_indices = sparse_volume.indices();

    const int threads_per_block = 128;
    dim3 grid_corners;
    at::cuda::getApplyGrid(N, grid_corners, masks.device().index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    extract_sparse_indices<<<grid_corners, threads, 0, stream>>>(
            sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            resolution_cells,
            sparse_indices_unraveled.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>());
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto sparse_field =
            torch::sparse_coo_tensor(sparse_indices_unraveled,
                                     sparse_volume.values(),
                                     { 1, resolution_cells.x + 1, resolution_cells.y + 1, resolution_cells.z + 1 },
                                     dtype_float)
                    .coalesce();

    return sparse_field;
}

template <typename scalar_t>
__global__ void
to_global_coordinates(torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> verts,
                      const glm::vec3 cube_corner_bfl,
                      const float cube_length,
                      const glm::i64vec3 resolution)
{
    const auto N = verts.size(0);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        verts[tid][0] = fmaf(cube_length / static_cast<float>(resolution.x),
                             static_cast<float>(verts[tid][0]),
                             cube_corner_bfl.x);
        verts[tid][1] = fmaf(cube_length / static_cast<float>(resolution.y),
                             static_cast<float>(verts[tid][1]),
                             cube_corner_bfl.y);
        verts[tid][2] = fmaf(cube_length / static_cast<float>(resolution.z),
                             static_cast<float>(verts[tid][2]),
                             cube_corner_bfl.z);
    }
}

template <typename scalar_t>
__global__ void
flip_faces(torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> faces)
{
    const auto N = faces.size(0);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        // Swap entries
        auto temp = faces[tid][1];
        faces[tid][1] = faces[tid][2];
        faces[tid][2] = temp;
    }
}

void
to_global_coordinates_and_flip_faces_(std::tuple<torch::Tensor, torch::Tensor>& self,
                                      const std::array<float, 3>& cube_corner_bfl,
                                      const float cube_length,
                                      const glm::i64vec3& resolution)
{
    // Use std::tie as capturing variables from structured bindings in a lambda requires C++20
    auto verts = torch::Tensor{};
    auto faces = torch::Tensor{};
    std::tie(verts, faces) = self;

    TORCH_CHECK_EQ(verts.device(), faces.device());
    TORCH_CHECK_EQ(verts.size(1), 3);
    TORCH_CHECK_EQ(faces.size(1), 3);
    TORCH_CHECK_GT(cube_length, 0.f);

    at::cuda::CUDAGuard device_guard{ verts.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    auto cube_corner_bfl_cuda = glm::vec3{ cube_corner_bfl[0], cube_corner_bfl[1], cube_corner_bfl[2] };

    const auto N_verts = verts.size(0);
    if (N_verts > 0)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                verts.scalar_type(),
                "to_global_coordinates",
                [&]()
                {
                    const int threads_per_block = 128;
                    dim3 grid;
                    at::cuda::getApplyGrid(N_verts, grid, verts.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    to_global_coordinates<<<grid, threads, 0, stream>>>(
                            verts.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                            cube_corner_bfl_cuda,
                            cube_length,
                            resolution);
                    AT_CUDA_CHECK(cudaGetLastError());
                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                });
    }

    const auto N_faces = faces.size(0);
    if (N_faces > 0)
    {
        AT_DISPATCH_INTEGRAL_TYPES(faces.scalar_type(),
                                   "flip_faces",
                                   [&]()
                                   {
                                       const int threads_per_block = 128;
                                       dim3 grid;
                                       at::cuda::getApplyGrid(N_faces, grid, faces.device().index(), threads_per_block);
                                       dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                                       flip_faces<<<grid, threads, 0, stream>>>(
                                               faces.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
                                       AT_CUDA_CHECK(cudaGetLastError());
                                       AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                   });
    }
}

__global__ void
to_wireframe(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
             const int64_t N,
             const glm::vec3 cube_corner_bfl,
             const float cube_length,
             const glm::i64vec3 resolution,
             torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> candidate_verts,
             torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> candidate_edges)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(candidates[tid], resolution);

        for (auto i = 0; i < 8; ++i)
        {
            auto v = cube_vertex(g, i);

            auto v_world = glm::vec3{ fmaf(cube_length / static_cast<float>(resolution.x), v.x, cube_corner_bfl.x),
                                      fmaf(cube_length / static_cast<float>(resolution.y), v.y, cube_corner_bfl.y),
                                      fmaf(cube_length / static_cast<float>(resolution.z), v.z, cube_corner_bfl.z) };

            auto index = 8 * tid + i;
            candidate_verts[index][0] = v_world.x;
            candidate_verts[index][1] = v_world.y;
            candidate_verts[index][2] = v_world.z;
        }

        for (auto i = 0; i < 12; ++i)
        {
            auto index = 12 * tid + i;
            candidate_edges[index][0] = 8 * tid + edge_to_vertex_table[i][0];
            candidate_edges[index][1] = 8 * tid + edge_to_vertex_table[i][1];
        }
    }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>>
candidate_voxels_to_wireframes_cuda(const std::vector<torch::Tensor>& candidate_voxels,
                                    const std::array<float, 3>& cube_corner_bfl,
                                    const float cube_length)
{
    for (auto& candidates : candidate_voxels)
    {
        TORCH_CHECK_EQ(candidates.device(), candidate_voxels[0].device());
    }

    at::cuda::CUDAGuard device_guard{ candidate_voxels[0].device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(candidate_voxels[0].device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(candidate_voxels[0].device());

    auto cube_corner_bfl_cuda = glm::vec3{ cube_corner_bfl[0], cube_corner_bfl[1], cube_corner_bfl[2] };

    auto wireframes = std::vector<std::tuple<torch::Tensor, torch::Tensor>>{};

    for (int64_t i = 0; i < static_cast<int64_t>(candidate_voxels.size()); ++i)
    {
        auto candidates = candidate_voxels[i];
        const auto N = candidates.numel();

        auto candidate_verts = torch::empty({ 8 * N, 3 }, dtype_float);
        auto candidate_edges = torch::empty({ 12 * N, 2 }, dtype_int64);

        if (N > 0)
        {
            const int threads_per_block = 128;
            dim3 grid;
            at::cuda::getApplyGrid(N, grid, candidate_voxels[0].device().index(), threads_per_block);
            dim3 threads = at::cuda::getApplyBlock(threads_per_block);

            to_wireframe<<<grid, threads, 0, stream>>>(
                    candidates.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                    N,
                    cube_corner_bfl_cuda,
                    cube_length,
                    glm::i64vec3{ 1 << i },
                    candidate_verts.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                    candidate_edges.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>());
            AT_CUDA_CHECK(cudaGetLastError());
            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        wireframes.emplace_back(candidate_verts, candidate_edges);
    }

    return wireframes;
}

} // namespace torchhull
