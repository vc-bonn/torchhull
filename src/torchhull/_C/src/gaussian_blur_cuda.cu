#include <ATen/Dispatch.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cub/device/device_select.cuh>
#include <glm/vec2.hpp>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <torch/nn/functional.h>
#include <torch/types.h>

#include <torchhull/image_utils.h>
#include <torchhull/preprocessor.h>

namespace torchhull
{

template <typename KernelT>
__global__ void
gaussian_kernel(torch::PackedTensorAccessor64<KernelT, 2, torch::RestrictPtrTraits> kernel, const float sigma)
{
    const auto kernel_size = kernel.size(0);
    const auto N = kernel_size * kernel_size;

    const auto kernel_half_size = kernel_size / 2;

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto p = glm::i64vec2{ tid % kernel_size, tid / kernel_size }; // unravel_index

        kernel[p.y][p.x] = expf(-static_cast<float>((p.y - kernel_half_size) * (p.y - kernel_half_size) +
                                                    (p.x - kernel_half_size) * (p.x - kernel_half_size)) /
                                (2.f * sigma * sigma));
    }
}

torch::Tensor
create_gaussian_kernel(const int kernel_size, const float sigma, const torch::ScalarType& dtype)
{
    const auto dtype_result = torch::TensorOptions{}.dtype(dtype).device(torch::kCUDA);

    auto kernel = torch::empty({ kernel_size, kernel_size }, dtype_result);

    at::cuda::CUDAGuard device_guard{ kernel.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(kernel_size * kernel_size, grid, kernel.device().index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(kernel.scalar_type(),
                                        "gaussian_kernel",
                                        [&]()
                                        {
                                            auto kernel_ =
                                                    kernel.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();

                                            gaussian_kernel<<<grid, threads, 0, stream>>>(kernel_, sigma);
                                            AT_CUDA_CHECK(cudaGetLastError());
                                            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                        });

    kernel.div_(kernel.sum(torch::kFloat32).cpu().item<float>());

    return kernel;
}

torch::Tensor
gaussian_blur_cuda_dense(const torch::Tensor& images,
                         const int kernel_size,
                         const float sigma,
                         const std::optional<torch::ScalarType> dtype)
{
    TORCH_CHECK_GT(kernel_size, 0);
    TORCH_CHECK_EQ(kernel_size % 2, 1);
    TORCH_CHECK_GT(sigma, 0.f);

    at::cuda::CUDAGuard device_guard{ images.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_result = dtype ? *dtype : images.scalar_type();

    if (kernel_size == 1)
    {
        return images.clone().to(dtype_result);
    }

    auto kernel = create_gaussian_kernel(kernel_size, sigma, dtype_result);

    kernel = kernel.unsqueeze(0).unsqueeze(0);
    kernel = kernel.expand({ 1, images.size(3), kernel_size, kernel_size });
    kernel = kernel.to(images.device());

    auto images_permuted = images.permute({ 0, 3, 1, 2 }).to(dtype_result);
    auto padded_images = torch::nn::functional::pad(images_permuted, // B x C x H x W
                                                    torch::nn::functional::PadFuncOptions{ {
                                                                                                   kernel_size / 2, // W
                                                                                                   kernel_size / 2, // W
                                                                                                   kernel_size / 2, // H
                                                                                                   kernel_size / 2  // H
                                                                                           } }
                                                            .mode(torch::kReflect));

    auto blurred_images_permuted =
            torch::nn::functional::conv2d(padded_images, kernel, torch::nn::functional::Conv2dFuncOptions{});

    auto blurred_images = blurred_images_permuted.permute({ 0, 2, 3, 1 });

    return blurred_images;
}

template <typename KernelT>
__global__ void
edge_kernel(const torch::PackedTensorAccessor64<KernelT, 4, torch::RestrictPtrTraits> images,
            torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> is_edge)
{
    const auto N = is_edge.size(0);

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto B = images.size(0);
    const auto H = images.size(1);
    const auto W = images.size(2);

    const auto image_dims = glm::i64vec3{ W, H, B };

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto p = unravel_index(tid, image_dims);

        auto i_00 = images[p.z][p.y][p.x][0];
        auto i_10 = in_image(p.y, p.x + 1, H, W) ? images[p.z][p.y][p.x + 1][0] : i_00;
        auto i_01 = in_image(p.y + 1, p.x, H, W) ? images[p.z][p.y + 1][p.x][0] : i_00;

        is_edge[tid] = (i_10 != i_00) || (i_01 != i_00);
    }
}

__global__ void
all_tile_indices_kernel(const int64_t B,
                        const int64_t H,
                        const int64_t W,
                        const int64_t N,
                        const int tile_size,
                        const int window_size,
                        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> edge_indices,
                        torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> all_tile_indices)
{
    const auto image_dims = glm::i64vec3{ W, H, B };
    const auto tile_dims = glm::i64vec3{ (W / tile_size) + 1, (H / tile_size) + 1, B };

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto p = unravel_index(edge_indices[tid], image_dims);

        const auto window_half_size = window_size / 2;
        for (int i = -window_half_size; i <= window_half_size; ++i)
        {
            for (int j = -window_half_size; j <= window_half_size; ++j)
            {
                const auto index = (window_size * window_size) * tid + (i + window_half_size) * window_size +
                                   (j + window_half_size);
                all_tile_indices[index] =
                        ravel_multi_index(glm::i64vec3{ glm::clamp<int64_t>((p.x / tile_size) + i, 0, W / tile_size),
                                                        glm::clamp<int64_t>((p.y / tile_size) + j, 0, H / tile_size),
                                                        p.z },
                                          tile_dims);
            }
        }
    }
}

template <int kernel_size, typename ImageT, typename BlurredImageT>
__global__ void
tile_convolution_kernel_specialized(
        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> tile_indices,
        const int64_t N,
        const int tile_size,
        const float sigma,
        const torch::PackedTensorAccessor64<ImageT, 4, torch::RestrictPtrTraits> images,
        torch::PackedTensorAccessor64<BlurredImageT, 4, torch::RestrictPtrTraits> blurred_images)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto B = images.size(0);
    const auto H = images.size(1);
    const auto W = images.size(2);

    const auto image_dims = glm::i64vec3{ W, H, B };
    const auto tile_dims = glm::i64vec3{ (W / tile_size) + 1, (H / tile_size) + 1, B };

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto tile_id = tid / (tile_size * tile_size);
        auto local_pixel_id = tid % (tile_size * tile_size);

        auto r = unravel_index(tile_indices[tile_id], tile_dims);
        auto l = glm::i64vec2{ local_pixel_id % tile_size, local_pixel_id / tile_size }; // unravel_index

        auto p = glm::i64vec3{ r.x * tile_size + l.x, r.y * tile_size + l.y, r.z };
        if (!in_image(p.y, p.x, H, W))
        {
            continue;
        }

        const auto kernel_half_size = kernel_size / 2;
        auto blurred_val = BlurredImageT{ 0 };
        auto normalization = BlurredImageT{ 0 };
        for (auto i = -kernel_half_size; i <= kernel_half_size; ++i)
        {
            for (auto j = -kernel_half_size; j <= kernel_half_size; ++j)
            {
                auto image_val = static_cast<BlurredImageT>(sample_reflect_padding(images, p.y + j, p.x + i, p.z, 0));
                auto kernel_val =
                        static_cast<BlurredImageT>(expf(-static_cast<float>(i * i + j * j) / (2.f * sigma * sigma)));

                blurred_val += kernel_val * image_val;
                normalization += kernel_val;
            }
        }

        blurred_images[p.z][p.y][p.x][0] = blurred_val / normalization;
    }
}

template <typename ImageT, typename BlurredImageT>
__global__ void
tile_convolution_kernel(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> tile_indices,
                        const int64_t N,
                        const int tile_size,
                        const int kernel_size,
                        const float sigma,
                        const torch::PackedTensorAccessor64<ImageT, 4, torch::RestrictPtrTraits> images,
                        torch::PackedTensorAccessor64<BlurredImageT, 4, torch::RestrictPtrTraits> blurred_images)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto B = images.size(0);
    const auto H = images.size(1);
    const auto W = images.size(2);

    const auto image_dims = glm::i64vec3{ W, H, B };
    const auto tile_dims = glm::i64vec3{ (W / tile_size) + 1, (H / tile_size) + 1, B };

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto tile_id = tid / (tile_size * tile_size);
        auto local_pixel_id = tid % (tile_size * tile_size);

        auto r = unravel_index(tile_indices[tile_id], tile_dims);
        auto l = glm::i64vec2{ local_pixel_id % tile_size, local_pixel_id / tile_size }; // unravel_index

        auto p = glm::i64vec3{ r.x * tile_size + l.x, r.y * tile_size + l.y, r.z };
        if (!in_image(p.y, p.x, H, W))
        {
            continue;
        }

        const auto kernel_half_size = kernel_size / 2;
        auto blurred_val = BlurredImageT{ 0 };
        auto normalization = BlurredImageT{ 0 };
        for (auto i = -kernel_half_size; i <= kernel_half_size; ++i)
        {
            for (auto j = -kernel_half_size; j <= kernel_half_size; ++j)
            {
                auto image_val = static_cast<BlurredImageT>(sample_reflect_padding(images, p.y + j, p.x + i, p.z, 0));
                auto kernel_val =
                        static_cast<BlurredImageT>(expf(-static_cast<float>(i * i + j * j) / (2.f * sigma * sigma)));

                blurred_val += kernel_val * image_val;
                normalization += kernel_val;
            }
        }

        blurred_images[p.z][p.y][p.x][0] = blurred_val / normalization;
    }
}

torch::Tensor
gaussian_blur_cuda_sparse(const torch::Tensor& images,
                          const int kernel_size,
                          const float sigma,
                          const std::optional<torch::ScalarType> dtype)
{
    TORCH_CHECK_GT(kernel_size, 0);
    TORCH_CHECK_EQ(kernel_size % 2, 1);
    TORCH_CHECK_GT(sigma, 0.f);

    at::cuda::CUDAGuard device_guard{ images.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_result = dtype ? *dtype : images.scalar_type();
    TORCH_CHECK_EQ(torch::isFloatingType(dtype_result), true);

    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(images.device());
    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(images.device());

    if (kernel_size == 1)
    {
        return images.clone().to(dtype_result);
    }

    // 1. Detect edges
    const auto N = images.numel();
    auto is_edge = torch::empty({ N }, dtype_uint8).contiguous();

    const int threads_per_block = 128;
    dim3 grid_pixels;
    at::cuda::getApplyGrid(N, grid_pixels, images.device().index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    AT_DISPATCH_ALL_TYPES_AND(torch::ScalarType::Half,
                              images.scalar_type(),
                              "edge_kernel",
                              [&]()
                              {
                                  auto images_ = images.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();
                                  auto is_edge_ = is_edge.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>();

                                  edge_kernel<<<grid_pixels, threads, 0, stream>>>(images_, is_edge_);
                                  AT_CUDA_CHECK(cudaGetLastError());
                                  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                              });

    // 2. Get indices of edges
    auto edge_indices = torch::empty({ N }, dtype_int64).contiguous();

    // Flagged is limited to 32-bit indices at least up to cub 2.6
    TORCH_CHECK_LT(N, (static_cast<int64_t>(1) << 31));

    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

        AT_CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage,
                                                 temp_storage_bytes,
                                                 thrust::counting_iterator<int64_t>(0),
                                                 is_edge.data_ptr<uint8_t>(),
                                                 edge_indices.data_ptr<int64_t>(),
                                                 num_selected_out.data_ptr<int64_t>(),
                                                 N,
                                                 stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

        AT_CUDA_CHECK(cub::DeviceSelect::Flagged(temp_storage.data_ptr(),
                                                 temp_storage_bytes,
                                                 thrust::counting_iterator<int64_t>(0),
                                                 is_edge.data_ptr<uint8_t>(),
                                                 edge_indices.data_ptr<int64_t>(),
                                                 num_selected_out.data_ptr<int64_t>(),
                                                 N,
                                                 stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        edge_indices.resize_({ num_selected_out.cpu().item<int64_t>() });
    }

    if (edge_indices.numel() == 0)
    {
        return images.clone().to(dtype_result);
    }

    // 3. Compute tiles
    const auto B = images.size(0);
    const auto H = images.size(1);
    const auto W = images.size(2);

    // Whole band should have width of kernel_size, so each tile must be at least 1/2 the size of this
    const auto tile_size = kernel_size / 2;
    constexpr auto window_size = 3;
    auto all_tile_indices =
            torch::empty({ window_size * window_size * edge_indices.numel() }, dtype_int64).contiguous();
    auto tile_indices = torch::empty({ window_size * window_size * edge_indices.numel() }, dtype_int64).contiguous();

    dim3 grid_indices;
    at::cuda::getApplyGrid(edge_indices.numel(), grid_indices, images.device().index(), threads_per_block);

    auto edge_indices_ = edge_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    auto all_tile_indices_ = all_tile_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();

    all_tile_indices_kernel<<<grid_indices, threads, 0, stream>>>(B,
                                                                  H,
                                                                  W,
                                                                  edge_indices.numel(),
                                                                  tile_size,
                                                                  window_size,
                                                                  edge_indices_,
                                                                  all_tile_indices_);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::sort(policy,
                 all_tile_indices.data_ptr<int64_t>(),
                 all_tile_indices.data_ptr<int64_t>() + all_tile_indices.numel());

    // Unique is limited to 32-bit indices, at least up to cub 2.6
    TORCH_CHECK_LT(window_size * window_size * edge_indices.numel(), (static_cast<int64_t>(1) << 31));

    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

        AT_CUDA_CHECK(cub::DeviceSelect::Unique(d_temp_storage,
                                                temp_storage_bytes,
                                                all_tile_indices.data_ptr<int64_t>(),
                                                tile_indices.data_ptr<int64_t>(),
                                                num_selected_out.data_ptr<int64_t>(),
                                                all_tile_indices.numel(),
                                                stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

        AT_CUDA_CHECK(cub::DeviceSelect::Unique(temp_storage.data_ptr(),
                                                temp_storage_bytes,
                                                all_tile_indices.data_ptr<int64_t>(),
                                                tile_indices.data_ptr<int64_t>(),
                                                num_selected_out.data_ptr<int64_t>(),
                                                all_tile_indices.numel(),
                                                stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        tile_indices.resize_({ num_selected_out.cpu().item<int64_t>() });
    }

    // 4. Convolve pixels in tiles
    const auto M = tile_indices.numel() * tile_size * tile_size;
    auto blurred_images = images.clone().to(dtype_result);

    dim3 grid_convolution;
    at::cuda::getApplyGrid(M, grid_convolution, images.device().index(), threads_per_block);

    AT_DISPATCH_ALL_TYPES_AND(
            torch::ScalarType::Half,
            images.scalar_type(),
            "tile_convolution_kernel",
            [&]()
            {
                auto tile_indices_ = tile_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
                auto images_ = images.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        blurred_images.scalar_type(),
                        "tile_convolution_kernel",
                        [&]()
                        {
                            auto blurred_images_ =
                                    blurred_images.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

#define CASE_TILE_CONVOLUTION_KERNEL_SPECIALIZED(KERNEL_SIZE)                                                          \
    case KERNEL_SIZE:                                                                                                  \
    {                                                                                                                  \
        tile_convolution_kernel_specialized<KERNEL_SIZE><<<grid_convolution, threads, 0, stream>>>(tile_indices_,      \
                                                                                                   M,                  \
                                                                                                   tile_size,          \
                                                                                                   sigma,              \
                                                                                                   images_,            \
                                                                                                   blurred_images_);   \
        DEFER(AT_CUDA_CHECK(cudaGetLastError());)                                                                      \
        DEFER(AT_CUDA_CHECK(cudaStreamSynchronize(stream));)                                                           \
    }                                                                                                                  \
    break;

                            switch (kernel_size)
                            {
                                // Tested all possible values up to 21 for specialization.
                                // 7 and 9 are 10% faster, whereas all the others are either equally fast or become
                                // slightly slower.
                                CASE_TILE_CONVOLUTION_KERNEL_SPECIALIZED(7)
                                CASE_TILE_CONVOLUTION_KERNEL_SPECIALIZED(9)

                                default:
                                {
                                    tile_convolution_kernel<<<grid_convolution, threads, 0, stream>>>(tile_indices_,
                                                                                                      M,
                                                                                                      tile_size,
                                                                                                      kernel_size,
                                                                                                      sigma,
                                                                                                      images_,
                                                                                                      blurred_images_);
                                    AT_CUDA_CHECK(cudaGetLastError());
                                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                }
                            }
                        });
            });

#undef CASE_TILE_CONVOLUTION_KERNEL_SPECIALIZED

    return blurred_images;
}

torch::Tensor
gaussian_blur_cuda(const torch::Tensor& images,
                   const int kernel_size,
                   const float sigma,
                   const bool sparse,
                   const std::optional<torch::ScalarType> dtype)
{
    return sparse ? gaussian_blur_cuda_sparse(images, kernel_size, sigma, dtype)
                  : gaussian_blur_cuda_dense(images, kernel_size, sigma, dtype);
}

} // namespace torchhull
