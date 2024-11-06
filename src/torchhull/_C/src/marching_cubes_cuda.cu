//
// Inspired and largely modified version of pytorch3d's Marching Cubes CUDA implementation,
// which is available under the below BSD 3-Clause license.
//
// Improvements compared to pytorch3d (as of 0.7.8):
// - Significantly better performance (6x - 30x for NVIDIA RTX 4090)
// - Lower memory consumption
// - Large tensor processing via full 64-bit indexing support
//

// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the name Meta nor the names of its contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdint>
#include <optional>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <glm/gtc/epsilon.hpp>
#include <glm/vec3.hpp>
#include <stdgpu/memory.h>
#include <stdgpu/unordered_map.cuh>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
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

using sparse_volume_hash_map = stdgpu::unordered_map<int64_t,
                                                     int64_t,
                                                     stdgpu::hash<int64_t>,
                                                     stdgpu::equal_to<int64_t>,
                                                     StdgpuAllocator<stdgpu::pair<const int64_t, int64_t>>>;

inline C10_HOST_DEVICE glm::vec3
interpolate_vertex(const float isolevel,
                   const glm::i64vec3& p1,
                   const glm::i64vec3& p2,
                   const float val1,
                   const float val2)
{
    constexpr float epsilon = 1e-5f;
    if (glm::epsilonEqual(isolevel, val1, epsilon))
    {
        return glm::vec3{ p1 };
    }
    if (glm::epsilonEqual(isolevel, val2, epsilon))
    {
        return glm::vec3{ p2 };
    }
    if (glm::epsilonEqual(val1, val2, epsilon))
    {
        return glm::vec3{ p1 };
    }

    auto t = (isolevel - val1) / (val2 - val1);
    return glm::vec3{ lerp(static_cast<float>(p1.x), static_cast<float>(p2.x), t),
                      lerp(static_cast<float>(p1.y), static_cast<float>(p2.y), t),
                      lerp(static_cast<float>(p1.z), static_cast<float>(p2.z), t) };
}

inline C10_HOST_DEVICE int64_t
hash_edge(const int64_t v1_id, const int64_t v2_direction)
{
    CUDA_DEVICE_CHECK(v1_id >= 0);
    CUDA_DEVICE_CHECK(v1_id < (int64_t{ 1 } << 60));
    CUDA_DEVICE_CHECK(v2_direction >= 0);
    CUDA_DEVICE_CHECK(v2_direction < (int64_t{ 1 } << 3));

    auto edge_id = static_cast<int64_t>(static_cast<uint64_t>(v1_id) << 3 | static_cast<uint64_t>(v2_direction));

    CUDA_DEVICE_CHECK(edge_id >= 0);

    return edge_id;
}

__global__ void
hash_volume(const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> volume_indices,
            const glm::i64vec3 full_sizes,
            const int64_t N,
            torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
            sparse_volume_hash_map volume_hashed)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = glm::i64vec3{ volume_indices[3][tid], volume_indices[2][tid], volume_indices[1][tid] };
        auto g_id = ravel_multi_index(g, full_sizes);

        volume_hashed.emplace(g_id, tid);
        sparse_indices[tid] = g_id;
    }
}

__global__ void
hash_ravelled_volume(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                     const int64_t N,
                     sparse_volume_hash_map volume_hashed)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        volume_hashed.emplace(sparse_indices[tid], tid);
    }
}

template <typename scalar_t>
__global__ void
classify_voxels_dense(const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> volume,
                      torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> verts_per_voxel,
                      torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> occupied_voxel,
                      const float isolevel)
{
    // Skip last slices where no surface will be extracted
    const auto sizes = glm::i64vec3{ volume.size(1) - 1, volume.size(2) - 1, volume.size(3) - 1 };
    const auto N = numel(sizes);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(tid, sizes);

        auto cubeindex = uint8_t{ 0 };
        for (auto i = 0; i < 8; ++i)
        {
            auto v = cube_vertex(g, i);
            auto val_i = volume[0][v.z][v.y][v.x];
            if (val_i < isolevel)
            {
                cubeindex |= (1 << index_table[i]);
            }
        }

        auto num_verts = vertex_number_table[cubeindex];
        verts_per_voxel[tid] = num_verts;
        occupied_voxel[tid] = (num_verts > 0);
    }
}

template <typename scalar_t>
__global__ void
classify_voxels_sparse(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                       const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> sparse_values,
                       const sparse_volume_hash_map volume_hashed,
                       const glm::i64vec3 full_sizes,
                       const int64_t N,
                       torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> verts_per_voxel,
                       torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> occupied_voxel,
                       const float isolevel)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(sparse_indices[tid], full_sizes);

        auto cubeindex = uint8_t{ 0 };
        // Check for boundary vertices
        if (g.x < full_sizes.x - 1 && g.y < full_sizes.y - 1 && g.z < full_sizes.z - 1)
        {
            for (auto i = 0; i < 8; ++i)
            {
                auto v = cube_vertex(g, i);
                auto v_id = ravel_multi_index(v, full_sizes);

                auto it = volume_hashed.find(v_id);
                if (it == volume_hashed.end())
                {
                    cubeindex = uint8_t{ 0 };
                    break;
                }
                if (sparse_values[it->second] < isolevel)
                {
                    cubeindex |= (1 << index_table[i]);
                }
            }
        }

        auto num_verts = vertex_number_table[cubeindex];
        verts_per_voxel[tid] = num_verts;
        occupied_voxel[tid] = (num_verts > 0);
    }
}

__global__ void
compactify_voxels(torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> compact_voxel_array,
                  const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> occupied_voxel,
                  const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> occupied_voxel_scan,
                  const int64_t N)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        if (occupied_voxel[tid])
        {
            compact_voxel_array[occupied_voxel_scan[tid]] = tid;
        }
    }
}

template <typename scalar_t>
__global__ void
generate_faces_dense(torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> verts,
                     torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> ids,
                     const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> compact_voxel_array,
                     const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> verts_per_voxel_scan,
                     const int64_t num_active_voxels,
                     const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> volume,
                     const float isolevel,
                     const bool unique_verts)
{
    // Skip last slices where no surface will be extracted
    const auto sizes = glm::i64vec3{ volume.size(1) - 1, volume.size(2) - 1, volume.size(3) - 1 };
    const auto full_sizes = sizes + int64_t{ 1 };

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < num_active_voxels; tid += num_threads)
    {
        auto voxel_id = compact_voxel_array[tid];

        auto g = unravel_index(voxel_id, sizes);

        float val[8];
        auto cubeindex = uint8_t{ 0 };
        for (auto i = 0; i < 8; ++i)
        {
            auto v = cube_vertex(g, i);
            auto val_i = volume[0][v.z][v.y][v.x];
            if (val_i < isolevel)
            {
                cubeindex |= (1 << index_table[i]);
            }
            val[index_table[i]] = val_i;
        }

        auto num_verts = vertex_number_table[cubeindex];
        for (auto i = 0; i < num_verts; ++i)
        {
            auto index = verts_per_voxel_scan[voxel_id] + i;
            auto edge = face_table[cubeindex][i];

            auto v1 = edge_to_vertex_table[edge][0];
            auto v2 = edge_to_vertex_table[edge][1];
            auto p1 = cube_vertex(g, v1);
            auto p2 = cube_vertex(g, v2);

            // Interpolate on-the-fly since the average number of edges over all combinations is 9.685... < 12.
            // However, not all combinations are equally likely, so this is more efficient than computing
            // - all 12 edge vertices
            // - conditionally computing the minimum set of edge vertices using an edge table and branching
            auto v = interpolate_vertex(isolevel, p1, p2, val[index_table[v1]], val[index_table[v2]]);
            verts[index][0] = v.x;
            verts[index][1] = v.y;
            verts[index][2] = v.z;

            if (unique_verts)
            {
                auto v1_id = ravel_multi_index(p1, full_sizes);
                auto v2_direction = edge_direction_table[edge];
                ids[index] = hash_edge(v1_id, v2_direction);
            }
        }
    }
}

template <typename scalar_t>
__global__ void
generate_faces_sparse(torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> verts,
                      torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> ids,
                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> compact_voxel_array,
                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> verts_per_voxel_scan,
                      const int64_t num_active_voxels,
                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                      const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> sparse_values,
                      const glm::i64vec3 full_sizes,
                      const sparse_volume_hash_map volume_hashed,
                      const float isolevel,
                      const bool unique_verts)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < num_active_voxels; tid += num_threads)
    {
        auto voxel_id = compact_voxel_array[tid];

        auto g = unravel_index(sparse_indices[voxel_id], full_sizes);

        float val[8];
        auto cubeindex = uint8_t{ 0 };
        for (auto i = 0; i < 8; ++i)
        {
            auto v = cube_vertex(g, i);
            auto v_id = ravel_multi_index(v, full_sizes);

            // All neighbors must exists, so unconditionally access the values
            auto it = volume_hashed.find(v_id);
            if (sparse_values[it->second] < isolevel)
            {
                cubeindex |= (1 << index_table[i]);
            }
            val[index_table[i]] = sparse_values[it->second];
        }

        auto num_verts = vertex_number_table[cubeindex];
        for (auto i = 0; i < num_verts; ++i)
        {
            auto index = verts_per_voxel_scan[voxel_id] + i;
            auto edge = face_table[cubeindex][i];

            auto v1 = edge_to_vertex_table[edge][0];
            auto v2 = edge_to_vertex_table[edge][1];
            auto p1 = cube_vertex(g, v1);
            auto p2 = cube_vertex(g, v2);

            // Interpolate on-the-fly since the average number of edges over all combinations is 9.685... < 12.
            // However, not all combinations are equally likely, so this is more efficient than computing
            // - all 12 edge vertices
            // - conditionally computing the minimum set of edge vertices using an edge table and branching
            auto v = interpolate_vertex(isolevel, p1, p2, val[index_table[v1]], val[index_table[v2]]);
            verts[index][0] = v.x;
            verts[index][1] = v.y;
            verts[index][2] = v.z;

            if (unique_verts)
            {
                auto v1_id = ravel_multi_index(p1, full_sizes);
                auto v2_direction = edge_direction_table[edge];
                ids[index] = hash_edge(v1_id, v2_direction);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes_cuda_dense(const torch::Tensor& volume,
                          const float isolevel,
                          const bool return_local_coords,
                          const bool unique_verts)
{
    TORCH_CHECK_EQ(volume.dim(), 4);
    TORCH_CHECK_EQ(volume.size(0), 1);
    TORCH_CHECK_LE(volume.numel(), (static_cast<int64_t>(1) << 60));
    TORCH_CHECK_GE(isolevel, 0);

    at::cuda::CUDAGuard device_guard{ volume.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(volume.device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(volume.device());

    const auto sizes = glm::i64vec3{ volume.size(1) - 1, volume.size(2) - 1, volume.size(3) - 1 };
    const auto N = numel(sizes);

    // 1. Find active voxels
    auto verts_per_voxel = torch::empty({ N }, dtype_int64);
    auto occupied_voxel = torch::empty({ N }, dtype_int64);

    const int threads_per_block = 128;
    dim3 grid_volume;
    at::cuda::getApplyGrid(N, grid_volume, volume.device().index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            volume.scalar_type(),
            "classify_voxels_dense",
            [&]()
            {
                classify_voxels_dense<<<grid_volume, threads, 0, stream>>>(
                        volume.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        verts_per_voxel.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        occupied_voxel.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        isolevel);
                AT_CUDA_CHECK(cudaGetLastError());
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    // 2. Make compact buffer for active voxels ids
    auto occupied_voxel_scan = torch::empty({ N }, dtype_int64);

    thrust::exclusive_scan(policy,
                           occupied_voxel.data_ptr<int64_t>(),
                           occupied_voxel.data_ptr<int64_t>() + N,
                           occupied_voxel_scan.data_ptr<int64_t>());

    auto last_element = occupied_voxel[N - 1].cpu().item<int64_t>();
    auto last_scan = occupied_voxel_scan[N - 1].cpu().item<int64_t>();
    const auto num_active_voxels = last_element + last_scan;

    if (num_active_voxels == 0)
    {
        auto verts = torch::zeros({ 0, 3 }, dtype_float);
        auto faces = torch::zeros({ 0, 3 }, dtype_int64);

        return { verts, faces };
    }

    auto compact_voxel_array = torch::empty({ num_active_voxels }, dtype_int64);

    compactify_voxels<<<grid_volume, threads, 0, stream>>>(
            compact_voxel_array.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            occupied_voxel.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            occupied_voxel_scan.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            N);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Release no longer needed tensors early to reduce memory pressure
    occupied_voxel = torch::Tensor{};
    occupied_voxel_scan = torch::Tensor{};

    // 3. Compute per-voxel vertex buffer offsets
    auto verts_per_voxel_scan = torch::empty({ N }, dtype_int64);
    thrust::exclusive_scan(policy,
                           verts_per_voxel.data_ptr<int64_t>(),
                           verts_per_voxel.data_ptr<int64_t>() + N,
                           verts_per_voxel_scan.data_ptr<int64_t>());

    last_element = verts_per_voxel[N - 1].cpu().item<int64_t>();
    last_scan = verts_per_voxel_scan[N - 1].cpu().item<int64_t>();
    const auto num_total_verts = last_element + last_scan;

    // Release no longer needed tensors early to reduce memory pressure
    verts_per_voxel = torch::Tensor{};

    // 4. Generate triangle vertices
    auto verts = torch::Tensor{};
    auto ids = torch::empty({ unique_verts ? num_total_verts : 0 }, dtype_int64);
    verts = torch::empty({ num_total_verts, 3 }, dtype_float);

    dim3 grid_active_volume;
    at::cuda::getApplyGrid(num_active_voxels, grid_active_volume, volume.device().index(), threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            volume.scalar_type(),
            "generate_faces_dense",
            [&]()
            {
                generate_faces_dense<<<grid_active_volume, threads, 0, stream>>>(
                        verts.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                        ids.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                        compact_voxel_array.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        verts_per_voxel_scan.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        num_active_voxels,
                        volume.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        isolevel,
                        unique_verts);
                AT_CUDA_CHECK(cudaGetLastError());
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    // Release no longer needed tensors early to reduce memory pressure
    compact_voxel_array = torch::Tensor{};
    verts_per_voxel_scan = torch::Tensor{};

    // 5. Convert triangle soup to compact mesh
    auto verts_mesh = torch::Tensor{};
    auto faces_mesh = torch::Tensor{};
    if (unique_verts)
    {
        const auto sorted = true;
        const auto return_inverse = true;
        const auto return_counts = false;
        auto [unique_ids, inverse_idx, _] = torch::unique_dim(ids, 0, sorted, return_inverse, return_counts);

        verts_mesh = torch::empty({ unique_ids.size(0), 3 }, dtype_float);
        verts_mesh.index_put_({ inverse_idx }, verts);

        faces_mesh = inverse_idx.reshape({ num_total_verts / 3, 3 }).to(dtype_int64);
    }
    else
    {
        verts_mesh = verts;
        faces_mesh = torch::arange(num_total_verts, dtype_int64).reshape({ num_total_verts / 3, 3 });
    }

    if (return_local_coords)
    {
        const auto scale = torch::tensor({ 2.f / (static_cast<float>(volume.size(1) - 1)),
                                           2.f / (static_cast<float>(volume.size(2) - 1)),
                                           2.f / (static_cast<float>(volume.size(3) - 1)) },
                                         dtype_float);

        verts_mesh *= torch::unsqueeze(scale, 0);
        verts_mesh -= 1.f;
    }

    return { verts_mesh, faces_mesh };
}

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes_cuda_sparse_impl(const stdgpu::device_unique_object<sparse_volume_hash_map>& hashed_indices,
                                const RavelledSparseTensor& volume,
                                const float isolevel,
                                const bool return_local_coords,
                                const bool unique_verts)
{
    at::cuda::CUDAGuard device_guard{ volume.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(volume.device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(volume.device());

    const auto N = volume.values().numel();

    if (N == 0)
    {
        auto verts = torch::zeros({ 0, 3 }, dtype_float);
        auto faces = torch::zeros({ 0, 3 }, dtype_int64);

        return { verts, faces };
    }

    auto sparse_indices = volume.indices();
    auto sparse_values = volume.values();
    const auto full_sizes = glm::i64vec3{ volume.size(1), volume.size(2), volume.size(3) };

    // 1. Find active voxels
    auto verts_per_voxel = torch::empty({ N }, dtype_int64);
    auto occupied_voxel = torch::empty({ N }, dtype_int64);

    const int threads_per_block = 128;
    dim3 grid_volume;
    at::cuda::getApplyGrid(N, grid_volume, volume.device().index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            volume.scalar_type(),
            "classify_voxels_sparse",
            [&]()
            {
                classify_voxels_sparse<<<grid_volume, threads, 0, stream>>>(
                        sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        sparse_values.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        *hashed_indices,
                        full_sizes,
                        N,
                        verts_per_voxel.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        occupied_voxel.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        isolevel);
                AT_CUDA_CHECK(cudaGetLastError());
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    // 2. Make compact buffer for active voxels ids
    auto occupied_voxel_scan = torch::empty({ N }, dtype_int64);

    thrust::exclusive_scan(policy,
                           occupied_voxel.data_ptr<int64_t>(),
                           occupied_voxel.data_ptr<int64_t>() + N,
                           occupied_voxel_scan.data_ptr<int64_t>());

    auto last_element = occupied_voxel[N - 1].cpu().item<int64_t>();
    auto last_scan = occupied_voxel_scan[N - 1].cpu().item<int64_t>();
    const auto num_active_voxels = last_element + last_scan;

    if (num_active_voxels == 0)
    {
        auto verts = torch::zeros({ 0, 3 }, dtype_float);
        auto faces = torch::zeros({ 0, 3 }, dtype_int64);

        return { verts, faces };
    }

    auto compact_voxel_array = torch::empty({ num_active_voxels }, dtype_int64);

    compactify_voxels<<<grid_volume, threads, 0, stream>>>(
            compact_voxel_array.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            occupied_voxel.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            occupied_voxel_scan.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            N);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Release no longer needed tensors early to reduce memory pressure
    occupied_voxel = torch::Tensor{};
    occupied_voxel_scan = torch::Tensor{};

    // 3. Compute per-voxel vertex buffer offsets
    auto verts_per_voxel_scan = torch::empty({ N }, dtype_int64);
    thrust::exclusive_scan(policy,
                           verts_per_voxel.data_ptr<int64_t>(),
                           verts_per_voxel.data_ptr<int64_t>() + N,
                           verts_per_voxel_scan.data_ptr<int64_t>());

    last_element = verts_per_voxel[N - 1].cpu().item<int64_t>();
    last_scan = verts_per_voxel_scan[N - 1].cpu().item<int64_t>();
    const auto num_total_verts = last_element + last_scan;

    // Release no longer needed tensors early to reduce memory pressure
    verts_per_voxel = torch::Tensor{};

    // 4. Generate triangle vertices
    auto verts = torch::Tensor{};
    auto ids = torch::empty({ unique_verts ? num_total_verts : 0 }, dtype_int64);

    verts = torch::empty({ num_total_verts, 3 }, dtype_float);

    dim3 grid_active_volume;
    at::cuda::getApplyGrid(num_active_voxels, grid_active_volume, volume.device().index(), threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            volume.scalar_type(),
            "generate_faces_sparse",
            [&]()
            {
                generate_faces_sparse<<<grid_active_volume, threads, 0, stream>>>(
                        verts.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                        ids.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                        compact_voxel_array.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        verts_per_voxel_scan.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        num_active_voxels,
                        sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                        sparse_values.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        full_sizes,
                        *hashed_indices,
                        isolevel,
                        unique_verts);
                AT_CUDA_CHECK(cudaGetLastError());
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    // Release no longer needed tensors early to reduce memory pressure
    compact_voxel_array = torch::Tensor{};
    verts_per_voxel_scan = torch::Tensor{};

    // 5. Convert triangle soup to compact mesh
    auto verts_mesh = torch::Tensor{};
    auto faces_mesh = torch::Tensor{};
    if (unique_verts)
    {
        const auto sorted = true;
        const auto return_inverse = true;
        const auto return_counts = false;
        auto [unique_ids, inverse_idx, _] = torch::unique_dim(ids, 0, sorted, return_inverse, return_counts);

        verts_mesh = torch::empty({ unique_ids.size(0), 3 }, dtype_float);
        verts_mesh.index_put_({ inverse_idx }, verts);

        faces_mesh = inverse_idx.reshape({ num_total_verts / 3, 3 }).to(dtype_int64);
    }
    else
    {
        verts_mesh = verts;
        faces_mesh = torch::arange(num_total_verts, dtype_int64).reshape({ num_total_verts / 3, 3 });
    }

    if (return_local_coords)
    {
        const auto scale = torch::tensor({ 2.f / (static_cast<float>(volume.size(1) - 1)),
                                           2.f / (static_cast<float>(volume.size(2) - 1)),
                                           2.f / (static_cast<float>(volume.size(3) - 1)) },
                                         dtype_float);

        verts_mesh *= torch::unsqueeze(scale, 0);
        verts_mesh -= 1.f;
    }

    return { verts_mesh, faces_mesh };
}

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes_cuda_sparse(const RavelledSparseTensor& volume,
                           const float isolevel,
                           const bool return_local_coords,
                           const bool unique_verts)
{
    TORCH_CHECK_EQ(volume.size(0), 1);
    TORCH_CHECK_EQ(volume.sparse_dim(), 4);
    TORCH_CHECK_EQ(volume.dense_dim(), 0);
    TORCH_CHECK_LT(volume.values().numel(), (static_cast<int64_t>(1) << 31)); // stdgpu uses 32-bit indices
    TORCH_CHECK_GE(isolevel, 0);

    at::cuda::CUDAGuard device_guard{ volume.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto N = volume.values().numel();

    auto volume_hashed = stdgpu::device_unique_object<sparse_volume_hash_map>{ stdgpu::null_object };
    auto sparse_indices = volume.indices();

    if (N == 0)
    {
        volume_hashed = stdgpu::device_unique_object<sparse_volume_hash_map>{ policy, 1 }; // Need capacity > 0
    }
    else
    {
        volume_hashed = stdgpu::device_unique_object<sparse_volume_hash_map>{ policy, static_cast<int32_t>(N) };

        const int threads_per_block = 128;
        dim3 grid_volume;
        at::cuda::getApplyGrid(N, grid_volume, volume.device().index(), threads_per_block);
        dim3 threads = at::cuda::getApplyBlock(threads_per_block);

        hash_ravelled_volume<<<grid_volume, threads, 0, stream>>>(
                sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                N,
                *volume_hashed);
        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return marching_cubes_cuda_sparse_impl(volume_hashed, volume, isolevel, return_local_coords, unique_verts);
}

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes_cuda_sparse(const torch::Tensor& volume,
                           const float isolevel,
                           const bool return_local_coords,
                           const bool unique_verts)
{
    TORCH_CHECK_EQ(volume.size(0), 1);
    TORCH_CHECK_LE(volume.numel(), (static_cast<int64_t>(1) << 60));
    TORCH_CHECK_EQ(volume.layout(), torch::Layout::Sparse);
    TORCH_CHECK_EQ(volume.sparse_dim(), 4);
    TORCH_CHECK_EQ(volume.dense_dim(), 0);
    TORCH_CHECK_LT(volume.values().numel(), (static_cast<int64_t>(1) << 31)); // stdgpu uses 32-bit indices
    TORCH_CHECK_GE(isolevel, 0);

    at::cuda::CUDAGuard device_guard{ volume.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(volume.device());

    const auto N = volume.values().numel();
    const auto full_sizes = glm::i64vec3{ volume.size(1), volume.size(2), volume.size(3) };

    auto volume_hashed = stdgpu::device_unique_object<sparse_volume_hash_map>{ stdgpu::null_object };
    auto sparse_indices = torch::empty({ N }, dtype_int64);

    if (N == 0)
    {
        volume_hashed = stdgpu::device_unique_object<sparse_volume_hash_map>{ policy, 1 }; // Need capacity > 0
    }
    else
    {
        volume_hashed = stdgpu::device_unique_object<sparse_volume_hash_map>{ policy, static_cast<int32_t>(N) };

        const int threads_per_block = 128;
        dim3 grid_volume;
        at::cuda::getApplyGrid(N, grid_volume, volume.device().index(), threads_per_block);
        dim3 threads = at::cuda::getApplyBlock(threads_per_block);

        auto volume_indices = volume.indices();

        hash_volume<<<grid_volume, threads, 0, stream>>>(
                volume_indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                full_sizes,
                N,
                sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                *volume_hashed);
        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return marching_cubes_cuda_sparse_impl(
            volume_hashed,
            { sparse_indices, volume.values(), { 1, volume.size(1), volume.size(2), volume.size(3) } },
            isolevel,
            return_local_coords,
            unique_verts);
}

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes_cuda(const torch::Tensor& volume,
                    const float isolevel,
                    const bool return_local_coords,
                    const bool unique_verts)
{
    if (volume.is_sparse())
    {
        return marching_cubes_cuda_sparse(volume, isolevel, return_local_coords, unique_verts);
    }
    else
    {
        return marching_cubes_cuda_dense(volume, isolevel, return_local_coords, unique_verts);
    }
}

} // namespace torchhull
