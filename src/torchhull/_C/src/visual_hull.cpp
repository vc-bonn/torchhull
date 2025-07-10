#include <torchhull/visual_hull.h>

#include <memory>

#include <c10/util/Exception.h>
#include <glm/vec3.hpp>

#include <torchhull/ravelled_sparse_tensor.h>

namespace torchhull
{

std::tuple<RavelledSparseTensor, std::vector<torch::Tensor>>
sparse_visual_hull_field_cuda_ravelled(const torch::Tensor& masks,
                                       const torch::Tensor& transforms,
                                       const int level,
                                       const std::array<float, 3>& cube_corner_bfl,
                                       const float cube_length,
                                       const bool masks_partial);

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes_cuda_sparse(const RavelledSparseTensor& sparse_volume,
                           const float isolevel,
                           const bool return_local_coords,
                           const bool unique_verts);

void
to_global_coordinates_and_flip_faces_(std::tuple<torch::Tensor, torch::Tensor>& self,
                                      const std::array<float, 3>& cube_corner_bfl,
                                      const float cube_length,
                                      const glm::i64vec3& resolution);

std::tuple<torch::Tensor, torch::Tensor>
visual_hull_cuda(const torch::Tensor& masks,
                 const torch::Tensor& transforms,
                 const int level,
                 const std::array<float, 3>& cube_corner_bfl,
                 const float cube_length,
                 const bool masks_partial,
                 const bool unique_verts)
{
    auto [volume, _] = sparse_visual_hull_field_cuda_ravelled(masks,
                                                              transforms,
                                                              level,
                                                              cube_corner_bfl,
                                                              cube_length,
                                                              masks_partial);

    auto isolevel = 0.5f;
    auto mesh = marching_cubes_cuda_sparse(volume, isolevel, false, unique_verts);

    to_global_coordinates_and_flip_faces_(mesh, cube_corner_bfl, cube_length, glm::i64vec3{ 1 << level });

    return mesh;
}

std::tuple<torch::Tensor, torch::Tensor>
visual_hull(const torch::Tensor& masks,
            const torch::Tensor& transforms,
            const int level,
            const std::array<float, 3>& cube_corner_bfl,
            const float cube_length,
            const bool masks_partial,
            const bool unique_verts)
{
    if (masks.is_cuda())
    {
        return visual_hull_cuda(masks, transforms, level, cube_corner_bfl, cube_length, masks_partial, unique_verts);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + masks.device().str() + "\".");
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
visual_hull_cuda_with_candidate_voxels_cuda(const torch::Tensor& masks,
                                            const torch::Tensor& transforms,
                                            const int level,
                                            const std::array<float, 3>& cube_corner_bfl,
                                            const float cube_length,
                                            const bool masks_partial,
                                            const bool unique_verts)
{
    auto [volume, candidates_octree] = sparse_visual_hull_field_cuda_ravelled(masks,
                                                                              transforms,
                                                                              level,
                                                                              cube_corner_bfl,
                                                                              cube_length,
                                                                              masks_partial);

    auto isolevel = 0.5f;
    auto mesh = marching_cubes_cuda_sparse(volume, isolevel, false, unique_verts);

    to_global_coordinates_and_flip_faces_(mesh, cube_corner_bfl, cube_length, glm::i64vec3{ 1 << level });

    return { std::get<0>(mesh), std::get<1>(mesh), candidates_octree };
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
visual_hull_with_candidate_voxels(const torch::Tensor& masks,
                                  const torch::Tensor& transforms,
                                  const int level,
                                  const std::array<float, 3>& cube_corner_bfl,
                                  const float cube_length,
                                  const bool masks_partial,
                                  const bool unique_verts)
{
    if (masks.is_cuda())
    {
        return visual_hull_cuda_with_candidate_voxels_cuda(masks,
                                                           transforms,
                                                           level,
                                                           cube_corner_bfl,
                                                           cube_length,
                                                           masks_partial,
                                                           unique_verts);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + masks.device().str() + "\".");
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>>
candidate_voxels_to_wireframes_cuda(const std::vector<torch::Tensor>& candidate_voxels,
                                    const std::array<float, 3>& cube_corner_bfl,
                                    const float cube_length);

std::vector<std::tuple<torch::Tensor, torch::Tensor>>
candidate_voxels_to_wireframes(const std::vector<torch::Tensor>& candidate_voxels,
                               const std::array<float, 3>& cube_corner_bfl,
                               const float cube_length)
{
    TORCH_CHECK_EQ(candidate_voxels.empty(), false);

    if (candidate_voxels[0].is_cuda())
    {
        return candidate_voxels_to_wireframes_cuda(candidate_voxels, cube_corner_bfl, cube_length);
    }

    TORCH_CHECK(false,
                "No backend implementation available for device \"" + candidate_voxels[0].device().str() + "\".");
}

torch::Tensor
sparse_visual_hull_field_cuda(const torch::Tensor& masks,
                              const torch::Tensor& transforms,
                              const int level,
                              const std::array<float, 3>& cube_corner_bfl,
                              const float cube_length,
                              const bool masks_partial);

torch::Tensor
sparse_visual_hull_field(const torch::Tensor& masks,
                         const torch::Tensor& transforms,
                         const int level,
                         const std::array<float, 3>& cube_corner_bfl,
                         const float cube_length,
                         const bool masks_partial)
{
    if (masks.is_cuda())
    {
        return sparse_visual_hull_field_cuda(masks, transforms, level, cube_corner_bfl, cube_length, masks_partial);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + masks.device().str() + "\".");
}

} // namespace torchhull
