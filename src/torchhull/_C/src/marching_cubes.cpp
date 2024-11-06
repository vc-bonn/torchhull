#include <torchhull/marching_cubes.h>

#include <c10/util/Exception.h>

namespace torchhull
{

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes_cuda(const torch::Tensor& volume,
                    const float isolevel,
                    const bool return_local_coords,
                    const bool unique_verts);

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes(const torch::Tensor& volume,
               const float isolevel,
               const bool return_local_coords,
               const bool unique_verts)
{
    if (volume.is_cuda())
    {
        return marching_cubes_cuda(volume, isolevel, return_local_coords, unique_verts);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + volume.device().str() + "\".");
}

} // namespace torchhull
