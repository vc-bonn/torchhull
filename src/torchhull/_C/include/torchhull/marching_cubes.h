#pragma once

#include <tuple>

#include <torch/types.h>

namespace torchhull
{

std::tuple<torch::Tensor, torch::Tensor>
marching_cubes(const torch::Tensor& volume,
               const float isolevel,
               const bool return_local_coords,
               const bool unique_verts);

} // namespace torchhull
