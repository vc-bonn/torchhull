#pragma once

#include <optional>

#include <torch/types.h>

namespace torchhull
{

torch::Tensor
gaussian_blur(const torch::Tensor& images,
              const int kernel_size,
              const float sigma,
              const bool sparse = true,
              const std::optional<torch::ScalarType> dtype = std::nullopt);

} // namespace torchhull
