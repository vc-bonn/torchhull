#pragma once

#include <string>
#include <tuple>

#include <torch/types.h>

namespace torchhull
{

void
store_curve_network(const std::string& filename,
                    const std::tuple<torch::Tensor, torch::Tensor>& curve_network,
                    const bool verbose = false);

} // namespace torchhull
