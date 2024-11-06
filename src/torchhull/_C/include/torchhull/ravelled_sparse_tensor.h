#pragma once

#include <cstdint>
#include <vector>

#include <c10/util/Exception.h>
#include <torch/types.h>

namespace torchhull
{

class RavelledSparseTensor
{
public:
    RavelledSparseTensor() = default;

    inline RavelledSparseTensor(const torch::Tensor& indices,
                                const torch::Tensor& values,
                                const std::vector<int64_t>& sizes)
      : indices_(indices)
      , values_(values)
      , sizes_(sizes)
    {
        TORCH_CHECK_EQ(indices.device(), values.device());
        TORCH_CHECK_EQ(indices.numel(), values.numel());
        TORCH_CHECK_EQ(indices.dim(), 1);
        TORCH_CHECK_EQ(values.dim(), 1);
    }

    inline torch::Device
    device() const
    {
        return indices_.device();
    }

    inline torch::Tensor
    indices() const
    {
        return indices_;
    }

    inline torch::Tensor
    values() const
    {
        return values_;
    }

    inline int64_t
    sparse_dim() const
    {
        return sizes_.size();
    }

    inline int64_t
    dense_dim() const
    {
        return 0;
    }

    inline int64_t
    size(const int64_t i) const
    {
        TORCH_CHECK_GE(i, 0);
        TORCH_CHECK_LT(i, sparse_dim());

        return sizes_[i];
    }

    inline torch::ScalarType
    scalar_type() const
    {
        return values_.scalar_type();
    }

private:
    torch::Tensor indices_;
    torch::Tensor values_;
    std::vector<int64_t> sizes_;
};

} // namespace torchhull
