#pragma once

#include <c10/macros/Macros.h>
#include <glm/common.hpp>
#include <glm/vec3.hpp>
#include <torch/types.h>

#include <torchhull/cuda_utils.h>
#include <torchhull/math.h>

namespace torchhull
{

inline C10_HOST_DEVICE int64_t
numel(const glm::i64vec3& sizes)
{
    return int64_t{ sizes.x * sizes.y * sizes.z };
}

inline C10_HOST_DEVICE glm::i64vec3
unravel_index(const int64_t index, const glm::i64vec3& shape)
{
    CUDA_DEVICE_CHECK(index >= 0);
    CUDA_DEVICE_CHECK(index < numel(shape));

    auto multi_index = glm::i64vec3{ index % shape.x, index / shape.x % shape.y, index / (shape.x * shape.y) };

    CUDA_DEVICE_CHECK(multi_index.x >= 0);
    CUDA_DEVICE_CHECK(multi_index.x < shape.x);
    CUDA_DEVICE_CHECK(multi_index.y >= 0);
    CUDA_DEVICE_CHECK(multi_index.y < shape.y);
    CUDA_DEVICE_CHECK(multi_index.z >= 0);
    CUDA_DEVICE_CHECK(multi_index.z < shape.z);

    return multi_index;
}

inline C10_HOST_DEVICE int64_t
ravel_multi_index(const glm::i64vec3& multi_index, const glm::i64vec3& shape)
{
    CUDA_DEVICE_CHECK(multi_index.x >= 0);
    CUDA_DEVICE_CHECK(multi_index.x < shape.x);
    CUDA_DEVICE_CHECK(multi_index.y >= 0);
    CUDA_DEVICE_CHECK(multi_index.y < shape.y);
    CUDA_DEVICE_CHECK(multi_index.z >= 0);
    CUDA_DEVICE_CHECK(multi_index.z < shape.z);

    auto index = int64_t{ multi_index.x + multi_index.y * shape.x + multi_index.z * shape.x * shape.y };

    CUDA_DEVICE_CHECK(index >= 0);
    CUDA_DEVICE_CHECK(index < numel(shape));

    return index;
}

inline C10_HOST_DEVICE bool
in_image(const int64_t y, const int64_t x, const int64_t height, const int64_t width)
{
    return 0 <= y && y < height && 0 <= x && x < width;
}

// false refers to torch.nn.functional.grid_sample()'s align_corners=false
inline C10_HOST_DEVICE float
unnormalize_ndc_false(const float coordinate, const int64_t size)
{
    return (coordinate + 1.f) * static_cast<float>(size) / 2.f - 0.5f;
}

template <typename ValueT>
inline C10_DEVICE ValueT
sample_zeros_padding(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> image,
                     const int64_t y,
                     const int64_t x,
                     const int batch,
                     const int channel)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = image.size(1);
    const auto W = image.size(2);

    return in_image(y, x, H, W) ? image[batch][y][x][channel] : ValueT{ 0 };
}

template <typename ValueT>
inline C10_DEVICE ValueT
sample_ones_padding(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> image,
                    const int64_t y,
                    const int64_t x,
                    const int batch,
                    const int channel)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = image.size(1);
    const auto W = image.size(2);

    return in_image(y, x, H, W) ? image[batch][y][x][channel] : ValueT{ 1 };
}

template <typename ValueT>
inline C10_DEVICE ValueT
sample_border_padding(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> image,
                      const int64_t y,
                      const int64_t x,
                      const int batch,
                      const int channel)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = image.size(1);
    const auto W = image.size(2);

    return image[batch][glm::clamp<int64_t>(y, 0, H - 1)][glm::clamp<int64_t>(x, 0, W - 1)][channel];
}

template <typename ValueT>
inline C10_DEVICE ValueT
sample_reflect_padding(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> image,
                       const int64_t y,
                       const int64_t x,
                       const int batch,
                       const int channel)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = image.size(1);
    const auto W = image.size(2);

    // NOTE
    // ----
    // The border value is NOT reflected, only the "inner" part should be considered.
    //
    // Example: N = 5
    //
    // Input:     0 1 2 3 4
    // Reflected: 0 1 2 3 | 4 3 2 1 | 0 1 2 3 | 4 3 2 1 | 0 1 2 3 | ...
    //
    // The negative direction works exactly the same, but starts with 1 instead of 0 flips.
    const auto flips_y = y / (H - 1) + (y < 0);
    const auto flips_x = x / (W - 1) + (x < 0);
    const auto remainder_y = mod(y, H - 1);
    const auto remainder_x = mod(x, W - 1);

    const auto index_y = (flips_y % 2 == 0) ? remainder_y : (H - 1) - remainder_y;
    const auto index_x = (flips_x % 2 == 0) ? remainder_x : (W - 1) - remainder_x;

    return image[batch][index_y][index_x][channel];
}

template <typename ValueT>
inline C10_DEVICE float
sample_bilinear_mode_zeros_padding(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> image,
                                   const float y,
                                   const float x,
                                   const int batch,
                                   const int channel)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = image.size(1);
    const auto W = image.size(2);

    const auto ROUND_DOWN = -0.5f;
    auto y_pixel = static_cast<int64_t>(roundf(y + ROUND_DOWN));
    auto x_pixel = static_cast<int64_t>(roundf(x + ROUND_DOWN));

    auto image_00 = static_cast<float>(sample_zeros_padding(image, y_pixel, x_pixel, batch, channel));
    auto image_10 = static_cast<float>(sample_zeros_padding(image, y_pixel + 1, x_pixel, batch, channel));
    auto image_01 = static_cast<float>(sample_zeros_padding(image, y_pixel, x_pixel + 1, batch, channel));
    auto image_11 = static_cast<float>(sample_zeros_padding(image, y_pixel + 1, x_pixel + 1, batch, channel));

    auto t_y = y - static_cast<float>(y_pixel);
    auto t_x = x - static_cast<float>(x_pixel);

    return bilerp(image_00, image_10, image_01, image_11, t_y, t_x);
}

template <typename ValueT>
inline C10_DEVICE float
sample_bilinear_mode_ones_padding(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> image,
                                  const float y,
                                  const float x,
                                  const int batch,
                                  const int channel)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = image.size(1);
    const auto W = image.size(2);

    const auto ROUND_DOWN = -0.5f;
    auto y_pixel = static_cast<int64_t>(roundf(y + ROUND_DOWN));
    auto x_pixel = static_cast<int64_t>(roundf(x + ROUND_DOWN));

    auto image_00 = static_cast<float>(sample_ones_padding(image, y_pixel, x_pixel, batch, channel));
    auto image_10 = static_cast<float>(sample_ones_padding(image, y_pixel + 1, x_pixel, batch, channel));
    auto image_01 = static_cast<float>(sample_ones_padding(image, y_pixel, x_pixel + 1, batch, channel));
    auto image_11 = static_cast<float>(sample_ones_padding(image, y_pixel + 1, x_pixel + 1, batch, channel));

    auto t_y = y - static_cast<float>(y_pixel);
    auto t_x = x - static_cast<float>(x_pixel);

    return bilerp(image_00, image_10, image_01, image_11, t_y, t_x);
}

template <typename ValueT>
inline C10_DEVICE float
sample_bilinear_mode_border_padding(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> image,
                                    const float y,
                                    const float x,
                                    const int batch,
                                    const int channel)
{
    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = image.size(1);
    const auto W = image.size(2);

    const auto ROUND_DOWN = -0.5f;
    auto y_pixel = static_cast<int64_t>(roundf(y + ROUND_DOWN));
    auto x_pixel = static_cast<int64_t>(roundf(x + ROUND_DOWN));

    auto image_00 = static_cast<float>(sample_border_padding(image, y_pixel, x_pixel, batch, channel));
    auto image_10 = static_cast<float>(sample_border_padding(image, y_pixel + 1, x_pixel, batch, channel));
    auto image_01 = static_cast<float>(sample_border_padding(image, y_pixel, x_pixel + 1, batch, channel));
    auto image_11 = static_cast<float>(sample_border_padding(image, y_pixel + 1, x_pixel + 1, batch, channel));

    auto t_y = y - static_cast<float>(y_pixel);
    auto t_x = x - static_cast<float>(x_pixel);

    return bilerp(image_00, image_10, image_01, image_11, t_y, t_x);
}

} // namespace torchhull
