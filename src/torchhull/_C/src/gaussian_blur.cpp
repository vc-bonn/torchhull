#include <torchhull/gaussian_blur.h>

#include <c10/util/Exception.h>

namespace torchhull
{

torch::Tensor
gaussian_blur_cuda(const torch::Tensor& images,
                   const int kernel_size,
                   const float sigma,
                   const bool sparse,
                   const std::optional<torch::ScalarType> dtype);

torch::Tensor
gaussian_blur(const torch::Tensor& images,
              const int kernel_size,
              const float sigma,
              const bool sparse,
              const std::optional<torch::ScalarType> dtype)
{
    if (images.is_cuda())
    {
        return gaussian_blur_cuda(images, kernel_size, sigma, sparse, dtype);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + images.device().str() + "\".");
}

} // namespace torchhull
