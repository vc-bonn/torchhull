#pragma once

#include <cstdio>
#include <exception>

#include <c10/macros/Macros.h>

namespace torchhull
{

// ---
#define ENABLE_CUDA_DEVICE_CHECKS 0
// ---

inline C10_HOST_DEVICE void
throw_device_exception(const char* file, const int line, const char* function, const char* condition)
{
    printf("%s:%d: %s: \"%s\" failed!\n", file, line, function, condition);
#if defined(__CUDA_ARCH__)
    asm("trap;");
#else
    std::terminate();
#endif
}

#define REQUIRE_SEMICOLON (void)0

#if ENABLE_CUDA_DEVICE_CHECKS
    #define CUDA_DEVICE_CHECK(...)                                                                                     \
        if (!(__VA_ARGS__))                                                                                            \
        {                                                                                                              \
            throw_device_exception(__FILE__, __LINE__, __func__, #__VA_ARGS__);                                        \
        }                                                                                                              \
        REQUIRE_SEMICOLON
#else
    #define CUDA_DEVICE_CHECK(...)
#endif

#if defined(__CUDACC__)
    #define CUDA_CONSTANT __constant__
#else
    #define CUDA_CONSTANT
#endif

} // namespace torchhull
