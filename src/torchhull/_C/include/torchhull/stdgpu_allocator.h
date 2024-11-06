#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <cstddef>

#include <stdgpu/memory.h>

namespace torchhull
{

template <typename T>
class StdgpuAllocator
{
public:
    using value_type = T;

    StdgpuAllocator() noexcept = default;

    ~StdgpuAllocator() noexcept = default;

    StdgpuAllocator(const StdgpuAllocator&) noexcept = default;

    template <typename U>
    explicit StdgpuAllocator(const StdgpuAllocator<U>&) noexcept
    {
    }

    StdgpuAllocator&
    operator=(const StdgpuAllocator&) noexcept = default;

    StdgpuAllocator(StdgpuAllocator&&) noexcept = default;

    StdgpuAllocator&
    operator=(StdgpuAllocator&&) noexcept = default;

    [[nodiscard]] T*
    allocate(stdgpu::index64_t n)
    {
        T* p = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(static_cast<std::size_t>(n) * sizeof(T)));
        stdgpu::register_memory(p, n, stdgpu::dynamic_memory_type::device);
        return p;
    }

    void
    deallocate(T* p, stdgpu::index64_t n)
    {
        stdgpu::deregister_memory(p, n, stdgpu::dynamic_memory_type::device);
        c10::cuda::CUDACachingAllocator::raw_delete(p);
    }
};

} // namespace torchhull
