cmake_minimum_required(VERSION 3.27)

project(torchhull LANGUAGES CXX CUDA)


if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.1)
    message(FATAL_ERROR "CUDA 12.1+ required but only ${CMAKE_CUDA_COMPILER_VERSION} detected.")
endif()


option(TORCHHULL_BUILD_BINDINGS "Build the Python bindings" ON)


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND HOST_DEVICE_FLAGS "-Wall")
    list(APPEND HOST_DEVICE_FLAGS "-Wextra")
    list(APPEND HOST_DEVICE_FLAGS "-Wshadow")
    list(APPEND HOST_DEVICE_FLAGS "-Wsign-compare")
    list(APPEND HOST_DEVICE_FLAGS "-Wconversion")
    list(APPEND HOST_DEVICE_FLAGS "-Wfloat-equal")
    list(APPEND HOST_DEVICE_FLAGS "-Wundef")
    list(APPEND HOST_DEVICE_FLAGS "-Wdouble-promotion")
endif()


include(FetchContent)

if(NOT TARGET glm::glm)
    FetchContent_Declare(
        glm
        PREFIX glm
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
        URL_HASH SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
        DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/external/glm"
        SYSTEM
    )

    FetchContent_MakeAvailable(glm)
endif()

if(NOT TARGET stdgpu::stdgpu)
    FetchContent_Declare(
        stdgpu
        PREFIX stdgpu
        URL https://github.com/stotko/stdgpu/archive/3a0b20e77a5eac672162fa5f6173ce9a34303d7f.tar.gz
        URL_HASH SHA256=4723bba67ccb67f3a0218515f555c4ed385ae2f638cf668b81d6d490c1f47fbc
        DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/external/stdgpu"
        SYSTEM
    )

    # Calling thrust::transform_reduce in bitset::count(), which in turn gets called in a contract check,
    # results in CUDA invalidDeviceFunction errors, so disable the checks to reduce the attack surface.
    set(STDGPU_ENABLE_CONTRACT_CHECKS OFF CACHE INTERNAL "")

    set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
    set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

    FetchContent_MakeAvailable(stdgpu)
endif()


find_package(charonload)

if(charonload_FOUND)
    charonload_add_torch_library(torchhull_cpp STATIC)
    add_library(torchhull::torchhull_cpp ALIAS torchhull_cpp)

    target_sources(torchhull_cpp PRIVATE src/gaussian_blur_cuda.cu
                                         src/gaussian_blur.cpp
                                         src/io.cpp
                                         src/marching_cubes_cuda.cu
                                         src/marching_cubes.cpp
                                         src/visual_hull_cuda.cu
                                         src/visual_hull.cpp)

    target_include_directories(torchhull_cpp PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_compile_definitions(torchhull_cpp PRIVATE "__CUDA_NO_HALF_OPERATORS__")
    target_compile_features(torchhull_cpp PUBLIC cxx_std_17)
    target_compile_options(torchhull_cpp PRIVATE ${HOST_DEVICE_FLAGS})
    target_link_libraries(torchhull_cpp PRIVATE glm::glm stdgpu::stdgpu)


    if(TORCHHULL_BUILD_BINDINGS)
        charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)

        target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/bindings.cpp)

        target_compile_options(${TORCH_EXTENSION_NAME} PRIVATE ${HOST_DEVICE_FLAGS})
        target_link_libraries(${TORCH_EXTENSION_NAME} PRIVATE torchhull_cpp)
    endif()
endif()
