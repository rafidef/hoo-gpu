cmake_minimum_required(VERSION 3.18)
project(hoosat_h100_miner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# H100 = sm_90 (Hopper architecture)
set(CMAKE_CUDA_ARCHITECTURES 90)

# CUDA flags — NO --use_fast_math! HooHash requires exact FP64 precision
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Enable separable compilation so device functions work across .cu files
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find CUDA toolkit (for cuBLAS)
find_package(CUDAToolkit REQUIRED)

# Main miner executable
add_executable(hoosat_h100_miner
    src/main.cu
    src/hoohash.cu
    src/blake3_cuda.cu
    src/stratum.cpp
)

target_include_directories(hoosat_h100_miner PRIVATE src/)

target_link_libraries(hoosat_h100_miner PRIVATE
    CUDA::cublas
    CUDA::cudart
)

# Platform-specific networking
if(WIN32)
    target_link_libraries(hoosat_h100_miner PRIVATE ws2_32)
    target_compile_definitions(hoosat_h100_miner PRIVATE _WIN32_WINNT=0x0601)
else()
    target_link_libraries(hoosat_h100_miner PRIVATE pthread)
endif()

# Test executable
add_executable(test_hoohash
    test/test_hoohash.cu
    src/hoohash.cu
    src/blake3_cuda.cu
)

target_include_directories(test_hoohash PRIVATE src/)

target_link_libraries(test_hoohash PRIVATE
    CUDA::cublas
    CUDA::cudart
)

if(WIN32)
    target_link_libraries(test_hoohash PRIVATE ws2_32)
endif()
