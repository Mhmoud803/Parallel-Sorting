cmake_minimum_required(VERSION 3.18)
project(project2_sort LANGUAGES CXX CUDA)

# ── Language standards ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Build type default ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug|Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# GPU architecture — override at configure time with:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
# Common values: 75 (Turing/T4), 80 (Ampere/A100), 86 (RTX 30xx), 89 (Ada/RTX 40xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ── Source files ─────────────────────────────────────────────────────────────
set(CXX_SOURCES
    src/main.cpp
    src/cli/cli.cpp
    src/data/generator.cpp
    src/sort/serial_sort.cpp
    src/sort/omp_sort.cpp
)

set(CUDA_SOURCES
    src/sort/cuda_sort.cu
)

# ── Executable target ─────────────────────────────────────────────────────────
add_executable(sort_bench ${CXX_SOURCES} ${CUDA_SOURCES})

# Make all headers resolvable relative to the src/ tree
target_include_directories(sort_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(sort_bench
    PRIVATE
        OpenMP::OpenMP_CXX
        CUDA::cudart
)

target_compile_options(sort_bench PRIVATE
    # Host compiler flags
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -Wall
        -Wextra
        -Wpedantic
    >
    # Device compiler flags
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xcompiler=-Wall
    >
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS sort_bench DESTINATION bin)
