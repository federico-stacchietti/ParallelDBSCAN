cmake_minimum_required(VERSION 3.18)

project(ParallelDBSCAN LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(parallel_dbscan
    main.cpp
    sequential.cpp
    utils.cpp
    kernels.cu
)

target_compile_features(parallel_dbscan PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(parallel_dbscan PRIVATE CUDA::cublas CUDA::cudart)

if(MSVC)
    target_compile_options(parallel_dbscan PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(parallel_dbscan PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()
