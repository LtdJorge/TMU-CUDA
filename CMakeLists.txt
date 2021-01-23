cmake_minimum_required(VERSION 3.17)
project(TMU_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_library(TMU_CUDA SHARED library.cu library.cuh)

set_target_properties(
        TMU_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)