cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(hello main.cu)
add_executable(vector_add vector_add.cu)
add_executable(matrix_mul matrix_mul.cu)
add_executable(reduction reduction.cu)
add_executable(stream_example stream_example.cu)
add_executable(unified_memory unified_memory.cu)

set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_mul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(reduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stream_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unified_memory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

