cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(hello main.cu)
add_executable(vector_add vector_add.cu)



set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
