cmake_minimum_required(VERSION 3.8)
project(hpc3 CUDA)

set(CMAKE_HOME_DIRECTORY  /home-mscluster/malence/hpc3/cmake-build-debug-remote-host)
set(CMAKE_CUDA_STANDARD 14)
include_directories(/usr/local/cuda-10.0-alternative/NVIDIA_CUDA-10.0_Samples/common/inc)
add_executable(hpc3 main.cu)

set_target_properties(
        hpc3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)