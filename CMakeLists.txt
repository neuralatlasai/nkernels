cmake_minimum_required(VERSION 3.24)

project(nkernels LANGUAGES CXX)

option(NKERNELS_ENABLE_CUDA "Build CUDA kernel infrastructure." ON)
option(NKERNELS_ENABLE_ROCM "Enable ROCm-compatible runtime gates." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(NKERNELS_CORE_SOURCES
    csrc/extensions/api_surface.cpp
    csrc/extensions/registry.cpp)

if(NKERNELS_ENABLE_CUDA)
  enable_language(CUDA)
  list(APPEND NKERNELS_CORE_SOURCES
       csrc/hardware_architecture/device_profile.cu
       csrc/architecture/compound/scaled_gemm_dispatch.cu)
endif()

add_library(nkernels_core STATIC ${NKERNELS_CORE_SOURCES})

target_include_directories(
  nkernels_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

target_compile_features(nkernels_core PUBLIC cxx_std_17)
target_compile_definitions(
  nkernels_core
  PUBLIC $<$<BOOL:${NKERNELS_ENABLE_ROCM}>:NKERNELS_ENABLE_ROCM=1>)

set_target_properties(nkernels_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NKERNELS_ENABLE_CUDA)
  set_target_properties(
    nkernels_core
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
