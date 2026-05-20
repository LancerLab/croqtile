set(AMDGPU_TARGET_SOURCES
  hip_target.cpp
  hip_codegen.cpp
  hip_dma_plan.cpp
)

add_library(AMDGPUTarget OBJECT ${AMDGPU_TARGET_SOURCES})

target_include_directories(AMDGPUTarget
  PUBLIC include
  PRIVATE ${CMAKE_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/lib/Target/GPU
)

add_dependencies(AMDGPUTarget gpu_deps)

if(ENABLE_CUDA AND DEFINED CUDA_VERSION_MAJOR)
  math(EXPR _amdgpu_cuda_ver
       "${CUDA_VERSION_MAJOR} * 1000 + ${CUDA_VERSION_MINOR} * 10")
  target_compile_definitions(AMDGPUTarget PRIVATE
    CHOREO_CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}
    CHOREO_CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR}
    CHOREO_CUDA_VERSION=${_amdgpu_cuda_ver}
  )
endif()

if(ENABLE_HIP)
  target_include_directories(AMDGPUTarget PRIVATE ${_rocm_root}/include)

  target_compile_definitions(AMDGPUTarget PRIVATE
    __CHOREO_ROCM_DIR__=${_rocm_root}
  )

  if(HIP_VERSION_MAJOR AND HIP_VERSION_MINOR)
    math(EXPR HIP_VERSION_NUMBER
         "${HIP_VERSION_MAJOR} * 1000 + ${HIP_VERSION_MINOR} * 10")
    target_compile_definitions(AMDGPUTarget PRIVATE
      CHOREO_HIP_VERSION_MAJOR=${HIP_VERSION_MAJOR}
      CHOREO_HIP_VERSION_MINOR=${HIP_VERSION_MINOR}
      CHOREO_HIP_VERSION=${HIP_VERSION_NUMBER}
    )
  endif()
endif()
