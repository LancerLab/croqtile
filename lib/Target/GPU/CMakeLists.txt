add_custom_target(
  gpu_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/choreo_cute_header.inc
  ${CMAKE_BINARY_DIR}/choreo_header.inc
)

set(GPU_TARGET_SOURCES
  cute_codegen.cpp
  cute_target.cpp
)

add_library(GPUTarget OBJECT ${GPU_TARGET_SOURCES})
add_dependencies(GPUTarget gpu_deps)

target_include_directories(GPUTarget
  PUBLIC include
  PRIVATE ${CMAKE_SOURCE_DIR}
)

if(ENABLE_CUDA)
  target_include_directories(GPUTarget PRIVATE ${_cuda_root}/include)
  link_directories(${_cuda_root}/lib64 ${_cuda_root}/targets/x86_64-linux/lib)

  target_compile_definitions(GPUTarget PRIVATE
    __CHOREO_CUDA_DIR__=${_cuda_root}
  )

  math(EXPR CUDA_VERSION_NUMBER
       "${CUDA_VERSION_MAJOR} * 1000 + ${CUDA_VERSION_MINOR} * 10")
  target_compile_definitions(GPUTarget PRIVATE
      CHOREO_CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}
      CHOREO_CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR}
      CHOREO_CUDA_VERSION=${CUDA_VERSION_NUMBER}
  )
endif()

if(ENABLE_CUTE)
  target_include_directories(GPUTarget PRIVATE
    ${_cute_root}/include
    ${_cute_root}/tools/util/include  # for lightweight helpers (optional)
  )

  target_compile_definitions(GPUTarget PRIVATE
    __CHOREO_CUTE_DIR__=${_cute_root}
  )
endif()

