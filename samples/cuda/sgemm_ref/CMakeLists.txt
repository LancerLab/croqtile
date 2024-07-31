# generate executable
add_executable(sgemm
    sgemm_main.cu
    runner.cu
)
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
