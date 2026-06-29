
project(_gpusim2grid LANGUAGES CXX CUDA)
find_package(cudss 0.8.0 REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cublas cudart cusolver cusparse)
pybind11_add_module(
    _gpusim2grid
    python_bindings.cpp
    cuda_utils.cu
    acpf_nr.cu
    contingency_analysis_helper.cpp
    acpf_nr_kernels.cu
    contingency_analysis_session.cu
    contingency/batch_pf_driver.cu
    contingency/batch_sources/injection_batch.cu
    injection_sweep.cu
    injection_sweep_session.cu
    dlpack_export.cu)
target_link_libraries(_gpusim2grid PRIVATE CUDA::cusparse cudss)
install(TARGETS _gpusim2grid DESTINATION ${SKBUILD_PROJECT_NAME})
