#ifndef __CHOREO_CUTE_DEVICE_CODEGEN_HPP__
#define __CHOREO_CUTE_DEVICE_CODEGEN_HPP__

#include "codegen.hpp"
#include <string>
#include <vector>

namespace Choreo {

struct CuteDeviceCodeGen : public DeviceCodeGen {
  std::string DeviceName() const override { return "gpu"; }
  std::string TargetName() const override { return "cute"; }
  bool IsHostDevice() const override { return false; }

  void EmitHostIncludes(std::ostream& os) const override {
    os << "#include <cuda_runtime.h>\n";
  }

  void EmitSync(std::ostream& os, const std::string& indent) override {
    os << indent << "cudaDeviceSynchronize();\n";
  }

  std::string CompileCommand(const std::string& src,
                             const std::string& obj) const override {
    return "${NVCC:-nvcc} -arch " + arch + " -c " + src + " -o " + obj;
  }

  std::string LinkFlags() const override {
    return "-L${CUDA_LIB:-/usr/local/cuda/lib64} -lcudart";
  }

  void SetupBuildEnv(std::ostream& out) const override {
    out << R"script(
# Find CUDA / nvcc
if [[ -z "${CUDA_HOME}" ]]; then
  FOUND_PATH=$(which "nvcc" 2>/dev/null)
  if [ -n "$FOUND_PATH" ]; then
    CUDA_HOME=$(dirname "$(dirname "$FOUND_PATH")")
  elif [[ -f /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  fi
fi

if [[ -z "${CUDA_HOME}" ]]; then
  echo "failed to find nvcc. install CUDA or set CUDA_HOME."
  exit 1
fi

NVCC=${CUDA_HOME}/bin/nvcc
CUDA_LIB=${CUDA_HOME}/lib64
export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"

# Ensure the host compiler's libstdc++ is on LD_LIBRARY_PATH at runtime
_host_cxx=$( ${NVCC} -v /dev/null -o /dev/null 2>&1 | grep -oP '(?<=-ccbin\s)\S+' || true )
if [[ -z "${_host_cxx}" ]]; then _host_cxx=$(which g++ 2>/dev/null || true); fi
if [[ -n "${_host_cxx}" ]]; then
  _host_libdir=$(dirname "$("${_host_cxx}" -print-file-name=libstdc++.so 2>/dev/null)" || true)
  if [[ -d "${_host_libdir}" ]]; then
    export LD_LIBRARY_PATH="${_host_libdir}:${LD_LIBRARY_PATH}"
  fi
fi

# Detect GPU arch
gpu_arch="${GPU_ARCH:-}"
if [[ -z "${gpu_arch}" ]]; then
  _gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)
  if [[ -n "${_gpu_cc}" ]]; then
    gpu_arch="sm_${_gpu_cc}"
  else
    gpu_arch="sm_86"
  fi
fi

)script";
  }

  void EmitHostCompileCommand(std::ostream& out,
                              const std::string& build_path,
                              const std::string& src,
                              const std::string& obj) const override {
    out << "  ${NVCC} -arch ${gpu_arch} -std=c++17 -c -I" << build_path
        << " -I${CUDA_HOME}/include " << src << " -o " << obj
        << " || { echo 'Host compilation failed'; exit 1; }\n";
  }

  void EmitLinkCommand(std::ostream& out,
                       const std::vector<std::string>& obj_files,
                       const std::string& exe_file) const override {
    out << "  ${NVCC} -arch ${gpu_arch} -std=c++17";
    for (auto& o : obj_files) out << " " << o;
    out << " " << LinkFlags() << " -lm -lpthread -o " << exe_file
        << " || { echo 'Linking failed'; exit 1; }\n";
  }

  void SetArch(const std::string& a) { arch = a; }

private:
  std::string arch = "sm_86";
};

} // namespace Choreo

#endif // __CHOREO_CUTE_DEVICE_CODEGEN_HPP__
