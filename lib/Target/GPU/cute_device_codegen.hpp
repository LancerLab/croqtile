#ifndef __CHOREO_CUTE_DEVICE_CODEGEN_HPP__
#define __CHOREO_CUTE_DEVICE_CODEGEN_HPP__

#include "aux.hpp"
#include "choreo_cute_header.inc"
#include "choreo_types_cute_header.inc"
#include "codegen.hpp"
#include <string>
#include <vector>

namespace Choreo {

struct CuteDeviceCodeGen : public DeviceCodeGen {
  std::string DeviceName() const override { return "gpu"; }
  std::string TargetName() const override { return "cute"; }
  std::string SourceExtension() const override { return ".cu"; }
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
    out << "\n# Find CUDA / nvcc\n";
    out << "if [[ -z \"${CUDA_HOME}\" ]]; then\n";
#ifdef __CHOREO_CUDA_DIR__
    out << "  if [[ -d " << STRINGIZE(__CHOREO_CUDA_DIR__) << " ]]; then\n";
    out << "    CUDA_HOME=" << STRINGIZE(__CHOREO_CUDA_DIR__) << "\n";
    out << "  else\n";
#endif
    out << R"script(  FOUND_PATH=$(which "nvcc" 2>/dev/null)
  if [ -n "$FOUND_PATH" ]; then
    CUDA_HOME=$(dirname "$(dirname "$FOUND_PATH")")
  elif [[ -f /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  fi
)script";
#ifdef __CHOREO_CUDA_DIR__
    out << "  fi\n";
#endif
    out << R"script(fi

if [[ -z "${CUDA_HOME}" ]]; then
  echo "failed to find the CUDA installation."
  echo "install CUDA or set CUDA_HOME to CUDA installation directory."
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

    // CUTE_HOME for cutlass/cute headers
    out << "if [[ -z \"${CUTE_HOME}\" ]]; then\n";
#ifdef __CHOREO_CUTE_DIR__
    out << "  export CUTE_HOME=" << STRINGIZE(__CHOREO_CUTE_DIR__) << "\n";
#else
    out << R"script(  echo "CUTE_HOME is not set and no default is configured."
  echo "set CUTE_HOME to the cute/cutlass header library path."
  exit 1
)script";
#endif
    out << "fi\n\n";

    out << R"script(if [ ! -f ${CUTE_HOME}/include/cutlass/cutlass.h ]; then
  echo "CUTE_HOME=${CUTE_HOME} does not contain cutlass headers."
  echo "set CUTE_HOME to the cute/cutlass header library path."
  exit 1
fi
)script";
  }

  void EmitSetupFiles(std::ostream& out,
                      const std::string& build_path) const override {
    auto tdir = TargetBuildDir(build_path);
    out << "mkdir -p " << tdir << "\n";
    out << "cat <<'EOF' > " << tdir << "/choreo_types_cute.h\n";
    out << __choreo_types_cute_header_as_string << "\nEOF\n\n";
    out << "cat <<'EOF' > " << tdir << "/choreo_cute.h\n";
    out << __choreo_cute_header_as_string << "\nEOF\n\n";
    out << "cp " << build_path << "/choreo.h " << tdir << "/\n";
    out << "cp " << build_path << "/choreo_types.h " << tdir << "/\n\n";
  }

  void EmitDeviceCompileCommand(std::ostream& out,
                                const std::string& build_path,
                                const std::string& src,
                                const std::string& obj) const override {
    out << "  ${NVCC} -arch ${gpu_arch} -std=c++17"
        << " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__"
        << " -D__USE_CUDA_TYPE__"
        << " -c -I" << TargetBuildDir(build_path) << " -I" << build_path
        << " -I${CUDA_HOME}/include -I${CUTE_HOME}/include " << src << " -o "
        << obj << " || { echo 'Device compilation failed'; exit 1; }\n";
  }

  void EmitHostCompileCommand(std::ostream& out, const std::string& build_path,
                              const std::string& src,
                              const std::string& obj) const override {
    out << "  ${NVCC} -arch ${gpu_arch} -std=c++17"
        << " -c -I" << build_path << " -I${CUDA_HOME}/include " << src << " -o "
        << obj << " || { echo 'Host compilation failed'; exit 1; }\n";
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
