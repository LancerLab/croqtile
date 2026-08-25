#ifndef __CHOREO_TOPSCC_DEVICE_CODEGEN_HPP__
#define __CHOREO_TOPSCC_DEVICE_CODEGEN_HPP__

#include "aux.hpp"
#include "codegen.hpp"
#include "topscc_header.inc"
#include <string>
#include <vector>

namespace Choreo {

struct TopsccDeviceCodeGen : public DeviceCodeGen {
  std::string DeviceName() const override { return "gcu"; }
  std::string TargetName() const override { return "topscc"; }
  bool IsHostDevice() const override { return false; }

  void EmitHostIncludes(std::ostream& os) const override {
    os << "#include \"tops/tops_ext.h\"\n";
    os << "#include \"tops/tops_runtime.h\"\n";
    os << "#if __GCU_ARCH__ >= 300\n";
    os << "#include \"tcle.h\"\n";
    os << "#endif\n";
  }

  void EmitSync(std::ostream& os, const std::string& indent) override {
    os << indent << "choreo::abend_true(topsDeviceSynchronize());\n";
  }

  std::string CompileCommand(const std::string& src,
                             const std::string& obj) const override {
    return "${TOPSCC:-topscc} -arch " + arch + " -c " + src + " -o " + obj;
  }

  std::string LinkFlags() const override {
    return "-L${TOPSCC_LIB:-/opt/tops/lib} -ltops";
  }

  void SetupBuildEnv(std::ostream& out) const override {
    // Pre-set the arch from an explicit -arch flag so the generated script
    // skips JIT device detection (which otherwise falls back to gcu300 and
    // drops arch-gated builtins such as tcle::atomic_*).
    if (CCtx().IsArchSet())
      out << "export GCU_ARCH=" << CCtx().GetArch() << "\n";
    out << "\n# Find topscc\n";
    out << "if [[ -z \"${TOPSCC_INSTALL}\" ]]; then\n";
#ifdef __CHOREO_TOPSCC_SIM_DIR__
    if (CCtx().IsSimArch()) {
      // Prefer the isolated simulator toolchain when compiling for a sim arch,
      // falling back to the native toolchain when the sim dir has no topscc
      // (e.g. the gcu5 simulator reuses the native topscc).
      out << "  if [[ -f "
          << STRINGIZE(__CHOREO_TOPSCC_SIM_DIR__) << "/bin/topscc ]]; then\n";
      out << "    TOPSCC_INSTALL="
          << STRINGIZE(__CHOREO_TOPSCC_SIM_DIR__) << "\n";
      out << "  elif [[ -d "
          << STRINGIZE(__CHOREO_TOPSCC_DIR__) << " ]]; then\n";
      out << "    TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__) << "\n";
      out << "  else\n";
      out << "if [[ -z \"${GCU_SIM_LIB}\" ]]; then\n";
      out << "  GCU_SIM_LIB="
          << STRINGIZE(__CHOREO_TOPSCC_SIM_DIR__) << "/lib/\n";
      out << "fi\n";
      out << "export LD_LIBRARY_PATH=\"${GCU_SIM_LIB}${LD_LIBRARY_PATH:+:${"
             "LD_LIBRARY_PATH}}\"\n";

    } else {
      out << "  if [[ -d " << STRINGIZE(__CHOREO_TOPSCC_DIR__) << " ]]; then\n";
      out << "    TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__) << "\n";
      out << "  else\n";
    }
#else
    out << "  if [[ -d " << STRINGIZE(__CHOREO_TOPSCC_DIR__) << " ]]; then\n";
    out << "    TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__) << "\n";
    out << "  else\n";
#endif
    out << R"script(  FOUND_PATH=$(which "topscc" 2>/dev/null)
  if [ -n "$FOUND_PATH" ]; then
    TOPSCC_INSTALL=$(dirname "$(dirname "$FOUND_PATH")")
  elif [[ -f /opt/tops/bin/topscc ]]; then
    TOPSCC_INSTALL=/opt/tops
  fi
)script";
    out << "  fi\n";
    out << R"script(fi

if [[ -z "${TOPSCC_INSTALL}" ]]; then
  echo "failed to find topscc. install topscc or set TOPSCC_INSTALL."
  exit 1
fi

TOPSCC=${TOPSCC_INSTALL}/bin/topscc
TOPSCC_LIB=${TOPSCC_INSTALL}/lib
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:}${TOPSCC_LIB}"

# Detect GCU arch
gcu_arch="${GCU_ARCH:-}"
if [[ -z "${gcu_arch}" ]]; then
  GCU_DEVICE_STR="$(lspci 2>/dev/null | grep -iE 'Enflame' | head -1 || true)"
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_arch=gcu300
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
    gcu_arch=gcu300
    export TOPS_VISIBLE_DEVICES=1
  elif [[ "${GCU_DEVICE_STR}" == *"S60"* ]]; then
    gcu_arch=gcu300
  elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
    gcu_arch=gcu210
  elif [[ "$(lspci 2>/dev/null | grep -iE 'Tencent')" != "" ]]; then
    gcu_arch=gcu210
  else
    gcu_arch=gcu300
  fi
fi

)script";

#ifdef __CHOREO_GCU_ACORE_DIR__
    out << R"script(
if [[ -z "${ACORE_INSTALL}" ]]; then
  ACORE_INSTALL=)script";
    out << STRINGIZE(__CHOREO_GCU_ACORE_DIR__) << "\n";
    out << "fi\n";
#endif
    out << R"script(
if [[ ! -z "${ACORE_INSTALL}" ]]; then
  GCU_ACORE_INCLUDE=${ACORE_INSTALL}/include
  GCU_ACORE_LIB_PATH=${ACORE_INSTALL}/lib
  GCU_ACORE_LIB=libacoreop.bc
fi
)script";

    // Match standalone topscc CFLAGS so hetero offload gets the same DMA
    // setup (GCU300+ needs TOPSCC_PRIVATE_DTE_AUTO_INIT for async DMA).
    out << R"script(
export CFLAGS="-arch ${gcu_arch} -std=c++17 -D__TOPSCC__ -ltops -lm -O3 -fPIC"
if [[ "${gcu_arch}" == gcu300* || "${gcu_arch}" == gcu400* || "${gcu_arch}" == gcu500* ]]; then
  export CFLAGS="${CFLAGS} -DTOPSCC_PRIVATE_DTE_AUTO_INIT"
fi
)script";
#ifdef __CHOREO_GCU_ACORE_DIR__
    out << R"script(
if [[ ! -z "${ACORE_INSTALL}" ]]; then
  export CFLAGS="${CFLAGS} --tops-device-lib-path=${GCU_ACORE_LIB_PATH} --tops-device-lib=${GCU_ACORE_LIB} -I${GCU_ACORE_INCLUDE} -D__ACORE_OP__"
fi
)script";
#endif
    out << R"script(
export CFLAGS="${CFLAGS} ${EXTRA_TARGET_CFLAGS:-}"

)script";
  }

  void EmitSetupFiles(std::ostream& out,
                      const std::string& build_path) const override {
    auto tdir = TargetBuildDir(build_path);
    out << "mkdir -p " << tdir << "\n";
    out << "cat <<'DEFEOF' > " << tdir << "/private_target0_defines.h\n";
    out << "#ifdef __TOPSCC__\n";
    out << "#define __CHOREO_PRIVATE_TGT0__\n";
    out << "#endif\n";
    out << "#define __CHOREO_TGT0_ARCH__ __GCU_ARCH__\n";
    out << "#define tgt0HostMalloc topsHostMalloc\n";
    out << "#define tgt0HostFree topsHostFree\n";
    out << "DEFEOF\n\n";
    out << "cat <<'RTEOF' > " << tdir << "/private_target0_runtime.h\n";
    out << __topscc_header_as_string << "\nRTEOF\n\n";
  }

  void EmitDeviceCompileCommand(std::ostream& out,
                                const std::string& build_path,
                                const std::string& src,
                                const std::string& obj) const override {
    EmitTopsccCompileCommand(out, build_path, src, obj,
                             "Device compilation failed");
  }

  void EmitHostCompileCommand(std::ostream& out, const std::string& build_path,
                              const std::string& src,
                              const std::string& obj) const override {
    EmitTopsccCompileCommand(out, build_path, src, obj,
                             "Host compilation failed");
  }

  void EmitLinkCommand(std::ostream& out,
                       const std::vector<std::string>& obj_files,
                       const std::string& exe_file) const override {
    out << R"script(  _STDCXX_PATH="$(${TOPSCC} -print-file-name=libstdc++.so 2>/dev/null)"
  RPATH_FLAG=""
  if [ -n "$_STDCXX_PATH" ] && [ -f "$_STDCXX_PATH" ]; then
    CXX_LIBDIR="$(cd "$(dirname "$_STDCXX_PATH")" && realpath "$(dirname "$(readlink -f "$_STDCXX_PATH")")" 2>/dev/null)"
    if [ -n "$CXX_LIBDIR" ] && [ -d "$CXX_LIBDIR" ]; then
      RPATH_FLAG="-Wl,-rpath,$CXX_LIBDIR"
    fi
  fi
)script";
    out << "  ${TOPSCC} ${CFLAGS}";
    for (auto& o : obj_files) out << " " << o;
    out << " -lm -lpthread ${RPATH_FLAG} -o " << exe_file
        << " || { echo 'Linking failed'; exit 1; }\n";
  }

  void SetArch(const std::string& a) { arch = a; }

private:
  void EmitTopsccCompileCommand(std::ostream& out,
                                const std::string& build_path,
                                const std::string& src, const std::string& obj,
                                const char* fail_msg) const {
    out << "  ${TOPSCC} ${CFLAGS} -c -I" << TargetBuildDir(build_path) << " -I"
        << build_path << " -I${TOPSCC_INSTALL}/include " << src << " -o " << obj
        << " || { echo '" << fail_msg << "'; exit 1; }\n";
  }

  std::string arch = "gcu300";
};

} // namespace Choreo

#endif // __CHOREO_TOPSCC_DEVICE_CODEGEN_HPP__
