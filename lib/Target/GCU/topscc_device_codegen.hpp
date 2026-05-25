#ifndef __CHOREO_TOPSCC_DEVICE_CODEGEN_HPP__
#define __CHOREO_TOPSCC_DEVICE_CODEGEN_HPP__

#include "codegen.hpp"
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
    out << R"script(
# Find topscc
if [[ -z "${TOPSCC_INSTALL}" ]]; then
  FOUND_PATH=$(which "topscc" 2>/dev/null)
  if [ -n "$FOUND_PATH" ]; then
    TOPSCC_INSTALL=$(dirname "$(dirname "$FOUND_PATH")")
  elif [[ -f /opt/tops/bin/topscc ]]; then
    TOPSCC_INSTALL=/opt/tops
  fi
fi

if [[ -z "${TOPSCC_INSTALL}" ]]; then
  echo "failed to find topscc. install topscc or set TOPSCC_INSTALL."
  exit 1
fi

TOPSCC=${TOPSCC_INSTALL}/bin/topscc
TOPSCC_LIB=${TOPSCC_INSTALL}/lib
export LD_LIBRARY_PATH="${TOPSCC_LIB}:${LD_LIBRARY_PATH:-}"

# Detect GCU arch
gcu_arch="${GCU_ARCH:-}"
if [[ -z "${gcu_arch}" ]]; then
  _gcu_dstr="$(lspci 2>/dev/null | grep -iE '(Enflame|Tencent)' | head -1)"
  case "${_gcu_dstr}" in
    *S60G*|*c035*|*S60*) gcu_arch="gcu300" ;;
    *I20*|*Tencent*)      gcu_arch="gcu210" ;;
    *)                    gcu_arch="gcu300" ;;
  esac
fi

)script";
  }

  void EmitHostCompileCommand(std::ostream& out,
                              const std::string& build_path,
                              const std::string& src,
                              const std::string& obj) const override {
    out << "  ${TOPSCC} -arch ${gcu_arch} -std=c++17 -c -I" << build_path
        << " -I${TOPSCC_INSTALL}/include " << src << " -o " << obj
        << " || { echo 'Host compilation failed'; exit 1; }\n";
  }

  void EmitLinkCommand(std::ostream& out,
                       const std::vector<std::string>& obj_files,
                       const std::string& exe_file) const override {
    out << "  ${TOPSCC} -arch ${gcu_arch} -std=c++17";
    for (auto& o : obj_files) out << " " << o;
    out << " " << LinkFlags() << " -lm -lpthread -o " << exe_file
        << " || { echo 'Linking failed'; exit 1; }\n";
  }

  void SetArch(const std::string& a) { arch = a; }

private:
  std::string arch = "gcu300";
};

} // namespace Choreo

#endif // __CHOREO_TOPSCC_DEVICE_CODEGEN_HPP__
