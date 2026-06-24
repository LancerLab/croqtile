//===- NVPTXCodeGen.cpp - CoIR -> PTX code generator ----------------------===//
//
// NVPTX code generator for the "gpu" target. Lowers CoIR through GPU, NVVM,
// and LLVM dialects, then translates to PTX via the NVPTX backend.
//
// Lower()      -> no-op (NVPTX runs its own MLIR lowering pipeline internally)
// EmitSource() -> raw PTX assembly (-es).
// EmitScript() -> self-contained bash script: PTX embedded as C string
//                 + host stubs + user host code, compiled with g++/nvcc,
//                 linked with -lcuda, executed on --execute.
// Compile()    -> full in-memory pipeline to binary.
//
//===----------------------------------------------------------------------===//

#include "CodeGen/GPU/CompilePipeline.h"
#include "CodeGen/GPU/EmitHostStubs.h"
#include "CodeGen/GPU/NativePipeline.h"
#include "Dialect/CoIR/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static std::string escapeCString(llvm::StringRef s) {
  std::string out;
  out.reserve(s.size() + s.size() / 8);
  for (char c : s) {
    switch (c) {
    case '\\': out += "\\\\"; break;
    case '"': out += "\\\""; break;
    case '\n': out += "\\n\\\n"; break;
    default: out += c; break;
    }
  }
  return out;
}

class NVPTXCodeGen : public CoIR::CodeGen {
public:
  // NVPTX runs its own MLIR-native lowering pipeline (ConvertToGPU ->
  // gpu-to-nvvm -> llvm) inside emitPTXFromCoIR on a cloned module.
  // The shared DMA/MMA lowering is not needed here.

  int EmitSource(ModuleOp module, llvm::StringRef arch,
                 llvm::raw_ostream &os) override {
    std::string a = arch.empty() ? "sm_80" : arch.str();
    std::string ptx = coir::gpu::emitPTXFromCoIR(module, a);
    if (ptx.empty()) {
      llvm::errs() << "nvptx codegen: PTX emission failed\n";
      return 1;
    }
    os << ptx;
    return 0;
  }

  int EmitScript(ModuleOp module, llvm::StringRef arch,
                 llvm::raw_ostream &os) override {
    std::string a = arch.empty() ? "sm_80" : arch.str();

    std::string ptx = coir::gpu::emitPTXFromCoIR(module, a);
    if (ptx.empty()) {
      llvm::errs() << "nvptx codegen: PTX emission failed\n";
      return 1;
    }

    std::string stubs = coir::gpu::emitHostStubs(module);

    auto &sctx = CoIR::ScriptContext::Get();

    os << "#!/usr/bin/env bash\n";
    os << "# CoIR generated script -- NVPTX: compile + execute\n";
    os << "set -eo pipefail\n\n";

    os << "TMPDIR=$(mktemp -d /tmp/cocc_gpu_XXXXXX)\n";
    os << "trap 'rm -rf $TMPDIR' EXIT\n\n";

    if (sctx.types_header && sctx.runtime_header) {
      os << "cat > \"$TMPDIR/choreo_types.h\" << '__COCC_TYPES_HEADER__'\n";
      os << sctx.types_header;
      os << "\n__COCC_TYPES_HEADER__\n\n";

      os << "cat > \"$TMPDIR/choreo.h\" << '__COCC_CHOREO_HEADER__'\n";
      os << sctx.runtime_header;
      os << "\n__COCC_CHOREO_HEADER__\n\n";
    }

    if (!sctx.build_env.empty()) {
      os << sctx.build_env;
    } else {
      os << "CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n";
      os << "if [[ ! -d \"$CUDA_HOME\" ]]; then\n";
      os << "  echo \"Error: CUDA_HOME not found at $CUDA_HOME\"; exit 1\n";
      os << "fi\n";
    }

    if (!sctx.arch_override.empty())
      os << "gpu_arch=\"" << sctx.arch_override << "\"\n\n";
    else
      os << "gpu_arch=\"" << a << "\"\n\n";

    os << "cat > \"$TMPDIR/host.cpp\" << '__COCC_HOST_SOURCE__'\n";

    os << "static const char __coir_ptx_string[] = \""
       << escapeCString(ptx) << "\";\n\n";

    if (sctx.types_header)
      os << "#include \"choreo_types.h\"\n";
    if (sctx.runtime_header)
      os << "#include \"choreo.h\"\n";
    os << "\n";

    os << stubs;

    auto hostAttr =
        module->getAttrOfType<mlir::StringAttr>("coir.host_code");
    if (hostAttr && !hostAttr.getValue().empty())
      os << hostAttr.getValue();

    os << "\n__COCC_HOST_SOURCE__\n\n";

    os << "g++ -std=c++17 -I\"$TMPDIR\" "
          "-I\"${CUDA_HOME}/include\" "
          "-o \"$TMPDIR/runner\" \"$TMPDIR/host.cpp\" "
          "-L\"${CUDA_HOME}/lib64\" -lcuda 2>&1\n\n";

    os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
    os << "  shift\n";
    os << "  \"$TMPDIR/runner\" \"$@\"\n";
    os << "fi\n";

    return 0;
  }

  int Compile(ModuleOp module, llvm::StringRef arch,
              llvm::StringRef outputPath) override {
    auto &sctx = CoIR::ScriptContext::Get();
    return coir::gpu::compileToExecutable(module, arch, outputPath,
                                          sctx.types_header,
                                          sctx.runtime_header, sctx.cuda_home);
  }
};

static bool registered = [] {
  CoIR::CodeGenRegistry::Register("gpu",
                                  [] { return std::make_unique<NVPTXCodeGen>(); });
  return true;
}();

} // namespace
