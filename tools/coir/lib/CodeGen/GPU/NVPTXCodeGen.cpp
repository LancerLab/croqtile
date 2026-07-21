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
// Compile()    -> default generated-script compilation to executable.
//
//===----------------------------------------------------------------------===//

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

    emitScriptPrologue(os, "NVPTX: compile + execute", "_gpu");

    if (sctx.build_env.empty()) {
      os << "CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n";
      os << "if [[ ! -d \"$CUDA_HOME\" ]]; then\n";
      os << "  echo \"Error: CUDA_HOME not found at $CUDA_HOME\"; exit 1\n";
      os << "fi\n";
    }

    if (!sctx.arch_override.empty())
      os << "gpu_arch=\"" << sctx.arch_override << "\"\n\n";
    else
      os << "gpu_arch=\"" << a << "\"\n\n";

    os << "BINFILE=\"$TMPDIR/runner\"\n\n";
    os << "cat > \"$TMPDIR/host.cpp\" << '__COCC_HOST_SOURCE__'\n";

    os << "static const char __coir_ptx_string[] = \""
       << escapeCString(ptx) << "\";\n\n";

    if (sctx.runtime_header)
      os << "#include \"choreo.h\"\n";
    os << "\n";

    os << stubs;

    emitUserCppCode(module, os);

    os << "\n__COCC_HOST_SOURCE__\n\n";

    os << "g++ -std=c++17 -I\"$TMPDIR\" "
          "-I\"${CUDA_HOME}/include\" "
          "-o \"$BINFILE\" \"$TMPDIR/host.cpp\" "
          "-L\"${CUDA_HOME}/lib64\" -lcuda 2>&1\n\n";

    emitScriptExecuteBlock(os);

    return 0;
  }

};

static bool registered = [] {
  CoIR::CodeGenRegistry::Register("gpu",
                                  [] { return std::make_unique<NVPTXCodeGen>(); });
  return true;
}();

} // namespace
