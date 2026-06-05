#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/Passes.h"
#ifdef COIR_HAS_AUX_TARGET
#include "EmitTopscc.h"
#endif

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<std::string> targetArch("target-arch",
                                       cl::desc("Target GPU architecture"),
                                       cl::init("sm_90a"));

static cl::opt<std::string> target("target",
    cl::desc("Emission target: gpu (default)"
#ifdef COIR_HAS_AUX_TARGET
             ", or auxiliary targets"
#endif
    ), cl::init("gpu"));

static cl::opt<bool> runFullPipeline(
    "full-pipeline",
    cl::desc("Run full CoIR lowering pipeline before emitting C++"),
    cl::init(false));

static cl::opt<bool> generateScript(
    "gs", cl::desc("Generate a self-contained bash script (compile + execute)"),
    cl::init(false));

static void emitScriptHeader(llvm::raw_ostream &os) {
  os << "#!/usr/bin/env bash\n";
  os << "# CoIR generated script -- compile and execute CUDA kernel\n";
  os << "set -euo pipefail\n\n";
  os << "CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n";
  os << "NVCC=\"${CUDA_HOME}/bin/nvcc\"\n\n";
  os << "if [[ ! -x \"$NVCC\" ]]; then\n";
  os << "  echo \"Error: nvcc not found at $NVCC\"; exit 1\n";
  os << "fi\n\n";
  os << "# Auto-detect GPU arch if not specified\n";
  os << "if [[ -z \"${GPU_ARCH:-}\" ]]; then\n";
  os << "  _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader "
        "2>/dev/null | head -1 | tr -d '.' || true)\n";
  os << "  GPU_ARCH=\"sm_${_cc:-86}\"\n";
  os << "fi\n\n";
  os << "# Find choreo runtime header\n";
  os << "if [[ -z \"${CHOREO_ROOT:-}\" ]]; then\n";
  os << "  _coir_bin=$(which coir-codegen 2>/dev/null || true)\n";
  os << "  if [[ -n \"$_coir_bin\" ]]; then\n";
  os << "    CHOREO_ROOT=\"$(cd \"$(dirname \"$_coir_bin\")\" && "
        "git rev-parse --show-toplevel 2>/dev/null || true)\"\n";
  os << "  fi\n";
  os << "  if [[ -z \"${CHOREO_ROOT:-}\" ]]; then\n";
  os << "    CHOREO_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && "
        "git rev-parse --show-toplevel 2>/dev/null || "
        "echo \"$(dirname \"${BASH_SOURCE[0]}\")\")\" \n";
  os << "  fi\n";
  os << "fi\n";
  os << "CHOREO_RT=\"${CHOREO_ROOT}/runtime\"\n\n";
  os << "TMPDIR=$(mktemp -d /tmp/coir_XXXXXX)\n";
  os << "TMPFILE=\"$TMPDIR/kernel.cu\"\n";
  os << "BINFILE=\"$TMPDIR/kernel\"\n";
  os << "trap 'rm -rf $TMPDIR' EXIT\n\n";
  os << "cat > \"$TMPFILE\" << '__COIR_CUDA_SOURCE__'\n";
}

static void emitHostCode(llvm::raw_ostream &os, ModuleOp module) {
  auto hostCodeAttr = module->getAttrOfType<StringAttr>("coir.host_code");
  if (!hostCodeAttr) return;
  os << "\n" << hostCodeAttr.getValue() << "\n";
}

static void emitScriptFooter(llvm::raw_ostream &os) {
  os << "__COIR_CUDA_SOURCE__\n\n";
  os << "# Compile and optionally execute\n";
  os << "\"$NVCC\" -std=c++17 -arch=\"$GPU_ARCH\" -I\"$CHOREO_RT\" "
        "-o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
  os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
  os << "  shift\n";
  os << "  \"$BINFILE\" \"$@\"\n";
  os << "fi\n";
}


int main(int argc, char** argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "CoIR codegen -- emit target source from CoIR IR\n"
      "  --target=gpu     Emit CUDA/CUTE C++ (default)\n"
#ifdef COIR_HAS_AUX_TARGET
      "  --target=<name>  Emit auxiliary target C++\n"
#endif
  );

  DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<gpu::GPUDialect>();
  registry.insert<coir::CoIRDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse the input MLIR
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    errs() << "coir-codegen: " << errorMessage << "\n";
    return 1;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    errs() << "coir-codegen: failed to parse input MLIR\n";
    return 1;
  }

  // Optionally run the full lowering pipeline
  if (runFullPipeline) {
    PassManager pm(&context);
    pm.addPass(coir::createClassifyCopiesPass());
    pm.addPass(coir::createLowerMMAPass());
    pm.addPass(coir::createLowerCopyPass());

    if (failed(pm.run(*module))) {
      errs() << "coir-codegen: lowering pipeline failed\n";
      return 1;
    }
  }

  // Emit based on target
#ifdef COIR_HAS_AUX_TARGET
  if (target.getValue() != "gpu") {
    if (generateScript)
      coir::emitTopsccScript(*module, llvm::outs());
    else
      coir::emitTopscc(*module, llvm::outs());
    return 0;
  }
#endif

  if (generateScript)
    emitScriptHeader(llvm::outs());

  PassManager emitPM(&context);
  emitPM.addPass(coir::createEmitCUDAPass());
  if (failed(emitPM.run(*module))) {
    errs() << "coir-codegen: C++ emission failed\n";
    return 1;
  }

  if (generateScript) {
    emitHostCode(llvm::outs(), *module);
    emitScriptFooter(llvm::outs());
  }

  return 0;
}
