#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
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
    cl::desc("Emission target (e.g. gpu, cute)"), cl::init("gpu"));

static cl::opt<bool> runFullPipeline(
    "lower",
    cl::desc("Run CoIR lowering pipeline before emitting C++"),
    cl::init(false));

static cl::opt<bool> generateScript(
    "gs", cl::desc("Generate a self-contained bash script (compile + execute)"),
    cl::init(false));



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

  // Normalize target aliases
  std::string tgt = target.getValue();
  if (tgt == "gpu") tgt = "cute";

  CoIR::Pipeline pipeline(*module, context);

  if (runFullPipeline) {
    if (!pipeline.Lower()) {
      errs() << "coir-codegen: lowering failed\n";
      return 1;
    }
  }

  return pipeline.EmitSource(tgt, generateScript, "");
}
