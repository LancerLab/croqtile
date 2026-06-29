//===- GCUCodeGen.cpp - GCU native codegen via external toolchain ---------===//
//
// Implements the "gcu" CodeGen backend for CoIR. This backend:
//   1. Lowers CoIR ops to GCU-compatible GPU MLIR (in-process pass)
//   2. Invokes gcu-compiler-opt and gcu-compiler-kernel as subprocesses
//      to produce a GCU binary (fatbin)
//
// Only GCU300+ architectures are supported. The -es and -gs options are
// not available for this target.
//
//===----------------------------------------------------------------------===//

#include "Target/CodeGen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace coir {
std::unique_ptr<mlir::Pass> createConvertToGCUPass();
} // namespace coir

using namespace mlir;

namespace {

#ifdef __CHOREO_KURAMA_DIR__
static constexpr const char *kKuramaDir = __CHOREO_KURAMA_DIR__;
#else
static constexpr const char *kKuramaDir = nullptr;
#endif

#ifdef __CHOREO_TOPSCC_DIR__
static constexpr const char *kTopsccDir = __CHOREO_TOPSCC_DIR__;
#else
static constexpr const char *kTopsccDir = nullptr;
#endif

std::string findTool(llvm::StringRef name) {
  auto checkDir = [&](const char *dir) -> std::string {
    if (!dir) return {};
    llvm::SmallString<256> path(dir);
    llvm::sys::path::append(path, "bin", name);
    if (llvm::sys::fs::exists(path))
      return std::string(path);
    return {};
  };
  if (auto p = checkDir(kKuramaDir); !p.empty()) return p;
  if (auto p = checkDir(kTopsccDir); !p.empty()) return p;
  if (auto p = checkDir("/opt/tops"); !p.empty()) return p;
  if (auto found = llvm::sys::findProgramByName(name))
    return std::string(*found);
  return {};
}

class GCUNativeCodeGen : public CoIR::CodeGen {
public:
  bool Lower(mlir::ModuleOp module) override {
    mlir::PassManager pm(module.getContext());
    pm.addPass(coir::createConvertToGCUPass());
    return mlir::succeeded(pm.run(module));
  }

  int EmitSource(mlir::ModuleOp, llvm::StringRef,
                 llvm::raw_ostream &os) override {
    os << "error: -t gcu does not support -es (emit source); "
          "use -t topscc for text emission\n";
    return 1;
  }

  int EmitScript(mlir::ModuleOp, llvm::StringRef,
                 llvm::raw_ostream &os) override {
    os << "error: -t gcu does not support -gs (generate script); "
          "use -t topscc for script generation\n";
    return 1;
  }

  int Compile(mlir::ModuleOp module, llvm::StringRef arch,
              llvm::StringRef outputPath) override {
    std::string optTool = findTool("gcu-compiler-opt");
    if (optTool.empty()) {
      llvm::errs() << "error: gcu-compiler-opt not found. "
                      "Set up the GCU compiler toolchain or add it to PATH.\n";
      return 1;
    }
    std::string kernelTool = findTool("gcu-compiler-kernel");
    if (kernelTool.empty()) {
      llvm::errs() << "error: gcu-compiler-kernel not found.\n";
      return 1;
    }

    // Write lowered MLIR to temp file.
    llvm::SmallString<128> mlirFile;
    if (auto ec = llvm::sys::fs::createTemporaryFile(
            "choreo-gcu", "mlir", mlirFile)) {
      llvm::errs() << "error: cannot create temp file: " << ec.message()
                   << "\n";
      return 1;
    }

    {
      std::error_code ec;
      llvm::raw_fd_ostream os(mlirFile, ec);
      if (ec) {
        llvm::errs() << "error: cannot write " << mlirFile << ": "
                     << ec.message() << "\n";
        return 1;
      }
      module.print(os);
    }

    // Step 1: Lower GPU MLIR to GCU LLVM dialect
    llvm::SmallString<128> loweredMlir;
    if (auto ec = llvm::sys::fs::createTemporaryFile(
            "choreo-gcu-low", "mlir", loweredMlir)) {
      llvm::errs() << "error: cannot create temp file: " << ec.message()
                   << "\n";
      llvm::sys::fs::remove(mlirFile);
      return 1;
    }

    {
      llvm::SmallVector<llvm::StringRef, 16> args = {
          optTool,
          "-convert-memref-to-gcu",
          "-kernel-memory-alloc",
          "-convert-scf-to-cf",
          "-convert-gpu-to-gcu",
          "-reconcile-unrealized-casts"};
      llvm::SmallString<64> attachFlag;
      if (!arch.empty()) {
        attachFlag = "--gcu-attach-target=arch=";
        attachFlag += arch;
        args.push_back(attachFlag);
      }
      args.push_back(mlirFile);
      args.push_back("-o");
      args.push_back(loweredMlir);

      std::string errMsg;
      int rc = llvm::sys::ExecuteAndWait(optTool, args, /*Env=*/std::nullopt,
                                         /*Redirects=*/{}, /*SecondsToWait=*/0,
                                         /*MemLimit=*/0, &errMsg);
      if (rc != 0) {
        llvm::errs() << "error: gcu-compiler-opt (lowering) failed";
        if (!errMsg.empty()) llvm::errs() << ": " << errMsg;
        llvm::errs() << "\n";
        llvm::sys::fs::remove(mlirFile);
        llvm::sys::fs::remove(loweredMlir);
        return 1;
      }
    }

    // Step 2: Serialize GPU module to binary object
    llvm::SmallString<128> binaryMlir;
    if (auto ec = llvm::sys::fs::createTemporaryFile(
            "choreo-gcu-bin", "mlir", binaryMlir)) {
      llvm::errs() << "error: cannot create temp file: " << ec.message()
                   << "\n";
      llvm::sys::fs::remove(mlirFile);
      llvm::sys::fs::remove(loweredMlir);
      return 1;
    }

    {
      llvm::SmallVector<llvm::StringRef, 8> args = {
          optTool, "-gpu-module-to-binary", loweredMlir, "-o", binaryMlir};

      std::string errMsg;
      int rc = llvm::sys::ExecuteAndWait(optTool, args, /*Env=*/std::nullopt,
                                         /*Redirects=*/{}, /*SecondsToWait=*/0,
                                         /*MemLimit=*/0, &errMsg);
      if (rc != 0) {
        llvm::errs() << "error: gcu-compiler-opt (binary) failed";
        if (!errMsg.empty()) llvm::errs() << ": " << errMsg;
        llvm::errs() << "\n";
        llvm::sys::fs::remove(mlirFile);
        llvm::sys::fs::remove(loweredMlir);
        llvm::sys::fs::remove(binaryMlir);
        return 1;
      }
    }

    llvm::sys::fs::remove(loweredMlir);

    // Step 2: gcu-compiler-kernel <binary.mlir> -o <output>
    {
      llvm::SmallVector<llvm::StringRef, 6> args = {
          kernelTool, binaryMlir, "-o", outputPath};

      std::string errMsg;
      int rc = llvm::sys::ExecuteAndWait(kernelTool, args,
                                         /*Env=*/std::nullopt,
                                         /*Redirects=*/{}, /*SecondsToWait=*/0,
                                         /*MemLimit=*/0, &errMsg);
      if (rc != 0) {
        llvm::errs() << "error: gcu-compiler-kernel failed";
        if (!errMsg.empty()) llvm::errs() << ": " << errMsg;
        llvm::errs() << "\n";
        llvm::sys::fs::remove(mlirFile);
        llvm::sys::fs::remove(binaryMlir);
        return 1;
      }
    }

    // Cleanup temp files.
    llvm::sys::fs::remove(mlirFile);
    llvm::sys::fs::remove(binaryMlir);
    return 0;
  }
};

static bool registered_gcu = [] {
  CoIR::CodeGenRegistry::Register("gcu", [] {
    return std::make_unique<GCUNativeCodeGen>();
  });
  return true;
}();

} // namespace
