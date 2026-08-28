//===- GCUCodeGen.cpp - GCU native codegen via external toolchain ---------===//
//
// Implements the "gcu" CodeGen backend for CoIR. This backend:
//   1. Lowers CoIR ops to GCU-compatible GPU MLIR (in-process pass)
//   2. Invokes gcu-compiler-opt to lower and serialize GPU modules
//   3. Invokes gcu-compiler-compile to produce a GCU device binary
//
// Only GCU300+ architectures are supported.
//
//===----------------------------------------------------------------------===//

#include "Target/CodeGen.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
namespace mgpu = mlir::gpu;

namespace {

#ifdef __CHOREO_KURAMA_DIR__
static constexpr const char* kKuramaDir = __CHOREO_KURAMA_DIR__;
#else
static constexpr const char* kKuramaDir = nullptr;
#endif

#ifdef __CHOREO_TOPSCC_DIR__
static constexpr const char* kTopsccDir = __CHOREO_TOPSCC_DIR__;
#else
static constexpr const char* kTopsccDir = nullptr;
#endif

std::string findTool(llvm::StringRef name, bool verbose = false) {
  auto checkDir = [&](const char* label, const char* dir) -> std::string {
    if (!dir) return {};
    llvm::SmallString<256> path(dir);
    llvm::sys::path::append(path, "bin", name);
    if (verbose) llvm::errs() << "  " << label << ": " << path;
    if (llvm::sys::fs::exists(path)) {
      if (verbose) llvm::errs() << " [found]\n";
      return std::string(path);
    }
    if (verbose) llvm::errs() << " [not found]\n";
    return {};
  };
  if (verbose) llvm::errs() << "findTool(\"" << name << "\") search order:\n";
  if (auto p = checkDir("KURAMA_DIR", kKuramaDir); !p.empty()) return p;
  if (auto p = checkDir("TOPSCC_DIR", kTopsccDir); !p.empty()) return p;
  if (auto p = checkDir("/opt/tops", "/opt/tops"); !p.empty()) return p;
  if (auto found = llvm::sys::findProgramByName(name)) {
    if (verbose) llvm::errs() << "  PATH: " << *found << " [found]\n";
    return std::string(*found);
  }
  if (verbose) llvm::errs() << "  PATH: [not found]\n";
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
                 llvm::raw_ostream& os) override {
    os << "error: -t gcu does not support -es (emit source); "
          "use -t topscc for text emission\n";
    return 1;
  }

  int EmitScript(mlir::ModuleOp module, llvm::StringRef arch,
                 llvm::raw_ostream& os) override {
    emitScriptPrologue(os, "GCU native: compile device binary via kurama");

    std::string optPath = findTool("gcu-compiler-opt");
    std::string compilePath = findTool("gcu-compiler-compile");
    if (optPath.empty() || compilePath.empty()) {
      llvm::errs() << "error: required tools not found for GCU script.\n";
      if (optPath.empty()) findTool("gcu-compiler-opt", true);
      if (compilePath.empty()) findTool("gcu-compiler-compile", true);
      return 1;
    }

    std::string a = arch.empty() ? "gcu300" : arch.str();

    // Emit explicit device code (__cok__ / __device__ blocks) as a separate
    // file for topscc compilation (Phase 1: script-based integration).
    auto explicitDeviceAttr =
        module->getAttrOfType<mlir::StringAttr>("coir.explicit_device_code");
    if (explicitDeviceAttr && !explicitDeviceAttr.getValue().empty()) {
      os << "# Explicit device code (__cok__ / __device__ blocks)\n";
      os << "EXPLICIT_DEVICE_FILE=\"$TMPDIR/explicit_device.cc\"\n";
      os << "cat > \"$EXPLICIT_DEVICE_FILE\""
         << " << '__COIR_EXPLICIT_DEVICE__'\n";
      os << explicitDeviceAttr.getValue();
      os << "\n__COIR_EXPLICIT_DEVICE__\n\n";
    }

    // Emit user C++ host code as a separate file.
    auto userCppAttr =
        module->getAttrOfType<mlir::StringAttr>("coir.user_cpp_code");
    if (userCppAttr && !userCppAttr.getValue().empty()) {
      os << "# User C++ host code (main, helpers, etc.)\n";
      os << "USER_CPP_FILE=\"$TMPDIR/user_host.cc\"\n";
      os << "cat > \"$USER_CPP_FILE\" << '__COIR_USER_CPP__'\n";
      os << userCppAttr.getValue();
      os << "\n__COIR_USER_CPP__\n\n";
    }

    os << "MLIRFILE=\"$TMPDIR/kernel.mlir\"\n";
    os << "LOWFILE=\"$TMPDIR/kernel_low.mlir\"\n";
    os << "BINMLIR=\"$TMPDIR/kernel_bin.mlir\"\n";
    os << "BINFILE=\"$TMPDIR/kernel.devbin\"\n\n";

    os << "cat > \"$MLIRFILE\" << '__COIR_GCU_MLIR__'\n";
    module.print(os);
    os << "\n__COIR_GCU_MLIR__\n\n";

    os << "\"" << optPath << "\""
       << " -convert-memref-to-gcu"
       << " -kernel-memory-alloc"
       << " -convert-scf-to-cf"
       << " -convert-gpu-to-gcu"
       << " -reconcile-unrealized-casts"
       << " --gcu-attach-target=arch=" << a
       << " \"$MLIRFILE\" -o \"$LOWFILE\" || exit 1\n\n";

    os << "\"" << optPath << "\""
       << " -gpu-module-to-binary=format=llvm"
       << " \"$LOWFILE\" -o \"$BINMLIR\" || exit 1\n\n";

    os << "\"" << compilePath << "\""
       << " \"$BINMLIR\" -a " << a << " -o \"$BINFILE\" || exit 1\n\n";

    // If explicit device code was emitted, compile it with topscc.
    if (explicitDeviceAttr && !explicitDeviceAttr.getValue().empty()) {
      os << "# Compile explicit device code with topscc\n";
      os << "\"${TOPSCC:-topscc}\" --cuda-device-only -c -emit-llvm"
         << " -arch " << a << " -o \"$TMPDIR/explicit_device.bc\""
         << " \"$EXPLICIT_DEVICE_FILE\" 2>&1\n";
      os << "# TODO(Phase 1): link explicit_device.bc with kurama device"
         << " binary via llvm-link\n";
    }

    return 0;
  }

  int Compile(mlir::ModuleOp module, llvm::StringRef arch,
              llvm::StringRef outputPath) override {
    std::string optTool = findTool("gcu-compiler-opt");
    if (optTool.empty()) {
      llvm::errs() << "error: gcu-compiler-opt not found.\n";
      findTool("gcu-compiler-opt", /*verbose=*/true);
      return 1;
    }
    std::string compileTool = findTool("gcu-compiler-compile");
    if (compileTool.empty()) {
      llvm::errs() << "error: gcu-compiler-compile not found.\n";
      findTool("gcu-compiler-compile", /*verbose=*/true);
      return 1;
    }

    llvm::SmallString<128> mlirFile;
    if (auto ec = llvm::sys::fs::createTemporaryFile("choreo-gcu", "mlir",
                                                     mlirFile)) {
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
    if (auto ec = llvm::sys::fs::createTemporaryFile("choreo-gcu-low", "mlir",
                                                     loweredMlir)) {
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
    if (auto ec = llvm::sys::fs::createTemporaryFile("choreo-gcu-bin", "mlir",
                                                     binaryMlir)) {
      llvm::errs() << "error: cannot create temp file: " << ec.message()
                   << "\n";
      llvm::sys::fs::remove(mlirFile);
      llvm::sys::fs::remove(loweredMlir);
      return 1;
    }

    {
      llvm::SmallVector<llvm::StringRef, 8> args = {
          optTool, "-gpu-module-to-binary=format=llvm", loweredMlir, "-o",
          binaryMlir};

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

    // Step 3: gcu-compiler-compile <binary.mlir> -o <output>
    {
      llvm::SmallVector<llvm::StringRef, 8> args = {compileTool, binaryMlir,
                                                    "-o", outputPath};
      llvm::SmallString<32> archFlag;
      if (!arch.empty()) {
        archFlag = "-a";
        args.push_back(archFlag);
        args.push_back(arch);
      }

      std::string errMsg;
      int rc = llvm::sys::ExecuteAndWait(compileTool, args,
                                         /*Env=*/std::nullopt,
                                         /*Redirects=*/{}, /*SecondsToWait=*/0,
                                         /*MemLimit=*/0, &errMsg);
      if (rc != 0) {
        llvm::errs() << "error: gcu-compiler-compile failed";
        if (!errMsg.empty()) llvm::errs() << ": " << errMsg;
        llvm::errs() << "\n";
        llvm::sys::fs::remove(mlirFile);
        llvm::sys::fs::remove(binaryMlir);
        return 1;
      }
    }

    llvm::sys::fs::remove(mlirFile);
    llvm::sys::fs::remove(binaryMlir);
    return 0;
  }
};

static bool registered_gcu = [] {
  CoIR::CodeGenRegistry::Register(
      "gcu", [] { return std::make_unique<GCUNativeCodeGen>(); });
  return true;
}();

} // namespace
