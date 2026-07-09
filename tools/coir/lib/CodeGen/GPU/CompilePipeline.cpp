//===- CompilePipeline.cpp - Full in-memory NVPTX compile pipeline --------===//
//
// Ties together:
//   - PTX emission (from NativePipeline)
//   - Host stub generation (from EmitHostStubs)
//   - Host compilation via clang-as-library (from HostCompile)
//   - PTX embedding in host LLVM IR
//   - x86 object emission
//   - External link to produce final executable
//
//===----------------------------------------------------------------------===//

#include "CodeGen/GPU/CompilePipeline.h"
#include "CodeGen/GPU/EmitHostStubs.h"
#include "CodeGen/GPU/HostCompile.h"
#include "CodeGen/GPU/NativePipeline.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

namespace coir {
namespace gpu {

void embedPTXInHostModule(llvm::Module &hostMod, llvm::StringRef ptxString,
                          llvm::StringRef globalName) {
  auto &ctx = hostMod.getContext();

  llvm::Constant *ptxInit =
      llvm::ConstantDataArray::getString(ctx, ptxString, /*AddNull=*/true);
  auto *ptxGV = new llvm::GlobalVariable(
      hostMod, ptxInit->getType(), /*isConstant=*/true,
      llvm::GlobalValue::PrivateLinkage, ptxInit, "__coir_ptx_data");
  ptxGV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  auto *externPTX = hostMod.getNamedGlobal(globalName);
  if (externPTX && externPTX != ptxGV) {
    auto *bc =
        llvm::ConstantExpr::getBitCast(ptxGV, externPTX->getType());
    externPTX->replaceAllUsesWith(bc);
    externPTX->eraseFromParent();
  }
  ptxGV->setName(globalName);
  ptxGV->setLinkage(llvm::GlobalValue::ExternalLinkage);
}

bool emitHostObjectFile(llvm::Module &hostMod, llvm::StringRef outputPath) {
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();

  auto triple = llvm::sys::getDefaultTargetTriple();
  hostMod.setTargetTriple(triple);

  std::string errStr;
  auto *target = llvm::TargetRegistry::lookupTarget(triple, errStr);
  if (!target) {
    llvm::errs() << "coir: cannot find x86 target: " << errStr << "\n";
    return false;
  }

  auto cpu = llvm::sys::getHostCPUName();
  llvm::TargetOptions tOpts;
  auto tm = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(triple, cpu, "", tOpts, llvm::Reloc::PIC_));
  if (!tm) {
    llvm::errs() << "coir: failed to create x86 TargetMachine\n";
    return false;
  }

  hostMod.setDataLayout(tm->createDataLayout());

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "coir: cannot open " << outputPath << ": "
                 << ec.message() << "\n";
    return false;
  }

  llvm::legacy::PassManager pm;
  if (tm->addPassesToEmitFile(pm, out, nullptr,
                              llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "coir: x86 target cannot emit object file\n";
    return false;
  }
  pm.run(hostMod);
  out.flush();
  return true;
}

bool linkHostExecutable(llvm::StringRef objectPath,
                        llvm::StringRef executablePath,
                        llvm::StringRef cudaHome) {
  auto gcc = llvm::sys::findProgramByName("g++");
  if (!gcc) {
    auto clangBin = llvm::sys::findProgramByName("clang++");
    if (!clangBin) {
      llvm::errs() << "coir: cannot find g++ or clang++ for linking\n";
      return false;
    }
    gcc = clangBin;
  }

  llvm::SmallVector<llvm::StringRef, 16> linkArgs;
  linkArgs.push_back(*gcc);
  linkArgs.push_back(objectPath);
  linkArgs.push_back("-o");
  linkArgs.push_back(executablePath);

  std::string cudaLibDir;
  if (!cudaHome.empty()) {
    cudaLibDir = (cudaHome + "/lib64").str();
    linkArgs.push_back("-L");
    linkArgs.push_back(cudaLibDir);
  }
  linkArgs.push_back("-lcuda");
  linkArgs.push_back("-lstdc++");
  linkArgs.push_back("-lm");

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(*gcc, linkArgs, /*Env=*/std::nullopt,
                                     /*Redirects=*/{}, /*SecondsToWait=*/60,
                                     /*MemoryLimit=*/0, &errMsg);
  if (rc != 0) {
    llvm::errs() << "coir: linking failed (exit " << rc << "): " << errMsg
                 << "\n";
    return false;
  }
  return true;
}

int compileToExecutable(mlir::ModuleOp module, llvm::StringRef arch,
                        llvm::StringRef outputPath,
                        const char *typesHeader, const char *runtimeHeader,
                        llvm::StringRef cudaHome) {
  std::string a = arch.empty() ? "sm_80" : arch.str();
  std::string ptx = emitPTXFromCoIR(module, a);
  if (ptx.empty()) {
    llvm::errs() << "coir compile: PTX emission failed\n";
    return 1;
  }

  std::string stubs = emitHostStubs(module);

  std::string hostSrc;
  {
    std::string tmp;
    llvm::raw_string_ostream oss(tmp);
    oss << "extern const char __coir_ptx_string[];\n\n";
    if (typesHeader)
      oss << "#include \"choreo_types.h\"\n";
    if (runtimeHeader)
      oss << "#include \"choreo.h\"\n";
    oss << "\n";
    oss << stubs;

    auto hostAttr =
        module->getAttrOfType<mlir::StringAttr>("coir.user_cpp_code");
    if (hostAttr && !hostAttr.getValue().empty())
      oss << hostAttr.getValue();
    oss << "\n";
    hostSrc = oss.str();
  }

  llvm::LLVMContext llvmCtx;
  HostCompileOptions opts;
  opts.cudaHome = cudaHome.str();
  if (typesHeader)
    opts.virtualFiles.emplace_back("choreo_types.h", typesHeader);
  if (runtimeHeader)
    opts.virtualFiles.emplace_back("choreo.h", runtimeHeader);

  auto hostMod = compileHostToLLVM(hostSrc, llvmCtx, opts);
  if (!hostMod) {
    llvm::errs() << "coir compile: host compilation failed\n";
    return 1;
  }

  embedPTXInHostModule(*hostMod, ptx, "__coir_ptx_string");

  llvm::SmallString<256> objPath(outputPath);
  objPath += ".o";
  if (!emitHostObjectFile(*hostMod, objPath)) {
    llvm::errs() << "coir compile: x86 object emission failed\n";
    return 1;
  }

  if (!linkHostExecutable(objPath, outputPath, cudaHome)) {
    llvm::errs() << "coir compile: linking failed\n";
    llvm::sys::fs::remove(objPath);
    return 1;
  }

  llvm::sys::fs::remove(objPath);
  return 0;
}

} // namespace gpu
} // namespace coir
