//===- HostCompile.cpp - In-memory host C++ -> LLVM IR via clang ----------===//
//
// Uses clang::CompilerInstance + clang::EmitLLVMOnlyAction with an in-memory
// VFS overlay to compile host C++ to LLVM IR without writing any temp files.
//
//===----------------------------------------------------------------------===//

#include "CodeGen/GPU/HostCompile.h"

#ifdef COIR_HAS_CLANG

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <vector>
#include <string>

using namespace llvm;
using namespace clang;

std::unique_ptr<llvm::Module>
coir::gpu::compileHostToLLVM(StringRef source, LLVMContext &ctx,
                             const HostCompileOptions &opts) {
  // Build an in-memory VFS overlaid on the real filesystem.
  auto realFS = vfs::getRealFileSystem();
  auto memFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  auto overlayFS = makeIntrusiveRefCnt<vfs::OverlayFileSystem>(realFS);
  overlayFS->pushOverlay(memFS);

  // Inject virtual files (choreo headers, etc.).
  for (auto &[name, contents] : opts.virtualFiles) {
    memFS->addFile(
        "/virtual_inc/" + name, 0,
        MemoryBuffer::getMemBufferCopy(contents, name));
  }

  // Inject the source file itself.
  std::string srcPath = "/virtual_src/host_main.cpp";
  memFS->addFile(srcPath, 0,
                 MemoryBuffer::getMemBufferCopy(source, "host_main.cpp"));

  // Compiler arguments.
  std::vector<std::string> argStrs = {
      "coir-clang",
      opts.standard,
      "-xc++",
      "-I/virtual_inc",
      srcPath,
  };

  if (!opts.cudaHome.empty()) {
    argStrs.push_back("-I" + opts.cudaHome + "/include");
  }

  // Suppress default system header warnings for generated code.
  argStrs.push_back("-w");

  std::vector<const char *> args;
  args.reserve(argStrs.size());
  for (auto &s : argStrs)
    args.push_back(s.c_str());

  // Diagnostics.
  auto diagOpts = makeIntrusiveRefCnt<DiagnosticOptions>();
  auto diagPrinter =
      std::make_unique<TextDiagnosticPrinter>(errs(), diagOpts.get());
  auto diagIDs = makeIntrusiveRefCnt<DiagnosticIDs>();
  DiagnosticsEngine diags(diagIDs, diagOpts.get(), diagPrinter.release());

  // Create invocation.
  auto invocation = std::make_shared<CompilerInvocation>();
  if (!CompilerInvocation::CreateFromArgs(*invocation, args, diags)) {
    errs() << "coir host compile: failed to create CompilerInvocation\n";
    return nullptr;
  }

  // Set up CompilerInstance.
  CompilerInstance ci;
  ci.setInvocation(std::move(invocation));
  ci.setDiagnostics(&diags);
  ci.createFileManager(overlayFS);

  // Target triple -- host.
  auto &targetOpts = ci.getTargetOpts();
  if (targetOpts.Triple.empty())
    targetOpts.Triple = sys::getDefaultTargetTriple();

  // Run EmitLLVMOnlyAction.
  auto act = std::make_unique<EmitLLVMOnlyAction>(&ctx);
  if (!ci.ExecuteAction(*act)) {
    errs() << "coir host compile: clang compilation failed\n";
    return nullptr;
  }

  return act->takeModule();
}

#else // !COIR_HAS_CLANG

std::unique_ptr<llvm::Module>
coir::gpu::compileHostToLLVM(llvm::StringRef, llvm::LLVMContext &,
                             const HostCompileOptions &) {
  llvm::errs() << "coir host compile: built without clang support "
                  "(COIR_HAS_CLANG not defined)\n";
  return nullptr;
}

#endif // COIR_HAS_CLANG
