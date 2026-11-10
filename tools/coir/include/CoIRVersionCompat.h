// CoIRVersionCompat.h -- Version-guard macros for LLVM/MLIR API differences.
//
// Different build environments may use different LLVM/MLIR 21.x versions.
// Several APIs changed across these versions; the macros below let the
// same source compile against either.
//
// Detection relies on LLVM_VERSION_MAJOR/MINOR/PATCH from
// llvm/Config/llvm-config.h, which is always available when linking
// against LLVM.
//
// Macro summary:
//   COIR_LLVM_PRE_21_1  -- defined when building against LLVM < 21.1.
//   COIR_LLVM_21_1_PLUS -- defined when building against LLVM >= 21.1.
//
// Helper wrappers:
//   COIR_SET_TARGET_TRIPLE(mod, tripleExpr)
//       mod.setTargetTriple(tripleExpr)            // pre-21.1: StringRef
//       mod.setTargetTriple(llvm::Triple(tripleExpr)) // 21.1+: Triple
//
//   COIR_CREATE_TARGET_MACHINE(target, tripleExpr, cpu, features, opts, reloc)
//       target->createTargetMachine(tripleExpr, ...)      // pre-21.1
//       target->createTargetMachine(llvm::Triple(tripleExpr), ...) // 21.1+
//
//   COIR_CONSTANT_INT(builder, loc, value, type)
//       arith::ConstantIntOp(loc, value, type)     // pre-21.1
//       arith::ConstantIntOp(loc, type, value)     // 21.1+
//
//   COIR_CONSTANT_FLOAT(builder, loc, value, type)
//       arith::ConstantFloatOp(loc, value, type)   // pre-21.1
//       arith::ConstantFloatOp(loc, type, value)   // 21.1+

#ifndef COIR_VERSION_COMPAT_H
#define COIR_VERSION_COMPAT_H

#include "llvm/Config/llvm-config.h"

// --- Version classification -----------------------------------------
// Versions before 21.1 use the older API; 21.1 and later use the new one.
#if defined(LLVM_VERSION_MAJOR) && defined(LLVM_VERSION_MINOR)
  #if LLVM_VERSION_MAJOR < 21 ||                                               \
      (LLVM_VERSION_MAJOR == 21 && LLVM_VERSION_MINOR < 1)
    #define COIR_LLVM_PRE_21_1 1
  #else
    #define COIR_LLVM_21_1_PLUS 1
  #endif
#else
  // Fallback: assume the newer public API.
  #define COIR_LLVM_21_1_PLUS 1
#endif

// --- LLVM Target / Triple helpers ----------------------------------

// Module::setTargetTriple: StringRef (pre-21.1) vs Triple (21.1+)
#define COIR_SET_TARGET_TRIPLE(mod, tripleExpr)                                \
  COIR_SET_TARGET_TRIPLE_IMPL(mod, tripleExpr)

#ifdef COIR_LLVM_PRE_21_1
  #define COIR_SET_TARGET_TRIPLE_IMPL(mod, tripleExpr)                         \
    (mod).setTargetTriple((tripleExpr))
#else
  #define COIR_SET_TARGET_TRIPLE_IMPL(mod, tripleExpr)                         \
    (mod).setTargetTriple(llvm::Triple((tripleExpr)))
#endif

// Target::createTargetMachine: StringRef (pre-21.1) vs Triple (21.1+)
#define COIR_CREATE_TARGET_MACHINE(target, tripleExpr, cpu, features, opts,    \
                                   reloc)                                      \
  (target)->createTargetMachine(COIR_TRIPLE_OR_REF(tripleExpr), (cpu),         \
                                (features), (opts), (reloc))

#ifdef COIR_LLVM_PRE_21_1
  #define COIR_TRIPLE_OR_REF(tripleExpr) (tripleExpr)
#else
  #define COIR_TRIPLE_OR_REF(tripleExpr) llvm::Triple((tripleExpr))
#endif

// --- MLIR Arith constant builders ---------------------------------
//
// ConstantIntOp::build:
//   pre-21.1:  (loc, value, type)
//   21.1+:     (loc, type, value)
//
// ConstantFloatOp::build:
//   pre-21.1:  (loc, value, type)
//   21.1+:     (loc, type, value)

#ifdef COIR_LLVM_PRE_21_1
  #define COIR_CONSTANT_INT(builder, loc, value, type)                         \
    (builder).create<mlir::arith::ConstantIntOp>((loc), (value), (type))
  #define COIR_CONSTANT_FLOAT(builder, loc, value, type)                       \
    (builder).create<mlir::arith::ConstantFloatOp>((loc), (value), (type))
#else
  #define COIR_CONSTANT_INT(builder, loc, value, type)                         \
    (builder).create<mlir::arith::ConstantIntOp>((loc), (type), (value))
  #define COIR_CONSTANT_FLOAT(builder, loc, value, type)                       \
    (builder).create<mlir::arith::ConstantFloatOp>((loc), (type), (value))
#endif

// --- MLIR memref::ReinterpretCastOp --------------------------------
//
// The OpFoldResult-based builder exists in both versions.
// The ValueRange-based builder also exists in both.  The difference is
// only in argument order for the constant ops used to build the
// sizes/strides, which is handled by COIR_CONSTANT_INT above.
//
// No separate macro is needed for ReinterpretCastOp itself.

// --- Clang frontend API --------------------------------------------
//
// TextDiagnosticPrinter ctor:
//   pre-21.1:  (os, DiagnosticOptions*)
//   21.1+:     (os, const DiagnosticOptions&)
//
// DiagnosticsEngine ctor:
//   pre-21.1:  (diagIDs, IntrusiveRefCntPtr<DiagnosticOptions>, consumer)
//   21.1+:     (diagIDs, const DiagnosticOptions&, consumer)
//
// CompilerInstance:
//   pre-21.1:  default ctor + setInvocation(invocation)
//   21.1+:     ctor(std::shared_ptr<CompilerInvocation>)
//   (handled with #ifdef in source, not a macro, because the pre-21.1
//    form requires two statements)

#ifdef COIR_LLVM_PRE_21_1
  // Use IntrusiveRefCntPtr + raw pointer API (older LLVM 21.x).
  #define COIR_DIAG_OPTS_TYPE()                                                \
    llvm::makeIntrusiveRefCnt<clang::DiagnosticOptions>()
  #define COIR_DIAG_OPTS_GET_PTR(opts) (opts).get()
  #define COIR_DIAG_OPTS_DEREF(opts) (opts).get()
  #define COIR_TEXT_DIAG_PRINTER(os, opts)                                     \
    std::make_unique<clang::TextDiagnosticPrinter>((os), (opts).get())
  #define COIR_DIAGS_ENGINE(diagIDs, opts, printer)                            \
    clang::DiagnosticsEngine((diagIDs), (opts), (printer).release())
#else
  // Use unique_ptr + reference API (LLVM 21.1+).
  #define COIR_DIAG_OPTS_TYPE() std::make_unique<clang::DiagnosticOptions>()
  #define COIR_DIAG_OPTS_GET_PTR(opts) (opts).get()
  #define COIR_DIAG_OPTS_DEREF(opts) *(opts)
  #define COIR_TEXT_DIAG_PRINTER(os, opts)                                     \
    std::make_unique<clang::TextDiagnosticPrinter>((os), *(opts))
  #define COIR_DIAGS_ENGINE(diagIDs, opts, printer)                            \
    clang::DiagnosticsEngine((diagIDs), *(opts), (printer).release())
#endif

#endif // COIR_VERSION_COMPAT_H
