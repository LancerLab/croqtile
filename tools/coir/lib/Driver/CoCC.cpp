//===- CoCC.cpp - CoIR Compiler Driver ------------------------------------===//
//
// End-to-end compiler driver for Choreo via CoIR path:
//   .co -> parse -> sema -> CoIR MLIR -> opt -> lower -> emit
//
//===----------------------------------------------------------------------===//

#include "ast.hpp"
#include "ASTCoIRGen.hpp"
#include "aux.hpp"
#include "choreo_api.hpp"
#include "codegen.hpp"
#include "command_line.hpp"
#include "context.hpp"
#include "io.hpp"
#include "options.hpp"

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "choreo_header.inc"
#include "choreo_types_header.inc"

using namespace Choreo;

extern Option<std::string> output;
extern Option<bool> emit_source;
extern Option<bool> generate_script;
extern Option<bool> dump_ast;

static Option<bool> emit_coir(OptionKind::User, "-emit-coir", "", false,
                              "Emit CoIR MLIR (like clang -emit-llvm).");
static Option<bool> no_opt(OptionKind::User, "--no-opt", "", false,
                           "Skip CoIR optimization passes.");

namespace {

bool ShouldEmitSource() { return emit_source || generate_script; }
bool ShouldGenerateBinary() { return false; /* TODO */ }
bool ShouldGenerateAssembly() { return false; /* TODO */ }

void SetupScriptContext() {
  auto &sctx = CoIR::ScriptContext::Get();
  sctx.types_header = __choreo_types_header_as_string;
  sctx.runtime_header = __choreo_header_as_string;
  auto dcg = CCtx().GetTarget().MakeDeviceCodeGen();
  if (dcg) {
    std::ostringstream env_ss;
    dcg->SetupBuildEnv(env_ss);
    sctx.build_env = env_ss.str();
  }
  auto arch_str = CCtx().GetArch();
  if (!arch_str.empty() && arch_str != "x86_64")
    sctx.arch_override = arch_str;
  else
    sctx.arch_override.clear();
}

} // namespace

int main(int argc, char *argv[]) {
  CommandLine cl;
  if (!cl.Parse(argc, argv)) return cl.ReturnCode();

  auto &reg = OptionRegistry::GetInstance();
  std::string input_file = reg.GetInputFileName();

  if (!emit_source && !generate_script && !emit_coir && !dump_ast)
    generate_script = true;

  // --- Frontend ---
  CompilerAPI api;
  int fe_status = api.RunFrontend(input_file);
  if (fe_status != 0) return fe_status;

  if (dump_ast) {
    CompilerAPI::GetAST().Print(dbgs());
    return 0;
  }

  // --- AST -> CoIR MLIR ---
  CoIR::IRSession::Get().ResetModule(input_file);
  CoIR::ASTCoIRGen translator;
  translator.suppress_output = true;
  if (!translator.RunOnProgram(CompilerAPI::GetAST())) {
    errs() << "cocc: IR generation failed\n";
    return 1;
  }

  auto &session = CoIR::IRSession::Get();
  CoIR::Pipeline pipeline(session.Module(), session.Context());

  // --- Emit IR early exit ---
  if (emit_coir) {
    pipeline.EmitCoIR(llvm::outs());
    return 0;
  }

  // --- Optimization (skipped with --no-opt or -O0) ---
  if (!no_opt && CCtx().GetOptimizationLevel() > 0) {
    if (!pipeline.Optimize()) {
      errs() << "cocc: optimization failed\n";
      return 1;
    }
  }

  // --- Stamp target info onto module for lowering passes ---
  {
    auto& target = CCtx().GetTarget();
    auto arch = CCtx().GetArch();
    CoIR::StampTargetOnModule(session.Module(), CCtx().TargetName(), arch,
                              target.MMATargetName(arch),
                              target.HasTMA(arch),
                              target.HasDMA(arch));
  }

  // --- Lowering ---
  if (!pipeline.Lower()) {
    errs() << "cocc: lowering failed\n";
    return 1;
  }

  // --- Output dispatch ---
  if (ShouldEmitSource()) {
    bool is_script = generate_script;
    if (is_script) SetupScriptContext();
    std::string out_path =
        output.WasExplicitlySet() ? output.GetValue() : "";
    return pipeline.EmitSource(CCtx().TargetName(), is_script, out_path);
  } else if (ShouldGenerateBinary()) {
    choreo_unreachable("binary generation not yet implemented");
  } else if (ShouldGenerateAssembly()) {
    choreo_unreachable("assembly generation not yet implemented");
  } else {
    choreo_unreachable("no output mode selected");
  }

  return 0;
}
