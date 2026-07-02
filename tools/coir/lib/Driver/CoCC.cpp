//===- CoCC.cpp - CoIR Compiler Driver ------------------------------------===//
//
// End-to-end compiler driver for Choreo via CoIR path:
//   .co -> parse -> sema -> CoIR MLIR -> opt -> lower -> target codegen
//
//===----------------------------------------------------------------------===//

#include "ast.hpp"
#include "ASTCoIRGen.hpp"
#include "assess.hpp"
#include "aux.hpp"
#include "choreo_api.hpp"
#include "codegen.hpp"
#include "command_line.hpp"
#include "context.hpp"
#include "io.hpp"
#include "options.hpp"
#include "pipeline.hpp"

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "choreo_header.inc"
#include "choreo_types_header.inc"
#include "choreo_types_cute_header.inc"
#include "choreo_cute_header.inc"

using namespace Choreo;

extern Option<std::string> output;
extern Option<bool> emit_source;
extern Option<bool> generate_script;
extern Option<bool> dump_ast;

static Option<bool> emit_coir(OptionKind::User, "-emit-coir", "", false,
                              "Emit CoIR MLIR (like clang -emit-llvm).");
static Option<bool> no_opt(OptionKind::User, "--no-opt", "", false,
                           "Skip CoIR optimization passes.");
static Option<bool> compile_binary(OptionKind::User, "--compile-binary", "-cb",
                                   false,
                                   "Compile to executable binary.");

namespace {

bool ShouldEmitSource() { return emit_source || generate_script; }
bool ShouldGenerateBinary() { return compile_binary; }
bool ShouldGenerateAssembly() { return false; /* TODO */ }

void SetupScriptContext() {
  auto &sctx = CoIR::ScriptContext::Get();
  sctx.types_header = __choreo_types_header_as_string;
  sctx.runtime_header = __choreo_header_as_string;
  auto tname = CCtx().TargetName();
  if (tname == "cute" || tname == "nvptx") {
    sctx.types_cute_header = __choreo_types_cute_header_as_string;
    sctx.cute_header = __choreo_cute_header_as_string;
  }
  auto dcg = CCtx().GetTarget().MakeDeviceCodeGen();
  if (dcg) {
    std::ostringstream env_ss;
    dcg->SetupBuildEnv(env_ss);
    sctx.build_env = env_ss.str();
    std::ostringstream setup_ss;
    dcg->EmitSetupFiles(setup_ss, "$TMPDIR");
    sctx.target_setup = setup_ss.str();
  }
  auto arch_str = CCtx().GetArch();
  if (!arch_str.empty() && arch_str != "x86_64")
    sctx.arch_override = arch_str;
  else
    sctx.arch_override.clear();

  // CUDA home for in-memory compilation.
  const char *env = std::getenv("CUDA_HOME");
  if (env)
    sctx.cuda_home = env;
  else
    sctx.cuda_home = "/usr/local/cuda";
}

/// Extract -mcoir / -mcoir= tokens from argv and forward them to the
/// LLVM/MLIR cl::opt registry (like clang's -mllvm).  Returns the
/// filtered argc/argv for Choreo's CommandLine parser, or {0, {}} on error.
std::pair<int, std::vector<char *>>
ProcessCoIROptions(int argc, char *argv[]) {
  coir::registerCoIRPasses();
  mlir::registerPassManagerCLOptions();

  std::vector<std::string> coir_args;
  std::vector<char *> filtered;
  filtered.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "-mcoir") {
      if (i + 1 >= argc) {
        errs() << "error: -mcoir requires an argument.\n";
        return {0, {}};
      }
      coir_args.push_back(argv[++i]);
    } else if (arg.starts_with("-mcoir=")) {
      coir_args.push_back(arg.drop_front(7).str());
    } else {
      filtered.push_back(argv[i]);
    }
  }

  if (!coir_args.empty()) {
    std::vector<const char *> fwd;
    fwd.push_back("cocc (CoIR option parsing)");
    for (auto &a : coir_args) fwd.push_back(a.c_str());
    llvm::cl::ParseCommandLineOptions(fwd.size(), fwd.data(),
                                      "CoIR/MLIR options via -mcoir\n");
  }

  return {static_cast<int>(filtered.size()), std::move(filtered)};
}

/// Run the pre-codegen AST stages (CodegenPrepare + target adaptor) so that
/// the assessor contains both sema-level and target-level (HW constraint)
/// assertions.  RunFrontend() sets NoCodegen(true), so PlanCodeGenStages()
/// never executes; we run the pre-codegen subset explicitly.
void RunPreCodegenOnAST() {
  ASTPipeline pre_cg;
  CCtx().GetTarget().PlanPreCodegenStages(pre_cg);
  pre_cg.RunOnProgram(CompilerAPI::GetAST());
}

/// Collect sema-level stats directly from the assessment log.
/// Walks each function's assessor log to tally total/static_true/
/// static_false/runtime counts without running the full AssertSite pass.
void CollectSemaStats() {
  auto& stats = CCtx().GetAssessmentStats();
  for (auto& [fname, fctx] : CCtx().GetAllFunctionContexts()) {
    for (auto& entry : fctx.GetAssessor().GetAssessmentLog()) {
      stats.total++;
      switch (entry.outcome) {
      case AssessOutcome::STATIC_TRUE: stats.static_true++; break;
      case AssessOutcome::STATIC_FALSE: stats.static_false++; break;
      case AssessOutcome::RUNTIME: stats.runtime_total++; break;
      }
      switch (entry.usage_type) {
      case UsageType::UnClassified: stats.unclassified_total++; break;
      case UsageType::ShapeCompatibility: stats.shape_compat_total++; break;
      case UsageType::ElementAccess: stats.elem_access_total++; break;
      case UsageType::LoopBound: stats.loop_bound_total++; break;
      case UsageType::HardwareConstraint: stats.hw_constraint_total++; break;
      }
      if (entry.outcome == AssessOutcome::RUNTIME) {
        switch (entry.usage_type) {
        case UsageType::UnClassified: stats.unclassified_runtime++; break;
        case UsageType::ShapeCompatibility: stats.shape_compat_runtime++; break;
        case UsageType::ElementAccess: stats.elem_access_runtime++; break;
        case UsageType::LoopBound: stats.loop_bound_runtime++; break;
        case UsageType::HardwareConstraint: stats.hw_constraint_runtime++; break;
        }
      }
    }
  }
}

/// Count cross-parameter symbolic dimension consistency checks recorded as
/// coir.dim_checks attributes on KernelOps.  Each check becomes a host-side
/// runtime_check with ShapeCompatibility / ENTRY classification.
void CollectDimCheckStats(mlir::ModuleOp module) {
  auto &stats = CCtx().GetAssessmentStats();
  module->walk([&](coir::KernelOp kernel) {
    auto attr = kernel->getAttrOfType<mlir::ArrayAttr>("coir.dim_checks");
    if (!attr) return;
    unsigned n = attr.size();
    stats.total += n;
    stats.shape_compat_total += n;
    stats.runtime_total += n;
    stats.shape_compat_runtime += n;
    stats.runtime_entry += n;
  });
}

} // namespace

int main(int argc, char *argv[]) {
  auto [filt_argc, filt_argv] = ProcessCoIROptions(argc, argv);
  if (filt_argc == 0) return 1;

  CommandLine cl;
  if (!cl.Parse(filt_argc, filt_argv.data())) return cl.ReturnCode();

  auto &reg = OptionRegistry::GetInstance();
  std::string input_file = reg.GetInputFileName();

  if (!emit_source && !generate_script && !emit_coir && !dump_ast &&
      !compile_binary) {
    if (CCtx().GetTarget().IsBinaryOnlyCodeGen())
      compile_binary = true;
    else
      generate_script = true;
  }

  if (CCtx().GetTarget().IsBinaryOnlyCodeGen()) {
    if (emit_source || generate_script) {
      errs() << "error: target '" << CCtx().TargetName()
             << "' does not support -es or -gs\n";
      return 1;
    }
  }

  // --- Frontend + pre-codegen ---
  // Suppress stats from AST pipelines; cocc prints unified stats later.
  bool want_stats = CCtx().PrintStats();
  CCtx().SetPrintStats(false);
  CompilerAPI api;
  int fe_status = api.RunFrontend(input_file);
  if (fe_status != 0) return fe_status;

  // Run pre-codegen AST passes (CodegenPrepare + target adaptor).
  // Populates CodeGenInfo (PBTree, launch config) and registers HW-constraint
  // assessments via SBE.  After this, the assessor contains RUNTIME assertions
  // from both SemaChecker and the target adaptor.
  RunPreCodegenOnAST();
  CollectSemaStats();
  CCtx().SetPrintStats(want_stats);

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

  // Count cross-parameter dimension consistency checks emitted as
  // coir.dim_checks on KernelOps during IR generation.
  CollectDimCheckStats(session.Module());
  int coirThreshold = static_cast<int>(CCtx().RuntimeCheckCostThreshold());
  CoIR::Pipeline pipeline(session.Module(), session.Context(),
                           coirThreshold, CCtx().PrintStats());

  // --- Emit IR early exit ---
  if (emit_coir) {
    pipeline.EmitCoIR(llvm::outs());
    return 0;
  }

  // --- Safety instrumentation (always runs) ---
  if (!pipeline.InstrumentSafety()) {
    errs() << "cocc: safety instrumentation failed\n";
    return 1;
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

  // --- Shared lowering (DMA/MMA classification, hoisting) ---
  if (!pipeline.Lower()) {
    errs() << "cocc: lowering failed\n";
    return 1;
  }

  // --- Target-specific codegen ---
  int result = 0;
  if (ShouldEmitSource()) {
    bool is_script = generate_script;
    if (is_script) SetupScriptContext();
    std::string out_path =
        output.WasExplicitlySet() ? output.GetValue() : "";
    std::string arch = CCtx().GetArch();
    result = pipeline.EmitSource(CCtx().TargetName(), is_script, out_path, arch);
  } else if (ShouldGenerateBinary()) {
    SetupScriptContext();
    std::string out_path =
        output.WasExplicitlySet() ? output.GetValue() : "a.out";
    std::string arch = CCtx().GetArch();
    result = pipeline.CompileBinary(CCtx().TargetName(), out_path, arch);
  } else if (ShouldGenerateAssembly()) {
    choreo_unreachable("assembly generation not yet implemented");
  } else {
    choreo_unreachable("no output mode selected");
  }

  // --- Stats reporting ---
  // CollectAssertStats already wrote directly into CCtx().GetAssessmentStats().
  if (CCtx().PrintStats())
    PrintAssessmentStats(CCtx().GetAssessmentStats());

  return result;
}
