#include "pipeline.hpp"
#include "codegen_prepare.hpp"
#include "earlysema.hpp"
#include "latenorm.hpp"
#include "liveness_analysis.hpp"
#include "loop_vectorize.hpp"
#include "mem_reuse.hpp"
#include "memcheck.hpp"
#include "normalize.hpp"
#include "semacheck.hpp"
#include "shapeinfer.hpp"
#include "typeinfer.hpp"
#include "visualize.hpp"
#include <iomanip>

using namespace Choreo;

std::once_flag ASTPipeline::init_flag;
std::unique_ptr<ASTPipeline> ASTPipeline::instance;

void ASTPipeline::Dump() const {
  dbgs() << "++ Pipeline Stages\n";
  for (auto& ps : pl)
    if ((ps.pred && ps.pred()) || !ps.pred)
      if (ps.v) dbgs() << " |-" << ps.v->GetName() << "\n";

  dbgs() << "++ END Pipeline\n";
}

bool ASTPipeline::RunOnProgram(AST::Node& root) {
  if (debug) Dump();
  // verify the input
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  for (auto& ps : pl) {
    if ((ps.pred && ps.pred()) || !ps.pred) {
      if (ps.v) {
        if (!ps.v->RunOnProgram(root)) {
          state = ps.v->Status();
          return false; // abend immediately
        }
        symtab = ps.v->SymTab();
        // TODO: force abend when failing on verifiers
        if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);
      }
      if (ps.cond_action) ps.cond_action(*this);
    }

    if (abend) return false;
    if (ps.action) ps.action(*this);
  }

  if (CCtx().PrintStats()) {
    const auto& s = CCtx().GetAssessmentStats();
    const char* sep =
        "===-------------------------------------------------------------------"
        "----===";
    errs() << "\n"
           << sep << "\n"
           << "                      ... Assessment Statistics ...\n"
           << sep << "\n";
    auto row = [&](size_t n, const char* desc) {
      errs() << std::right << std::setw(6) << n << "  assess  - " << desc
             << "\n";
    };
    row(s.total, "Assessments evaluated");
    row(s.static_true, "Resolved at compile time (static-true)");
    row(s.static_false, "Proven false at compile time (static-false)");
    row(s.runtime_total, "Runtime assertions generated");
    row(s.runtime_low, "Runtime assertions (low cost)");
    row(s.runtime_medium, "Runtime assertions (medium cost)");
    row(s.runtime_high, "Runtime assertions (high cost)");
    row(s.runtime_enabled, "Runtime assertions enabled");
    row(s.runtime_disabled, "Runtime assertions disabled by cost filter");
    errs() << "  ---\n";
    row(s.unclassified_total, "Assessments (unclassified)");
    row(s.shape_compat_total, "Assessments (shape-compatibility)");
    row(s.elem_access_total, "Assessments (element-access)");
    row(s.loop_bound_total, "Assessments (loop-bound)");
    row(s.hw_constraint_total, "Assessments (hw-constraint)");
    row(s.unclassified_runtime, "Runtime assertions (unclassified)");
    row(s.shape_compat_runtime, "Runtime assertions (shape-compatibility)");
    row(s.elem_access_runtime, "Runtime assertions (element-access)");
    row(s.loop_bound_runtime, "Runtime assertions (loop-bound)");
    row(s.hw_constraint_runtime, "Runtime assertions (hw-constraint)");
    errs() << sep << "\n";
  }

  return !abend;
}

ASTPipeline& ASTPipeline::PlanSemanticRoutine() {
  // Initialize the common ast pipeline
  // apply early semantics check without knowing type details
  AddStage<EarlySemantics>();
  // minor AST change: desugar for canonicalized AST
  AddStage<Normalizer>();
  // perform shape inference of mdspans, future, etc.
  AddStage<ShapeInference>();
  // inference all the unknown types - decls
  AddStage<TypeInference>();
  AddAction([](ASTPipeline& p) {
    if (CCtx().ShowInferredTypes() || CCtx().TraceValueNumbers()) p.SetAbend();
    CCtx().SetGlobalSymbolTable(p.LastSymTab());
  });
  // late normalize
  AddStage<LateNorm>();
  AddAction([](ASTPipeline& p) {
    CCtx().SetGlobalSymbolTable(p.LastSymTab());

    // debug: dump the symbol table
    if (std::getenv("DUMP_SYMTAB") || CCtx().DumpSymtab())
      CCtx().GetGlobalSymbolTable()->Print(dbgs());
  });
  // to visualize the dma
  if (std::getenv("VISUALIZE") || CCtx().Visualize())
    AddStageWithPost<Visualizer>([](ASTPipeline& p) { p.SetAbend(); });

  // early loop vectorizer for the certain target
  if (!CCtx().NoVectorize() && CCtx().TargetSupportVectorize()) {
    AddStage<LoopVectorizer>();
    if (CCtx().TraceVectorize())
      AddAction([](ASTPipeline& p) { p.SetAbend(); });
    AddAction(
        [](ASTPipeline& p) { CCtx().SetGlobalSymbolTable(p.LastSymTab()); });
  }

  if (CCtx().TargetSupportMemAlloc() && CCtx().MemReuse())
    AddStage<MemoryReuse>();

  // apply the semantic check
  AddStage<SemaChecker>();
  return *this;
}

ASTPipeline& ASTPipeline::PlanCodeGenRoutine() {
  if (CCtx().NoCodegen()) { // do not generate code
    AddAction([](ASTPipeline& p) { p.SetAbend(); });
  }

  AddStage<CodegenPrepare>();

  // delegate to target for codegen plan
  if (!CCtx().GetTarget().PlanCodeGenStages(*this)) {
    errs() << "Failed to initialize codegen for target '" << CCtx().TargetName()
           << "'\n";
    abend = true;
  }
  return *this;
}

ASTPipeline& ASTPipeline::Get() {
  std::call_once(init_flag,
                 []() { instance = std::make_unique<ASTPipeline>(); });
  return *instance;
}
