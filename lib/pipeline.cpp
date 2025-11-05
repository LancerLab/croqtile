#include "pipeline.hpp"
#include "codegen.hpp"
#include "codegen_cuda.hpp"
#include "codegen_cute.hpp"
#include "codegen_factor.hpp"
#include "codegen_prepare.hpp"
#include "codegen_topscc.hpp"
#include "earlysema.hpp"
#include "gcucheck.hpp"
#include "gpuadapt.hpp"
#include "latenorm.hpp"
#include "liveness_analysis.hpp"
#include "loop_vectorize.hpp"
#include "mem_reuse.hpp"
#include "memcheck.hpp"
#include "normalize.hpp"
#include "semacheck.hpp"
#include "shapeinfer.hpp"
#include "sym_replace.hpp"
#include "ttrans_factor.hpp"
#include "ttrans_topscc.hpp"
#include "typeinfer.hpp"
#include "visualize.hpp"

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
  return !abend;
}

ASTPipeline& ASTPipeline::PlanSemanticRoutine() {
  // Initialize the common ast pipeline
  // apply early semantics check without knowing type details
  AddStage<EarlySemantics>();
  // minor AST change: desugar for canonicalized AST
  AddStage<Normalizer>();
  if (CCtx().GetTarget() == CompileTarget::Factor) AddStage<SymReplace>();
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
  if (!CCtx().NoVectorize() && CCtx().GetTarget() == CompileTarget::Topscc) {
    AddStage<LoopVectorizer>();
    if (CCtx().TraceVectorize())
      AddAction([](ASTPipeline& p) { p.SetAbend(); });
    AddAction(
        [](ASTPipeline& p) { CCtx().SetGlobalSymbolTable(p.LastSymTab()); });
  }

  if ((CCtx().GetTarget() == CompileTarget::Topscc) && CCtx().MemReuse())
    AddStage<MemoryReuse>();
  if ((CCtx().GetTarget() == CompileTarget::Cute) && CCtx().MemReuse())
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

  switch (CCtx().GetTarget()) {
  case CompileTarget::Factor: {
    // apply the gcu specific checking
    AddStage<GCUCheck>();
    AddStage<FactorTrans>();
    AddStage<MemUsageCheck>();
    AddStage<Factor::FactorCodeGen>();
    break;
  }
  case CompileTarget::Topscc: {
    // apply GCU specific checks
    AddStage<GCUCheck>();
    AddStage<MemUsageCheck>();
    AddStage<Topscc::TopsccCodeGen>();
    break;
  }
  case CompileTarget::CUDA: {
    AddStage<MemUsageCheck>();
    AddStage<CUDA::CUDACodeGen>();
    break;
  }
  case CompileTarget::Cute: {
    AddStage<GPUAdaptor>();
    AddStage<MemUsageCheck>();
    AddStage<Cute::CuteCodeGen>();
    break;
  }
  default:
    errs() << "Invalid target: '" << STR(CCtx().GetTarget()) << "'\n";
    abend = true;
  }
  return *this;
}

ASTPipeline& ASTPipeline::Get() {
  std::call_once(init_flag,
                 []() { instance = std::make_unique<ASTPipeline>(); });
  return *instance;
}
