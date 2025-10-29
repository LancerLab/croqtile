#include "ast.hpp"
#include "codegen.hpp"
#include "codegen_cuda.hpp"
#include "codegen_cute.hpp"
#include "codegen_factor.hpp"
#include "codegen_prepare.hpp"
#include "codegen_topscc.hpp"
#include "command_line.hpp"
#include "earlysema.hpp"
#include "gcucheck.hpp"
#include "gpuadapt.hpp"
#include "latenorm.hpp"
#include "liveness_analysis.hpp"
#include "loop_vectorize.hpp"
#include "mem_reuse.hpp"
#include "memcheck.hpp"
#include "normalize.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "scanner.hpp"
#include "semacheck.hpp"
#include "shapeinfer.hpp"
#include "sym_replace.hpp"
#include "symtab.hpp"
#include "ttrans_factor.hpp"
#include "ttrans_topscc.hpp"
#include "typeinfer.hpp"
#include "types.hpp"
#include "visualize.hpp"
#include <cstdlib>
#include <getopt.h>

using namespace Choreo;

extern AST::Program root;

using namespace AST;
using namespace Choreo;

int main(int argc, char* argv[]) {
  CommandLine cl;
  if (!cl.Parse(argc, argv)) return cl.ReturnCode();

  auto& r = OptionRegistry::GetInstance();

  if (CCtx().DumpAst() && CCtx().NoCodegen())
    errs() << "warning: Semantic check is ignored since dumping AST is "
              "required.\n";

  if (CCtx().PrintPassNames())
    dbgs() << "<file: " << r.GetInputFileName() << ">\n";

  // Apply the preprocessing
  std::stringstream pps;
  std::stringstream cok_ss;
  if (!CCtx().NoPreProcess()) {
    if (CCtx().PrintPassNames()) dbgs() << "|- preprocess the choreo program\n";
    if (CCtx().GetOutputKind() == OutputKind::PreProcessedCode) {
      SimplePreprocessor spp(r.GetOutputStream());
      if (!spp.Process(r.GetInputStream())) return 1;
      return 0;
    } else {
      SimplePreprocessor spp(pps);
      if (!spp.Process(r.GetInputStream())) return 1;
      if (CCtx().GetTarget() == CompileTarget::Topscc)
        if (!spp.ExtractDeviceKernel(cok_ss)) return 1;
    }
  }

  if (CCtx().GetOutputKind() == OutputKind::PreProcessedCode) return 0;

  Scanner s;
  PContext pctx;
  Parser p(pctx, s);
  Parser cok_p(pctx, s);

  if (CCtx().DebugAll()) {
    dbgs() << "Choreo: Debug of parsing is switched on." << std::endl;
    p.set_debug_level(1); // Enable Bison debugging
    Scanner::SetDebug();
  }

  if (CCtx().DropComments()) Scanner::SetRemoveComments();

  if (CCtx().PrintPassNames()) dbgs() << "|- parse program into AST.\n";
  if (CCtx().GetTarget() == CompileTarget::Topscc) {
    Scanner::SetLocationUpdate(false);
    s.yyrestart(cok_ss);
    if (cok_p.parse() != 0 || pctx.HasError()) {
      errs() << "Parsing failed due to syntax errors." << std::endl;
      return 1;
    }
  }

  Scanner::SetLocationUpdate(true);
  s.yyrestart((CCtx().NoPreProcess()) ? r.GetInputStream() : pps);
  if (p.parse() != 0 || pctx.HasError()) {
    errs() << "Parsing failed due to syntax errors." << std::endl;
    return 1;
  }

  if (CCtx().DumpAst()) {
    root.Print(dbgs());
    return 0;
  }

  // Initialize the common ast pipeline
  ASTPipeline pl(CCtx().VerifyVisitors());
  // apply early semantics check without knowing type details
  pl.AddStage<EarlySemantics>();
  // minor AST change: desugar for canonicalized AST
  pl.AddStage<Normalizer>();
  if (CCtx().GetTarget() == CompileTarget::Factor) pl.AddStage<SymReplace>();
  // perform shape inference of mdspans, future, etc.
  pl.AddStage<ShapeInference>();
  // inference all the unknown types - decls
  pl.AddStage<TypeInference>();
  pl.AddAction([](ASTPipeline& p) {
    if (CCtx().ShowInferredTypes() || CCtx().TraceValueNumbers()) p.SetAbend();
    CCtx().SetGlobalSymbolTable(p.LastSymTab());
  });
  // late normalize
  pl.AddStage<LateNorm>();
  pl.AddAction([](ASTPipeline& p) {
    CCtx().SetGlobalSymbolTable(p.LastSymTab());

    // debug: dump the symbol table
    if (std::getenv("DUMP_SYMTAB") || CCtx().DumpSymtab())
      CCtx().GetGlobalSymbolTable()->Print(dbgs());
  });
  // to visualize the dma
  if (std::getenv("VISUALIZE") || CCtx().Visualize())
    pl.AddStageWithPost<Visualizer>([](ASTPipeline& p) { p.SetAbend(); });

  // early loop vectorizer for the certain target
  if (!CCtx().NoVectorize() && CCtx().GetTarget() == CompileTarget::Topscc) {
    pl.AddStage<LoopVectorizer>();
    if (CCtx().TraceVectorize())
      pl.AddAction([](ASTPipeline& p) { p.SetAbend(); });
    pl.AddAction(
        [](ASTPipeline& p) { CCtx().SetGlobalSymbolTable(p.LastSymTab()); });
  }

  if ((CCtx().GetTarget() == CompileTarget::Topscc) && CCtx().MemReuse())
    pl.AddStage<MemReuse>();
  // apply the semantic check
  pl.AddStage<SemaChecker>();

  if (!pl.RunOnProgram(root)) return pl.Status();

  // --------- Following passes generate codes -------- //

  if (CCtx().NoCodegen()) return 0; // do not generate code

  // collect information for codegen
  CodegenPrepare cgp;
  if (!cgp.RunOnProgram(root)) return cgp.Status();

  switch (CCtx().GetTarget()) {
  case CompileTarget::Factor: {
    // apply the gcu specific checking
    GCUCheck gcu_checker;
    if (!gcu_checker.RunOnProgram(root)) return gcu_checker.Status();

    FactorTrans trans;
    if (!trans.RunOnProgram(root)) return trans.Status();

    MemUsageCheck mem_usage_checker;
    if (!mem_usage_checker.RunOnProgram(root))
      return mem_usage_checker.Status();

    Choreo::Factor::FactorCodeGen codegen(cgp.GetASTInfo());
    if (!codegen.RunOnProgram(root)) return codegen.Status();
    break;
  }
  case CompileTarget::Topscc: {
    // apply GCU specific checks
    GCUCheck gcu_checker;
    if (!gcu_checker.RunOnProgram(root)) return gcu_checker.Status();

#if 0
    TopsccTrans trans;
    if (!trans.RunOnProgram(root)) return trans.Status();
#endif

    MemUsageCheck muc;
    if (!muc.RunOnProgram(root)) return muc.Status();

    Choreo::Topscc::TopsccCodeGen codegen(cgp.GetASTInfo());
    if (!codegen.RunOnProgram(root)) return codegen.Status();
    break;
  }
  case CompileTarget::CUDA: {
    Choreo::MemUsageCheck muc;
    if (!muc.RunOnProgram(root)) return muc.Status();

    Choreo::CUDA::CUDACodeGen codegen;
    if (!codegen.RunOnProgram(root)) return codegen.Status();
    break;
  }
  case CompileTarget::Cute: {
    GPUAdaptor gpu_adaptor;
    if (!gpu_adaptor.RunOnProgram(root)) return gpu_adaptor.Status();

    Choreo::MemUsageCheck muc;
    if (!muc.RunOnProgram(root)) return muc.Status();

    Choreo::Cute::CuteCodeGen codegen(cgp.GetASTInfo());
    if (!codegen.RunOnProgram(root)) return codegen.Status();
    break;
  }
  default:
    errs() << "Invalid target: '" << STR(CCtx().GetTarget()) << "'\n";
    return 1;
  }

  return 0;
}
