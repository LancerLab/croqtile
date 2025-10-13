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
#include "mem_reuse.hpp"
#include "memcheck.hpp"
#include "normalize.hpp"
#include "options.hpp"
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
#include "verifier.hpp"
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

  ASTVerify vf;

  // apply early semantics check without knowing type details
  EarlySemantics sv;
  if (!sv.RunOnProgram(root)) return sv.Status();
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  // minor AST change: desugar for canonicalized AST
  Normalizer ds;
  if (!ds.RunOnProgram(root)) return ds.Status();
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  if (CCtx().GetTarget() != CompileTarget::Topscc) {
    SymReplace sr;
    if (!sr.RunOnProgram(root)) return sr.Status();
    if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);
  }

  // perform shape inference of mdspans, future, etc.
  ShapeInference si;
  if (!si.RunOnProgram(root)) return si.Status();
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  // inference all the unknown types - decls
  TypeInference ti;
  if (!ti.RunOnProgram(root)) return ti.Status();
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  if (CCtx().ShowInferredTypes() || CCtx().TraceValueNumbers()) return 0;

  // late normalize
  LateNorm ln(ti.SymTab());
  if (!ln.RunOnProgram(root)) return ln.Status();
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  CCtx().SetGlobalSymbolTable(ln.SymTab());

  // debug: dump the symbol table
  if (std::getenv("DUMP_SYMTAB") || CCtx().DumpSymtab())
    CCtx().GetGlobalSymbolTable()->Print(dbgs());

  if (std::getenv("VISUALIZE") || CCtx().Visualize()) {
    Visualizer vl;
    if (!vl.RunOnProgram(root)) return vl.Status();
    return 0;
  }

  LivenessAnalyzer la;
  if (!la.RunOnProgram(root)) return la.Status();

  MemAnalyzer ma;
  if (!ma.RunOnProgram(root)) return ma.Status();
  if (CCtx().GetTarget() == CompileTarget::Topscc) {
    MemReuse mr(la, ma);
    if (CCtx().MemReuse() && !mr.RunOnProgram(root)) return mr.Status();
  }
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  // apply the semantic check
  SemaChecker sc;
  if (!sc.RunOnProgram(root)) return sc.Status();
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

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
    Choreo::CUDA::CUDACodeGen codegen;
    if (!codegen.RunOnProgram(root)) return codegen.Status();
    break;
  }
  case CompileTarget::Cute: {
    GPUAdaptor gpu_adaptor;
    if (!gpu_adaptor.RunOnProgram(root)) return gpu_adaptor.Status();

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
