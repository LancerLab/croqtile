#include "MemUsageCheck.hpp"
#include "ast.hpp"
#include "codegen.hpp"
#include "codegen_cuda.hpp"
#include "codegen_factor.hpp"
#include "dynshape.hpp"
#include "earlysema.hpp"
#include "enums.hpp"
#include "gcucheck.hpp"
#include "latenorm.hpp"
#include "normalize.hpp"
#include "options.hpp"
#include "scanner.hpp"
#include "sym_replace.hpp"
#include "symtab.hpp"
#include "ttrans_factor.hpp"
#include "typecheck.hpp"
#include "typeinfer.hpp"
#include "types.hpp"
#include "valno.hpp"
#include "visualize.hpp"
#include <cstdlib>
#include <getopt.h>

using namespace Choreo;

location loc;
AST::Program root(loc);
SymbolTable symtab;

using namespace AST;
using namespace Choreo;

int main(int argc, char* argv[]) {
  Option<std::string> arch("--architecture", "-arch", "gcu300");
  Option<std::string> output("--output", "-o", "", true);
  Option<std::string> target("--target", "-t", "factor", true);
  Option<std::string> abend_after("--stop-after", "-sa", "");
  Option<std::string> trace_visit("--trace-visit", "-tv", "");
  Option<std::string> debug_visit("--debug-visit", "-dv", "");
  Option<std::string> print_ahead("--print-before", "-pb", "");
  Option<std::string> print_after("--print-after", "-pa", "");
  Option<bool> print_ahead_all("--print-before-all", "-pba", false);
  Option<bool> print_after_all("--print-after-all", "-paa", false);
  Option<bool> cross_compile("--cross-compile", "-cc", false);
  Option<bool> debug_on("--debug", "-d", false);
  Option<bool> dump_ast("--dump-ast", "-e", false);
  Option<bool> print_vn("--print-valno", "-v", false);
  Option<bool> inf_type("--infer-types", "-i", false);
  Option<bool> dump_sym("--dump-symbol", "-l", false);
  Option<bool> visualiz("--visualize", "-u", false);
  Option<bool> ncodegen("--no-codegen", "-s", false);
  Option<bool> del_comm("--remove-comments", "-n", false);
  Option<bool> sym_repl("--print-sym-replace", "-sr", false);
  Option<bool> prt_pass("--show-passes", "-sp", false);

  // parse all the options
  OptionRegistry& r = OptionRegistry::GetInstance();
  if (!r.Parse(argc, argv)) {
    errs() << "Usage: " << argv[0] << " <filename>\n";
    exit(1);
  }
  r.SetOutputStream(output.GetValue());

  if (!trace_visit.GetValue().empty())
    setenv("CHOREO_TRACE_VISITOR", ToUpper(trace_visit.GetValue()).c_str(), 1);

  if (!debug_visit.GetValue().empty())
    setenv("CHOREO_DEBUG_VISITOR", ToUpper(debug_visit.GetValue()).c_str(), 1);

  if (!print_ahead.GetValue().empty())
    setenv("CHOREO_PRINT_BEFORE", ToUpper(print_ahead.GetValue()).c_str(), 1);

  if (print_ahead_all) setenv("CHOREO_PRINT_BEFORE", "ALLPASSES", 1);

  if (!print_after.GetValue().empty())
    setenv("CHOREO_PRINT_AFTER", ToUpper(print_after.GetValue()).c_str(), 1);

  if (print_after_all) setenv("CHOREO_PRINT_AFTER", "ALLPASSES", 1);

  if (prt_pass) setenv("CHOREO_PRINT_PASSES", "", 1);

  if (!abend_after.GetValue().empty())
    setenv("CHOREO_STOP_AFTER_PASS", ToUpper(abend_after.GetValue()).c_str(),
           1);

  if (dump_ast) {
    if (ncodegen)
      errs()
          << "Warning: Semantic check is ignored since dumping AST is required."
          << std::endl;
  }

  std::string filename = r.GetInputFileName();
  loc.begin.filename = loc.end.filename = &filename;

  if (prt_pass) dbgs() << "|- " << filename << "\n";

  Scanner s;
  s.yyrestart(r.GetInputStream());
  Parser p(s);

  if (debug_on) {
    dbgs() << "Choreo: Debug of parsing is switched on." << std::endl;
    p.set_debug_level(1); // Enable Bison debugging
    Scanner::SetDebug();
  }

  if (del_comm) Scanner::SetRemoveComments();

  if (prt_pass) dbgs() << "|- parse program into AST.\n";
  if (p.parse() != 0) {
    errs() << "Parsing failed due to syntax errors." << std::endl;
    return 1;
  }

  if (dump_ast) {
    root.Print(dbgs());
    return 0;
  }

  auto tgt = Choreo::Target::Unknown;
  if (target.GetValue() == "factor") tgt = Choreo::Target::Factor;
  if (target.GetValue() == "cuda") tgt = Choreo::Target::CUDA;

  // apply early semantics check without knowing type details
  EarlySemantics sv(tgt);
  if (!sv.RunProgram(root)) return sv.Status();

  // minor AST change: desugar for canonicalized AST
  Normalizer ds;
  if (!ds.RunProgram(root)) return ds.Status();

  SymReplace sr;
  if (!sr.RunProgram(root)) return sr.Status();

  // perform shape inference of mdspans, future, etc.
  ShapeInference si(print_vn);
  if (!si.RunProgram(root)) return si.Status();

  // inference all the unknown types - decls
  TypeInference ti(inf_type);
  if (!ti.RunProgram(root)) return ti.Status();
  if (inf_type || print_vn) return 0;

  // late normalize
  LateNorm ln(ti.SymTab());
  if (!ln.RunProgram(root)) return ln.Status();

  BufferInfoCollect bic(ln.SymTab());
  if (!bic.RunProgram(root)) return bic.Status();

  BufferGenerate bg(ln.SymTab(), bic.FBInfo());
  if (!bg.RunProgram(root)) return bg.Status();

  // debug: dump the symbol table
  if (std::getenv("DUMP_SYMTAB") || dump_sym) bg.SymTab()->Print(dbgs());

  if (std::getenv("VISUALIZE") || visualiz) {
    Visualizer vl(bg.SymTab());
    if (!vl.RunProgram(root)) return vl.Status();
    return 0;
  }

  // apply the type check
  TypeChecker sc(bg.SymTab(), tgt);
  if (!sc.RunProgram(root)) return sc.Status();

  // --------- Following passes generate codes -------- //

  if (ncodegen) return 0; // do not generate code

  // collect information for codegen
  CodegenPrepare cgp(sc.SymTab());
  if (!cgp.RunProgram(root)) return cgp.Status();

  switch (tgt) {
  case Target::Factor: {
    // apply the gcu specific checking
    GCUCheck gcu_checker(sc.SymTab());
    if (!gcu_checker.RunProgram(root)) return gcu_checker.Status();

    FactorTrans trans(sc.SymTab(), bg.FBInfo());
    trans.SetKind(FactorTrans::Kind::T_SELECT);
    if (!trans.RunProgram(root)) return trans.Status();
    trans.SetKind(FactorTrans::Kind::T_SWAP);
    if (!trans.RunProgram(root)) return trans.Status();

    assert(arch.GetValue().size() >= 3 &&
           arch.GetValue().substr(0, 3) == "gcu");
    MemUsageCheck mem_usage_checker(sc.SymTab(), Target::Factor,
                                    arch.GetValue());
    if (!mem_usage_checker.RunProgram(root)) return mem_usage_checker.Status();

    Choreo::Factor::FactorCodeGen codegen(
        sc.SymTab(), mem_usage_checker.GetRtMemUsageInfo(), bg.FBInfo(),
        cgp.GetASTInfo(), cross_compile);
    if (!codegen.RunProgram(root)) return codegen.Status();
    break;
  }
  case Target::CUDA: {
    Choreo::CUDA::CUDACodeGen codegen(sc.SymTab(), cross_compile);
    if (!codegen.RunProgram(root)) return codegen.Status();
    break;
  }
  case Target::Topscc: {
    errs() << "Target '" << target.GetValue()
           << "' has not been supported yet.\n";
    return 1;
  }
  case Target::Unknown: {
    errs() << "Invalid target: '" << target.GetValue() << "'\n";
    return 1;
  }
  default:
    errs() << "Invalid target: '" << target.GetValue() << "'\n";
    return 1;
  }

  return 0;
}

unsigned SymbolTable::anonymous_count = 0;
unsigned SymbolTable::anon_type_count = 0;
