#include <getopt.h>

#include <cstdlib>

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
  Option<bool> debug_on("--debug", "-d", false);
  Option<bool> cross_compile("--cross-compile", "-cc", false);
  Option<bool> dump_ast("--dump-ast", "-e", false);
  Option<bool> print_vn("--print-valno", "-v", false);
  Option<bool> inf_type("--infer-types", "-i", false);
  Option<bool> dump_sym("--dump-symbol", "-l", false);
  Option<bool> visualiz("--visualize", "-u", false);
  Option<bool> gen_none("--no-codegen", "-s", false);
  Option<bool> del_comm("--remove-comments", "-n", false);
  Option<bool> sym_repl("--print-sym-replace", "-sr", false);
  Option<bool> prt_pass("--show-passes", "-sp", false);

  // parse all the options
  OptionRegistry& r = OptionRegistry::GetInstance();
  if (!r.Parse(argc, argv)) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
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

  if (dump_ast) {
    if (gen_none)
      std::cerr
          << "Warning: Semantic check is ignored since dumping AST is required."
          << std::endl;
  }

  std::string filename = r.GetInputFileName();
  loc.begin.filename = loc.end.filename = &filename;

  if (prt_pass) std::cout << "|- " << filename << "\n";

  Scanner s;
  s.yyrestart(r.GetInputStream());
  Parser p(s);

  if (debug_on) {
    std::cout << "Choreo: Debug of parsing is switched on." << std::endl;
    p.set_debug_level(1); // Enable Bison debugging
    Scanner::SetDebug();
  }

  if (del_comm) Scanner::SetRemoveComments();

  if (prt_pass) std::cout << "|- parse program into AST.\n";
  if (p.parse() != 0) {
    std::cerr << "Parsing failed due to syntax errors." << std::endl;
    return 1;
  }

  if (dump_ast) {
    root.Print(std::cout);
    return 0;
  }

  std::string stop_after = ToUpper(abend_after.GetValue());

  auto tgt = Choreo::Target::Unknown;
  if (target.GetValue() == "factor") tgt = Choreo::Target::Factor;
  if (target.GetValue() == "cuda") tgt = Choreo::Target::CUDA;

  // apply early semantics check without knowing type details
  EarlySemantics sv(std::cout, tgt);
  if (prt_pass) std::cout << "|- " << sv.GetName() << "\n";
  root.accept(sv);
  if (sv.HasError()) return 1;
  if (stop_after == sv.GetName()) return 0;

  // minor AST change: desugar for canonicalized AST
  Normalizer ds(std::cout);
  if (prt_pass) std::cout << "|- " << ds.GetName() << "\n";
  root.accept(ds);
  if (stop_after == ds.GetName()) return 0;

  SymReplace sr(std::cout);
  if (prt_pass) std::cout << "|- " << sr.GetName() << "\n";
  root.accept(sr);
  if (stop_after == sr.GetName()) return 0;

  // perform shape inference of mdspans, future, etc.
  ShapeInference si(print_vn);
  if (prt_pass) std::cout << "|- " << si.GetName() << "\n";
  root.accept(si);
  if (si.HasError()) return 1;
  if (stop_after == si.GetName()) return 0;

  // inference all the unknown types - decls
  TypeInference ti(inf_type);
  if (prt_pass) std::cout << "|- " << ti.GetName() << "\n";
  root.accept(ti);
  if (ti.HasError()) return 1;
  if (inf_type || print_vn || (stop_after == ti.GetName())) return 0;

  // late normalize
  LateNorm ln(ti.SymTab(), std::cout);
  if (prt_pass) std::cout << "|- " << ln.GetName() << "\n";
  root.accept(ln);
  BufferInfoCollect bic(ln.SymTab());
  if (prt_pass) std::cout << "|- " << bic.GetName() << "\n";
  root.accept(bic);
  BufferGenerate bg(ln.SymTab(), bic.FBInfo());
  if (prt_pass) std::cout << "|- " << bg.GetName() << "\n";
  root.accept(bg);
  if (stop_after == ln.GetName()) return 0;

  // debug: dump the symbol table
  if (std::getenv("DUMP_SYMTAB") || dump_sym) ln.SymTab()->Print(std::cout);

  if (std::getenv("VISUALIZE") || visualiz) {
    Visualizer vl(ln.SymTab());
    if (prt_pass) std::cout << "|- " << vl.GetName() << "\n";
    root.accept(vl);
    return 0;
  }

  // apply the type check
  TypeChecker sc(ln.SymTab(), std::cout, tgt);
  if (prt_pass) std::cout << "|- " << sc.GetName() << "\n";
  root.accept(sc);
  if (sc.HasError()) return 1;
  if (gen_none || (stop_after == sc.GetName())) return 0;

  // collect information for dynamic/runtime shape handling
  ShapeDynamics sds(sc.SymTab());
  if (prt_pass) std::cout << "|- " << sds.GetName() << "\n";
  root.accept(sds);
  if (sds.HasError()) return 1;

  switch (tgt) {
  case Target::Factor: {
    // apply the gcu specific checking
    GCUCheck gcu_checker(sc.SymTab());
    if (prt_pass) std::cout << "|- " << gcu_checker.GetName() << "\n";
    root.accept(gcu_checker);
    if (gcu_checker.HasError()) return 1;
    if (stop_after == gcu_checker.GetName()) return 0;

    FactorTrans trans(sc.SymTab(), bg.FBInfo());
    if (prt_pass) std::cout << "|- " << trans.GetName() << "\n";
    trans.SetKind(FactorTrans::Kind::T_SELECT);
    root.accept(trans);
    trans.SetKind(FactorTrans::Kind::T_SWAP);
    root.accept(trans);
    if (trans.HasError()) return 1;
    if (stop_after == trans.GetName()) return 0;

    assert(arch.GetValue().size() >= 3 &&
           arch.GetValue().substr(0, 3) == "gcu");
    MemUsageCheck mem_usage_checker(sc.SymTab(), Target::Factor,
                                    arch.GetValue());
    if (prt_pass) std::cout << "|- " << mem_usage_checker.GetName() << "\n";
    root.accept(mem_usage_checker);
    if (mem_usage_checker.HasError()) return 1;
    if (stop_after == mem_usage_checker.GetName()) return 0;

    Choreo::Factor::FactorCodeGen codegen(std::cout, sc.SymTab(),
                                          mem_usage_checker.GetRtMemUsageInfo(),
                                          bg.FBInfo(), cross_compile);
    if (prt_pass) std::cout << "|- " << codegen.GetName() << "\n";
    root.accept(codegen);
    break;
  }
  case Target::CUDA: {
    Choreo::CUDA::CUDACodeGen codegen(std::cout, sc.SymTab(), cross_compile);
    if (prt_pass) std::cout << "|- " << codegen.GetName() << "\n";
    root.accept(codegen);
    break;
  }
  case Target::Topscc: {
    std::cerr << "Target '" << target.GetValue()
              << "' has not been supported yet.\n";
    return 1;
  }
  case Target::Unknown: {
    std::cerr << "Invalid target: '" << target.GetValue() << "'\n";
    return 1;
  }
  default:
    std::cerr << "Invalid target: '" << target.GetValue() << "'\n";
    return 1;
  }

  return 0;
}

unsigned SymbolTable::anonymous_count = 0;
unsigned SymbolTable::anon_type_count = 0;
