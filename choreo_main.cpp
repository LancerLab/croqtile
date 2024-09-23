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
#include "normalize.hpp"
#include "options.hpp"
#include "scanner.hpp"
#include "sym_replace.hpp"
#include "symtab.hpp"
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
  Option<std::string> output("--output", "-o", "", true);
  Option<std::string> target("--target", "-t", "factor", true);
  Option<std::string> stop_after("--stop-after", "-sa", "", true);
  Option<std::string> arch("--architecture", "-arch", "gcu300", true);
  Option<bool> debug_on("--debug", "-d", false, false);
  Option<bool> cross_compile("--cross-compile", "-cc", false, false);
  Option<bool> dump_ast("--dump-ast", "-e", false, false);
  Option<bool> print_vn("--print-valno", "-v", false, false);
  Option<bool> inf_type("--infer-types", "-i", false, false);
  Option<bool> dump_sym("--dump-symbol", "-l", false, false);
  Option<bool> visualiz("--visualize", "-u", false, false);
  Option<bool> gen_none("--no-codegen", "-s", false, false);
  Option<bool> del_comm("--remove-comments", "-n", false, false);
  Option<bool> mem_usag("--memory-usage-check", "-muc", false, false);
  Option<bool> sym_repl("--print-sym-replace", "-sr", false, false);
  Option<bool> prt_pass("--show-passes", "-sp", false, false);

  // parse all the options
  OptionRegistry& r = OptionRegistry::GetInstance();
  if (!r.Parse(argc, argv)) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    exit(1);
  }
  r.SetOutputStream(output.GetValue());

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
    p.set_debug_level(1);  // Enable Bison debugging
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

  // apply early semantics check without knowing type details
  EarlySemantics sv;
  if (prt_pass) std::cout << "|- " << sv.GetName() << "\n";
  root.accept(sv);
  if (sv.HasError()) return 1;
  if (stop_after.GetValue() == sv.GetName()) return 0;

  // minor AST change: desugar for canonicalized AST
  Normalizer ds(std::cout);
  if (prt_pass) std::cout << "|- " << ds.GetName() << "\n";
  root.accept(ds);
  if (stop_after.GetValue() == "norm") return 0;

  SymReplace sr(nullptr, sym_repl.GetValue(), std::cout);
  if (prt_pass) std::cout << "|- " << sr.GetName() << "\n";
  root.accept(sr);
  if (stop_after.GetValue() == "symreplace") return 0;

  // perform shape inference of mdspans, future, etc.
  ShapeInference si(print_vn);
  if (prt_pass) std::cout << "|- " << si.GetName() << "\n";
  root.accept(si);
  if (si.HasError()) return 1;
  if (stop_after.GetValue() == "shapeinfer") return 0;

  // inference all the unknown types - decls
  TypeInference ti(inf_type);
  if (prt_pass) std::cout << "|- " << ti.GetName() << "\n";
  root.accept(ti);
  if (ti.HasError()) return 1;
  if (inf_type || print_vn || (stop_after.GetValue() == "typeinf")) return 0;

  // debug: dump the symbol table
  if (std::getenv("DUMP_SYMTAB") || dump_sym) ti.SymTab()->Print(std::cout);

  if (std::getenv("VISUALIZE") || visualiz) {
    Visualizer vl(ti.SymTab());
    if (prt_pass) std::cout << "|- " << vl.GetName() << "\n";
    root.accept(vl);
    return 0;
  }

  // apply the type check
  TypeChecker sc(ti.SymTab());
  if (prt_pass) std::cout << "|- " << sc.GetName() << "\n";
  root.accept(sc);
  if (sc.HasError()) return 1;
  if (gen_none || (stop_after.GetValue() == "recheck")) return 0;

  // collect information for dynamic/runtime shape handling
  ShapeDynamics sds(ti.SymTab());
  if (prt_pass) std::cout << "|- " << sds.GetName() << "\n";
  root.accept(sds);
  if (sds.HasError()) return 1;

  auto tgt = Choreo::Target::Unknown;
  if (target.GetValue() == "factor") tgt = Choreo::Target::Factor;
  if (target.GetValue() == "cuda") tgt = Choreo::Target::CUDA;

  switch (tgt) {
    case Target::Factor: {
      // apply the gcu specific checking
      GCUCheck gcu_checker(sc.SymTab());
      if (prt_pass) std::cout << "|- " << gcu_checker.GetName() << "\n";
      root.accept(gcu_checker);
      if (gcu_checker.HasError()) return 1;
      if (stop_after.GetValue() == "gcucheck") return 0;

      assert(arch.GetValue().size() >= 3 &&
             arch.GetValue().substr(0, 3) == "gcu");
      MemUsageCheck mem_usage_checker(sc.SymTab(), Target::Factor,
                                      arch.GetValue(), mem_usag ? true : false);
      if (prt_pass) std::cout << "|- " << mem_usage_checker.GetName() << "\n";
      root.accept(mem_usage_checker);
      if (mem_usage_checker.HasError()) return 1;
      if (stop_after.GetValue() == "mucheck") return 0;

      Choreo::Factor::FactorCodeGen codegen(
          std::cout, sc.SymTab(), mem_usage_checker.GetRtMemUsageInfo(),
          cross_compile);
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
