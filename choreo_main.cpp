#include <getopt.h>

#include <cstdlib>

#include "enums.hpp"
#include "ast.hpp"
#include "codegen.hpp"
#include "desugar.hpp"
#include "options.hpp"
#include "scanner.hpp"
#include "symtab.hpp"
#include "symvalid.hpp"
#include "typecheck.hpp"
#include "typeinfer.hpp"
#include "types.hpp"
#include "valno.hpp"
#include "visualize.hpp"

using namespace Choreo;

location loc;
AST::Program root(loc);
SymbolTable symtab;
StringifyTable strtab;

using namespace AST;
using namespace Choreo;

int main(int argc, char* argv[]) {
  Option<std::string> output("--output", "-o", "", true);
  Option<bool> debug_on("--debug", "-d", false, false);
  Option<bool> dump_ast("--dump-ast", "-e", false, false);
  Option<bool> print_vn("--print-valno", "-v", false, false);
  Option<bool> dump_inf("--dump-infer", "-i", false, false);
  Option<bool> sema_chk("--sema-check", "-s", false, false);
  Option<bool> del_comm("--remove-comments", "-n", false, false);

  // parse all the options
  OptionRegistry& r = OptionRegistry::GetInstance();
  if (!r.Parse(argc, argv)) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    exit(1);
  }
  r.SetOutputStream(output.GetValue());

  std::string filename = r.GetInputFileName();
  loc.begin.filename = loc.end.filename = &filename;

  Scanner s;
  s.yyrestart(r.GetInputStream());
  Parser p(s);

  if (debug_on) {
    std::cout << "Choreo: Debug of parsing is switched on." << std::endl;
    p.set_debug_level(1);  // Enable Bison debugging
    Scanner::SetDebug();
  }

  if (del_comm) Scanner::SetRemoveComments();

  p.parse();

  if (dump_ast) {
    if (sema_chk)
      std::cerr
          << "Warning: Semantic check is ignored since dumping AST is required."
          << std::endl;

    root.Print(std::cout);
    return 0;
  }

  // verify symbol references inside scopes
  SymbolValidator sv;
  root.accept(sv);
  if (sv.HasError())
    return 1;

  // minor AST change: desugar for canonicalized AST
  DeSugaring ds;
  root.accept(ds);

  // perform shape inference of mdspans, future, etc.
  ShapeInference si(print_vn);
  root.accept(si);

  // inference all the unknown types - decls
  TypeInference ti(dump_inf);
  root.accept(ti);
  if (dump_inf) return 0;

  // debug: dump the symbol table
  if (std::getenv("DUMP_SYMTAB"))
    ti.SymTab()->Print(std::cout);

  if (std::getenv("VISUALIZE")) {
    Visualizer vl(ti.SymTab());
    root.accept(vl);
  }

  // apply type check and generate symbol table
  TypeChecker sc(ti.SymTab());
  root.accept(sc);

  if (sema_chk) return 0;

  Choreo::Target tgt = Choreo::Target::Factor;

  if (tgt == Target::Factor) {
    FactorCodeGen codegen(std::cout, sc.SymTab());
    root.accept(codegen);
  }

  return 0;
}

unsigned SymbolTable::anonymous_count = 0;
unsigned SymbolTable::anon_type_count = 0;
