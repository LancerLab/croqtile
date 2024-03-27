#include <getopt.h>

#include <cstdlib>

#include "ast.hpp"
#include "codegen.hpp"
#include "scanner.hpp"
#include "valno.hpp"
#include "semantics.hpp"
#include "symtab.hpp"
#include "typeinfer.hpp"
#include "types.hpp"

using namespace Choreo;

location loc;
AST::Program root(loc);
SymbolTable symtab;
StringifyTable strtab;

using namespace AST;
using namespace Choreo;

int main(int argc, char* argv[]) {
  std::string filename;
  bool debugMode = false;
  bool onlyDumpAST = false;
  bool showInferOnly = false;
  bool printValueNumbers = false;
  bool onlySemaCheck = false;
  bool removeComments = false;

  enum class Target { Factor, Topscc };

  Target tgt = Target::Factor;

  // Define long options
  static struct option long_options[] = {
      {"debug", no_argument, 0, 'd'},
      {"dump-ast", no_argument, 0, 'e'},
      {"print-valno", no_argument, 0, 'v'},
      {"dump-infer", no_argument, 0, 'i'},
      {"sema-check", no_argument, 0, 's'},
      {"remove-comments", no_argument, 0, 'n'},
      {0, 0, 0, 0}};

  // Parse command-line options
  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "deivsn", long_options, &option_index)) !=
         -1) {
    switch (opt) {
      case 'd':
        debugMode = true;
        break;
      case 'e':
        onlyDumpAST = true;
        break;
      case 'v':
        printValueNumbers = true;
        break;
      case 's':
        onlySemaCheck = true;
        break;
      case 'i':
        showInferOnly = true;
        break;
      case 'n':
        removeComments = true;
        break;
      case '?':
        // getopt_long already printed an error message
        return 1;
      default:
        break;
    }
  }

  if (optind >= argc) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    return 1;
  }

  filename = argv[optind];

  if (filename.empty()) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    return 1;
  }

  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return 1;
  }

  loc.begin.filename = loc.end.filename = &filename;

  Scanner s;
  s.yyrestart(file);
  Parser p(s);

  if (debugMode) {
    std::cout << "Choreo: Debug of parsing is switched on." << std::endl;
    p.set_debug_level(1);  // Enable Bison debugging
    Scanner::SetDebug();
  }

  if (removeComments) Scanner::SetRemoveComments();

  p.parse();

  if (onlyDumpAST) {
    if (onlySemaCheck)
      std::cerr
          << "Warning: Semantic check is ignored since dumping AST is required."
          << std::endl;

    root.Print(std::cout);
    return 0;
  }

  // value numbering to partially evaluate the mdspan type
  ValueNumberingVisitor vn(printValueNumbers);
  root.accept(vn);

  // infer the unknown types - decls
  TypeInference type_infer(showInferOnly);
  root.accept(type_infer);

  if (showInferOnly) return 0;

  // apply type check
  SemanticChecker sema_check;
  root.accept(sema_check);

  if (onlySemaCheck) return 0;

  if (tgt == Target::Factor) {
    FactorCodeGen codegen(std::cout);
    root.accept(codegen);
  }

  return 0;
}

unsigned SymbolTable::anonymous_count = 0;
unsigned SymbolTable::anon_type_count = 0;
