#include <getopt.h>

#include <cstdlib>

#include "ast.hpp"
#include "codegen.hpp"
#include "scanner.hpp"
#include "semantics.hpp"
#include "symtab.hpp"

Choreo::location loc;
AST::Program root;
AST::SymbolTable symtab;

using namespace AST;
using namespace Choreo;

int main(int argc, char* argv[]) {
  std::string filename;
  bool debugMode = false;
  bool onlyDumpAST = false;
  bool onlySemaCheck = false;
  bool removeComments = false;

  enum class Target { Factor, Topscc };

  Target tgt = Target::Factor;

  // Define long options
  static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                         {"dump-ast", no_argument, 0, 'e'},
                                         {"sema-check", no_argument, 0, 's'},
                                         {"remove-comments", no_argument, 0, 'n'},
                                         {0, 0, 0, 0}};

  // Parse command-line options
  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "desn", long_options, &option_index)) !=
         -1) {
    switch (opt) {
      case 'd':
        debugMode = true;
        break;
      case 'e':
        onlyDumpAST = true;
        break;
      case 's':
        onlySemaCheck = true;
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

  Choreo::Scanner s;
  s.yyrestart(file);
  Choreo::Parser p(s);

  if (debugMode) {
    std::cout << "Choreo: Debug of parsing is switched on." << std::endl;
    p.set_debug_level(1);  // Enable Bison debugging
    Choreo::Scanner::SetDebug();
  }

  if (removeComments)
    Choreo::Scanner::SetRemoveComments();

  p.parse();

  if (onlyDumpAST) {
    if (onlySemaCheck)
      std::cerr
          << "Warning: Semantic check is ignored since dumping AST is required."
          << std::endl;

    root.Print(std::cout);
    return 0;
  }

  SemanticChecker semachk;
  root.accept(semachk);

  if (onlySemaCheck) return 0;

  if (tgt == Target::Factor) {
    FactorCodeGen codegen(std::cout);
    root.accept(codegen);
  }

  return 0;
}

int AST::SymbolTable::anonymous_count = 0;
