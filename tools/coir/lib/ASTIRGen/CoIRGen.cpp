#include "ast.hpp"
#include "ASTCoIRGen.hpp"
#include "context.hpp"
#include "io.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "scanner.hpp"
#include "target_registry.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace Choreo;

extern AST::Program root;
extern location loc;

namespace {

struct GenOptions {
  std::string input_file;
  std::string output_file;
  bool dump_ast = false;
  bool show_help = false;
};

void PrintUsage(const char* prog) {
  std::cout << "Usage: " << prog << " [options] <input.co>\n"
            << "\n"
            << "Choreo IR generator -- translate .co programs into CoIR MLIR.\n"
            << "\n"
            << "Options:\n"
            << "  -e, --dump-ast      Dump the AST after semantic analysis\n"
            << "  -o, --output <file> Write MLIR output to file (default: stdout)\n"
            << "  -h, --help          Show this help message\n"
            << "\n"
            << "Examples:\n"
            << "  " << prog << " test.co              # emit CoIR to stdout\n"
            << "  " << prog << " -o test.mlir test.co # emit CoIR to file\n";
}

bool ParseArgs(int argc, char* argv[], GenOptions& opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      opts.show_help = true;
      return true;
    } else if (arg == "-e" || arg == "--dump-ast") {
      opts.dump_ast = true;
    } else if (arg == "-o" || arg == "--output") {
      if (i + 1 < argc)
        opts.output_file = argv[++i];
      else {
        std::cerr << "coir-gen: -o requires an output file argument\n";
        return false;
      }
    } else if (arg[0] == '-' && arg != "-") {
      std::cerr << "coir-gen: unknown option '" << arg << "'\n";
      std::cerr << "Try 'coir-gen --help' for more information.\n";
      return false;
    } else {
      if (!opts.input_file.empty()) {
        std::cerr << "coir-gen: multiple input files: '" << opts.input_file
                  << "' and '" << arg << "'\n";
        return false;
      }
      opts.input_file = arg;
    }
  }

  if (!opts.show_help && opts.input_file.empty()) {
    std::cerr << "coir-gen: no input file\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char* argv[]) {
  GenOptions opts;
  if (!ParseArgs(argc, argv, opts)) return 1;
  if (opts.show_help) {
    PrintUsage(argv[0]);
    return 0;
  }

  auto target = TargetRegistry::Create("cc");
  if (!target) {
    std::cerr << "coir-gen: internal error: 'cc' target not available\n";
    return 1;
  }
  CCtx().SetTarget(std::move(target));
  CCtx().SetNoCodegen(true);
  CCtx().SetDropComments(true);
  CCtx().SetDumpAst(opts.dump_ast);

  auto& reg = OptionRegistry::GetInstance();
  reg.Reset();
  reg.SetInputFileDirect(opts.input_file);

  bool stdin_mode = (opts.input_file == "-");

  if (!stdin_mode) {
    std::ifstream probe(opts.input_file);
    if (!probe.good()) {
      std::cerr << "coir-gen: cannot open '" << opts.input_file << "'\n";
      return 1;
    }
    loc.begin.filename = loc.end.filename = opts.input_file;
    std::ifstream src(opts.input_file);
    CCtx().ReadSourceLines(src);
  }

  // Preprocessing
  std::stringstream pps;
  auto spp = CCtx().GetTarget().MakePP(pps);
  if (!spp->Process(reg.GetInputStream())) return 1;

  // Parse
  Scanner s;
  PContext pctx;
  Parser p(pctx, s);

  Scanner::SetRemoveComments();
  Scanner::SetLocationUpdate(true);
  s.yyrestart(pps);
  if (p.parse() != 0 || pctx.HasError()) {
    errs() << "coir-gen: parsing failed due to syntax errors.\n";
    return 1;
  }

  if (opts.dump_ast) {
    root.Print(dbgs());
    dbgs() << "\n--- after semantic analysis ---\n\n";
  }

  // Semantic analysis
  auto& pl = ASTPipeline::Get().PlanSemanticRoutine();
  if (!pl.RunOnProgram(root)) return pl.Status();

  if (opts.dump_ast) {
    root.Print(dbgs());
    return 0;
  }

  // Reset MLIR module with input filename for location tracking
  CoIR::IRSession::Get().ResetModule(opts.input_file);

  // Translate AST to CoIR MLIR
  CoIR::ASTCoIRGen translator;
  if (!translator.RunOnProgram(root)) return 1;

  return 0;
}
