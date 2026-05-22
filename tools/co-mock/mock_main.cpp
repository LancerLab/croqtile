#include "ast.hpp"
#include "context.hpp"
#include "debugger.hpp"
#include "io.hpp"
#include "mock_interp.hpp"
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

struct MockOptions {
  std::string input_file;
  std::string script_file;
  bool dump_ast = false;
  bool interactive = false;
  bool show_help = false;
};

void PrintUsage(const char* prog) {
  std::cout << "Usage: " << prog << " [options] <input.co>\n"
            << "\n"
            << "Choreo mock interpreter -- execute .co programs without\n"
            << "hardware. Useful for testing program logic, DMA semantics,\n"
            << "and control flow.\n"
            << "\n"
            << "Options:\n"
            << "  -e, --dump-ast      Dump the AST after semantic analysis\n"
            << "  -i, --interactive   Start in interactive debugger mode\n"
            << "  -s, --script <file> Run debugger commands from a script\n"
            << "  -h, --help          Show this help message\n"
            << "\n"
            << "Interactive debugger commands:\n"
            << "  s, step             Step into next statement\n"
            << "  n, next             Step over (execute block as one unit)\n"
            << "  c, continue         Run until next breakpoint or end\n"
            << "  p <var>             Print variable value (or p var[i])\n"
            << "  l, list             Show source around current line\n"
            << "  b <line>            Set breakpoint at line number\n"
            << "  d <line>            Delete breakpoint (or 'd' for all)\n"
            << "  info                Show all variables in scope\n"
            << "  info break          Show all breakpoints\n"
            << "  q, quit             Exit\n"
            << "\n"
            << "Examples:\n"
            << "  " << prog << " test.co            # run directly\n"
            << "  " << prog << " -i test.co         # interactive debug\n"
            << "  " << prog << " -s cmds.txt test.co  # scripted debug\n"
            << "  " << prog << " -e test.co         # dump AST only\n";
}

bool ParseArgs(int argc, char* argv[], MockOptions& opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      opts.show_help = true;
      return true;
    } else if (arg == "-e" || arg == "--dump-ast") {
      opts.dump_ast = true;
    } else if (arg == "-i" || arg == "--interactive") {
      opts.interactive = true;
    } else if (arg == "-s" || arg == "--script") {
      if (i + 1 < argc)
        opts.script_file = argv[++i];
      else {
        std::cerr << "co-mock: -s requires a script file argument\n";
        return false;
      }
      opts.interactive = true;
    } else if (arg[0] == '-' && arg != "-") {
      std::cerr << "co-mock: unknown option '" << arg << "'\n";
      std::cerr << "Try 'co-mock --help' for more information.\n";
      return false;
    } else {
      if (!opts.input_file.empty()) {
        std::cerr << "co-mock: multiple input files: '" << opts.input_file
                  << "' and '" << arg << "'\n";
        return false;
      }
      opts.input_file = arg;
    }
  }

  if (!opts.show_help && opts.input_file.empty()) {
    std::cerr << "co-mock: no input file\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char* argv[]) {
  MockOptions opts;
  if (!ParseArgs(argc, argv, opts)) return 1;
  if (opts.show_help) {
    PrintUsage(argv[0]);
    return 0;
  }

  // Use "cc" as the backing target -- it provides the broadest CPU-like
  // semantic model (parallel levels, memory capacities) without requiring
  // any real hardware toolchain.
  auto target = TargetRegistry::Create("cc");
  if (!target) {
    std::cerr << "co-mock: internal error: 'cc' target not available\n";
    return 1;
  }
  CCtx().SetTarget(std::move(target));
  CCtx().SetNoCodegen(true);
  CCtx().SetDropComments(true);
  CCtx().SetDumpAst(opts.dump_ast);

  // Set up input
  auto& reg = OptionRegistry::GetInstance();
  reg.Reset();
  reg.SetInputFileDirect(opts.input_file);

  bool stdin_mode = (opts.input_file == "-");

  if (!stdin_mode) {
    std::ifstream probe(opts.input_file);
    if (!probe.good()) {
      std::cerr << "co-mock: cannot open '" << opts.input_file << "'\n";
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
    errs() << "co-mock: parsing failed due to syntax errors.\n";
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

  // Interpret
  Mock::MockInterpreter interp;

  Mock::Debugger dbg(interp.GetMemory(), interp);
  std::ifstream script_stream;
  if (opts.interactive) {
    dbg.SetActive(true);
    interp.SetDebugger(&dbg);
    if (!opts.script_file.empty()) {
      script_stream.open(opts.script_file);
      if (!script_stream.good()) {
        std::cerr << "co-mock: cannot open script '" << opts.script_file
                  << "'\n";
        return 1;
      }
      dbg.SetInputStream(&script_stream);
      std::cout << "co-mock script mode: " << opts.script_file << "\n";
    } else {
      std::cout << "co-mock debugger -- type 'h' for help\n";
    }
    std::cout << "Debugging: " << opts.input_file << "\n\n";
  }

  if (!interp.RunOnProgramImpl(root)) return 1;

  return 0;
}
