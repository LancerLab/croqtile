#include "ast.hpp"
#include "codegen.hpp"
#include "codegen_cuda.hpp"
#include "codegen_factor.hpp"
#include "codegen_prepare.hpp"
#include "earlysema.hpp"
#include "gcucheck.hpp"
#include "latenorm.hpp"
#include "memcheck.hpp"
#include "normalize.hpp"
#include "options.hpp"
#include "preprocess.hpp"
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
#include <filesystem>
#include <getopt.h>

using namespace Choreo;

location loc;
AST::Program root(loc);
SymbolTable symtab;

using namespace AST;
using namespace Choreo;

// Major available options
Option<std::string> target(
    OptionKind::User, "--target", "-t", "factor",
    "Set the compilation target. The 'platform' includes <factor|topscc|cuda>.",
    "--target <platform>", true);
Option<std::string> arch(OptionKind::User, "-arch", "", "gcu300",
                         "Set the architecture to execute the binary code.",
                         "-arch=<processor>");
Option<std::string> output(OptionKind::User, "-o", "", "",
                           "Place the output into <file>.", "-o <file>", true);

Option<bool>
    emit_source(OptionKind::User, "-es", "", false,
                "Emit target source file without target source compilation.");
Option<bool> compile_only(
    OptionKind::User, "-c", "", false,
    "Compile choreo code and the generated target code; Without linking.");
Option<bool> generate_script(OptionKind::User, "-gs", "", false,
                             "Generate target script.");

Option<bool>
    del_comm(OptionKind::User, "--remove-comments", "-n", false,
             "Remove all comments in non-choreo code. (Useful for FileCheck)");
Option<bool> inf_type(OptionKind::User, "--infer-types", "-i", false,
                      "Show the result of type inference.");
Option<bool> pp_only(OptionKind::User, "-E", "", false,
                     "Preprocess only; do not compile.");
Option<bool> no_pp(OptionKind::Hidden, "--no-preprocess", "-npp", false,
                   "Donnot invoke Choreo Proprocessor to compile.");
Option<bool> use_kernel_template(
    OptionKind::Hidden, "--use_kernel_template", "-kt", false,
    "(Experimental) Allow choreo code to instantiate C++ template functions.");
Option<bool>
    native_f16(OptionKind::User, "--native-f16", "-f16n", false,
               "Utilize native f16 type when target platform support.");

Option<std::string> abend_after(OptionKind::Hidden, "--stop-after", "-sa", "",
                                "Stop compilation after the visit pass.",
                                "--stop-after=<pass>");
Option<std::string> trace_visit(
    OptionKind::Hidden, "--trace-visit", "-tv", "",
    "Enable tracing of node visits during AST traversal by the visit pass.",
    "--trace-visit=<pass>");
Option<std::string>
    debug_visit(OptionKind::Hidden, "--debug-visit", "-dv", "",
                "Enable debugging during AST traversal by the visit pass.",
                "--debug-visit=<pass>");
Option<std::string> print_ahead(OptionKind::Hidden, "--print-before", "-pb", "",
                                "Print AST ahead of the visit pass.",
                                "--print-before=<pass>");
Option<std::string> print_after(OptionKind::Hidden, "--print-after", "-pa", "",
                                "Print AST after the visit pass.",
                                "--print-after=<pass>");
Option<std::string> dsyms_after(OptionKind::Hidden, "--dump-symbol-after",
                                "-ds", "",
                                "Dump the symbol table after the visit pass.",
                                "--dump-symbol-after=<pass>");
Option<bool> print_ahead_all(OptionKind::Hidden, "--print-before-all", "-pba",
                             false, "Print AST ahead of all the visit passes.");
Option<bool> print_after_all(OptionKind::Hidden, "--print-after-all", "-paa",
                             false, "Print AST after all the visit passes.");
Option<bool> cross_compile(OptionKind::Hidden, "--cross-compile", "-cc",
                           false); // useful?
Option<bool> debug_on(OptionKind::Hidden, "--debug", "-d", false,
                      "Enable Debugging of all the visit passes.");
Option<bool> dump_ast(OptionKind::User, "--dump-ast", "-e", false,
                      "Dump the Abstract Syntax Tree (AST) after parsing.");
Option<bool> print_vn(OptionKind::Hidden, "--print-valno", "-v", false,
                      "Trace the value numbering process.");
Option<bool> dump_sym(OptionKind::Hidden, "--dump-symbol", "-l", false,
                      "Dump the symbol table after LATENORM.");
Option<bool> visualiz(OptionKind::Hidden, "--visualize", "-u", false,
                      "Visualize the data movement of DMAs.");
Option<bool> ncodegen(OptionKind::Hidden, "--no-codegen", "-s", false,
                      "Do not generate Code.");
Option<bool> sym_repl(OptionKind::Hidden, "--print-sym-replace", "-sr", false,
                      "Trace the symbol replace process.");
Option<bool> prt_pass(OptionKind::Hidden, "--show-passes", "-sp", false,
                      "Show the visit pass pipeline.");
Option<bool> save_temps(OptionKind::Hidden, "--save-temps", "", false,
                        "Save the temporal files.");

int main(int argc, char* argv[]) {
  // parse all the options
  auto& r = OptionRegistry::GetInstance();
  if (!r.Parse(argc, argv)) {
    if (!r.Message().empty()) errs() << r.Message() << "\n";
    exit(r.ReturnCode());
  }

  // set the compilation targets
  if (ToUpper(target.GetValue()) == "FACTOR")
    CCtx().SetTarget(CompileTarget::Factor);
  else if (ToUpper(target.GetValue()) == "TOPSCC")
    CCtx().SetTarget(CompileTarget::Topscc);
  else if (ToUpper(target.GetValue()) == "CUDA")
    CCtx().SetTarget(CompileTarget::CUDA);
  else {
    errs() << "Compile Target '" << target.GetValue()
           << "' is invalid. Compilation abort.\n";
    exit(1);
  }

  // set the arch to compile
  if (ToUpper(arch.GetValue()) == "GCU200")
    CCtx().SetArch(TargetArch::GCU20);
  else if (ToUpper(arch.GetValue()) == "GCU210")
    CCtx().SetArch(TargetArch::GCU21);
  else if (ToUpper(arch.GetValue()) == "GCU300")
    CCtx().SetArch(TargetArch::GCU3);
  else if (ToUpper(arch.GetValue()) == "GPU")
    CCtx().SetArch(TargetArch::GPU);
  else {
    errs() << "Arch '" << arch.GetValue()
           << "' is invalid. Compilation abort.\n";
    exit(1);
  }

  if (pp_only)
    CCtx().SetOutputKind(OutputKind::PreProcessedCode);
  else if (emit_source)
    CCtx().SetOutputKind(OutputKind::TargetSourceCode);
  else if (compile_only) {
    CCtx().SetOutputKind(OutputKind::TargetModule);
    if (output.GetValue().empty()) output = "a.o"; // default module name
  } else if (generate_script)
    CCtx().SetOutputKind(OutputKind::ShellScript);
  else {
    CCtx().SetOutputKind(OutputKind::TargetExecutable);
    if (output.GetValue().empty()) output = "a.out"; // default exe name
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

  if (!dsyms_after.GetValue().empty())
    setenv("CHOREO_DUMP_SYMTAB_AFTER", ToUpper(dsyms_after.GetValue()).c_str(),
           1);

  if (print_after_all) setenv("CHOREO_PRINT_AFTER", "ALLPASSES", 1);

  if (prt_pass) setenv("CHOREO_PRINT_PASSES", "", 1);

  if (!abend_after.GetValue().empty())
    setenv("CHOREO_STOP_AFTER_PASS", ToUpper(abend_after.GetValue()).c_str(),
           1);

  if (dump_ast) {
    if (ncodegen)
      errs()
          << "warning: Semantic check is ignored since dumping AST is required."
          << std::endl;
  }

  std::string filename = r.GetInputFileName();
  if (!std::filesystem::exists(filename)) {
    errs() << "error: The input file '" << filename << "' does not exist."
           << std::endl;
    return 1;
  }

  loc.begin.filename = loc.end.filename = &filename;

  if (prt_pass) dbgs() << "|- " << filename << "\n";

  // Apply the preprocessing
  std::stringstream pps;
  if (!no_pp) {
    if (prt_pass) dbgs() << "|- preprocess the choreo program\n";
    if (CCtx().GetOutputKind() == OutputKind::PreProcessedCode) {
      SimplePreprocessor spp(r.GetOutputStream());
      if (!spp.Process(r.GetInputStream())) return 1;
      return 0;
    } else {
      SimplePreprocessor spp(pps);
      if (!spp.Process(r.GetInputStream())) return 1;
    }
  }

  Scanner s;
  s.yyrestart((no_pp) ? r.GetInputStream() : pps);
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

  // apply early semantics check without knowing type details
  EarlySemantics sv;
  if (!sv.RunOnProgram(root)) return sv.Status();

  // minor AST change: desugar for canonicalized AST
  Normalizer ds;
  if (!ds.RunOnProgram(root)) return ds.Status();

  SymReplace sr;
  if (!sr.RunOnProgram(root)) return sr.Status();

  // perform shape inference of mdspans, future, etc.
  ShapeInference si(print_vn);
  if (!si.RunOnProgram(root)) return si.Status();

  // inference all the unknown types - decls
  TypeInference ti(inf_type);
  if (!ti.RunOnProgram(root)) return ti.Status();
  if (inf_type || print_vn) return 0;

  // late normalize
  LateNorm ln(ti.SymTab());
  if (!ln.RunOnProgram(root)) return ln.Status();

  // debug: dump the symbol table
  if (std::getenv("DUMP_SYMTAB") || dump_sym) ln.SymTab()->Print(dbgs());

  if (std::getenv("VISUALIZE") || visualiz) {
    Visualizer vl(ln.SymTab());
    if (!vl.RunOnProgram(root)) return vl.Status();
    return 0;
  }

  // apply the type check
  TypeChecker sc(ln.SymTab());
  if (!sc.RunOnProgram(root)) return sc.Status();

  // --------- Following passes generate codes -------- //

  if (ncodegen) return 0; // do not generate code

  // collect information for codegen
  CodegenPrepare cgp(sc.SymTab());
  if (!cgp.RunOnProgram(root)) return cgp.Status();

  switch (CCtx().GetTarget()) {
  case CompileTarget::Factor: {
    // apply the gcu specific checking
    GCUCheck gcu_checker(sc.SymTab());
    if (!gcu_checker.RunOnProgram(root)) return gcu_checker.Status();

    FactorTrans trans(sc.SymTab());
    if (!trans.RunOnProgram(root)) return trans.Status();

    MemUsageCheck mem_usage_checker(sc.SymTab());
    if (!mem_usage_checker.RunOnProgram(root))
      return mem_usage_checker.Status();

    Choreo::Factor::FactorCodeGen codegen(
        sc.SymTab(), mem_usage_checker.GetRtMemUsageInfo(), cgp.GetASTInfo(),
        cross_compile, use_kernel_template);
    if (!codegen.RunOnProgram(root)) return codegen.Status();
    break;
  }
  case CompileTarget::CUDA: {
    Choreo::CUDA::CUDACodeGen codegen(sc.SymTab(), cross_compile);
    if (!codegen.RunOnProgram(root)) return codegen.Status();
    break;
  }
  case CompileTarget::Topscc: {
    errs() << "Target '" << target.GetValue()
           << "' has not been supported yet.\n";
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
