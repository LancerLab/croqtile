#include "command_line.hpp"
#include "context.hpp"
#include <fstream>
#include <sys/stat.h>

using namespace Choreo;

#ifndef __CHOREO_DEFAULT_TARGET__
#error "no default target is specified."
#endif

extern location loc;

// Major available options
Option<std::string> target(OptionKind::User, "--target", "-t",
                           __CHOREO_DEFAULT_TARGET__,
                           "Set the compilation target. Use '--help-target' to "
                           "show current supported targets.",
                           "--target <platform>", true);
Option<std::string> arch(OptionKind::User, "-arch", "", "" /*default empty*/,
                         "Set the architecture to execute the binary code.",
                         "-arch=<processor>");
Option<std::string> output(OptionKind::User, "-o", "", "",
                           "Place the output into <file>.", "-o <file>", true);
Option<std::string>
    api_mode(OptionKind::User, "--api", "-api", "cffi",
             "Select API mode for generated code (cffi|sglang).",
             "--api=<mode>", true);

Option<std::string>
    debug_file_dir(OptionKind::User, "-ddir", "", "./build/",
                   "Place compiler debug artifacts under <dir>.",
                   "--ddir=<dir>");
Option<bool>
    emit_source(OptionKind::User, "-es", "", false,
                "Emit target source file without target source compilation.");
Option<bool> compile_only(
    OptionKind::User, "--compile", "-c", false,
    "Compile choreo code and the generated target code; Without linking.");
Option<bool> generate_script(OptionKind::User, "--generate-script", "-gs",
                             false, "Generate target script.");
namespace Choreo {
Option<bool>
    sim_sparse(OptionKind::User, "--sim", "-sim", false,
               "Enable simulated sparse DMA encode/decode (non-production).");
} // namespace Choreo
Option<bool> generate_debug_info(OptionKind::User, "-g", "", false,
                                 "Generate source-level debug information.");

Option<bool>
    del_comm(OptionKind::User, "--remove-comments", "-n", false,
             "Remove all comments in non-choreo code. (Useful for FileCheck)");
Option<bool> inf_type(OptionKind::User, "--infer-types", "-i", false,
                      "Show the result of type inference.");
Option<bool> pp_only(OptionKind::User, "-E", "", false,
                     "Preprocess only; do not compile.");
Option<bool> no_pp(OptionKind::Hidden, "--no-preprocess", "-npp", false,
                   "Do not invoke Choreo Preprocessor to compile.");
Option<bool> use_kernel_template(
    OptionKind::Hidden, "--use-kernel-template", "-kt", false,
    "(Experimental) Allow choreo code to instantiate C++ template functions.");
Option<bool> use_hetero_tileflow(
    OptionKind::Hidden, "--use-hetero-tileflow", "-ht", false,
    "(Experimental) Allow choreo code to apply implicit/aggressive tileflow"
    "optimisation under heterogeneous scenario.");
Option<bool>
    use_pic(OptionKind::Hidden, "--use-pic", "-fpic", false,
            "Generate position-independent code if possible (small mode).");
Option<bool> simplify_fp_valno(
    OptionKind::Hidden, "--simplify-fp-valno", "-sfv", false,
    "(Experimental) Simplify the value numbering for floating point types.");
Option<bool>
    native_f16(OptionKind::User, "--native-f16", "-f16n", false,
               "Utilize native f16 type when target platform support.");

Option<bool>
    native_bf16(OptionKind::User, "--native-bf16", "-bf16n", false,
                "Utilize native bf16 type when target platform support.");

Option<bool> verbose(OptionKind::User, "--verbose", "-v", false,
                     "Display the programs invoked by the compiler.");
Option<bool> inhibit_warning(OptionKind::User, "-w", "", false,
                             "Inhibit all warning messages.");
Option<bool> warning_as_error(OptionKind::User, "-Werror", "", false,
                              "Make all warnings into errors.");
Option<bool> disable_runtime_check(OptionKind::User, "--disable-runtime-check",
                                   "", false, "Disable all runtime checks.");

Option<std::string>
    target_options(OptionKind::Hidden, "--target-options", "-tos", "",
                   "Extra target options used for target compilation.", "");
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
Option<std::string> disable_pass(OptionKind::Hidden, "--disable-visit", "-dp",
                                 "", "Disable the visit pass.",
                                 "--disable-visit=<pass>");
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
Option<bool> print_vn(OptionKind::Hidden, "--print-valno", "-vn", false,
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
Option<bool> liveness(OptionKind::Hidden, "--liveness", "", true,
                      "Analyze the liveness of the program.");
Option<bool> mem_reuse(OptionKind::Hidden, "--mem-reuse", "", true,
                       "Analyze the memory usage, then perform memory reuse.");
Option<bool> diag_dma(OptionKind::Hidden, "--diag-dma", "-dd", true,
                      "Enable runtime DMA diagnosis.");
Option<bool> print_node_type(OptionKind::Hidden, "--print-node-type", "-pnt",
                             false, "Print node with its type.");
Option<bool> verify_visitors(OptionKind::Hidden, "--verify", "-vf", false,
                             "verify all visitors for legality.");

// TODO: add mechanism to handle GCC-style "-f" options
Option<bool> no_show_source(
    OptionKind::Hidden, "-fno-show-source-location", "", false,
    "Do not show the source code location when error/warning/etc..");
Option<bool> analyze_device_functions(
    OptionKind::Hidden, "--analyze-device-functions", "-adf", true,
    "Analyze the device functions in the choreo code. (Experimental)");
Option<bool> branch_norm(OptionKind::Hidden, "--branch-norm", "-bn", false,
                         "Normalize the if-else branches in the choreo code.");
Option<bool> loop_norm(OptionKind::Hidden, "--loop-norm", "-ln", false,
                       "Normalize the loops in the choreo code.");
Option<bool> debug_vectorize(OptionKind::Hidden, "--debug-vectorize", "-dvec",
                             false, "debug loop vectorization process.");
Option<bool> no_vectorize(OptionKind::Hidden, "--no-vectorize", "-nm", false,
                          "Do not vectorize any foreach loop.");
Option<bool> vectorize(OptionKind::Hidden, "--vectorize", "-vec", false,
                       "Enable loop vectorization.");
Option<size_t> max_local_mem_capacity(
    OptionKind::Hidden, "--max-local-mem-capacity", "-fmax-local", 0,
    "Set the max local memory capacity (in bytes) per thread. 0 means use "
    "default value.");
Option<bool> mem_default_aligned(OptionKind::Hidden, "--mem-default-aligned",
                                 "-fmem_aligned", true,
                                 "Use the default alignment in memory reuse.");

// Some system missed c++17 filesystem support. Use POSIX instead
inline bool file_exists(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

bool CommandLine::Parse(int argc, char** argv) {

  // parse all the options
  auto& r = OptionRegistry::GetInstance();
  r.Reset();
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "-D") { // macros definitions
      auto pos = arg.find('=');
      if (pos != std::string::npos) {
        auto name = arg.substr(2, pos - 2);
        auto val = arg.substr(pos + 1);
        CCtx().GetCLMacros()[name] = val;
      } else
        CCtx().GetCLMacros()[arg.substr(2)] = "";
    } else if (arg.substr(0, 2) == "-I") { // include path
      CCtx().GetIncPaths().push_back(arg.substr(2));
    } else if (arg.substr(0, 2) == "-L") { // library path
      CCtx().GetLibPaths().push_back(arg.substr(2));
    } else if (arg.substr(0, 2) == "-l") { // library path
      CCtx().GetLibs().push_back(arg.substr(2));
    } else if (arg.substr(0, 2) == "-O") { // optimization level
      int level = arg[2] - '0';
      if (arg.size() != 3 || level > 3 || level < 0) {
        std::cerr << "Invalid optimization level: " << arg << ".\n";
        ret_code = 1;
        return false;
      }
      CCtx().SetOptimizationLevel(level);
    } else if (!r.Parse(argc, argv, i)) {
      if (!r.Message().empty()) errs() << r.Message() << "\n";
      exit(r.ReturnCode());
    }
  }

  if (!r.StdinAsInput() && r.GetInputFileName().empty()) {
    std::cerr << "error: no input file.\n";
    ret_code = 1;
    return false;
  }

  // set the compilation targets
  if (!CCtx().SetTarget(TargetRegistry::Create(ToLower(target.GetValue())))) {
    errs() << "Compile Target '" << target.GetValue()
           << "' is invalid. Compilation abort.\n";
    exit(1);
  }

  // set the arch to compile
  if (!arch.GetValue().empty()) CCtx().AddArch(ToLower(arch.GetValue()));

  // set API mode
  {
    auto api = ToLower(api_mode.GetValue());
    if (api != "cffi" && api != "sglang") {
      errs() << "Invalid --api value: '" << api_mode.GetValue()
             << "'. Supported values: cffi, sglang.\n";
      exit(1);
    }
    CCtx().SetApiMode(api);
  }

  if (pp_only) {
    if (no_pp) {
      errs() << "option '-E' can not work with '--no-preprocess'. Compilation "
                "abort.\n";
      exit(1);
    }
    CCtx().SetOutputKind(OutputKind::PreProcessedCode);
  } else if (emit_source)
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

  // save the options to the global context
  CCtx().SetGenDebugInfo(generate_debug_info.GetValue());
  CCtx().SetDumpAst(dump_ast.GetValue());
  CCtx().SetNoCodegen(ncodegen.GetValue());
  CCtx().SetPrintPassNames(prt_pass.GetValue());
  CCtx().SetNoPreProcess(no_pp.GetValue());
  CCtx().SetDropComments(del_comm.GetValue());
  CCtx().SetDebugAll(debug_on.GetValue());
  CCtx().SetShowInferredTypes(inf_type.GetValue());
  CCtx().SetDumpSymtab(dump_sym.GetValue());
  CCtx().SetVisualize(visualiz.GetValue());
  CCtx().SetCrossCompile(cross_compile.GetValue());
  CCtx().SetTraceValueNumbers(print_vn.GetValue());
  CCtx().SetTraceVectorize(debug_vectorize.GetValue());
  CCtx().SetNoVectorize(no_vectorize.GetValue());
  CCtx().SetVectorize(vectorize.GetValue());
  CCtx().SetShowSourceLocation(!no_show_source.GetValue());
  CCtx().SetLivenessAnalysis(liveness.GetValue());
  CCtx().SetMemReuse(mem_reuse.GetValue());
  CCtx().SetSimplifyFpValno(simplify_fp_valno.GetValue());
  CCtx().SetVerifyVisitors(verify_visitors.GetValue());
  CCtx().SetDMADiagnosis(diag_dma.GetValue());
  CCtx().SetLoopNorm(loop_norm.GetValue());
  CCtx().SetMaxLocalMemCapacityPerThread(max_local_mem_capacity.GetValue());
  CCtx().SetMemDefaultAligned(mem_default_aligned.GetValue());
  CCtx().SetInhibitWarning(inhibit_warning.GetValue());
  CCtx().SetWarningAsError(warning_as_error.GetValue());
  CCtx().SetDisableRuntimeCheck(disable_runtime_check.GetValue());
  CCtx().SetDebugFileDir(debug_file_dir.GetValue());

  if (!trace_visit.GetValue().empty())
    setenv("CHOREO_TRACE_VISITOR", ToUpper(trace_visit.GetValue()).c_str(), 1);

  if (!debug_visit.GetValue().empty())
    setenv("CHOREO_DEBUG_VISITOR", ToUpper(debug_visit.GetValue()).c_str(), 1);

  if (!print_ahead.GetValue().empty())
    setenv("CHOREO_PRINT_BEFORE", ToUpper(print_ahead.GetValue()).c_str(), 1);

  if (print_ahead_all) setenv("CHOREO_PRINT_BEFORE", "ALLPASSES", 1);

  if (!print_after.GetValue().empty())
    setenv("CHOREO_PRINT_AFTER", ToUpper(print_after.GetValue()).c_str(), 1);

  if (!disable_pass.GetValue().empty())
    setenv("CHOREO_DISABLE_VISIT", ToUpper(disable_pass.GetValue()).c_str(), 1);

  if (!dsyms_after.GetValue().empty())
    setenv("CHOREO_DUMP_SYMTAB_AFTER", ToUpper(dsyms_after.GetValue()).c_str(),
           1);

  if (print_after_all) setenv("CHOREO_PRINT_AFTER", "ALLPASSES", 1);

  if (print_node_type) setenv("CHOREO_PRINT_NODETYPE", "", 1);

  if (prt_pass) setenv("CHOREO_PRINT_PASSES", "", 1);

  if (!abend_after.GetValue().empty())
    setenv("CHOREO_STOP_AFTER_PASS", ToUpper(abend_after.GetValue()).c_str(),
           1);

  if (analyze_device_functions)
    setenv("CHOREO_ANALYZE_DEVICE_FUNCTIONS", "", 1);

  if (!r.StdinAsInput()) {
    std::string filename = r.GetInputFileName();
    if (!file_exists(filename)) {
      errs() << "error: The input file '" << filename << "' does not exist."
             << std::endl;
      ret_code = 1;
      return false;
    }

    loc.begin.filename = loc.end.filename = filename;

    // read the source file into memory
    std::ifstream ifs(filename);
    CCtx().ReadSourceLines(ifs);
  }

  return true;
}
