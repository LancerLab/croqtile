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
    sim_sparse(OptionKind::Hidden, "--sim", "-sim", false,
               "Enable simulated sparse DMA encode/decode (non-production).");
// Enable host prepacked-u32 metadata path when generating device code.
Option<bool>
    use_prepack(OptionKind::Hidden, "--use-prepack", "", false,
                "Enable host prepacked-u32 metadata handling (prepack)");
Option<bool>
    use_prepack_v2(OptionKind::User, "--use-prepack-v2", "", false,
                   "Enable host prepacked-v2 metadata (fully coalesced loads)");
} // namespace Choreo
Option<bool> generate_debug_info(OptionKind::User, "-g", "", false,
                                 "Generate source-level debug information.");
Option<bool> target_generate_debug_info(
    OptionKind::User, "--target-debug", "-tg", false,
    "Generate target compiler debug information only.");
Option<std::string> debug_line_path_mode(
    OptionKind::User, "--debug-line-path", "", "relative",
    "Set #line file path mode when '-g' is enabled (relative|absolute).",
    "--debug-line-path=<relative|absolute>");

Option<bool>
    del_comm(OptionKind::User, "--remove-comments", "-n", false,
             "Remove all comments in non-choreo code. (Useful for FileCheck)");
Option<bool> inf_type(OptionKind::User, "--infer-types", "-i", false,
                      "Show the result of type inference.");
Option<bool> inf_ty_strd(OptionKind::User, "--infer-types-with-strides", "-ii",
                         false,
                         "Show the result of type inference (with strides).");
Option<bool> show_strides(OptionKind::User, "--always-show-strides", "-ass",
                          false,
                          "Always show the strides when printing types.");
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

Option<bool>
    print_features(OptionKind::User, "--print-features", "", false,
                   "Print the supported features for the selected target "
                   "and exit. One feature per line (upper-cased).");
Option<bool> verbose(OptionKind::User, "--verbose", "-v", false,
                     "Display the programs invoked by the compiler.");
Option<bool> inhibit_warning(OptionKind::User, "-w", "", false,
                             "Inhibit all warning messages.");
Option<bool> warning_as_error(OptionKind::User, "-Werror", "", false,
                              "Make all warnings into errors.");
Option<bool> disable_runtime_check(OptionKind::User, "--disable-runtime-check",
                                   "", false, "Disable all runtime checks.");
Option<std::string> runtime_check_level(OptionKind::User, "--runtime-check",
                                        "-rtc", "entry",
                                        "Control runtime assertion insertion. "
                                        "(none|entry|low|medium|high|all)");
Option<bool> show_assess(
    OptionKind::User, "--show-assess", "-sass", false,
    "Print a report of all generated assessments after the hoisting pass: "
    "assessment inputs, final assertion sites, hoist locations, and "
    "runtime cost estimates.");
Option<bool> trace_assess(OptionKind::Hidden, "--trace-assess", "-tass", false,
                          "Trace the assessment generation, resolution, and "
                          "corresoponding assertion site determination.");
Option<bool> print_stats(
    OptionKind::User, "--stats", "", false,
    "Print aggregate assertion/assessment statistics after compilation.");
Option<bool> disable_cuda_runtime_env_check(
    OptionKind::Hidden, "--disable-cuda-runtime-env-check", "", false,
    "Do not emit cuda runtime enviroment check.");
Option<bool> zero_cost(OptionKind::User, "--zero-cost", "-zero-cost", false,
                       "Zero-overhead mode: disable all runtime checks "
                       "(-rtc=none), DMA diagnosis (-dd=false), and CUDA "
                       "runtime environment check.");
#ifdef CHOREO_FAST_COMPILE_DEFAULT
constexpr bool kFastCompileDefault = true;
#else
constexpr bool kFastCompileDefault = false;
#endif
Option<bool>
    fast_compile(OptionKind::User, "--fast-compile", "-fc", kFastCompileDefault,
                 "Use separate compilation with a cached precompiled CuTe "
                 "runtime for faster nvcc compilation. The precompiled "
                 "runtime is built automatically on first use and cached in "
                 "$XDG_CACHE_HOME/choreo/ (or ~/.cache/choreo/).");
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
Option<bool>
    time_passes(OptionKind::User, "--time-passes", "-tp", false,
                "Measure and display the time spent in each compiler pass.");
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
Option<size_t> shared_mem_alignment(OptionKind::Hidden,
                                    "--shared-mem-alignment", "-fsmem-align",
                                    true,
                                    "Set the alignment of shared memory.");
Option<bool> use_warpspec(OptionKind::Hidden, "--use-warpspec", "", false,
                          "Enable warp-specialized synchronization for shared "
                          "event/full-empty pipelines.");
Option<bool> hoist_wgmma_arrive(
    OptionKind::User, "--hoist-wgmma-arrive", "", false,
    "Hoist warpgroup_arrive() before unrolled foreach loops containing "
    "WGMMA exec without commit, enabling batched arrive for the entire "
    "unrolled loop body.");
Option<bool> single_thread_producer(
    OptionKind::User, "--single-thread-producer", "", true,
    "When used with --use-warpspec, keep the producer inthreads scope single-"
    "threaded. Set to false to instead single-guard producer TMA/event "
    "operations individually.");
Option<bool> skip_epilogue_group_sync(
    OptionKind::User, "--skip-epilogue-group-sync", "", false,
    "Skip wg_barrier.sync() before shared-to-global TMA copies in warpspec "
    "mode. Safe when the writing warpgroup exclusively owns the shared buffer "
    "and fence_proxy_async_shared_cta() provides sufficient ordering.");
Option<bool>
    use_target_lib(OptionKind::User, "--use-target-lib", "-utl", false,
                   "Lower __lib_* builtins to target-specific library calls. "
                   "Overrides the target's default when explicitly set.");

// Some system missed c++17 filesystem support. Use POSIX instead
inline bool file_exists(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

bool CommandLine::Parse(int argc, char** argv) {

  // parse all the options
  auto& r = OptionRegistry::GetInstance();
  r.Reset();
  bool end_of_options = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (end_of_options) {
      // After '--', treat everything as positional (input file)
      if (!r.GetInputFileName().empty()) {
        errs() << "error: set input file twice: '" << r.GetInputFileName()
               << "' and '" << arg << "'.\n";
        ret_code = 1;
        return false;
      }
      r.SetInputFileDirect(arg);
      continue;
    }

    if (arg == "--") {
      end_of_options = true;
      continue;
    }

    if (arg.substr(0, 2) == "-D") { // macros definitions
      if (arg.size() == 2) {
        std::cerr << "error: missing macro name after '-D'.\n";
        ret_code = 1;
        return false;
      }
      auto pos = arg.find('=');
      if (pos != std::string::npos) {
        auto name = arg.substr(2, pos - 2);
        if (name.empty()) {
          std::cerr << "error: missing macro name in '" << arg << "'.\n";
          ret_code = 1;
          return false;
        }
        auto val = arg.substr(pos + 1);
        CCtx().GetCLMacros()[name] = val;
      } else
        CCtx().GetCLMacros()[arg.substr(2)] = "";
    } else if (arg.substr(0, 2) == "-I") { // include path
      if (arg.size() == 2) {
        std::cerr << "error: missing path after '-I'.\n";
        ret_code = 1;
        return false;
      }
      CCtx().GetIncPaths().push_back(arg.substr(2));
    } else if (arg.substr(0, 2) == "-L") { // library path
      if (arg.size() == 2) {
        std::cerr << "error: missing path after '-L'.\n";
        ret_code = 1;
        return false;
      }
      CCtx().GetLibPaths().push_back(arg.substr(2));
    } else if (arg.substr(0, 2) == "-l") { // library
      if (arg.size() == 2) {
        std::cerr << "error: missing library name after '-l'.\n";
        ret_code = 1;
        return false;
      }
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

  if (!print_features.GetValue() && !r.StdinAsInput() &&
      r.GetInputFileName().empty()) {
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

  if (print_features.GetValue()) {
    auto& tgt = CCtx().GetTarget();
    auto arch_id =
        arch.GetValue().empty() ? tgt.DefaultArch() : ToLower(arch.GetValue());
    for (auto& ft : tgt.SupportedFeatures(arch_id))
      std::cout << ToUpper(ft.name) << "\n";
    exit(0);
  }

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

  {
    auto dlpm = ToLower(debug_line_path_mode.GetValue());
    if (dlpm != "relative" && dlpm != "absolute") {
      errs() << "Invalid --debug-line-path value: '"
             << debug_line_path_mode.GetValue()
             << "'. Supported values: relative, absolute.\n";
      exit(1);
    }
    CCtx().SetDebugLinePathMode(dlpm == "absolute"
                                    ? DebugLinePathMode::Absolute
                                    : DebugLinePathMode::WorkspaceRelative);
  }

  if (generate_debug_info.GetValue() && target_generate_debug_info.GetValue()) {
    errs() << "option '-g' cannot be used together with '-tg'. "
              "Please choose exactly one mode.\n";
    exit(1);
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
  CCtx().SetTargetDebugInfo(generate_debug_info.GetValue() ||
                            target_generate_debug_info.GetValue());
  CCtx().SetDumpAst(dump_ast.GetValue());
  CCtx().SetNoCodegen(ncodegen.GetValue());
  CCtx().SetPrintPassNames(prt_pass.GetValue());
  CCtx().SetTimePasses(time_passes.GetValue());
  CCtx().SetNoPreProcess(no_pp.GetValue());
  CCtx().SetDropComments(del_comm.GetValue());
  CCtx().SetDebugAll(debug_on.GetValue());
  CCtx().SetShowInferredTypes(inf_type.GetValue() || inf_ty_strd.GetValue());
  CCtx().SetShowStrides(inf_ty_strd.GetValue() || show_strides.GetValue());
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
  CCtx().SetSharedMemAlignment(shared_mem_alignment.GetValue());
  CCtx().SetUseWarpSpec(use_warpspec.GetValue());
  CCtx().SetHoistWGMMAArrive(hoist_wgmma_arrive.GetValue());
  CCtx().SetSingleThreadProducer(single_thread_producer.GetValue());
  CCtx().SetSkipEpilogueGroupSync(skip_epilogue_group_sync.GetValue());
  if (use_target_lib.WasExplicitlySet())
    CCtx().SetUseTargetLib(use_target_lib.GetValue());
  else
    CCtx().SetUseTargetLib(CCtx().GetTarget().DefaultUseTargetLib());
  CCtx().SetInhibitWarning(inhibit_warning.GetValue());
  CCtx().SetWarningAsError(warning_as_error.GetValue());

  // --runtime-check=<none|entry|low|medium|high|all> controls assertion
  // granularity and cost threshold. Each level is a superset of the previous:
  //   entry  - host-side ENTRY assertions only (default, cheapest)
  //   low    - entry + device-side hoist/use-site with LOW cost threshold
  //   medium - entry + device-side hoist/use-site with MEDIUM cost threshold
  //   high   - entry + device-side hoist/use-site with HIGH cost threshold
  //   all    - same as high
  //   none   - disable all runtime assertions
  // --disable-runtime-check overrides to "none" for backward compatibility.
  {
    auto rtc = ToLower(runtime_check_level.GetValue());
    if (disable_runtime_check.GetValue()) rtc = "none";

    if (rtc == "none") {
      CCtx().SetRuntimeCheckCostThreshold(AssertionCost::NONE);
    } else if (rtc == "entry") {
      CCtx().SetRuntimeCheckCostThreshold(AssertionCost::ENTRY);
    } else if (rtc == "low") {
      CCtx().SetRuntimeCheckCostThreshold(AssertionCost::LOW);
    } else if (rtc == "medium") {
      CCtx().SetRuntimeCheckCostThreshold(AssertionCost::MEDIUM);
    } else if (rtc == "high" || rtc == "all") {
      CCtx().SetRuntimeCheckCostThreshold(AssertionCost::HIGH);
    } else {
      errs() << "error: unsupported --runtime-check value: '" << rtc
             << "'. Use none, entry, low, medium, high, or all.\n";
      return false;
    }
  }

  if (zero_cost.GetValue()) {
    CCtx().SetRuntimeCheckCostThreshold(AssertionCost::NONE);
    CCtx().SetDMADiagnosis(false);
    CCtx().SetDisableCudaRuntimeEnvCheck(true);
  }

  CCtx().SetShowAssess(show_assess.GetValue());
  CCtx().SetTraceAssess(trace_assess.GetValue());
  CCtx().SetPrintStats(print_stats.GetValue());
  if (!zero_cost.GetValue())
    CCtx().SetDisableCudaRuntimeEnvCheck(
        disable_cuda_runtime_env_check.GetValue());
  CCtx().SetFastCompile(fast_compile.GetValue());
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
  if (time_passes) setenv("CHOREO_TIME_PASSES", "1", 1);

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
