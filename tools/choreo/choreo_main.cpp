#include "ast.hpp"
#include "choreo_api.hpp"
#include "command_line.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "scanner.hpp"
#include <cstdlib>
#include <filesystem>
#include <getopt.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace Choreo;
using namespace AST;

extern Option<bool> make_lib;
extern Option<std::string> target;
extern Option<std::string> arch;
extern Option<size_t> parallel_jobs;

static std::string GetSelfPath(const char* argv0) {
  // Resolve via /proc/self/exe first (Linux)
  char buf[4096];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len > 0) {
    buf[len] = '\0';
    return std::string(buf);
  }
  return std::string(argv0);
}

static std::string FileStem(const std::string& path) {
  return std::filesystem::path(path).stem().string();
}

static bool IsCppFile(const std::string& path) {
  auto ext = std::filesystem::path(path).extension().string();
  return ext == ".cpp" || ext == ".cc" || ext == ".cxx" || ext == ".c";
}

// Compile multiple .co/.cpp files into a single static library.
// .co files are compiled via choreo, .cpp files via the target's host compiler.
// All objects are archived with `ar rcs`.
static int MultiFileLibDriver(int argc, char* argv[]) {
  auto& r = OptionRegistry::GetInstance();
  auto& inputs = r.GetInputFileNames();
  auto output = r.GetOutputFileName();
  auto self = GetSelfPath(argv[0]);
  size_t jobs = parallel_jobs.GetValue();
  if (jobs == 0) jobs = 1;

  // Create temp directory for object files
  std::string tmp_dir = "/tmp/choreo_multilib_XXXXXX";
  char* tmp_result = mkdtemp(tmp_dir.data());
  if (!tmp_result) {
    errs() << "error: failed to create temporary directory.\n";
    return 1;
  }
  tmp_dir = std::string(tmp_result);

  // Build the base args for choreo subprocess invocations (.co files)
  std::vector<std::string> base_args;
  base_args.push_back(self);
  base_args.push_back("--suppress-main");
  base_args.push_back("-c");

  // Collect arch and extra flags for C++ compilation
  std::string target_arch;
  std::vector<std::string> extra_flags;

  {
    std::set<std::string> skip_flag = {"--lib", "--suppress-main", "-c",
                                       "--compile"};
    std::set<std::string> skip_with_arg = {"-o"};
    std::set<std::string> pass_with_arg = {"--target", "-t"};
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--") break;

      if (a[0] != '-') continue;

      auto base_opt = a;
      if (auto pos = base_opt.find("="); pos != std::string::npos)
        base_opt = a.substr(0, pos);

      if (skip_flag.count(base_opt)) continue;
      if (skip_with_arg.count(base_opt)) {
        if (a.find("=") == std::string::npos && i + 1 < argc) ++i;
        continue;
      }
      if (pass_with_arg.count(base_opt)) {
        base_args.push_back(a);
        if (a.find("=") == std::string::npos && i + 1 < argc) {
          ++i;
          base_args.push_back(argv[i]);
        }
        continue;
      }

      if (a.substr(0, 2) == "-j") continue;

      // Capture -arch for target C++ compiler
      if (base_opt == "-arch" || base_opt == "--arch") {
        if (a.find("=") != std::string::npos)
          target_arch = a.substr(a.find("=") + 1);
        else if (i + 1 < argc)
          target_arch = argv[i + 1];
      }

      // Capture -I, -D flags for target C++ compiler
      if (a.substr(0, 2) == "-I" || a.substr(0, 2) == "-D")
        extra_flags.push_back(a);

      base_args.push_back(a);
    }
  }

  // Compile each input to .o
  struct CompileTask {
    std::string input;
    std::string obj;
    bool is_cpp;
  };
  std::vector<std::string> obj_files;
  std::vector<CompileTask> compile_tasks;
  for (auto& input : inputs) {
    auto stem = FileStem(input);
    auto obj = tmp_dir + "/" + stem + ".o";
    obj_files.push_back(obj);
    compile_tasks.push_back({input, obj, IsCppFile(input)});
  }

  auto cxx = CCtx().GetTarget().HostCXXCompiler();

  auto RunCompile = [&](size_t idx) -> int {
    auto& task = compile_tasks[idx];
    std::string cmd;

    if (task.is_cpp) {
      cmd = "'" + cxx + "' -c -fPIC -std=c++17";
      if (!target_arch.empty()) cmd += " -arch " + target_arch;
      auto choreo_dir = std::filesystem::path(self).parent_path().parent_path();
      cmd += " -I'" + choreo_dir.string() + "/runtime'";
      auto tgt_dir = choreo_dir / "lib" / "Target";
      if (std::filesystem::is_directory(tgt_dir)) {
        for (auto& entry : std::filesystem::directory_iterator(tgt_dir)) {
          auto rt = entry.path() / "runtime";
          if (std::filesystem::is_directory(rt))
            cmd += " -I'" + rt.string() + "'";
        }
      }
      for (auto& f : extra_flags) cmd += " '" + f + "'";
      cmd += " '" + task.input + "' -o '" + task.obj + "'";
    } else {
      // Compile .co with choreo
      for (auto& a : base_args) { cmd += "'" + a + "' "; }
      cmd += "'" + task.input + "' -o '" + task.obj + "'";
    }

    if (CCtx().PrintPassNames() || getenv("CHOREO_VERBOSE"))
      errs() << "[" << (idx + 1) << "/" << compile_tasks.size() << "] " << cmd
             << "\n";

    int rc = system(cmd.c_str());
    if (rc != 0) {
      errs() << "error: compilation of '" << task.input << "' failed.\n";
    }
    return WEXITSTATUS(rc);
  };

  int fail_count = 0;

  if (jobs <= 1) {
    // Sequential
    for (size_t i = 0; i < compile_tasks.size(); ++i) {
      if (RunCompile(i) != 0) fail_count++;
    }
  } else {
    // Parallel: dispatch in batches of `jobs`
    for (size_t batch_start = 0; batch_start < compile_tasks.size();
         batch_start += jobs) {
      size_t batch_end = std::min(batch_start + jobs, compile_tasks.size());
      std::vector<std::thread> threads;
      std::vector<int> results(batch_end - batch_start, 0);

      for (size_t i = batch_start; i < batch_end; ++i) {
        size_t local_idx = i - batch_start;
        threads.emplace_back(
            [&, i, local_idx]() { results[local_idx] = RunCompile(i); });
      }
      for (auto& t : threads) t.join();
      for (auto rc : results) {
        if (rc != 0) fail_count++;
      }
      if (fail_count > 0) break;
    }
  }

  if (fail_count > 0) {
    // Cleanup
    std::filesystem::remove_all(tmp_dir);
    return 1;
  }

  // Archive all .o files into the output .a
  std::string ar_cmd = "ar rcs '" + output + "'";
  for (auto& obj : obj_files) ar_cmd += " '" + obj + "'";

  if (CCtx().PrintPassNames() || getenv("CHOREO_VERBOSE"))
    errs() << "[archive] " << ar_cmd << "\n";

  int ar_rc = system(ar_cmd.c_str());
  if (ar_rc != 0) {
    errs() << "error: archiving failed.\n";
    std::filesystem::remove_all(tmp_dir);
    return 1;
  }

  errs() << "Library generated: " << output << " (" << obj_files.size()
         << " objects)\n";

  // Cleanup temp
  std::filesystem::remove_all(tmp_dir);
  return 0;
}

static int SingleFileCompile() {
  auto& r = OptionRegistry::GetInstance();

  if (CCtx().DumpAst() && CCtx().NoCodegen())
    errs() << "warning: Semantic check is ignored since dumping AST is "
              "required.\n";

  if (CCtx().PrintPassNames())
    dbgs() << "<file: " << r.GetInputFileName() << ">\n";

  // Apply the preprocessing
  std::stringstream pps;
  std::stringstream cok_ss;
  if (!CCtx().NoPreProcess()) {
    if (CCtx().PrintPassNames()) dbgs() << "|- preprocess the choreo program\n";
    if (CCtx().GetOutputKind() == OutputKind::PreProcessedCode) {
      auto spp = CCtx().GetTarget().MakePP(r.GetOutputStream());
      if (!spp->Process(r.GetInputStream())) return 1;
      return 0;
    } else {
      auto spp = CCtx().GetTarget().MakePP(pps);
      if (!spp->Process(r.GetInputStream())) return 1;
      if (CCtx().HasFeature(ChoreoFeature::HDRPARSE))
        if (!spp->ExtractDeviceKernel(cok_ss)) return 1;
    }
  }

  if (CCtx().GetOutputKind() == OutputKind::PreProcessedCode) return 0;

  Scanner s;
  PContext pctx;
  Parser p(pctx, s);
  Parser cok_p(pctx, s);

  if (CCtx().DebugAll()) {
    dbgs() << "Choreo: Debug of parsing is switched on." << std::endl;
    p.set_debug_level(1);
    Scanner::SetDebug();
  }

  if (CCtx().DropComments()) Scanner::SetRemoveComments();

  if (CCtx().PrintPassNames()) dbgs() << "|- parse program into AST.\n";
  if (CCtx().HasFeature(ChoreoFeature::HDRPARSE)) {
    Scanner::SetLocationUpdate(false);
    s.yyrestart(cok_ss);
    if (cok_p.parse() != 0 || pctx.HasError()) {
      errs() << "Parsing failed due to syntax errors." << std::endl;
      return 1;
    }
  }

  Scanner::SetLocationUpdate(true);
  s.yyrestart((CCtx().NoPreProcess()) ? r.GetInputStream() : pps);
  if (p.parse() != 0 || pctx.HasError()) {
    errs() << "Parsing failed due to syntax errors." << std::endl;
    return 1;
  }

  if (CCtx().DumpAst()) {
    CompilerAPI::GetAST().Print(dbgs());
    return 0;
  }

  auto& pl = ASTPipeline::Get().PlanAllRoutines();
  pl.ValidatePassNames();

  if (!pl.RunOnProgram(CompilerAPI::GetAST())) return pl.Status();

  return 0;
}

int main(int argc, char* argv[]) {
  CommandLine cl;
  if (!cl.Parse(argc, argv)) return cl.ReturnCode();

  auto& r = OptionRegistry::GetInstance();

  // Multi-file --lib mode: compile each .co to .o, then archive
  if (r.HasMultipleInputs() &&
      CCtx().GetOutputKind() == OutputKind::TargetLibrary) {
    return MultiFileLibDriver(argc, argv);
  }

  return SingleFileCompile();
}
