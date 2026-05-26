#include <emscripten/bind.h>
#include <emscripten.h>
#include <sstream>
#include <string>
#include <vector>

#include "ast.hpp"
#include "choreo_header.inc"
#include "command_line.hpp"
#include "context.hpp"
#include "io.hpp"
#include "mock_interp.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "scanner.hpp"
#include "target_registry.hpp"

using namespace Choreo;

extern AST::Program root;
extern location loc;

namespace {

struct CompileResult {
  std::string output;
  std::string errors;
  bool success;
};

bool runtime_initialized = false;

void initRuntime() {
  if (runtime_initialized) return;
  runtime_initialized = true;

  EM_ASM({
    if (typeof FS !== 'undefined') {
      FS.writeFile('/choreo.h', UTF8ToString($0));
    }
  }, __choreo_header_as_string);
}

void resetGlobalState() {
  initRuntime();
  loc = location();
  root = AST::Program(loc);
  ASTPipeline::ResetInstance();
}

std::vector<std::string> splitFlags(const std::string& flags) {
  std::vector<std::string> result;
  std::istringstream iss(flags);
  std::string token;
  while (iss >> token) result.push_back(token);
  return result;
}

CompileResult runCompilation(const std::string& source,
                             const std::string& target,
                             const std::string& flags) {
  CompileResult result;
  result.success = false;

  resetGlobalState();

  std::vector<std::string> argStrings;
  argStrings.push_back("choreo");

  if (!target.empty()) {
    argStrings.push_back("-t");
    argStrings.push_back(target);
  } else {
    argStrings.push_back("-t");
    argStrings.push_back("cc");
  }

  auto extraFlags = splitFlags(flags);
  for (auto& f : extraFlags) argStrings.push_back(f);

  argStrings.push_back("-");

  std::vector<char*> argv;
  for (auto& s : argStrings) argv.push_back(const_cast<char*>(s.c_str()));
  int argc = static_cast<int>(argv.size());

  std::ostringstream outputCapture;
  std::ostringstream errorCapture;

  auto& reg = OptionRegistry::GetInstance();
  reg.Reset();

  CommandLine cl;
  if (!cl.Parse(argc, argv.data())) {
    result.errors = "Failed to parse command line arguments.";
    return result;
  }

  std::istringstream sourceStream(source);
  auto origCin = std::cin.rdbuf();
  auto origCout = std::cout.rdbuf();
  auto origCerr = std::cerr.rdbuf();
  std::cin.rdbuf(sourceStream.rdbuf());
  std::cout.rdbuf(outputCapture.rdbuf());
  std::cerr.rdbuf(errorCapture.rdbuf());

  int exitCode = 0;
  try {
    std::stringstream pps;
    std::stringstream cok_ss;

    if (!CCtx().NoPreProcess()) {
      if (CCtx().GetOutputKind() == OutputKind::PreProcessedCode) {
        auto spp = CCtx().GetTarget().MakePP(outputCapture);
        if (!spp->Process(reg.GetInputStream())) exitCode = 1;
        else result.success = true;
        goto cleanup;
      } else {
        auto spp = CCtx().GetTarget().MakePP(pps);
        if (!spp->Process(reg.GetInputStream())) {
          exitCode = 1;
          goto cleanup;
        }
      }
    }

    {
      Scanner s;
      PContext pctx;
      Parser p(pctx, s);

      if (CCtx().DropComments()) Scanner::SetRemoveComments();
      Scanner::SetLocationUpdate(true);
      s.yyrestart((CCtx().NoPreProcess()) ? reg.GetInputStream() : pps);

      if (p.parse() != 0 || pctx.HasError()) {
        exitCode = 1;
        goto cleanup;
      }
    }

    if (CCtx().DumpAst()) {
      root.Print(outputCapture);
      result.success = true;
      goto cleanup;
    }

    {
      auto& pl = ASTPipeline::Get().PlanAllRoutines();
      if (!pl.RunOnProgram(root)) {
        exitCode = pl.Status();
        goto cleanup;
      }
    }
    result.success = true;
  } catch (const std::exception& e) {
    errorCapture << "Exception: " << e.what() << "\n";
    exitCode = 1;
  } catch (...) {
    errorCapture << "Unknown exception occurred.\n";
    exitCode = 1;
  }

cleanup:
  std::cin.rdbuf(origCin);
  std::cout.rdbuf(origCout);
  std::cerr.rdbuf(origCerr);

  result.output = outputCapture.str();
  result.errors = errorCapture.str();
  if (exitCode != 0) result.success = false;

  return result;
}

CompileResult compile(const std::string& source, const std::string& target,
                      const std::string& flags) {
  std::string tgt = target.empty() ? "cc" : target;
  std::string allFlags = "-es";
  if (!flags.empty()) allFlags += " " + flags;
  return runCompilation(source, tgt, allFlags);
}

CompileResult mockRun(const std::string& source) {
  CompileResult result;
  result.success = false;

  resetGlobalState();

  auto target = TargetRegistry::Create("cc");
  if (!target) {
    result.errors = "Internal error: 'cc' target not available.";
    return result;
  }
  CCtx().SetTarget(std::move(target));
  CCtx().SetNoCodegen(true);
  CCtx().SetDropComments(true);

  auto& reg = OptionRegistry::GetInstance();
  reg.Reset();

  std::vector<std::string> argStrings = {"choreo", "-t", "cc", "-s", "-"};
  std::vector<char*> argv;
  for (auto& s : argStrings) argv.push_back(const_cast<char*>(s.c_str()));
  CommandLine cl;
  if (!cl.Parse(static_cast<int>(argv.size()), argv.data())) {
    result.errors = "Failed to parse command line arguments.";
    return result;
  }

  std::istringstream sourceStream(source);
  auto origCin = std::cin.rdbuf();
  auto origCout = std::cout.rdbuf();
  auto origCerr = std::cerr.rdbuf();
  std::ostringstream outputCapture;
  std::ostringstream errorCapture;
  std::cin.rdbuf(sourceStream.rdbuf());
  std::cout.rdbuf(outputCapture.rdbuf());
  std::cerr.rdbuf(errorCapture.rdbuf());

  try {
    std::stringstream pps;
    auto spp = CCtx().GetTarget().MakePP(pps);
    if (!spp->Process(reg.GetInputStream())) goto cleanup;

    {
      Scanner s;
      PContext pctx;
      Parser p(pctx, s);
      Scanner::SetRemoveComments();
      Scanner::SetLocationUpdate(true);
      s.yyrestart(pps);
      if (p.parse() != 0 || pctx.HasError()) {
        errorCapture << "Parsing failed.\n";
        goto cleanup;
      }
    }

    {
      auto& pl = ASTPipeline::Get().PlanSemanticRoutine();
      if (!pl.RunOnProgram(root)) {
        errorCapture << "Semantic analysis failed.\n";
        goto cleanup;
      }
    }

    {
      Mock::MockInterpreter interp;
      if (!interp.RunOnProgramImpl(root)) {
        errorCapture << "Interpretation failed.\n";
        goto cleanup;
      }
    }
    result.success = true;
  } catch (const std::exception& e) {
    errorCapture << "Exception: " << e.what() << "\n";
  } catch (...) {
    errorCapture << "Unknown exception occurred.\n";
  }

cleanup:
  std::cin.rdbuf(origCin);
  std::cout.rdbuf(origCout);
  std::cerr.rdbuf(origCerr);

  result.output = outputCapture.str();
  result.errors = errorCapture.str();
  return result;
}

CompileResult dumpAST(const std::string& source) {
  return runCompilation(source, "cc", "-e");
}

std::string getVersion() {
#ifdef CHOREO_SDK_VERSION
  return CHOREO_SDK_VERSION;
#else
  return "co-web-dev";
#endif
}

} // anonymous namespace

EMSCRIPTEN_BINDINGS(co_web) {
  emscripten::value_object<CompileResult>("CompileResult")
      .field("output", &CompileResult::output)
      .field("errors", &CompileResult::errors)
      .field("success", &CompileResult::success);

  emscripten::function("compile", &compile);
  emscripten::function("mockRun", &mockRun);
  emscripten::function("dumpAST", &dumpAST);
  emscripten::function("getVersion", &getVersion);
}
