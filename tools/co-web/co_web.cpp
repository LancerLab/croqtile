#include <emscripten/bind.h>
#include <sstream>
#include <string>

#include "ast.hpp"
#include "context.hpp"
#include "io.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "scanner.hpp"
#include "target_registry.hpp"
#include "choreo_sdk.hpp"

using namespace Choreo;
extern AST::Program root;

namespace {

std::streambuf* orig_cout = nullptr;
std::streambuf* orig_cerr = nullptr;

void capture_streams(std::ostringstream& out, std::ostringstream& err) {
  orig_cout = std::cout.rdbuf(out.rdbuf());
  orig_cerr = std::cerr.rdbuf(err.rdbuf());
}

void restore_streams() {
  if (orig_cout) std::cout.rdbuf(orig_cout);
  if (orig_cerr) std::cerr.rdbuf(orig_cerr);
  orig_cout = nullptr;
  orig_cerr = nullptr;
}

// Parse source code only (preprocess + scan + parse).
// Returns 0 on success, non-zero on error.
int parse_source(const std::string& source, const std::string& target) {
  auto& r = OptionRegistry::GetInstance();
  r.Reset();

  // Reset the global AST root to avoid accumulating state across calls.
  location fresh_loc;
  root.SetBody(AST::Make<AST::MultiNodes>(fresh_loc));

  auto& ctx = CCtx();
  ctx.SetDumpAst(false);
  ctx.SetNoCodegen(true);

  if (!target.empty()) {
    if (!ctx.SetTarget(TargetRegistry::Create(target))) {
      std::cerr << "Unknown target: " << target << "\n";
      return 1;
    }
  } else {
    ctx.SetTarget(TargetRegistry::Create("cute"));
  }

  std::istringstream source_stream(source);
  auto old_cin = std::cin.rdbuf(source_stream.rdbuf());

  int result = 0;

  std::stringstream pps;
  {
    auto spp = ctx.GetTarget().MakePP(pps);
    if (!spp->Process(source_stream)) {
      result = 1;
      goto cleanup;
    }
  }

  {
    Scanner s;
    PContext pctx;
    Parser p(pctx, s);

    Scanner::SetLocationUpdate(true);
    s.yyrestart(pps);
    if (p.parse() != 0 || pctx.HasError()) {
      std::cerr << "Parsing failed due to syntax errors.\n";
      result = 1;
    }
  }

cleanup:
  std::cin.rdbuf(old_cin);
  return result;
}

} // anonymous namespace

namespace CoWeb {

emscripten::val compile(const std::string& source, const std::string& target,
                        const std::string& /*flags*/) {
  std::ostringstream out_stream, err_stream;
  capture_streams(out_stream, err_stream);

  int rc = parse_source(source, target);
  if (rc == 0) {
    root.Print(std::cout);
  }
  restore_streams();

  emscripten::val result = emscripten::val::object();
  result.set("output", out_stream.str());
  result.set("errors", err_stream.str());
  result.set("success", rc == 0);
  return result;
}

emscripten::val dumpAST(const std::string& source) {
  std::ostringstream out_stream, err_stream;
  capture_streams(out_stream, err_stream);

  int rc = parse_source(source, "");
  if (rc == 0) {
    root.Print(std::cout);
  }
  restore_streams();

  emscripten::val result = emscripten::val::object();
  result.set("output", out_stream.str());
  result.set("errors", err_stream.str());
  result.set("success", rc == 0);
  return result;
}

emscripten::val mockRun(const std::string& source) {
  std::ostringstream out_stream, err_stream;
  capture_streams(out_stream, err_stream);

  int rc = parse_source(source, "cute");
  restore_streams();

  std::string output = out_stream.str();
  if (rc == 0 && output.empty()) {
    output = "[mock] Program parsed and validated successfully.\n"
             "[mock] No runtime output (mock interpreter not yet available).\n";
  }

  emscripten::val result = emscripten::val::object();
  result.set("output", output);
  result.set("errors", err_stream.str());
  result.set("success", rc == 0);
  return result;
}

std::string getVersion() { return Choreo::SDK::Version(); }

} // namespace CoWeb

EMSCRIPTEN_BINDINGS(co_web) {
  emscripten::function("compile", &CoWeb::compile);
  emscripten::function("dumpAST", &CoWeb::dumpAST);
  emscripten::function("mockRun", &CoWeb::mockRun);
  emscripten::function("getVersion", &CoWeb::getVersion);
}
