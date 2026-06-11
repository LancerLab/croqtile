#ifndef __CHOREO_API_HPP__
#define __CHOREO_API_HPP__

#include "context.hpp"
#include "io.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "scanner.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace Choreo {
namespace AST {
class Program;
} // namespace AST

/// Choreo compiler API -- modular frontend services for tools that consume
/// .co files. After CommandLine::Parse() has configured CCtx() (target,
/// arch, options), call RunFrontend() to execute the frontend pipeline.
///
/// Usage:
///   CommandLine cl;
///   cl.Parse(argc, argv);
///   auto& ast = Choreo::CompilerAPI::GetAST();  // after frontend
///
struct CompilerAPI {
  /// Get the global AST root.
  static AST::Program& GetAST();

  /// Get the global source location.
  static location& GetLoc();

  /// Open the input file and set up CCtx source lines and location.
  bool OpenInput(const std::string& filename) {
    auto& reg = OptionRegistry::GetInstance();
    reg.SetInputFileDirect(filename);

    std::ifstream probe(filename);
    if (!probe.good()) {
      errs() << "error: cannot open '" << filename << "'\n";
      return false;
    }
    GetLoc().begin.filename = GetLoc().end.filename = filename;
    std::ifstream src(filename);
    CCtx().ReadSourceLines(src);
    return true;
  }

  /// Preprocess the input through the current target's preprocessor.
  bool Preprocess(std::stringstream& output) {
    auto& reg = OptionRegistry::GetInstance();
    auto spp = CCtx().GetTarget().MakePP(output);
    return spp->Process(reg.GetInputStream());
  }

  /// Parse the preprocessed source into the AST.
  bool Parse(std::stringstream& preprocessed) {
    Scanner s;
    PContext pctx;
    Parser p(pctx, s);
    Scanner::SetRemoveComments();
    Scanner::SetLocationUpdate(true);
    s.yyrestart(preprocessed);
    if (p.parse() != 0 || pctx.HasError()) {
      errs() << "error: parsing failed\n";
      return false;
    }
    return true;
  }

  /// Run semantic analysis pipeline on the AST.
  int RunSema() {
    auto& pl = ASTPipeline::Get().PlanSemanticRoutine();
    if (!pl.RunOnProgram(GetAST())) return pl.Status();
    return 0;
  }

  /// Run full frontend pipeline: open -> preprocess -> parse -> sema.
  /// CCtx() must already be configured (target, arch) via CommandLine::Parse.
  /// Sets NoCodegen(true) since this only runs up to semantic analysis.
  int RunFrontend(const std::string& filename) {
    CCtx().SetNoCodegen(true);
    if (!OpenInput(filename)) return 1;

    std::stringstream pps;
    if (!Preprocess(pps)) return 1;
    if (!Parse(pps)) return 1;
    return RunSema();
  }
};

} // namespace Choreo

#endif // __CHOREO_API_HPP__
