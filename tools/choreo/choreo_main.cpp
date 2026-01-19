#include "ast.hpp"
#include "command_line.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "scanner.hpp"
#include <cstdlib>
#include <getopt.h>

using namespace Choreo;

extern AST::Program root;

using namespace AST;
using namespace Choreo;

int main(int argc, char* argv[]) {
  CommandLine cl;
  if (!cl.Parse(argc, argv)) return cl.ReturnCode();

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
    p.set_debug_level(1); // Enable Bison debugging
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
    root.Print(dbgs());
    return 0;
  }

  auto& pl = ASTPipeline::Get().PlanAllRoutines();

  if (!pl.RunOnProgram(root)) return pl.Status();

  return 0;
}
