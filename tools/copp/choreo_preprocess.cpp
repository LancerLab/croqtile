#include "command_line.hpp"
#include "context.hpp"
#include "io.hpp"
#include "options.hpp"
#include "preprocess.hpp"

using namespace Choreo;
extern Option<std::string> output;

int main(int argc, char* argv[]) {
  CommandLine cl;
  if (!cl.Parse(argc, argv)) return cl.ReturnCode();

  // parse all the options
  OptionRegistry& r = OptionRegistry::GetInstance();

  auto pp = CCtx().GetTarget().MakePP(r.GetOutputStream());
  if (!pp->Process(r.GetInputStream())) return 1;

  return 0;
}
