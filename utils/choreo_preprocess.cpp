#include "preprocess.hpp"
#include "options.hpp"
#include "io.hpp"

using namespace Choreo;

int main(int argc, char* argv[]) {
  Option<std::string> output("--output", "-o", "", true);
  // parse all the options
  OptionRegistry& r = OptionRegistry::GetInstance();
  if (!r.Parse(argc, argv)) {
    errs() << "Usage: " << argv[0] << " <filename>\n";
    exit(1);
  }

  r.SetOutputStream(output.GetValue());

  SimplePreprocessor pp;
  pp.process(r.GetInputStream(), r.GetOutputStream());

  return 0;
}
