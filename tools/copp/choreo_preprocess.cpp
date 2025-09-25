#include "io.hpp"
#include "options.hpp"
#include "preprocess.hpp"

using namespace Choreo;
extern Option<std::string> output;

int main(int argc, char* argv[]) {
  // parse all the options
  OptionRegistry& r = OptionRegistry::GetInstance();
  for (int i = 1; i < argc; ++i) {
    if (!r.Parse(argc, argv, i)) {
      if (!r.Message().empty()) errs() << r.Message() << "\n";
      exit(r.ReturnCode());
    }
  }
  r.SetOutputStream(output.GetValue());

  SimplePreprocessor pp(r.GetOutputStream());
  if (!pp.Process(r.GetInputStream())) return 1;

  return 0;
}
