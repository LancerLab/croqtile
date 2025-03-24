#include "io.hpp"
#include "options.hpp"
#include "preprocess.hpp"

using namespace Choreo;

int main(int argc, char* argv[]) {
  Option<std::string> output(OptionKind::User, "--output", "-o", "",
                             "Place the output into <file>.", "-o <file>",
                             true);
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
