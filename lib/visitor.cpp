#include "visitor.hpp"
#include <set>

using namespace Choreo;

std::unordered_set<std::string> Choreo::Visitor::AllVisitors;
std::unordered_set<std::string> Choreo::Visitor::MatchedEnvPasses;
std::unordered_set<std::string> Choreo::Visitor::KnownPassNames;

void Visitor::ValidatePassEnvVars() {
  struct EnvCheck {
    const char* env;
    const char* opt;
  };
  EnvCheck checks[] = {
      {"CHOREO_STOP_AFTER_PASS", "--stop-after"},
      {"CHOREO_DEBUG_VISITOR", "--debug-visit"},
      {"CHOREO_TRACE_VISITOR", "--trace-visit"},
      {"CHOREO_PRINT_BEFORE", "--print-before"},
      {"CHOREO_PRINT_AFTER", "--print-after"},
      {"CHOREO_DUMP_SYMTAB_AFTER", "--dump-symbol-after"},
      {"CHOREO_DISABLE_VISIT", "--disable-visit"},
  };

  for (auto& [env, opt] : checks) {
    const char* val = std::getenv(env);
    if (!val || std::string(val).empty()) continue;
    std::string upper_val(val);

    if (upper_val == "ALLPASSES") continue;

    auto tokens = SplitStringByDelimiter(upper_val, ",");
    for (auto& tok : tokens) {
      if (tok.empty()) continue;
      std::string key = std::string(env) + ":" + tok;
      if (MatchedEnvPasses.count(key)) continue;

      std::string best;
      size_t best_dist = (size_t)-1;
      for (auto& k : KnownPassNames) {
        size_t d = EditDistance(tok, k);
        if (d < best_dist) {
          best_dist = d;
          best = k;
        }
      }
      errs() << "warning: unknown pass name '" << ToLower(tok) << "' for "
             << opt << ".";
      size_t threshold = std::max((size_t)2, tok.size() / 2);
      if (best_dist <= threshold)
        errs() << " Did you mean '" << ToLower(best) << "'?";
      errs() << "\n";

      std::set<std::string> sorted(KnownPassNames.begin(),
                                   KnownPassNames.end());
      errs() << "  Known passes:";
      for (auto& k : sorted) errs() << " " << ToLower(k);
      errs() << "\n";
    }
  }
}

location loc;
AST::Program root(loc);
