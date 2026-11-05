#include "agent_diagnostic.hpp"

#include "assess.hpp"
#include "context.hpp"
#include "parallel_level.hpp"
#include "target.hpp"
#include "types.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <ostream>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace Choreo {

Option<std::string> agent_diagnostics(
    OptionKind::User, "-agent-diagnostics", "--agent-diagnostics", "",
    "Write agent-facing compiler messages (constraint/observed/location/repair) "
    "as XML.",
    "-agent-diagnostics=<file.xml>", true);

bool AgentDiagnosticsEnabled() {
  return !agent_diagnostics.GetValue().empty();
}

namespace {

// One rejection must stay inside the ~100-token envelope budget of the
// compiler-harness co-design; the whole report is bounded as well so a broken
// kernel cannot flood an agent's context window.
constexpr size_t kTokenBudget = 100;
constexpr size_t kMaxDiagnostics = 8;

struct Entry {
  std::string file;
  int line = 0;
  int column = 0;
  std::string severity;
  std::string message;
  std::string source_line;
  AgentFacts facts;
  size_t occurrences = 0;
};

using EntryKey =
    std::tuple<std::string, int, int, std::string, std::string>;
std::map<EntryKey, Entry> agent_entries;

// Legality boundary of the selected target plus the state of the symbolic
// assessment engine. Both come from the same sources as -target-info and
// -static-check-info, so the envelope stays consistent with those reports.
struct TargetFacts {
  std::string name;
  std::string arch;
  size_t max_threads_per_block = 0;
  size_t shared_mem_bytes = 0;
  size_t proved = 0;
  size_t violated = 0;
  size_t runtime = 0;
  size_t unproven = 0;
};

TargetFacts CollectTargetFacts() {
  TargetFacts f;
  const auto& target = CCtx().GetTarget();
  f.name = target.Name();
  f.arch = CCtx().GetArch();
  f.max_threads_per_block = target.GetMaxThreadsPerBlock(f.arch);
  // Every target exposes the plain SHARED tier; GROUP_SHARED is opt-in only.
  f.shared_mem_bytes = target.GetMemCapacity(Storage::SHARED, f.arch);
  for (const auto& [function, context] : CCtx().GetAllFunctionContexts()) {
    (void)function;
    for (const auto& check : context.GetAssessor().GetAssessmentLog()) {
      switch (check.outcome) {
      case AssessOutcome::STATIC_TRUE: ++f.proved; break;
      case AssessOutcome::STATIC_FALSE: ++f.violated; break;
      case AssessOutcome::RUNTIME: ++f.runtime; break;
      case AssessOutcome::UNKNOWN: ++f.unproven; break;
      }
    }
  }
  return f;
}

bool Contains(const std::string& text, const std::string& needle) {
  auto lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return s;
  };
  return lower(text).find(lower(needle)) != std::string::npos;
}

std::string ParenAfter(const std::string& text, const std::string& marker) {
  const size_t pos = text.find(marker);
  if (pos == std::string::npos) return "";
  const size_t start = pos + marker.size();
  const size_t end = text.find(')', start);
  if (end == std::string::npos) return "";
  return text.substr(start, end - start);
}

std::string NumberAfter(const std::string& text, const std::string& marker) {
  const size_t pos = text.find(marker);
  if (pos == std::string::npos) return "";
  size_t start = pos + marker.size();
  while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])))
    ++start;
  size_t end = start;
  while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end])))
    ++end;
  return text.substr(start, end - start);
}

std::string FirstClause(const std::string& text) {
  const size_t pos = text.find_first_of(".;");
  return text.substr(0, pos == std::string::npos ? text.size() : pos);
}

// Batch 2 lives here: each rule maps a family of existing compiler messages
// onto the four agent-facing blocks. Migrating a check to explicit AgentFacts
// is an alternative; rules keep all 353 checks covered in the meantime.
AgentFacts Classify(const std::string& message) {
  AgentFacts f;

  if (Contains(message, "Type inconsistent between DMA") ||
      Contains(message, "rank mismatch between 'from' and 'to'")) {
    f.category = "shape-compat";
    f.constraint = "DMA shape compatibility";
    f.predicate = "from.d[i] == to.d[i]";
    const std::string from = ParenAfter(message, "'from'(");
    const std::string to = ParenAfter(message, "'to'(");
    if (!from.empty()) f.observed.push_back("from = " + from);
    if (!to.empty()) f.observed.push_back("to = " + to);
    if (f.observed.empty())
      f.observed.push_back("message = " + FirstClause(message));
    f.valid = "from.d[i] == to.d[i]";
    f.repair_do = "Append .zfill, or make the from/to extents equal.";
    f.repair_why = "Source tile is smaller than the destination.";
    f.repair_alt = "Resize the destination or use .pad.";
    f.confidence = "certain";
    return f;
  }

  if (Contains(message, "Dimensions could be inconsistent") ||
      Contains(message, "consider adding .zfill")) {
    f.category = "shape-compat";
    f.constraint = "Symbolic shape derivation for a DMA tile";
    f.predicate = "from.d[i] == to.d[i]";
    f.observed.push_back("message = " + FirstClause(message));
    f.valid = "to.d[i] must equal from.d[i] for every dimension";
    f.repair_do = "Append .zfill to the DMA to zero-fill tail tiles.";
    f.repair_why = "The source tile is smaller than the destination when the "
                   "extent is not a multiple of the tile size.";
    f.repair_alt = "Set the tile extent to cdiv(extent, tile) * tile.";
    f.confidence = "certain";
    return f;
  }

  if (Contains(message, "total thread dimension must be a multiple of")) {
    const std::string n = NumberAfter(message, "multiple of ");
    const std::string value = n.empty() ? "32" : n;
    f.category = "thread-topology";
    f.constraint = "Warp-group thread topology";
    f.predicate = "total_threads % " + value + " == 0";
    f.observed.push_back("total_threads is not a multiple of " + value);
    const TargetFacts tf = CollectTargetFacts();
    f.valid = "total_threads = k * " + value;
    if (tf.max_threads_per_block)
      f.valid += " and <= " + std::to_string(tf.max_threads_per_block);
    f.repair_do = "Set the thread 'by' bound to a multiple of " + value + ".";
    f.repair_why = "Warp-group levels require complete groups of " + value +
                   " threads.";
    f.repair_alt = "Drop the group-4 level (max_threads_per_block=" +
                   std::to_string(tf.max_threads_per_block) + ").";
    f.confidence = "certain";
    return f;
  }

  if (Contains(message, "is not supported by current architecture") ||
      (Contains(message, "MMA [") && Contains(message, "not supported"))) {
    f.category = "mma-shape";
    f.constraint = "Target-supported MMA/WGMMA shape and precision";
    f.predicate = "mma.shape in target.supported_shapes(arch)";
    f.observed.push_back("shape = " + FirstClause(message));
    const TargetFacts tf = CollectTargetFacts();
    f.valid = "mma_supported=" +
              std::string(CCtx().GetTarget().IsMMASupported() ? "true" : "false") +
              " on " + tf.name + "/" + tf.arch;
    f.repair_do = "Use an MMA shape/precision listed by -target-info.";
    f.repair_why = "The requested WGMMA configuration has no lowering on the "
                   "selected architecture.";
    f.repair_alt = "Switch -arch from " + tf.name + "/" + tf.arch + ".";
    f.confidence = "likely";
    return f;
  }

  if (Contains(message, "trigger-after cannot bind") ||
      Contains(message, "explicitly waited")) {
    f.category = "async-ordering";
    f.constraint = "Async wait/trigger ordering";
    f.predicate = "trigger(event) directly follows wait(event)";
    f.observed.push_back("future was already explicitly waited");
    f.valid = "an explicit wait followed by a direct trigger";
    f.repair_do = "Replace trigger-after with an explicit wait followed by a "
                  "direct trigger.";
    f.repair_why = "trigger-after cannot re-bind a future that has already "
                   "been waited on.";
    f.repair_alt = "Remove the redundant wait and keep trigger-after.";
    f.confidence = "certain";
    return f;
  }

  if (Contains(message, "invalid step 0 in the iteration space") ||
      Contains(message, "empty iteration space")) {
    f.category = "loop-range";
    f.constraint = "foreach iteration space";
    f.predicate = "step and extent positive";
    f.observed.push_back("message = " + FirstClause(message));
    f.valid = "step at least 1";
    f.repair_do = "Use a positive step and a non-empty range.";
    f.repair_why = "Zero step or empty range never advances the loop.";
    f.repair_alt = "Guard dynamic ranges with a runtime check.";
    f.confidence = "certain";
    return f;
  }

  // Fallback: every check still gets all four blocks, so an agent never has
  // to parse prose to find the location and a next action.
  const std::string brief = message.substr(0, 160);
  f.category = "general";
  f.constraint = FirstClause(brief);
  if (f.constraint.empty())
    f.constraint = "compiler check rejected this construct";
  f.predicate = "check failed";
  f.observed.push_back("message = " + brief);
  f.valid = "see -target-info";
  f.repair_do = "Fix the construct at this location.";
  f.repair_why = brief;
  f.repair_alt = "Consult -target-info and simplify.";
  f.priority = "medium";
  f.confidence = "unknown";
  return f;
}

std::string Trim(std::string text) {
  const size_t begin = text.find_first_not_of(" \t");
  if (begin == std::string::npos) return "";
  const size_t end = text.find_last_not_of(" \t");
  return text.substr(begin, end - begin + 1);
}

std::string XmlEscape(const std::string& text) {
  std::ostringstream os;
  for (char c : text) {
    switch (c) {
    case '&': os << "&amp;"; break;
    case '<': os << "&lt;"; break;
    case '>': os << "&gt;"; break;
    case '"': os << "&quot;"; break;
    case '\'': os << "&apos;"; break;
    default: os << c; break;
    }
  }
  return os.str();
}

// Word-granularity estimate, matching the ~74-token envelope in the paper.
// A characters/4 floor would over-count XML markup and strip the observed
// and repair fields that carry the actual repair signal.
size_t EstimateTokens(const std::string& text) {
  size_t tokens = 0;
  bool in_word = false;
  for (unsigned char c : text) {
    if (std::isalnum(c) || c == '_' || c == '.' || c == '/') {
      if (!in_word) ++tokens;
      in_word = true;
    } else {
      in_word = false;
    }
  }
  return tokens;
}

std::string Join(const std::vector<std::string>& parts) {
  std::string out;
  for (const auto& part : parts) {
    if (!out.empty()) out += "; ";
    out += part;
  }
  return out;
}

std::string RenderDiagnostic(const Entry& e) {
  const AgentFacts& f = e.facts;
  std::ostringstream os;
  os << "<diagnostic severity=\"" << XmlEscape(e.severity)
     << "\" category=\"" << XmlEscape(f.category) << "\" occurrences=\""
     << e.occurrences << "\">\n";
  os << "  <constraint>" << XmlEscape(f.constraint) << "; predicate: "
     << XmlEscape(f.predicate) << "</constraint>\n";
  const std::string observed = Join(f.observed);
  os << "  <observed>" << XmlEscape(observed);
  if (!f.valid.empty()) {
    if (!observed.empty()) os << "; ";
    os << "valid: " << XmlEscape(f.valid);
  }
  os << "</observed>\n";
  os << "  <location file=\"" << XmlEscape(e.file) << "\" line=\"" << e.line
     << "\" column=\"" << e.column << "\">" << XmlEscape(e.source_line)
     << "</location>\n";
  os << "  <repair priority=\"" << XmlEscape(f.priority)
     << "\" confidence=\"" << XmlEscape(f.confidence) << "\"><do>"
     << XmlEscape(f.repair_do) << "</do><why>" << XmlEscape(f.repair_why)
     << "</why><alt>" << XmlEscape(f.repair_alt) << "</alt></repair>\n";
  os << "</diagnostic>";
  return os.str();
}

void Shorten(std::string& text) {
  if (text.size() <= 4) {
    text.clear();
    return;
  }
  size_t keep = text.size() * 2 / 3;
  const size_t space = text.find_last_of(' ', keep);
  if (space != std::string::npos && space > keep / 2) keep = space;
  // Adding the ellipsis must still make the field strictly shorter, otherwise
  // repeated shrinking oscillates and the token budget is never reached.
  if (keep + 3 >= text.size()) keep = text.size() - 4;
  text = text.substr(0, keep) + "...";
}

// Drop the least actionable fields first; constraint, observed, source
// statement and do survive because they carry the directive the agent acts on.
void Shrink(Entry& e) {
  Shorten(e.facts.repair_alt);
  Shorten(e.facts.repair_why);
}

void InsertTokenCount(std::string& body, size_t tokens) {
  const size_t pos = body.find('>');
  if (pos != std::string::npos)
    body.insert(pos, " tokens=\"" + std::to_string(tokens) + "\"");
}

} // namespace

void RecordAgentDiagnostic(const std::string& file, int line, int column,
                           const std::string& severity,
                           const std::string& message,
                           const AgentFacts* facts) {
  if (!AgentDiagnosticsEnabled()) return;

  const EntryKey key{file, line, column, severity, message};
  auto it = agent_entries.find(key);
  if (it == agent_entries.end()) {
    Entry e;
    e.file = file;
    e.line = line;
    e.column = column;
    e.severity = severity;
    e.message = message;
    e.source_line = Trim(CCtx().GetSourceLine(line)).substr(0, 120);
    e.facts = facts ? *facts : Classify(message);
    e.occurrences = 1;
    agent_entries.emplace(key, std::move(e));
    return;
  }
  if (facts) {
    // A pass enriching an already-emitted message is not a new occurrence.
    it->second.facts = *facts;
  } else {
    ++it->second.occurrences;
  }
}

void EmitAgentDiagnostics(std::ostream& os, const std::string& status,
                          const std::string& stage, int exit_code) {
  // A report is published once. Reset on the in_progress barrier so a driver
  // that compiles several inputs in one process never mixes two candidates.
  if (status == "in_progress") agent_entries.clear();
  std::vector<std::string> bodies;
  size_t total_tokens = 0;
  for (const auto& [key, original] : agent_entries) {
    (void)key;
    if (bodies.size() == kMaxDiagnostics) break;
    Entry e = original;
    std::string body = RenderDiagnostic(e);
    for (int attempt = 0;
         attempt < 6 && EstimateTokens(body) > kTokenBudget; ++attempt) {
      Shrink(e);
      body = RenderDiagnostic(e);
    }
    if (EstimateTokens(body) > kTokenBudget) {
      e.facts.repair_alt.clear();
      e.facts.repair_why.clear();
      body = RenderDiagnostic(e);
    }
    const size_t tokens = EstimateTokens(body);
    InsertTokenCount(body, tokens);
    bodies.push_back(std::move(body));
    total_tokens += tokens;
  }

  const size_t omitted = agent_entries.size() - bodies.size();
  const TargetFacts tf = CollectTargetFacts();
  // Target limits and assessment outcomes travel as envelope attributes (not
  // another line), so the diagnostics stay the first lines an agent reads.
  os << "<compile_result status=\"" << XmlEscape(status) << "\" target=\""
     << XmlEscape(tf.name) << "\" arch=\"" << XmlEscape(tf.arch)
     << "\" max_threads_per_block=\"" << tf.max_threads_per_block
     << "\" shared_mem_bytes=\"" << tf.shared_mem_bytes
     << "\" assessments_proved=\"" << tf.proved
     << "\" assessments_violated=\"" << tf.violated
     << "\" assessments_runtime=\"" << tf.runtime << "\" stage=\""
     << XmlEscape(stage) << "\" exit_code=\"" << exit_code
     << "\" diagnostics=\"" << bodies.size() << "\" omitted=\"" << omitted
     << "\" tokens=\"" << total_tokens << "\" budget_per_diagnostic=\""
     << kTokenBudget << "\">\n";
  for (const auto& body : bodies) os << body << "\n";
  os << "</compile_result>\n";
  agent_entries.clear();
}

} // namespace Choreo
