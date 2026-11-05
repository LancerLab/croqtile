#ifndef __CHOREO_AGENT_DIAGNOSTIC_HPP__
#define __CHOREO_AGENT_DIAGNOSTIC_HPP__

#include "options.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace Choreo {

// Destination of the agent-facing report; "-" means stdout. Keeping this in
// the driver (like -static-check-info) makes the envelope available to a
// tuning harness without changing the human-readable stderr diagnostics.
extern Option<std::string> agent_diagnostics;

// Structured facts of one rejection, following the compiler-harness
// co-design: <constraint>, <observed>, <location>, <repair>.
struct AgentFacts {
  std::string category = "general";
  std::string constraint;
  std::string predicate;
  std::vector<std::string> observed;
  std::string valid;
  std::string repair_do;
  std::string repair_why;
  std::string repair_alt;
  std::string priority = "high";
  std::string confidence = "likely";
};

bool AgentDiagnosticsEnabled();

// Collect one compiler message. Explicit facts take precedence over the
// message classifier; passes can migrate to explicit facts batch by batch
// while every remaining check still gets a complete envelope.
void RecordAgentDiagnostic(const std::string& file, int line, int column,
                           const std::string& severity,
                           const std::string& message,
                           const AgentFacts* facts = nullptr);

// Render the <compile_result> envelope. `status` is one of in_progress,
// accepted, rejected, incomplete.
void EmitAgentDiagnostics(std::ostream& os, const std::string& status,
                          const std::string& stage, int exit_code);

} // namespace Choreo

#endif // __CHOREO_AGENT_DIAGNOSTIC_HPP__
