#ifndef __CHOREO_STATIC_CHECK_INFO_HPP__
#define __CHOREO_STATIC_CHECK_INFO_HPP__

#include "options.hpp"

#include <functional>
#include <iosfwd>
#include <string>

namespace Choreo {

class Target;

// User-facing report options. They intentionally live in the core driver so
// target-info is available independently of the selected backend.
extern Option<std::string> target_info;
extern Option<std::string> static_check_info;
extern Option<std::string> performance_remarks;
extern Option<bool> static_check_details;

void RecordStaticCheckDependency(const std::string& path);
void RecordStaticCheckPreprocessedInput(const std::string& input);
void EmitAssessmentFactsJson(std::ostream& os, bool finalized = false);
void EmitMemoryFactsJson(std::ostream& os);
void RecordBackendCompilation(bool succeeded);
void EmitBackendFactsJson(std::ostream& os);

// Error collection is enabled only for report requests. Repeated template
// diagnostics are grouped by source location and message.
void RecordStaticCheckError(const std::string& file, int line, int column,
                            const std::string& message);
void EmitIncompleteStaticCheckInfo(std::ostream& os, const std::string& status,
                                   const std::string& stage, int exit_code);

// Emit target-derived facts for the currently selected target/architecture.
void EmitTargetInfoJson(std::ostream& os);

// Write a report to a file, or to stdout when path is "-".
bool WriteReportFile(const std::string& path,
                     const std::function<void(std::ostream&)>& emit);

// Emit source and invocation identity shared by target-specific reports.
void EmitStaticCheckContextJson(std::ostream& os);

// Emit the generic report used when the selected target has not implemented
// target-specific static facts yet.
void EmitGenericStaticCheckInfo(std::ostream& os);

} // namespace Choreo

#endif // __CHOREO_STATIC_CHECK_INFO_HPP__
