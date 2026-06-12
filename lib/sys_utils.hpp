#ifndef __CHOREO_SYS_UTILS_HPP__
#define __CHOREO_SYS_UTILS_HPP__

#include <string>

namespace Choreo {

// Execute a shell command and capture its stdout (trimmed of trailing
// newlines). Returns empty string on failure.
std::string ExecCapture(const std::string& cmd);

// Find a toolchain binary: check configured_dir/bin/<name> first, then PATH.
// Returns full path or empty string if not found.
std::string FindToolchain(const std::string& configured_dir,
                          const std::string& binary_name);

// Create a unique temporary file path with the given suffix.
// Returns empty string on failure.
std::string MakeTempFile(const std::string& suffix);

// Write source to a temp file, compile with `compiler`, run, return stdout.
// Returns empty string on failure.
std::string CompileAndRun(const std::string& compiler,
                          const std::string& source, const std::string& suffix,
                          const std::string& extra_flags = "");

} // namespace Choreo

#endif // __CHOREO_SYS_UTILS_HPP__
