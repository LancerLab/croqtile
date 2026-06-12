#include "sys_utils.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

namespace Choreo {

std::string MakeTempFile(const std::string& suffix) {
  std::error_code ec;
  auto tmp = fs::temp_directory_path(ec);
  if (ec) return "";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(100000, 999999);
  for (int i = 0; i < 16; ++i) {
    auto p = tmp / ("choreo_detect_" + std::to_string(dis(gen)) + suffix);
    if (!fs::exists(p, ec) && !ec) return p.string();
  }
  return "";
}

std::string ExecCapture(const std::string& cmd) {
  auto out = MakeTempFile(".out");
  if (out.empty()) return "";
#ifdef _WIN32
  std::string full_cmd = cmd + " > \"" + out + "\" 2>NUL";
#else
  std::string full_cmd = cmd + " > '" + out + "' 2>/dev/null";
#endif
  int rc = std::system(full_cmd.c_str());
  std::string result;
  if (rc == 0) {
    std::ifstream f(out);
    if (f) {
      std::ostringstream ss;
      ss << f.rdbuf();
      result = ss.str();
    }
  }
  std::error_code ec;
  fs::remove(out, ec);
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
    result.pop_back();
  return result;
}

std::string FindToolchain(const std::string& configured_dir,
                          const std::string& binary_name) {
  std::error_code ec;
  if (!configured_dir.empty()) {
    fs::path p = fs::path(configured_dir) / "bin" / binary_name;
    if (fs::is_regular_file(p, ec)) return p.string();
  }
  const char* path_env = std::getenv("PATH");
  if (!path_env) return "";
  std::string path_str{path_env};
  std::istringstream stream(path_str);
  std::string dir;
#ifdef _WIN32
  const char delim = ';';
#else
  const char delim = ':';
#endif
  while (std::getline(stream, dir, delim)) {
    fs::path p = fs::path(dir) / binary_name;
    if (fs::is_regular_file(p, ec)) return p.string();
  }
  return "";
}

std::string CompileAndRun(const std::string& compiler,
                          const std::string& source, const std::string& suffix,
                          const std::string& extra_flags) {
  auto src = MakeTempFile(suffix);
  if (src.empty()) return "";
  auto exe = src.substr(0, src.size() - suffix.size());
  {
    std::ofstream f(src);
    if (!f) return "";
    f << source;
    if (!f) {
      std::error_code ec;
      fs::remove(src, ec);
      return "";
    }
  }
#ifdef _WIN32
  std::string cmd = "\"" + compiler + "\" -o \"" + exe + "\" \"" + src + "\"";
  if (!extra_flags.empty()) cmd += " " + extra_flags;
  cmd += " 2>NUL && \"" + exe + "\" 2>NUL";
#else
  std::string cmd = "'" + compiler + "' -o '" + exe + "' '" + src + "'";
  if (!extra_flags.empty()) cmd += " " + extra_flags;
  cmd += " 2>/dev/null && '" + exe + "' 2>/dev/null";
#endif
  auto result = ExecCapture(cmd);
  std::error_code ec;
  fs::remove(src, ec);
  fs::remove(exe, ec);
  return result;
}

} // namespace Choreo
