#include "codegen.hpp"

#include <cctype>
#include <filesystem>
#include <sstream>
#include <system_error>

using namespace Choreo;

std::once_flag CodeGenInfo::init_flag;
std::unique_ptr<CodeGenInfo> CodeGenInfo::instance;
int TMADesc::index = 0;

Choreo::CodeGenInfo& CodeGenInfo::Get() {
  std::call_once(init_flag,
                 []() { instance = std::make_unique<CodeGenInfo>(); });
  return *instance;
}

std::string Choreo::EscapeLinePathForDirective(const std::string& path) {
  std::string escaped;
  escaped.reserve(path.size());
  for (auto ch : path) {
    if (ch == '\\' || ch == '"') escaped.push_back('\\');
    escaped.push_back(ch);
  }
  return escaped;
}

std::string Choreo::ResolveDebugLinePath(const location& loc,
                                         DebugLinePathMode mode) {
  auto file = loc.begin.filename;
  if (file.empty()) return "";

  std::filesystem::path file_path(file);
  std::error_code ec;
  if (mode == DebugLinePathMode::Absolute) {
    if (file_path.is_relative())
      file_path = std::filesystem::current_path(ec) / file_path;
    auto abs_path = std::filesystem::weakly_canonical(file_path, ec);
    if (!ec)
      file_path = abs_path;
    else
      file_path = file_path.lexically_normal();
    return file_path.string();
  }

  if (file_path.is_relative())
    file_path = std::filesystem::current_path(ec) / file_path;
  auto abs_path = std::filesystem::weakly_canonical(file_path, ec);
  if (!ec)
    file_path = abs_path;
  else
    file_path = file_path.lexically_normal();

  std::error_code rec;
  auto rel_path = std::filesystem::relative(
      file_path, std::filesystem::current_path(rec), rec);
  if (!rec && !rel_path.empty()) return rel_path.string();
  return file_path.string();
}

namespace {
inline bool ParseLineDirective(const std::string& line, int& line_no,
                               std::string& filename) {
  size_t pos = 0;
  while (pos < line.size() && std::isspace((unsigned char)line[pos])) ++pos;
  if (line.compare(pos, 5, "#line") != 0) return false;
  pos += 5;
  while (pos < line.size() && std::isspace((unsigned char)line[pos])) ++pos;

  size_t num_begin = pos;
  while (pos < line.size() && std::isdigit((unsigned char)line[pos])) ++pos;
  if (num_begin == pos) return false;
  line_no = std::stoi(line.substr(num_begin, pos - num_begin));

  while (pos < line.size() && std::isspace((unsigned char)line[pos])) ++pos;
  if (pos >= line.size() || line[pos] != '"') return false;
  ++pos;
  size_t file_begin = pos;
  while (pos < line.size() && line[pos] != '"') ++pos;
  if (pos >= line.size()) return false;
  filename = line.substr(file_begin, pos - file_begin);
  return true;
}
} // namespace

std::string Choreo::PinLineDirectivePerGeneratedLine(const std::string& code) {
  std::istringstream iss(code);
  std::ostringstream oss;

  bool active = false;
  int active_line = -1;
  std::string active_file;
  std::string line;
  while (std::getline(iss, line)) {
    int parsed_line = -1;
    std::string parsed_file;
    if (ParseLineDirective(line, parsed_line, parsed_file)) {
      active = true;
      active_line = parsed_line;
      active_file = parsed_file;
      oss << line << "\n";
      continue;
    }

    bool non_empty = false;
    for (auto ch : line) {
      if (!std::isspace((unsigned char)ch)) {
        non_empty = true;
        break;
      }
    }
    if (active && non_empty) {
      oss << "#line " << active_line << " \""
          << EscapeLinePathForDirective(active_file) << "\"\n";
    }
    oss << line << "\n";
  }

  return oss.str();
}
