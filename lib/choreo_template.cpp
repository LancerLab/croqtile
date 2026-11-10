#include "choreo_template.hpp"

#include "io.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Choreo {
namespace {

enum class ParamKind { Type, Int, Bool };

struct TemplateParam {
  ParamKind kind;
  std::string name;
  std::string default_arg;
};

struct BoundArg {
  ParamKind kind;
  std::string dsl;
  std::string cpp;
  std::string canonical;
};

struct FunctionSyntax {
  size_t begin = 0;
  size_t end = 0;
  size_t name_begin = 0;
  size_t name_end = 0;
  size_t template_args_begin = std::string::npos;
  size_t template_args_end = std::string::npos;
  std::string name;
  std::vector<std::string> template_args;
};

struct TemplateDefinition {
  size_t begin = 0;
  size_t end = 0;
  size_t line = 0;
  size_t function_begin = 0;
  size_t function_line = 0;
  std::string name;
  std::string function_text;
  size_t function_name_begin = 0;
  size_t function_name_end = 0;
  size_t function_template_args_begin = std::string::npos;
  size_t function_template_args_end = std::string::npos;
  std::vector<TemplateParam> params;
  std::vector<std::string> specialization_args;

  bool IsSpecialization() const { return params.empty(); }
};

struct InstantiationRequest {
  size_t begin = 0;
  size_t end = 0;
  size_t line = 0;
  std::string name;
  std::vector<std::string> args;
};

struct Instance {
  std::vector<BoundArg> args;
  std::string key;
  std::string internal_name;
  std::string display_name;
  size_t request_offset = 0;
  size_t request_line = 0;
  const TemplateDefinition* specialization = nullptr;
};

struct Replacement {
  size_t begin;
  size_t end;
  std::string text;
};

struct TypeName {
  const char* canonical;
  const char* dsl;
  const char* cpp;
};

thread_local const std::string* diagnostic_source = nullptr;
thread_local std::map<std::string, ChoreoTemplateInstanceInfo>
    template_instances;
thread_local std::string active_template_instance;

struct ScopedDiagnosticSource {
  const std::string* previous;

  explicit ScopedDiagnosticSource(const std::string& source)
      : previous(diagnostic_source) {
    diagnostic_source = &source;
  }

  ~ScopedDiagnosticSource() { diagnostic_source = previous; }
};

const std::unordered_map<std::string, TypeName>& TypeNames() {
  static const std::unordered_map<std::string, TypeName> names = {
      {"f64", {"f64", "f64", "double"}},
      {"double", {"f64", "f64", "double"}},
      {"f32", {"f32", "f32", "float"}},
      {"float", {"f32", "f32", "float"}},
      {"tf32", {"tf32", "tf32", "choreo::tf32"}},
      {"f16", {"f16", "f16", "choreo::f16"}},
      {"half", {"f16", "f16", "choreo::f16"}},
      {"bf16", {"bf16", "bf16", "choreo::bf16"}},
      {"bfp16", {"bf16", "bf16", "choreo::bf16"}},
      {"f8", {"f8_e4m3", "f8_e4m3", "choreo::f8_e4m3"}},
      {"f8_e4m3", {"f8_e4m3", "f8_e4m3", "choreo::f8_e4m3"}},
      {"f8_e5m2", {"f8_e5m2", "f8_e5m2", "choreo::f8_e5m2"}},
      {"f8_ue4m3", {"f8_ue4m3", "f8_ue4m3", "choreo::f8_ue4m3"}},
      {"f8_ue8m0", {"f8_ue8m0", "f8_ue8m0", "choreo::f8_ue8m0"}},
      {"f6_e2m3", {"f6_e2m3", "f6_e2m3", "choreo::f6_e2m3"}},
      {"f6_e3m2", {"f6_e3m2", "f6_e3m2", "choreo::f6_e3m2"}},
      {"f4_e2m1", {"f4_e2m1", "f4_e2m1", "choreo::f4_e2m1"}},
      {"u64", {"u64", "u64", "uint64_t"}},
      {"s64", {"s64", "s64", "int64_t"}},
      {"u32", {"u32", "u32", "unsigned int"}},
      {"s32", {"s32", "s32", "int"}},
      {"int", {"s32", "s32", "int"}},
      {"u16", {"u16", "u16", "unsigned short"}},
      {"s16", {"s16", "s16", "short"}},
      {"u8", {"u8", "u8", "unsigned char"}},
      {"s8", {"s8", "s8", "choreo::s8"}},
      {"u6", {"u6", "u6", "choreo::u6"}},
      {"s6", {"s6", "s6", "choreo::s6"}},
      {"u4", {"u4", "u4", "choreo::u4"}},
      {"s4", {"s4", "s4", "choreo::s4"}},
      {"u2", {"u2", "u2", "choreo::u2"}},
      {"s2", {"s2", "s2", "choreo::s2"}},
      {"bin1", {"bin1", "bin1", "choreo::bin1"}},
      {"u1", {"u1", "u1", "choreo::u1"}},
      {"bool", {"bool", "bool", "bool"}},
  };
  return names;
}

bool IsIdentStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool IsIdentChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string Trim(const std::string& text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin])))
    ++begin;
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
    --end;
  return text.substr(begin, end - begin);
}

size_t LineAt(const std::string& text, size_t pos) {
  return 1 + static_cast<size_t>(
                 std::count(text.begin(), text.begin() + pos, '\n'));
}

size_t ColumnAt(const std::string& text, size_t pos) {
  if (pos == 0) return 1;
  size_t line_begin = text.rfind('\n', pos - 1);
  return line_begin == std::string::npos ? pos + 1 : pos - line_begin;
}

std::string SourceLine(size_t line) {
  if (!diagnostic_source || line == 0) return "";
  size_t begin = 0;
  for (size_t current = 1; current < line; ++current) {
    begin = diagnostic_source->find('\n', begin);
    if (begin == std::string::npos) return "";
    ++begin;
  }
  size_t end = diagnostic_source->find('\n', begin);
  if (end == std::string::npos) end = diagnostic_source->size();
  return diagnostic_source->substr(begin, end - begin);
}

bool DiagnoseAt(const std::string& filename, size_t line, size_t column,
                const std::string& message, const char* level = "error") {
  if (!filename.empty()) errs() << filename << ":";
  if (line) errs() << line << "." << std::max<size_t>(column, 1) << ": ";
  errs() << level << ": " << message << "\n";
  auto source_line = SourceLine(line);
  if (!source_line.empty()) {
    errs() << "  " << source_line << "\n  ";
    for (size_t i = 1; i < std::max<size_t>(column, 1); ++i) errs() << " ";
    errs() << "^\n";
  }
  return false;
}

bool DiagnoseOffset(const std::string& filename, size_t offset,
                    const std::string& message, const char* level = "error") {
  if (!diagnostic_source) return DiagnoseAt(filename, 0, 1, message, level);
  offset = std::min(offset, diagnostic_source->size());
  return DiagnoseAt(filename, LineAt(*diagnostic_source, offset),
                    ColumnAt(*diagnostic_source, offset), message, level);
}

bool DiagnoseExpanded(const std::string& filename, const std::string& text,
                      size_t offset, size_t base_line,
                      const std::string& message) {
  return DiagnoseAt(filename, base_line + LineAt(text, offset) - 1,
                    ColumnAt(text, offset), message);
}

void SkipQuoted(const std::string& text, size_t& pos, char quote) {
  ++pos;
  while (pos < text.size()) {
    if (text[pos] == '\\') {
      pos = std::min(pos + 2, text.size());
      continue;
    }
    if (text[pos] == quote) {
      ++pos;
      return;
    }
    ++pos;
  }
}

void SkipRawString(const std::string& text, size_t& pos) {
  size_t delimiter_begin = pos + 2;
  size_t open = text.find('(', delimiter_begin);
  if (open == std::string::npos) {
    pos = text.size();
    return;
  }
  std::string delimiter = text.substr(delimiter_begin, open - delimiter_begin);
  std::string close = ")" + delimiter + "\"";
  size_t end = text.find(close, open + 1);
  pos = end == std::string::npos ? text.size() : end + close.size();
}

bool SkipCommentOrLiteral(const std::string& text, size_t& pos) {
  if (pos + 1 < text.size() && text[pos] == '/' && text[pos + 1] == '/') {
    size_t end = text.find('\n', pos + 2);
    pos = end == std::string::npos ? text.size() : end;
    return true;
  }
  if (pos + 1 < text.size() && text[pos] == '/' && text[pos + 1] == '*') {
    size_t end = text.find("*/", pos + 2);
    pos = end == std::string::npos ? text.size() : end + 2;
    return true;
  }
  if (pos + 1 < text.size() && text[pos] == 'R' && text[pos + 1] == '"') {
    SkipRawString(text, pos);
    return true;
  }
  if (text[pos] == '"' || text[pos] == '\'') {
    SkipQuoted(text, pos, text[pos]);
    return true;
  }
  return false;
}

size_t SkipTrivia(const std::string& text, size_t pos) {
  while (pos < text.size()) {
    if (std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
      continue;
    }
    size_t old = pos;
    if (SkipCommentOrLiteral(text, pos)) {
      if (old + 1 < text.size() && text[old] == '/' &&
          (text[old + 1] == '/' || text[old + 1] == '*'))
        continue;
      return old;
    }
    break;
  }
  return pos;
}

bool IsKeywordAt(const std::string& text, size_t pos,
                 const std::string& keyword) {
  if (text.compare(pos, keyword.size(), keyword) != 0) return false;
  if (pos > 0 && IsIdentChar(text[pos - 1])) return false;
  size_t end = pos + keyword.size();
  return end >= text.size() || !IsIdentChar(text[end]);
}

std::optional<std::pair<std::string, size_t>>
ParseIdentifier(const std::string& text, size_t pos) {
  pos = SkipTrivia(text, pos);
  if (pos >= text.size() || !IsIdentStart(text[pos])) return std::nullopt;
  size_t end = pos + 1;
  while (end < text.size() && IsIdentChar(text[end])) ++end;
  return std::make_pair(text.substr(pos, end - pos), end);
}

std::optional<size_t> FindMatching(const std::string& text, size_t open_pos,
                                   char open, char close) {
  if (open_pos >= text.size() || text[open_pos] != open) return std::nullopt;
  int depth = 1;
  int paren = 0;
  int bracket = 0;
  int brace = 0;
  size_t pos = open_pos + 1;
  while (pos < text.size()) {
    if (SkipCommentOrLiteral(text, pos)) continue;
    if (open == '<') {
      // In value arguments, angle characters inside a delimited expression
      // are operators (for example, `1 << 2`), not template delimiters.
      switch (text[pos]) {
      case '(': ++paren; break;
      case ')': --paren; break;
      case '[': ++bracket; break;
      case ']': --bracket; break;
      case '{': ++brace; break;
      case '}': --brace; break;
      default: break;
      }
      if (paren != 0 || bracket != 0 || brace != 0) {
        ++pos;
        continue;
      }
    }
    if (text[pos] == open)
      ++depth;
    else if (text[pos] == close && --depth == 0)
      return pos;
    ++pos;
  }
  return std::nullopt;
}

std::vector<std::string> SplitTopLevel(const std::string& text, char delim) {
  std::vector<std::string> parts;
  size_t begin = 0;
  int angle = 0;
  int paren = 0;
  int bracket = 0;
  int brace = 0;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t before = pos;
    if (SkipCommentOrLiteral(text, pos)) continue;
    if (before != pos) continue;
    switch (text[pos]) {
    case '<':
      if (paren == 0 && bracket == 0 && brace == 0) ++angle;
      break;
    case '>':
      if (paren == 0 && bracket == 0 && brace == 0) --angle;
      break;
    case '(': ++paren; break;
    case ')': --paren; break;
    case '[': ++bracket; break;
    case ']': --bracket; break;
    case '{': ++brace; break;
    case '}': --brace; break;
    default: break;
    }
    if (text[pos] == delim && angle == 0 && paren == 0 && bracket == 0 &&
        brace == 0) {
      parts.push_back(Trim(text.substr(begin, pos - begin)));
      begin = pos + 1;
    }
    ++pos;
  }
  std::string tail = Trim(text.substr(begin));
  if (!tail.empty() || !parts.empty()) parts.push_back(tail);
  return parts;
}

std::optional<size_t> FindTopLevelEqual(const std::string& text) {
  int angle = 0;
  int paren = 0;
  int bracket = 0;
  int brace = 0;
  size_t pos = 0;
  while (pos < text.size()) {
    if (SkipCommentOrLiteral(text, pos)) continue;
    switch (text[pos]) {
    case '<':
      if (paren == 0 && bracket == 0 && brace == 0) ++angle;
      break;
    case '>':
      if (paren == 0 && bracket == 0 && brace == 0) --angle;
      break;
    case '(': ++paren; break;
    case ')': --paren; break;
    case '[': ++bracket; break;
    case ']': --bracket; break;
    case '{': ++brace; break;
    case '}': --brace; break;
    case '=':
      if (angle == 0 && paren == 0 && bracket == 0 && brace == 0) return pos;
      break;
    default: break;
    }
    ++pos;
  }
  return std::nullopt;
}

bool ParseTemplateParams(const std::string& text,
                         std::vector<TemplateParam>& params,
                         const std::string& filename, size_t offset) {
  auto parts = SplitTopLevel(text, ',');
  if (parts.empty())
    return DiagnoseOffset(
        filename, offset,
        "a primary Choreo template needs at least one parameter");

  std::set<std::string> names;
  bool saw_default = false;
  static const std::regex param_re(
      R"(^\s*(typename|class|int|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$)");
  for (const auto& raw : parts) {
    auto equal = FindTopLevelEqual(raw);
    std::string declaration = Trim(raw.substr(0, equal.value_or(raw.size())));
    std::string default_arg =
        equal ? Trim(raw.substr(*equal + 1)) : std::string();
    std::smatch match;
    if (!std::regex_match(declaration, match, param_re))
      return DiagnoseOffset(filename, offset,
                            "unsupported Choreo template parameter '" + raw +
                                "' (expected typename, class, int, or bool)");
    if (equal && default_arg.empty())
      return DiagnoseOffset(filename, offset,
                            "empty template parameter default");
    if (!equal && saw_default)
      return DiagnoseOffset(
          filename, offset,
          "a template parameter without a default follows one with a default");
    saw_default |= equal.has_value();

    TemplateParam param;
    std::string kind = match[1].str();
    param.kind = kind == "int"    ? ParamKind::Int
                 : kind == "bool" ? ParamKind::Bool
                                  : ParamKind::Type;
    param.name = match[2].str();
    param.default_arg = default_arg;
    if (!names.insert(param.name).second)
      return DiagnoseOffset(filename, offset,
                            "duplicate template parameter '" + param.name +
                                "'");
    params.push_back(std::move(param));
  }
  return true;
}

bool ParseFunction(const std::string& text, size_t co_begin,
                   FunctionSyntax& syntax, const std::string& filename) {
  syntax.begin = co_begin;
  size_t pos = co_begin + std::string("__co__").size();
  auto ret = ParseIdentifier(text, pos);
  if (!ret)
    return DiagnoseOffset(filename, pos,
                          "expected a return type after '__co__'");
  pos = ret->second;
  auto name = ParseIdentifier(text, pos);
  if (!name)
    return DiagnoseOffset(filename, pos,
                          "expected a function name in Choreo template");
  syntax.name = name->first;
  syntax.name_begin = SkipTrivia(text, pos);
  syntax.name_end = name->second;
  pos = SkipTrivia(text, name->second);

  if (pos < text.size() && text[pos] == '<') {
    auto close = FindMatching(text, pos, '<', '>');
    if (!close)
      return DiagnoseOffset(filename, pos,
                            "unterminated template argument list on '" +
                                syntax.name + "'");
    syntax.template_args_begin = pos;
    syntax.template_args_end = *close + 1;
    syntax.template_args =
        SplitTopLevel(text.substr(pos + 1, *close - pos - 1), ',');
    pos = SkipTrivia(text, *close + 1);
  }

  if (pos >= text.size() || text[pos] != '(')
    return DiagnoseOffset(filename, pos,
                          "expected '(' after Choreo template function name");
  auto close_paren = FindMatching(text, pos, '(', ')');
  if (!close_paren)
    return DiagnoseOffset(filename, pos,
                          "unterminated Choreo template parameter list");
  pos = SkipTrivia(text, *close_paren + 1);
  if (pos >= text.size() || text[pos] != '{')
    return DiagnoseOffset(filename, pos,
                          "expected a function body for Choreo template '" +
                              syntax.name + "'");
  auto close_brace = FindMatching(text, pos, '{', '}');
  if (!close_brace)
    return DiagnoseOffset(filename, pos,
                          "unterminated body of Choreo template '" +
                              syntax.name + "'");
  syntax.end = *close_brace + 1;
  return true;
}

bool CollectTemplates(const std::string& input,
                      std::vector<TemplateDefinition>& definitions,
                      std::vector<InstantiationRequest>& requests,
                      const std::string& filename) {
  size_t pos = 0;
  while (pos < input.size()) {
    if (SkipCommentOrLiteral(input, pos)) continue;
    if (!IsKeywordAt(input, pos, "template")) {
      ++pos;
      continue;
    }

    size_t template_begin = pos;
    size_t line = LineAt(input, pos);
    size_t next = SkipTrivia(input, pos + std::string("template").size());
    if (next >= input.size()) break;

    if (input[next] == '<') {
      auto close = FindMatching(input, next, '<', '>');
      if (!close)
        return DiagnoseOffset(filename, next,
                              "unterminated Choreo template parameter list");
      size_t co = SkipTrivia(input, *close + 1);
      if (!IsKeywordAt(input, co, "__co__")) {
        pos = *close + 1;
        continue;
      }

      FunctionSyntax function;
      if (!ParseFunction(input, co, function, filename)) return false;
      std::string header = Trim(input.substr(next + 1, *close - next - 1));
      TemplateDefinition definition;
      definition.begin = template_begin;
      definition.end = function.end;
      definition.line = line;
      definition.function_begin = function.begin;
      definition.function_line = LineAt(input, function.begin);
      definition.name = function.name;
      definition.function_text =
          input.substr(function.begin, function.end - function.begin);
      definition.function_name_begin = function.name_begin - function.begin;
      definition.function_name_end = function.name_end - function.begin;
      if (function.template_args_begin != std::string::npos) {
        definition.function_template_args_begin =
            function.template_args_begin - function.begin;
        definition.function_template_args_end =
            function.template_args_end - function.begin;
      }

      if (header.empty()) {
        if (function.template_args.empty())
          return DiagnoseOffset(
              filename, template_begin,
              "'template <>' requires explicit arguments on the Choreo "
              "function name");
        definition.specialization_args = function.template_args;
      } else {
        if (!function.template_args.empty())
          return DiagnoseOffset(
              filename, template_begin,
              "partial Choreo template specialization is not supported");
        if (!ParseTemplateParams(header, definition.params, filename, next + 1))
          return false;
      }
      definitions.push_back(std::move(definition));
      pos = function.end;
      continue;
    }

    if (IsKeywordAt(input, next, "__co__")) {
      size_t cursor = next + std::string("__co__").size();
      auto name = ParseIdentifier(input, cursor);
      if (!name)
        return DiagnoseOffset(
            filename, cursor,
            "expected a function name in explicit Choreo template "
            "instantiation");
      cursor = SkipTrivia(input, name->second);
      if (cursor >= input.size() || input[cursor] != '<')
        return DiagnoseOffset(
            filename, cursor,
            "explicit Choreo template instantiation requires an argument "
            "list");
      auto close = FindMatching(input, cursor, '<', '>');
      if (!close)
        return DiagnoseOffset(filename, cursor,
                              "unterminated explicit template argument list");
      size_t semicolon = SkipTrivia(input, *close + 1);
      if (semicolon >= input.size() || input[semicolon] != ';')
        return DiagnoseOffset(
            filename, semicolon,
            "expected ';' after explicit Choreo template instantiation");
      InstantiationRequest request;
      request.begin = template_begin;
      request.end = semicolon + 1;
      request.line = line;
      request.name = name->first;
      request.args =
          SplitTopLevel(input.substr(cursor + 1, *close - cursor - 1), ',');
      requests.push_back(std::move(request));
      pos = semicolon + 1;
      continue;
    }

    ++pos;
  }
  return true;
}

std::string StripChoreoNamespace(std::string name) {
  name = Trim(name);
  static const std::string prefix = "choreo::";
  if (name.compare(0, prefix.size(), prefix) == 0) name.erase(0, prefix.size());
  return name;
}

bool ParseIntArg(const std::string& raw, BoundArg& result,
                 const std::string& filename, size_t offset) {
  std::string text = Trim(raw);
  if (text.empty())
    return DiagnoseOffset(filename, offset, "empty int template argument");
  size_t suffix = text.size();
  while (suffix > 0 && (text[suffix - 1] == 'u' || text[suffix - 1] == 'U' ||
                        text[suffix - 1] == 'l' || text[suffix - 1] == 'L'))
    --suffix;
  std::string number = text.substr(0, suffix);
  char* end = nullptr;
  errno = 0;
  long long value = std::strtoll(number.c_str(), &end, 0);
  if (errno != 0 || end == number.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max())
    return DiagnoseOffset(filename, offset,
                          "invalid int template argument '" + raw + "'");
  result = {ParamKind::Int, std::to_string(value), std::to_string(value),
            "i:" + std::to_string(value)};
  return true;
}

bool ParseBoolArg(const std::string& raw, BoundArg& result,
                  const std::string& filename, size_t offset) {
  std::string text = Trim(raw);
  if (text == "true" || text == "1") {
    result = {ParamKind::Bool, "true", "true", "b:true"};
    return true;
  }
  if (text == "false" || text == "0") {
    result = {ParamKind::Bool, "false", "false", "b:false"};
    return true;
  }
  return DiagnoseOffset(filename, offset,
                        "invalid bool template argument '" + raw + "'");
}

bool ParseTypeArg(const std::string& raw, BoundArg& result,
                  const std::string& filename, size_t offset) {
  std::string text = StripChoreoNamespace(raw);
  auto found = TypeNames().find(text);
  if (found == TypeNames().end())
    return DiagnoseOffset(filename, offset,
                          "unsupported Choreo type template argument '" + raw +
                              "'");
  result = {ParamKind::Type, found->second.dsl, found->second.cpp,
            "t:" + std::string(found->second.canonical)};
  return true;
}

bool BindArguments(const TemplateDefinition& definition,
                   const std::vector<std::string>& provided, size_t offset,
                   std::vector<BoundArg>& bound, std::string& key,
                   const std::string& filename) {
  if (provided.size() > definition.params.size())
    return DiagnoseOffset(filename, offset,
                          "too many template arguments for '" +
                              definition.name + "'");
  for (size_t i = 0; i < definition.params.size(); ++i) {
    const auto& param = definition.params[i];
    std::string raw = i < provided.size() ? provided[i] : param.default_arg;
    if (raw.empty())
      return DiagnoseOffset(filename, offset,
                            "missing template argument for '" + param.name +
                                "' in '" + definition.name + "'");
    BoundArg arg;
    bool ok = param.kind == ParamKind::Type
                  ? ParseTypeArg(raw, arg, filename, offset)
              : param.kind == ParamKind::Int
                  ? ParseIntArg(raw, arg, filename, offset)
                  : ParseBoolArg(raw, arg, filename, offset);
    if (!ok) return false;
    if (!key.empty()) key += ",";
    key += arg.canonical;
    bound.push_back(std::move(arg));
  }
  return true;
}

uint64_t StableHash(const std::string& text) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : text) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string InternalName(const std::string& name, const std::string& key) {
  std::ostringstream os;
  os << "__choreo_template_" << name << "_" << std::hex << std::setw(16)
     << std::setfill('0') << StableHash(name + "<" + key + ">");
  return os.str();
}

std::string DisplayName(const std::string& name,
                        const std::vector<BoundArg>& args) {
  std::ostringstream os;
  os << name << "<";
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) os << ", ";
    os << args[i].dsl;
  }
  os << ">";
  return os.str();
}

std::string ReplaceIdentifiers(
    const std::string& text,
    const std::unordered_map<std::string, std::string>& replacements) {
  std::string result;
  result.reserve(text.size());
  size_t pos = 0;
  while (pos < text.size()) {
    size_t begin = pos;
    if (SkipCommentOrLiteral(text, pos)) {
      result += text.substr(begin, pos - begin);
      continue;
    }
    if (IsIdentStart(text[pos])) {
      size_t end = pos + 1;
      while (end < text.size() && IsIdentChar(text[end])) ++end;
      std::string name = text.substr(pos, end - pos);
      auto found = replacements.find(name);
      result += found == replacements.end() ? name : found->second;
      pos = end;
      continue;
    }
    result.push_back(text[pos++]);
  }
  return result;
}

class ConstexprEvaluator {
  const std::string& text;
  size_t pos = 0;
  std::string error;

  void SkipSpace() {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos])))
      ++pos;
  }

  bool Consume(const std::string& token) {
    SkipSpace();
    if (text.compare(pos, token.size(), token) != 0) return false;
    pos += token.size();
    return true;
  }

  int64_t ParseOr(bool evaluate = true) {
    int64_t value = ParseAnd(evaluate);
    while (error.empty() && Consume("||")) {
      bool lhs = value != 0;
      int64_t rhs = ParseAnd(evaluate && !lhs);
      if (evaluate) value = lhs || rhs;
    }
    return value;
  }

  int64_t ParseAnd(bool evaluate = true) {
    int64_t value = ParseEquality(evaluate);
    while (error.empty() && Consume("&&")) {
      bool lhs = value != 0;
      int64_t rhs = ParseEquality(evaluate && lhs);
      if (evaluate) value = lhs && rhs;
    }
    return value;
  }

  int64_t ParseEquality(bool evaluate = true) {
    int64_t value = ParseRelational(evaluate);
    while (error.empty()) {
      bool equal = Consume("==");
      bool not_equal = !equal && Consume("!=");
      if (!equal && !not_equal) break;
      int64_t rhs = ParseRelational(evaluate);
      if (evaluate) value = equal ? value == rhs : value != rhs;
    }
    return value;
  }

  int64_t ParseRelational(bool evaluate = true) {
    int64_t value = ParseAdditive(evaluate);
    while (error.empty()) {
      enum class Op { None, LE, GE, LT, GT } op = Op::None;
      if (Consume("<="))
        op = Op::LE;
      else if (Consume(">="))
        op = Op::GE;
      else if (Consume("<"))
        op = Op::LT;
      else if (Consume(">"))
        op = Op::GT;
      if (op == Op::None) break;
      int64_t rhs = ParseAdditive(evaluate);
      if (!evaluate) continue;
      if (op == Op::LE)
        value = value <= rhs;
      else if (op == Op::GE)
        value = value >= rhs;
      else if (op == Op::LT)
        value = value < rhs;
      else
        value = value > rhs;
    }
    return value;
  }

  int64_t ParseAdditive(bool evaluate = true) {
    int64_t value = ParseMultiplicative(evaluate);
    while (error.empty()) {
      bool add = Consume("+");
      bool subtract = !add && Consume("-");
      if (!add && !subtract) break;
      int64_t rhs = ParseMultiplicative(evaluate);
      if (evaluate) {
        int64_t next = 0;
        bool overflow = add ? __builtin_add_overflow(value, rhs, &next)
                            : __builtin_sub_overflow(value, rhs, &next);
        if (overflow) {
          error = "integer overflow";
          return 0;
        }
        value = next;
      }
    }
    return value;
  }

  int64_t ParseMultiplicative(bool evaluate = true) {
    int64_t value = ParseUnary(evaluate);
    while (error.empty()) {
      if (Consume("*")) {
        int64_t rhs = ParseUnary(evaluate);
        if (evaluate) {
          int64_t next = 0;
          if (__builtin_mul_overflow(value, rhs, &next)) {
            error = "integer overflow";
            return 0;
          }
          value = next;
        }
      } else if (Consume("/")) {
        int64_t rhs = ParseUnary(evaluate);
        if (evaluate && rhs == 0) {
          error = "division by zero";
          return 0;
        }
        if (evaluate && value == std::numeric_limits<int64_t>::min() &&
            rhs == -1) {
          error = "integer overflow";
          return 0;
        }
        if (evaluate) value /= rhs;
      } else if (Consume("%")) {
        int64_t rhs = ParseUnary(evaluate);
        if (evaluate && rhs == 0) {
          error = "remainder by zero";
          return 0;
        }
        if (evaluate && value == std::numeric_limits<int64_t>::min() &&
            rhs == -1) {
          error = "integer overflow";
          return 0;
        }
        if (evaluate) value %= rhs;
      } else {
        break;
      }
    }
    return value;
  }

  int64_t ParseUnary(bool evaluate = true) {
    if (Consume("!")) {
      int64_t value = ParseUnary(evaluate);
      return evaluate ? !value : 0;
    }
    if (Consume("-")) {
      int64_t value = ParseUnary(evaluate);
      if (evaluate && value == std::numeric_limits<int64_t>::min()) {
        error = "integer overflow";
        return 0;
      }
      return evaluate ? -value : 0;
    }
    if (Consume("+")) return ParseUnary(evaluate);
    return ParsePrimary(evaluate);
  }

  std::string ParseTypeName() {
    SkipSpace();
    size_t begin = pos;
    while (pos < text.size()) {
      char c = text[pos];
      if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == ':')
        ++pos;
      else
        break;
    }
    return StripChoreoNamespace(text.substr(begin, pos - begin));
  }

  int64_t ParsePrimary(bool evaluate = true) {
    SkipSpace();
    if (Consume("(")) {
      int64_t value = ParseOr(evaluate);
      if (!Consume(")") && error.empty()) error = "expected ')'";
      return value;
    }
    if (text.compare(pos, 4, "true") == 0 &&
        (pos + 4 == text.size() || !IsIdentChar(text[pos + 4]))) {
      pos += 4;
      return evaluate ? 1 : 0;
    }
    if (text.compare(pos, 5, "false") == 0 &&
        (pos + 5 == text.size() || !IsIdentChar(text[pos + 5]))) {
      pos += 5;
      return 0;
    }
    if (text.compare(pos, 9, "__is_same") == 0 &&
        (pos + 9 == text.size() || !IsIdentChar(text[pos + 9]))) {
      pos += 9;
      if (!Consume("<")) {
        error = "expected '<' after __is_same";
        return 0;
      }
      std::string lhs = ParseTypeName();
      if (!Consume(",")) {
        error = "expected ',' in __is_same";
        return 0;
      }
      std::string rhs = ParseTypeName();
      if (!Consume(">")) {
        error = "expected '>' after __is_same";
        return 0;
      }
      if (lhs.empty() || rhs.empty()) {
        error = "expected two scalar types in __is_same";
        return 0;
      }
      auto lhs_type = TypeNames().find(lhs);
      auto rhs_type = TypeNames().find(rhs);
      if (lhs_type == TypeNames().end() || rhs_type == TypeNames().end()) {
        error = "unsupported type in __is_same";
        return 0;
      }
      return evaluate && std::string(lhs_type->second.canonical) ==
                             rhs_type->second.canonical;
    }

    SkipSpace();
    size_t begin = pos;
    if (pos + 1 < text.size() && text[pos] == '0' &&
        (text[pos + 1] == 'x' || text[pos + 1] == 'X')) {
      pos += 2;
      while (pos < text.size() &&
             std::isxdigit(static_cast<unsigned char>(text[pos])))
        ++pos;
    } else {
      while (pos < text.size() &&
             std::isdigit(static_cast<unsigned char>(text[pos])))
        ++pos;
    }
    size_t number_end = pos;
    while (pos < text.size() && (text[pos] == 'u' || text[pos] == 'U' ||
                                 text[pos] == 'l' || text[pos] == 'L'))
      ++pos;
    if (number_end == begin) {
      error = "expected a constant expression";
      return 0;
    }
    std::string number = text.substr(begin, number_end - begin);
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(number.c_str(), &end, 0);
    if (errno == ERANGE || end == number.c_str() || *end != '\0') {
      error = "invalid integer literal";
      return 0;
    }
    return evaluate ? value : 0;
  }

public:
  explicit ConstexprEvaluator(const std::string& expression)
      : text(expression) {}

  bool Evaluate(bool& value, std::string& message) {
    int64_t result = ParseOr();
    SkipSpace();
    if (error.empty() && pos != text.size())
      error = "unexpected token '" + text.substr(pos) + "'";
    if (!error.empty()) {
      message = error;
      return false;
    }
    value = result != 0;
    return true;
  }
};

bool DiagnoseInstance(const std::string& filename, const std::string& text,
                      size_t offset, size_t base_line,
                      const std::string& message, const Instance* instance) {
  DiagnoseExpanded(filename, text, offset, base_line, message);
  if (instance)
    DiagnoseOffset(filename, instance->request_offset,
                   "in instantiation of Choreo template '" +
                       instance->display_name + "' requested here",
                   "note");
  return false;
}

std::string PreserveLayout(const std::string& text) {
  std::string result = text;
  for (char& c : result)
    if (c != '\n' && c != '\r') c = ' ';
  return result;
}

bool FoldStaticAssertions(const std::string& text, std::string& result,
                          const std::string& filename, size_t base_line,
                          const Instance* instance) {
  size_t cursor = 0;
  size_t pos = 0;
  while (pos < text.size()) {
    if (SkipCommentOrLiteral(text, pos)) continue;
    if (!IsKeywordAt(text, pos, "static_assert")) {
      ++pos;
      continue;
    }

    size_t open_paren = SkipTrivia(text, pos + 13);
    if (open_paren >= text.size() || text[open_paren] != '(')
      return DiagnoseInstance(filename, text, pos, base_line,
                              "expected '(' after 'static_assert'", instance);
    auto close_paren = FindMatching(text, open_paren, '(', ')');
    if (!close_paren)
      return DiagnoseInstance(filename, text, pos, base_line,
                              "unterminated 'static_assert'", instance);
    size_t semicolon = SkipTrivia(text, *close_paren + 1);
    if (semicolon >= text.size() || text[semicolon] != ';')
      return DiagnoseInstance(filename, text, pos, base_line,
                              "expected ';' after 'static_assert'", instance);

    auto argument_text =
        text.substr(open_paren + 1, *close_paren - open_paren - 1);
    std::vector<size_t> commas;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    for (size_t argument_pos = 0; argument_pos < argument_text.size();) {
      if (SkipCommentOrLiteral(argument_text, argument_pos)) continue;
      switch (argument_text[argument_pos]) {
      case '(': ++paren_depth; break;
      case ')': --paren_depth; break;
      case '[': ++bracket_depth; break;
      case ']': --bracket_depth; break;
      case '{': ++brace_depth; break;
      case '}': --brace_depth; break;
      case ',':
        if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0)
          commas.push_back(argument_pos);
        break;
      default: break;
      }
      ++argument_pos;
    }

    std::string condition = Trim(argument_text);
    std::string raw_message;
    for (auto comma = commas.rbegin(); comma != commas.rend(); ++comma) {
      auto candidate_message = Trim(argument_text.substr(*comma + 1));
      if (candidate_message.size() >= 2 && candidate_message.front() == '"' &&
          candidate_message.back() == '"') {
        condition = Trim(argument_text.substr(0, *comma));
        raw_message = std::move(candidate_message);
        break;
      }
      bool ignored_value = false;
      std::string ignored_error;
      ConstexprEvaluator candidate(Trim(argument_text.substr(0, *comma)));
      if (!candidate.Evaluate(ignored_value, ignored_error)) continue;
      condition = Trim(argument_text.substr(0, *comma));
      raw_message = std::move(candidate_message);
      break;
    }
    if (condition.empty())
      return DiagnoseInstance(filename, text, pos, base_line,
                              "Choreo 'static_assert' requires a condition",
                              instance);

    bool value = false;
    std::string eval_error;
    ConstexprEvaluator evaluator(condition);
    if (!evaluator.Evaluate(value, eval_error))
      return DiagnoseInstance(
          filename, text, pos, base_line,
          "invalid 'static_assert' condition: " + eval_error, instance);

    std::string assertion_message;
    if (!raw_message.empty()) {
      if (raw_message.size() < 2 || raw_message.front() != '"' ||
          raw_message.back() != '"')
        return DiagnoseInstance(
            filename, text, pos, base_line,
            "Choreo 'static_assert' message must be a string literal",
            instance);
      assertion_message = raw_message.substr(1, raw_message.size() - 2);
    }
    if (!value) {
      std::string message = "static assertion failed";
      if (instance) message += " for '" + instance->display_name + "'";
      if (!assertion_message.empty()) message += ": " + assertion_message;
      return DiagnoseInstance(filename, text, pos, base_line, message,
                              instance);
    }

    result += text.substr(cursor, pos - cursor);
    result += PreserveLayout(text.substr(pos, semicolon + 1 - pos));
    cursor = semicolon + 1;
    pos = cursor;
  }
  result += text.substr(cursor);
  return true;
}

std::optional<size_t> FindConstexprIfEnd(const std::string& text,
                                         size_t if_pos) {
  if (!IsKeywordAt(text, if_pos, "if")) return std::nullopt;
  size_t constexpr_pos = SkipTrivia(text, if_pos + 2);
  if (!IsKeywordAt(text, constexpr_pos, "constexpr")) return std::nullopt;
  size_t open_paren = SkipTrivia(text, constexpr_pos + 9);
  if (open_paren >= text.size() || text[open_paren] != '(') return std::nullopt;
  auto close_paren = FindMatching(text, open_paren, '(', ')');
  if (!close_paren) return std::nullopt;
  size_t then_open = SkipTrivia(text, *close_paren + 1);
  if (then_open >= text.size() || text[then_open] != '{') return std::nullopt;
  auto then_close = FindMatching(text, then_open, '{', '}');
  if (!then_close) return std::nullopt;

  size_t end = *then_close + 1;
  size_t else_pos = SkipTrivia(text, end);
  if (!IsKeywordAt(text, else_pos, "else")) return end;
  size_t else_body = SkipTrivia(text, else_pos + 4);
  if (else_body < text.size() && text[else_body] == '{') {
    auto else_close = FindMatching(text, else_body, '{', '}');
    return else_close ? std::optional<size_t>(*else_close + 1) : std::nullopt;
  }
  return FindConstexprIfEnd(text, else_body);
}

bool FoldConstexpr(const std::string& text, std::string& result,
                   const std::string& filename, size_t base_line,
                   const Instance* instance) {
  size_t cursor = 0;
  size_t pos = 0;
  while (pos < text.size()) {
    if (SkipCommentOrLiteral(text, pos)) continue;
    if (!IsKeywordAt(text, pos, "if")) {
      ++pos;
      continue;
    }
    size_t constexpr_pos = SkipTrivia(text, pos + 2);
    if (!IsKeywordAt(text, constexpr_pos, "constexpr")) {
      ++pos;
      continue;
    }
    size_t open_paren = SkipTrivia(text, constexpr_pos + 9);
    if (open_paren >= text.size() || text[open_paren] != '(')
      return DiagnoseInstance(filename, text, pos, base_line,
                              "expected '(' after 'if constexpr'", instance);
    auto close_paren = FindMatching(text, open_paren, '(', ')');
    if (!close_paren)
      return DiagnoseInstance(filename, text, pos, base_line,
                              "unterminated 'if constexpr' condition",
                              instance);
    std::string condition =
        text.substr(open_paren + 1, *close_paren - open_paren - 1);
    bool selected = false;
    std::string eval_error;
    ConstexprEvaluator evaluator(condition);
    if (!evaluator.Evaluate(selected, eval_error))
      return DiagnoseInstance(filename, text, pos, base_line,
                              "invalid 'if constexpr' condition: " + eval_error,
                              instance);

    size_t then_open = SkipTrivia(text, *close_paren + 1);
    if (then_open >= text.size() || text[then_open] != '{')
      return DiagnoseInstance(filename, text, pos, base_line,
                              "Choreo 'if constexpr' requires a braced body",
                              instance);
    auto then_close = FindMatching(text, then_open, '{', '}');
    if (!then_close)
      return DiagnoseInstance(filename, text, pos, base_line,
                              "unterminated 'if constexpr' body", instance);

    size_t whole_end = *then_close + 1;
    std::string chosen;
    size_t chosen_anchor = then_open;
    if (selected)
      chosen = text.substr(then_open + 1, *then_close - then_open - 1);

    size_t else_pos = SkipTrivia(text, whole_end);
    if (IsKeywordAt(text, else_pos, "else")) {
      size_t else_open = SkipTrivia(text, else_pos + 4);
      if (IsKeywordAt(text, else_open, "if")) {
        auto nested_end = FindConstexprIfEnd(text, else_open);
        if (!nested_end)
          return DiagnoseInstance(filename, text, pos, base_line,
                                  "malformed 'else if constexpr' branch",
                                  instance);
        if (!selected) {
          chosen = text.substr(else_open, *nested_end - else_open);
          chosen_anchor = else_open;
        }
        whole_end = *nested_end;
      } else if (else_open >= text.size() || text[else_open] != '{') {
        return DiagnoseInstance(
            filename, text, pos, base_line,
            "Choreo 'if constexpr' requires a braced 'else'", instance);
      } else {
        auto else_close = FindMatching(text, else_open, '{', '}');
        if (!else_close)
          return DiagnoseInstance(filename, text, pos, base_line,
                                  "unterminated 'if constexpr' else body",
                                  instance);
        if (!selected) {
          chosen = text.substr(else_open + 1, *else_close - else_open - 1);
          chosen_anchor = else_open;
        }
        whole_end = *else_close + 1;
      }
    }

    std::string folded_prefix;
    auto prefix = text.substr(cursor, pos - cursor);
    if (!FoldStaticAssertions(prefix, folded_prefix, filename,
                              base_line + LineAt(text, cursor) - 1, instance))
      return false;

    std::string folded_chosen;
    if (!FoldConstexpr(chosen, folded_chosen, filename,
                       base_line + LineAt(text, chosen_anchor) - 1, instance))
      return false;
    result += folded_prefix;
    // Keep a lexical scope and a statically true predicate. The inactive
    // branch has already disappeared before parsing.
    std::string replacement = "if (true) {" + folded_chosen + "}";
    result += replacement;
    auto original_newlines = static_cast<size_t>(
        std::count(text.begin() + pos, text.begin() + whole_end, '\n'));
    auto replacement_newlines = static_cast<size_t>(
        std::count(replacement.begin(), replacement.end(), '\n'));
    if (original_newlines > replacement_newlines)
      result.append(original_newlines - replacement_newlines, '\n');
    cursor = whole_end;
    pos = whole_end;
  }
  std::string folded_suffix;
  auto suffix = text.substr(cursor);
  if (!FoldStaticAssertions(suffix, folded_suffix, filename,
                            base_line + LineAt(text, cursor) - 1, instance))
    return false;
  result += folded_suffix;
  return true;
}

std::string InstantiateFunction(const TemplateDefinition& primary,
                                const Instance& instance) {
  const TemplateDefinition& source =
      instance.specialization ? *instance.specialization : primary;
  std::string function = source.function_text;

  if (source.function_template_args_begin != std::string::npos) {
    function.erase(source.function_template_args_begin,
                   source.function_template_args_end -
                       source.function_template_args_begin);
  }
  function.replace(source.function_name_begin,
                   source.function_name_end - source.function_name_begin,
                   instance.internal_name);

  if (!instance.specialization) {
    std::unordered_map<std::string, std::string> substitutions;
    for (size_t i = 0; i < primary.params.size(); ++i)
      substitutions.emplace(primary.params[i].name, instance.args[i].dsl);
    function = ReplaceIdentifiers(function, substitutions);
  }
  return function;
}

std::string Dispatcher(const TemplateDefinition& primary,
                       const std::vector<Instance>& instances) {
  std::ostringstream os;
  os << "\n#include <type_traits>\n#include <utility>\n";
  os << "template <";
  for (size_t i = 0; i < primary.params.size(); ++i) {
    if (i) os << ", ";
    const auto& param = primary.params[i];
    if (param.kind == ParamKind::Type)
      os << "typename ";
    else if (param.kind == ParamKind::Int)
      os << "int ";
    else
      os << "bool ";
    os << param.name;
    if (!param.default_arg.empty()) {
      BoundArg value;
      if (param.kind == ParamKind::Type)
        ParseTypeArg(param.default_arg, value, "", 0);
      else if (param.kind == ParamKind::Int)
        ParseIntArg(param.default_arg, value, "", 0);
      else
        ParseBoolArg(param.default_arg, value, "", 0);
      os << " = " << value.cpp;
    }
  }
  if (!primary.params.empty()) os << ", ";
  os << "typename... __ChoreoArgs>\n";
  os << "decltype(auto) " << primary.name
     << "(__ChoreoArgs&&... __choreo_args) {\n";
  for (size_t instance_index = 0; instance_index < instances.size();
       ++instance_index) {
    os << (instance_index == 0 ? "  if constexpr (" : "  else if constexpr (");
    const auto& instance = instances[instance_index];
    for (size_t i = 0; i < primary.params.size(); ++i) {
      if (i) os << " && ";
      const auto& param = primary.params[i];
      const auto& arg = instance.args[i];
      if (param.kind == ParamKind::Type)
        os << "std::is_same<" << param.name << ", " << arg.cpp << ">::value";
      else
        os << param.name << " == " << arg.cpp;
    }
    os << ") {\n    return " << instance.internal_name
       << "(std::forward<__ChoreoArgs>(__choreo_args)...);\n  }\n";
  }
  os << (instances.empty() ? "  {\n" : "  else {\n")
     << "    static_assert(sizeof...(__ChoreoArgs) != "
        "sizeof...(__ChoreoArgs),\n"
     << "                  \"Choreo template '" << primary.name
     << "' was not explicitly instantiated for these template arguments\");\n"
     << "  }\n}\n";
  return os.str();
}

std::string PreserveNewlines(const std::string& text) {
  return std::string(
      static_cast<size_t>(std::count(text.begin(), text.end(), '\n')), '\n');
}

std::string EscapeLineFilename(const std::string& filename) {
  std::string result;
  for (char c : filename) {
    if (c == '\\' || c == '"') result.push_back('\\');
    result.push_back(c);
  }
  return result;
}

} // namespace

bool IsChoreoTemplateInstantiation(const std::string& line) {
  static const std::regex pattern(R"(^\s*template\s+__co__(?:\s|$).*$)");
  return std::regex_match(line, pattern);
}

bool ExpandChoreoTemplates(const std::string& input, std::ostream& output,
                           const std::string& filename) {
  ScopedDiagnosticSource diagnostic_scope(input);
  template_instances.clear();
  active_template_instance.clear();
  std::vector<TemplateDefinition> definitions;
  std::vector<InstantiationRequest> requests;
  if (!CollectTemplates(input, definitions, requests, filename)) return false;
  if (definitions.empty() && requests.empty()) {
    output << input;
    return true;
  }

  std::map<std::string, const TemplateDefinition*> primaries;
  std::map<std::string, std::vector<const TemplateDefinition*>> specializations;
  for (const auto& definition : definitions) {
    if (!definition.IsSpecialization()) {
      if (!primaries.emplace(definition.name, &definition).second)
        return DiagnoseOffset(
            filename, definition.begin,
            "Choreo template overloading is not supported for '" +
                definition.name + "'");
    } else {
      specializations[definition.name].push_back(&definition);
    }
  }

  for (const auto& request : requests)
    if (!primaries.count(request.name))
      return DiagnoseOffset(
          filename, request.begin,
          "explicit instantiation of undefined Choreo template '" +
              request.name + "'");
  for (const auto& item : specializations)
    if (!primaries.count(item.first))
      return DiagnoseOffset(filename, item.second.front()->begin,
                            "specialization of undefined Choreo template '" +
                                item.first + "'");

  for (const auto& item : primaries) {
    for (const auto& param : item.second->params) {
      if (param.default_arg.empty()) continue;
      BoundArg value;
      bool ok = param.kind == ParamKind::Type
                    ? ParseTypeArg(param.default_arg, value, filename,
                                   item.second->begin)
                : param.kind == ParamKind::Int
                    ? ParseIntArg(param.default_arg, value, filename,
                                  item.second->begin)
                    : ParseBoolArg(param.default_arg, value, filename,
                                   item.second->begin);
      if (!ok) return false;
    }
  }

  std::map<std::string, std::vector<Instance>> all_instances;
  for (const auto& primary_item : primaries) {
    const auto& name = primary_item.first;
    const auto& primary = *primary_item.second;
    std::map<std::string, Instance> unique;

    for (const auto& request : requests) {
      if (request.name != name) continue;
      Instance instance;
      if (!BindArguments(primary, request.args, request.begin, instance.args,
                         instance.key, filename))
        return false;
      instance.request_offset = request.begin;
      instance.request_line = request.line;
      instance.internal_name = InternalName(name, instance.key);
      instance.display_name = DisplayName(name, instance.args);
      unique.emplace(instance.key, std::move(instance));
    }

    std::map<std::string, const TemplateDefinition*> specs_by_key;
    for (const auto* specialization : specializations[name]) {
      Instance specialized;
      if (!BindArguments(primary, specialization->specialization_args,
                         specialization->begin, specialized.args,
                         specialized.key, filename))
        return false;
      if (!specs_by_key.emplace(specialized.key, specialization).second)
        return DiagnoseOffset(
            filename, specialization->begin,
            "duplicate full specialization of Choreo template '" + name + "'");
      specialized.request_line = specialization->line;
      specialized.request_offset = specialization->begin;
      specialized.internal_name = InternalName(name, specialized.key);
      specialized.display_name = DisplayName(name, specialized.args);
      specialized.specialization = specialization;
      auto inserted = unique.emplace(specialized.key, specialized);
      if (!inserted.second)
        inserted.first->second.specialization = specialization;
    }

    for (auto& item : unique) {
      auto spec = specs_by_key.find(item.first);
      if (spec != specs_by_key.end()) item.second.specialization = spec->second;
      all_instances[name].push_back(std::move(item.second));
    }
    std::sort(all_instances[name].begin(), all_instances[name].end(),
              [](const Instance& lhs, const Instance& rhs) {
                return lhs.request_line < rhs.request_line;
              });
  }

  std::map<std::string, std::string> symbol_keys;
  for (const auto& [name, instances] : all_instances) {
    for (const auto& instance : instances) {
      auto identity = name + "<" + instance.key + ">";
      auto inserted = symbol_keys.emplace(instance.internal_name, identity);
      if (!inserted.second && inserted.first->second != identity)
        return DiagnoseOffset(
            filename, instance.request_offset,
            "internal symbol collision between Choreo template instances '" +
                inserted.first->second + "' and '" + identity + "'");
      template_instances[instance.internal_name] = {
          instance.display_name,
          filename,
          instance.request_line,
          ColumnAt(input, instance.request_offset),
          instance.specialization ? instance.specialization->function_line
                                  : primaries[name]->function_line,
          (instance.specialization ? instance.specialization->function_line
                                   : primaries[name]->function_line) +
              LineAt(instance.specialization
                         ? instance.specialization->function_text
                         : primaries[name]->function_text,
                     (instance.specialization
                          ? instance.specialization->function_text
                          : primaries[name]->function_text)
                         .size()) -
              1};
    }
  }

  std::vector<Replacement> replacements;
  for (const auto& definition : definitions) {
    if (definition.IsSpecialization()) {
      replacements.push_back(
          {definition.begin, definition.end,
           PreserveNewlines(input.substr(definition.begin,
                                         definition.end - definition.begin))});
      continue;
    }

    std::ostringstream expanded;
    for (const auto& instance : all_instances[definition.name]) {
      size_t source_line = instance.specialization
                               ? instance.specialization->function_line
                               : definition.function_line;
      expanded << "// __choreo_template_instance " << instance.internal_name
               << "\n";
      if (!filename.empty())
        expanded << "#line " << source_line << " \""
                 << EscapeLineFilename(filename) << "\"\n";
      std::string concrete = InstantiateFunction(definition, instance);
      std::string folded;
      if (!FoldConstexpr(concrete, folded, filename, source_line, &instance))
        return false;
      expanded << folded << "\n";
    }
    if (!filename.empty())
      expanded << "#line " << definition.line << " \""
               << EscapeLineFilename(filename) << "\"\n";
    expanded << Dispatcher(definition, all_instances[definition.name]);
    size_t after_line = definition.line +
                        LineAt(input.substr(definition.begin,
                                            definition.end - definition.begin),
                               definition.end - definition.begin) -
                        1;
    if (!filename.empty())
      expanded << "#line " << after_line << " \""
               << EscapeLineFilename(filename) << "\"\n";
    replacements.push_back({definition.begin, definition.end, expanded.str()});
  }
  for (const auto& request : requests)
    replacements.push_back({request.begin, request.end,
                            PreserveNewlines(input.substr(
                                request.begin, request.end - request.begin))});

  std::sort(replacements.begin(), replacements.end(),
            [](const Replacement& lhs, const Replacement& rhs) {
              return lhs.begin < rhs.begin;
            });
  size_t cursor = 0;
  for (const auto& replacement : replacements) {
    if (replacement.begin < cursor)
      return DiagnoseOffset(filename, replacement.begin,
                            "overlapping Choreo template declarations");
    output << input.substr(cursor, replacement.begin - cursor);
    output << replacement.text;
    cursor = replacement.end;
  }
  output << input.substr(cursor);
  return true;
}

const ChoreoTemplateInstanceInfo*
FindChoreoTemplateInstance(const std::string& internal_name) {
  auto found = template_instances.find(internal_name);
  return found == template_instances.end() ? nullptr : &found->second;
}

void SetActiveChoreoTemplateInstance(const std::string& internal_name) {
  active_template_instance = internal_name;
}

const ChoreoTemplateInstanceInfo*
FindActiveChoreoTemplateInstance(const std::string& filename, size_t line) {
  auto* instance = FindChoreoTemplateInstance(active_template_instance);
  if (!instance || instance->filename != filename ||
      line < instance->definition_begin_line ||
      line > instance->definition_end_line)
    return nullptr;
  return instance;
}

} // namespace Choreo
