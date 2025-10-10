#ifndef __CHOREO_PRE_PROCESS_HPP__
#define __CHOREO_PRE_PROCESS_HPP__

#include "aux.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "context.hpp"
#include "io.hpp"
#include <filesystem>
#include <iostream>
#include <istream>
#include <regex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Choreo {

Option<bool>
    applyDefinesInCO(OptionKind::Internal, "--apply-defines-in-co", "", true,
                     "Apply #define macros inside of Choreo Function.");
Option<bool>
    applyDefinesInCOK(OptionKind::Internal, "--apply-defines-in-cok", "", true,
                      "Apply #define macros inside of Choreo Function.");

Option<bool> debugPP(OptionKind::Internal, "--debug-pp", "", false,
                     "Debug the preprocessor.");

class SimplePreprocessor {
private:
  using DefineMap = std::unordered_map<std::string, std::string>;
  using FuncMap =
      std::unordered_map<std::string, std::tuple<std::string, std::string>>;

  std::ostream& output;
  DefineMap globalDefines;
  DefineMap localDefines;
  FuncMap globalDefinedFuncs;
  FuncMap localDefinedFuncs;

  int choreo_brace_count = 0;
  int kernel_brace_count = 0;

  int uc_if_count = 0;
  int co_if_count = 0;

  bool co_skip_line = false;
  bool uc_skip_line = false;

  std::stack<bool> co_condition_stack;
  std::stack<bool> co_skip_stack;

  std::stack<bool> uc_condition_stack;
  std::stack<bool> uc_skip_stack;

  enum CodePartition {
    CP_CHOREO,
    CP_KERNEL,
    CP_USER,
  };

  CodePartition code_partition = CP_USER;

  bool c_skip = false;
  size_t line_num = 1;

  bool has_cok = false;
  std::string build_path;
  std::string cc_file;
  std::string pp_file;
  std::vector<std::string> include_lines;
  std::vector<std::string> cok_codes;

  struct Bundle {
    size_t start;
    size_t end;
  };

  struct Range {
    size_t start{}, end{};
  };

  bool debug = false;

public:
  SimplePreprocessor(std::ostream& o) : output(o), debug(debugPP) {
    // Replicate the target-specific macros
    switch (CCtx().GetTarget()) {
    case CompileTarget::Topscc: globalDefines.emplace("__TOPSCC__", ""); break;
    default: break;
    }
    switch (CCtx().GetArch()) {
    case TargetArch::GCU20: globalDefines.emplace("__GCU_ARCH__", "200"); break;
    case TargetArch::GCU21: globalDefines.emplace("__GCU_ARCH__", "210"); break;
    case TargetArch::GCU3: globalDefines.emplace("__GCU_ARCH__", "300"); break;
    case TargetArch::GCU4: globalDefines.emplace("__GCU_ARCH__", "400"); break;
    default: break;
    }
    // command-line macros override
    for (auto& item : CCtx().GetCLMacros())
      globalDefines[item.first] = item.second;
  }

private:
  std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";

    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
  }

  std::vector<Bundle> find_bundles(const std::string& code) {
    std::vector<Bundle> bundles;
    std::istringstream iss(code);
    std::string line;
    size_t offset = 0;
    size_t start_offset = std::string::npos;
    while (getline(iss, line)) {
      // include newline length
      size_t line_len = line.size() + 1;
      if (line.rfind(
              "// __CLANG_OFFLOAD_BUNDLE____START__ tops-dtu-enflame-tops",
              0) == 0) {
        start_offset = offset + line_len; // content starts after this line
      } else if (line.rfind(
                     "// __CLANG_OFFLOAD_BUNDLE____END__ tops-dtu-enflame-tops",
                     0) == 0) {
        if (start_offset != std::string::npos) {
          bundles.push_back(
              {start_offset, offset}); // content ends before this line
          start_offset = std::string::npos;
        }
      }
      offset += line_len;
    }
    return bundles;
  }

  static inline bool isIdent(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
  }

  // Advance i while predicate holds; returns new index
  template <class Pred>
  static inline size_t advance_while(const std::string& s, size_t i, Pred p) {
    while (i < s.size() && p(s[i])) ++i;
    return i;
  }

  // Skip whitespace and comments starting at i. Returns index of first
  // non-space/comment char.
  static size_t skip_ws_and_comments(const std::string& s, size_t i) {
    for (;;) {
      // skip whitespace
      i = advance_while(s, i, [](char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' ||
               c == '\v';
      });
      if (i >= s.size()) return i;
      // line comment
      if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/') {
        i += 2;
        while (i < s.size() && s[i] != '\n') ++i;
        continue;
      }
      // block comment
      if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '*') {
        i += 2;
        while (i + 1 < s.size() && !(s[i] == '*' && s[i + 1] == '/')) ++i;
        if (i + 1 < s.size()) i += 2;
        continue;
      }
      break;
    }
    return i;
  }

  static void skip_string_literal(const std::string& s, size_t& i) {
    // assumes s[i] == '"'
    ++i; // past opening quote
    while (i < s.size()) {
      if (s[i] == '\\') {
        i += 2;
        continue;
      }
      if (s[i] == '"') {
        ++i;
        break;
      }
      ++i;
    }
  }

  static void skip_char_literal(const std::string& s, size_t& i) {
    // assumes s[i] == '\''
    ++i;
    while (i < s.size()) {
      if (s[i] == '\\') {
        i += 2;
        continue;
      }
      if (s[i] == '\'') {
        ++i;
        break;
      }
      ++i;
    }
  }

  static void skip_raw_string(const std::string& s, size_t& i) {
    // assumes s[i] == 'R' and s[i+1] == '"'
    i += 2; // position at start of delimiter
    std::string delim;
    while (i < s.size() && s[i] != '(') {
      delim.push_back(s[i]);
      ++i;
    }
    if (i < s.size() && s[i] == '(') ++i; // consume '('
    // find ")delim"
    for (;;) {
      if (i >= s.size()) return; // unmatched, give up
      if (s[i] == ')') {
        size_t j = i + 1;
        // check for closing delimiter delim then '"'
        if (j + delim.size() < s.size() &&
            s.compare(j, delim.size(), delim) == 0 &&
            j + delim.size() < s.size() && s[j + delim.size()] == '"') {
          i = j + delim.size() + 1; // past closing '"'
          return;
        }
      }
      ++i;
    }
  }

  std::vector<Range> extract_cok_sections(const std::string& code,
                                          size_t base_offset = 0) {
    std::vector<Range> ranges;
    const std::string token = "__cok__";
    size_t i = 0, N = code.size();

    while (i < N) {
      char c = code[i];

      // skip comments/strings
      if (c == '/' && i + 1 < N) {
        if (code[i + 1] == '/') {
          i += 2;
          while (i < N && code[i] != '\n') ++i;
          continue;
        }
        if (code[i + 1] == '*') {
          i += 2;
          while (i + 1 < N && !(code[i] == '*' && code[i + 1] == '/')) ++i;
          if (i + 1 < N) i += 2;
          continue;
        }
      }
      if (c == '"') {
        skip_string_literal(code, i);
        continue;
      }
      if (c == '\'') {
        skip_char_literal(code, i);
        continue;
      }
      if (c == 'R' && i + 1 < N && code[i + 1] == '"') {
        skip_raw_string(code, i);
        continue;
      }

      // found __cok__
      if (c == '_' && i + token.size() <= N &&
          code.compare(i, token.size(), token) == 0) {
        bool left_ok = (i == 0) || !isIdent(code[i - 1]);
        bool right_ok =
            (i + token.size() >= N) || !isIdent(code[i + token.size()]);
        if (left_ok && right_ok) {
          size_t j = i + token.size();
          j = skip_ws_and_comments(code, j);
          if (j < N && code[j] == '{') {
            size_t block_start = j + 1;
            int depth = 1;
            ++j;

            size_t content_start = block_start;
            while (j < N && depth > 0) {
              char d = code[j];

              // handle nested comments/strings
              if (d == '/' && j + 1 < N) {
                if (code[j + 1] == '/') {
                  j += 2;
                  while (j < N && code[j] != '\n') ++j;
                  continue;
                }
                if (code[j + 1] == '*') {
                  j += 2;
                  while (j + 1 < N && !(code[j] == '*' && code[j + 1] == '/'))
                    ++j;
                  if (j + 1 < N) j += 2;
                  continue;
                }
              }
              if (d == '"') {
                skip_string_literal(code, j);
                continue;
              }
              if (d == '\'') {
                skip_char_literal(code, j);
                continue;
              }
              if (d == 'R' && j + 1 < N && code[j + 1] == '"') {
                skip_raw_string(code, j);
                continue;
              }

              // detect nested __cok__
              if (d == '_' && j + token.size() <= N &&
                  code.compare(j, token.size(), token) == 0) {
                // flush parent content before nested
                if (j > content_start) {
                  ranges.push_back(
                      {base_offset + content_start, base_offset + j - 1});
                }
                // recursively extract inner __cok__ blocks
                auto inner =
                    extract_cok_sections(code.substr(j), base_offset + j);
                ranges.insert(ranges.end(), inner.begin(), inner.end());

                // skip whole nested block
                return ranges; // stop here, recursion will handle rest
              }

              if (d == '{') {
                ++depth;
                ++j;
                continue;
              }
              if (d == '}') {
                --depth;
                ++j;
                continue;
              }
              ++j;
            }
            if (depth == 0 && j > block_start) {
              ranges.push_back(
                  {base_offset + block_start, base_offset + j - 2});
            }
            i = j;
            continue;
          }
        }
      }
      ++i;
    }
    return ranges;
  }

  void EmitScript(std::ostream& os) {
    os << R"script(#!/usr/bin/env bash

# This is the choreo generated bash script to compile topscc code
TOPSCC_INSTALL=/opt/tops;
)script";

    os << R"script(
if [[ -z "${TOPSCC_INSTALL}" ]]; then
  echo "failed to find the topscc installation."
  echo "install topscc or set TOPSCC_INSTALL to topscc installation directory."
  exit 1
fi

TOPSCC=${TOPSCC_INSTALL}/bin/topscc
TOPSCC_LIB=${TOPSCC_INSTALL}/lib
)script";
    os << "rm -fr " << build_path << "\n";
    os << "mkdir -p " << build_path << "\n\n";
    os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
    os << __choreo_header_as_string << "\nEOF\n\n";
    os << "cat <<'EOF' > " << cc_file << "\n";
    for (auto& line : include_lines) { os << line << "\n"; }
    if (!cok_codes.empty()) {
      os << "\n__cok__ {\n";
      for (auto& line : cok_codes) { os << line << "\n"; }
      os << "}\n";
    }
    os << "\nEOF\n\n";
    os << R"(export CFLAGS=")";
    std::filesystem::path cwd = std::filesystem::current_path();
    const std::string input_file =
        OptionRegistry::GetInstance().GetInputFileName();
    std::filesystem::path input_rel_path(input_file);
    auto input_abs_path =
        std::filesystem::weakly_canonical(cwd / input_rel_path)
            .parent_path()
            .string();
    os << " -I" << input_abs_path;
    for (auto inc_path : CCtx().GetIncPaths()) os << " -I" << inc_path;
    for (auto lib_path : CCtx().GetLibPaths()) os << " -L" << lib_path;
    os << "\"";

    os << "\n ${TOPSCC} -E ${CFLAGS}"
       << " " << cc_file << " -o " << pp_file << "\n";
  }

  std::string SubStituteMacroFuncs(const std::string& line,
                                   const FuncMap& funcs, bool& changed) {
    if (!changed) return line;

    auto extractArgs = [this](const std::string& line,
                              const std::string& func_name, std::string& prefix,
                              std::string& suffix) {
      std::regex pattern("\\b" + func_name + "\\b");
      std::smatch match;
      size_t pos = std::string::npos;
      if (std::regex_search(line, match, pattern)) {
        pos = match.position(0);
      } else
        return std::vector<std::string>();

      std::vector<std::string> args;
      int paren_count = 0;
      int start_pos = pos + func_name.length();
      prefix = line.substr(0, pos);
      for (size_t i = pos + func_name.length(); i < line.length(); ++i) {
        char c = line[i];
        if (c == '(') {
          paren_count++;
          if (paren_count == 1) {
            start_pos = i + 1; // start after the opening parenthesis
          }
        } else if (c == ')') {
          paren_count--;
          if (paren_count == 0) {
            // found the closing parenthesis, extract the last argument
            std::string arg = line.substr(start_pos, i - start_pos);
            args.push_back(trim(arg));
            suffix = line.substr(i + 1);
            break;
          }
          if (paren_count < 0) {
            errs() << "error: unmatched parenthesis in macro function call '"
                   << func_name << "'\n";
            return std::vector<std::string>();
          } // unmatched parenthesis
        } else if (c == ',' && paren_count == 1) {
          // found a comma at the top level, extract the argument
          std::string arg = line.substr(start_pos, i - start_pos);
          args.push_back(trim(arg));
          start_pos = i + 1; // move to the next character after the comma
        }
      }
      return args;
    };

    std::string result;

    for (const auto& func : funcs) {
      const auto& name = func.first;
      const auto& args_str = std::get<0>(func.second);
      const auto& body_str = std::get<1>(func.second);
      // build regex for function arguments

      std::stringstream ss(args_str);
      std::string arg;
      std::vector<std::string> arg_template;
      while (std::getline(ss, arg, ',')) { arg_template.push_back(trim(arg)); }
      std::string prefix, suffix;
      auto args = extractArgs(line, name, prefix, suffix);
      if (args.empty() || args.size() != arg_template.size()) continue;
      changed = true;
      std::string substituted = body_str;
      for (size_t i = 0; i < args.size(); ++i) {
        auto& arg_name = arg_template[i];
        auto& arg_value = args[i];
        // replace the argument in the macro function body
        size_t pos = 0;
        while ((pos = substituted.find(arg_name, pos)) != std::string::npos) {
          substituted.replace(pos, arg_name.length(), arg_value);
          pos += arg_value.length();
        }
      }
      return prefix + substituted + suffix;
    }
    changed = false;
    return line;
  }

  std::string SubStituteDefines(const std::string& line,
                                const DefineMap& defines,
                                const FuncMap& funcs) {
    std::string result;
    std::string current_token;
    bool in_string = false;
    bool in_raw_string = false;
    bool escape = false;
    size_t paren_count = 0;
    for (size_t i = 0; i < line.length(); ++i) {
      char c = line[i];
      // raw string begins
      if (!in_string && !in_raw_string && i + 2 < line.length() &&
          line[i] == 'R' && line[i + 1] == '"' && line[i + 2] == '(') {
        i += 2;
        in_raw_string = true;
        result += "R\"(";
        paren_count++;
        continue;
      }

      // raw string ends
      if (in_raw_string && paren_count > 0) {
        if (c == ')' && i + 1 < line.length() && line[i + 1] == '"') {
          paren_count--;
          if (paren_count == 0) in_raw_string = false;
          result += c;
          result += '"';
          i++;
          continue;
        } else {
          // append char in raw string
          result += c;
          continue;
        }
      }

      // string begins & ends
      if (c == '"' && !in_raw_string && !escape) {
        in_string = !in_string;
        result += c;
        continue;
      }

      // escape character
      if (in_string && !in_raw_string) {
        if (c == '\\') {
          escape = true;
          result += c;
          continue;
        } else
          escape = false;
      }

      // append char in string
      if (in_string || in_raw_string) {
        result += c;
        continue;
      }

      if (std::isalnum(c) || c == '_')
        current_token += c;
      else {
        // substitute token
        if (!current_token.empty()) {
          auto it = defines.find(current_token);
          auto func_it = funcs.find(current_token);
          if (it != defines.end() && func_it == funcs.end())
            result += it->second;
          else
            result += current_token;
          current_token.clear();
        }
        result += c;
      }
    }

    // last token
    if (!current_token.empty()) {
      auto it = defines.find(current_token);
      if (it != defines.end())
        result += it->second;
      else
        result += current_token;
    }

    return result;
  }

  std::string SubstituteGlobalDefines(const std::string& line) {
    return SubStituteDefines(line, globalDefines, globalDefinedFuncs);
  }

  std::string SubstituteLocalDefines(const std::string& line) {
    return SubStituteDefines(line, localDefines, localDefinedFuncs);
  }

  std::string SubstituteGlobalMacroFuncs(const std::string& line,
                                         bool& changed) {
    return SubStituteMacroFuncs(line, globalDefinedFuncs, changed);
  }

  std::string SubstituteLocalMacroFuncs(const std::string& line,
                                        bool& changed) {
    return SubStituteMacroFuncs(line, localDefinedFuncs, changed);
  }

  bool isDirective(const std::string& line, const std::string& directive,
                   bool blank = true) {
    auto d = directive;
    if (blank) d += " ";
    return line == directive || line.rfind(d, 0) == 0;
  }

private:
  std::string
  preprocessBooleanExpression(const std::string& expr, const DefineMap& defines,
                              std::map<std::string, bool>& macroMap) {
    std::regex defRegex("defined\\s*(?:\\((\\w+)\\)|(\\w+))");
    std::smatch match;
    std::string result = expr;
    std::map<size_t, std::string> replacements;
    auto begin = std::sregex_iterator(expr.begin(), expr.end(), defRegex);
    auto end = std::sregex_iterator();
    std::vector<std::pair<size_t, std::string>> replaceList;
    for (auto i = begin; i != end; ++i) {
      auto m = *i;
      auto varName = m[1].str().empty() ? m[2].str() : m[1].str();
      auto fullMatch = m[0].str();

      auto replacement = "def_" + varName;
      size_t pos = m.position();

      replaceList.emplace_back(pos, replacement);
      macroMap[replacement] = defines.find(varName) != defines.end();
    }
    for (auto it = replaceList.rbegin(); it != replaceList.rend(); ++it) {
      size_t pos = it->first;
      const std::string& replacement = it->second;
      size_t len = expr.substr(pos).find_first_of(")");
      std::smatch subMatch;
      if (regex_search(expr.begin() + pos, expr.end(), subMatch, defRegex)) {
        auto matched_str = subMatch[0];
        len = matched_str.length();
      }
      result.replace(pos, len, replacement);
    }

    return result;
  }

  bool EvaluateBooleanExpression(const std::string& condition_expr,
                                 const DefineMap& defines) {
    std::map<std::string, bool> macroMap;
    auto expr = preprocessBooleanExpression(condition_expr, defines, macroMap);
    auto precedence = [](const std::string& op) {
      if (op == "!") return 3;
      if (op == "&&") return 2;
      if (op == "||") return 1;
      return 0;
    };

    auto applyOp = [](const std::string& op, bool b, bool a) {
      if (op == "&&") return a && b;
      if (op == "||") return a || b;
      return false;
    };

    auto nextToken = [](const std::string& expr,
                        size_t& pos) -> const std::string {
      while (pos < expr.size() && isspace(expr[pos])) ++pos;
      if (pos >= expr.size()) return "";

      char c = expr[pos];

      if (c == '(' || c == ')' || c == '!') {
        ++pos;
        return std::string(1, c);
      }

      if (c == '&' || c == '|') {
        if (pos + 1 < expr.size() && expr[pos + 1] == c) {
          pos += 2;
          return std::string(2, c);
        }
      }

      std::string token;
      while ((pos < expr.size() && isalnum(expr[pos])) || expr[pos] == '_') {
        token += expr[pos++];
      }
      return token;
    };

    std::stack<bool> values;
    std::stack<std::string> ops;

    size_t pos = 0;
    std::string token;
    while (!(token = nextToken(expr, pos)).empty()) {
      if (token == "(") {
        ops.push(token);
      } else if (token == ")") {
        while (!ops.empty() && ops.top() != "(") {
          auto op = ops.top();
          ops.pop();
          if (op == "!") {
            if (values.empty()) errs() << "Invalid expression\n";
            bool a = values.top();
            values.pop();
            values.push(!a);
          } else {
            if (values.size() < 2) errs() << "Invalid expression\n";
            bool b = values.top();
            values.pop();
            bool a = values.top();
            values.pop();
            values.push(applyOp(op, b, a));
          }
        }
        if (!ops.empty()) ops.pop();
      } else if (token[0] == '&' || token[0] == '|' || token[0] == '!') {
        while (!ops.empty() && precedence(ops.top()) >= precedence(token)) {
          std::string op = ops.top();
          ops.pop();
          if (op == "!") {
            if (values.empty()) errs() << "Invalid expression\n";
            bool a = values.top();
            values.pop();
            values.push(!a);
          } else {
            if (values.size() < 2) errs() << "Invalid expression\n";
            bool b = values.top();
            values.pop();
            bool a = values.top();
            values.pop();
            values.push(applyOp(op, b, a));
          }
        }
        ops.push(token);
      } else {
        auto it = macroMap.find(token);
        if (it == macroMap.end()) errs() << "Invalid expression\n";
        values.push(it->second);
      }
    }

    while (!ops.empty()) {
      std::string op = ops.top();
      ops.pop();
      if (op == "!") {
        if (values.empty()) errs() << "Invalid expression\n";
        bool a = values.top();
        values.pop();
        values.push(!a);
      } else {
        if (values.size() < 2) errs() << "Invalid expression\n";
        bool b = values.top();
        values.pop();
        bool a = values.top();
        values.pop();
        values.push(applyOp(op, b, a));
      }
    }

    if (values.size() != 1) errs() << "Invalid expression\n";

    return values.top();
  }

  const std::string HandleCComments(const std::string& line) {
    std::string work_string = line;
    std::string result_line;
    while (!work_string.empty()) {
      if (c_skip) {
        auto pos = std::string::npos;
        if (pos = work_string.find("*/"); pos != std::string::npos) {
          // strip out the comment part
          work_string = work_string.substr(pos + 2);
        } else
          work_string.clear();
        if (pos != std::string::npos) c_skip = false;
      } else {
        auto pos = std::string::npos;
        if (pos = work_string.find("/*"); pos != std::string::npos) {
          result_line += work_string.substr(0, pos);
          work_string = work_string.substr(pos + 2);
        } else {
          result_line += work_string;
          work_string.clear();
        }
        if (pos != std::string::npos) c_skip = true;
      }
    }
    return result_line;
  }

  void HandleOneUserLine(const std::string& line) {
    if (debug) dbgs() << "[U] " << line << " [U]\n";

    assert(kernel_brace_count == 0 && "expect no kernel brace in host code.");
    assert(choreo_brace_count == 0 && "expect no kernel brace.");

    bool skip_line = c_skip;
    // Strip any comments to avoid incorrect analysis of braces
    auto rline = std::regex_replace(line, std::regex("//.*"), "");
    auto aline = HandleCComments(rline);
    if (skip_line && aline.empty()) {
      if (debug) dbgs() << " - skipped\n";
      output << line << '\n';
      return;
    }

    auto bline = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
    bool cur_cond = uc_condition_stack.empty() || uc_condition_stack.top();
    bool cur_skip = !uc_skip_stack.empty() && uc_skip_stack.top();
    if (isDirective(bline, "#ifdef")) {
      std::regex ifdefRegex("#ifdef\\s+(\\w+)");
      std::smatch match;
      uc_if_count++;
      if (std::regex_match(bline, match, ifdefRegex)) {
        bool condition = globalDefines.find(match[1]) != globalDefines.end();
        uc_condition_stack.push(condition);
        uc_skip_stack.push(uc_skip_line);
        uc_skip_line = uc_skip_line || !condition;
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#ifndef")) {
      std::regex ifndefRegex("#ifndef\\s+(\\w+)");
      std::smatch match;
      uc_if_count++;
      if (std::regex_match(bline, match, ifndefRegex)) {
        bool condition = globalDefines.find(match[1]) == globalDefines.end();
        uc_condition_stack.push(condition);
        uc_skip_stack.push(uc_skip_line);
        uc_skip_line = uc_skip_line || !condition;
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#define")) {
      if (cur_cond && !cur_skip) {
        std::regex defineRegex(
            R"(^\s*#define\s+(\w+)(?:\s+([^/]*?))?\s*(?://.*|/\*.*\*/)?\s*$)");
        std::smatch match;
        if (std::regex_match(bline, match, defineRegex))
          globalDefines[match[1]] = match[2].matched ? match[2].str() : "1";
        else {
          defineRegex = std::regex(R"(#define\s+(\w+)\(([^)]*)\)\s+(.*))");
          if (std::regex_match(bline, match, defineRegex)) {
            assert(match[2].matched && match[3].matched &&
                   "Expecting a function-like macro definition.");
            auto macro_func_name = match[1].str();
            auto macro_func_params = match[2].str();
            auto macro_func_body = match[3].str();
            if (globalDefines.count(macro_func_name)) {
              errs() << "copp: in line " << line_num
                     << ": error: redefinition of macro function '"
                     << macro_func_name << "'\n";
              abort();
            } else {
              globalDefines[macro_func_name] = macro_func_body;
              globalDefinedFuncs[macro_func_name] =
                  std::make_tuple(macro_func_params, macro_func_body);
            }
          }
        }
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#undef")) {
      if (cur_cond && !cur_skip) {
        std::regex undefRegex("#undef\\s+(\\w+)");
        std::smatch match;
        if (std::regex_match(bline, match, undefRegex)) {
          globalDefines.erase(match[1]);
          globalDefinedFuncs.erase(match[1]);
        }
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#if")) {
      std::regex ifRegex("#if\\s+(.*)");
      std::smatch match;
      uc_if_count++;
      if (std::regex_match(bline, match, ifRegex)) {
        auto uc_code = match[1].str();
        std::smatch uc_code_match;

        std::regex uc_code_regex(R"(!?defined\s*(?:\(\w+\)|\w+)\s*.*)");
        if (std::regex_match(uc_code, uc_code_match, uc_code_regex)) {
          bool condition = EvaluateBooleanExpression(uc_code, globalDefines);
          uc_condition_stack.push(condition);
          uc_skip_stack.push(uc_skip_line);
          uc_skip_line = uc_skip_line || !condition;
        } else {
          try {
            bool condition = std::stoi(match[1].str()) != 0;
            uc_condition_stack.push(condition);
            uc_skip_stack.push(uc_skip_line);
            uc_skip_line = uc_skip_line || !condition;
          } catch (...) {
            uc_condition_stack.push(false);
            uc_skip_stack.push(uc_skip_line);
            uc_skip_line = true;
          }
        }
      }

      output << line << '\n';
      return;
    } else if (isDirective(bline, "#elif")) {
      if (!uc_condition_stack.empty())
        uc_skip_line = uc_skip_stack.top() || uc_condition_stack.top();

      std::regex ifRegex("#elif\\s+(.*)");
      std::smatch match;
      if (std::regex_match(bline, match, ifRegex)) {
        auto uc_code = match[1].str();
        std::smatch uc_code_match;

        std::regex uc_code_regex(R"(!?defined\s*(?:\(\w+\)|\w+)\s*.*)");
        if (std::regex_match(uc_code, uc_code_match, uc_code_regex)) {
          bool condition = EvaluateBooleanExpression(uc_code, globalDefines);
          uc_condition_stack.push(condition);
          uc_skip_stack.push(uc_skip_line);
          uc_skip_line = uc_skip_line || !condition;
        } else {
          try {
            bool condition = std::stoi(match[1].str()) != 0;
            uc_condition_stack.push(condition);
            uc_skip_stack.push(uc_skip_line);
            uc_skip_line = uc_skip_line || !condition;
          } catch (...) {
            uc_condition_stack.push(false);
            uc_skip_stack.push(uc_skip_line);
            uc_skip_line = true;
          }
        }
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#else")) {
      if (!uc_condition_stack.empty()) {
        bool currentCondition = uc_condition_stack.top();
        uc_condition_stack.top() = !currentCondition;
        uc_skip_line = uc_skip_stack.top() || !uc_condition_stack.top();
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#endif")) {
      uc_if_count--;
      if (uc_if_count < 0) {
        errs() << "copp: in line " << line_num
               << ": error: redundant '#endif'\n";
        abort();
      }
      if (!uc_condition_stack.empty()) {
        uc_condition_stack.pop();
        uc_skip_line = uc_skip_stack.top();
        uc_skip_stack.pop();
      }
      output << line << '\n';
      return;
    }

    // Make substitution for the further work
    auto sline = SubstituteGlobalDefines(aline);
    bool changed = true;
    while (changed) { sline = SubstituteGlobalMacroFuncs(sline, changed); }

    // Check if entering a __co__ function
    if (sline.find("__co__ ") != std::string::npos) {
      if (kernel_brace_count) {
        errs() << "copp: in line " << line_num
               << ": error: '__co__' function inside '__cok__' is illegal.\n";
        abort();
      }
      code_partition = CP_CHOREO;
      localDefines = globalDefines;
      localDefinedFuncs = globalDefinedFuncs;

      auto c_pos = sline.find_first_of('(');
      if (c_pos != std::string::npos) {
        auto b_pos = sline.find("__co__ ");
        auto co_decl = sline.substr(b_pos, c_pos - b_pos);
        auto co_code = sline.substr(c_pos);

        // Output the function declaration. Append any leading C comment. C
        // comments in the middle of decl are ignored
        if (!uc_skip_line) {
          if (auto pos = line.find("__co__ "))
            output << line.substr(0, pos) << co_decl;
          else
            output << co_decl;
        }

        HandleOneChoreoLine(co_code, false);
      } else if (!uc_skip_line)
        output << line << '\n'; // output the original line

      return;
    }

    // Check if entering a '__cok__' partition
    if (sline.find("__cok__ ") != std::string::npos) {
      if (choreo_brace_count) {
        errs() << "copp: in line " << line_num
               << ": error: '__cok__' code inside '__co__' function is "
                  "illegal.\n";
        abort();
      }
      code_partition = CP_KERNEL;
      has_cok = true;
      auto k_pos = sline.find_first_of('{');
      if (k_pos != std::string::npos) {
        auto kernel_decl = sline.substr(0, k_pos);
        // put the best effort to get the original line
        auto kline = line.substr(line.find("__cok__ "));
        auto kernel_code = kline.substr(kline.find("{"));

        // Output the __cok__. Append any leading C comment.
        if (auto pos = line.find("__cok__ "))
          output << line.substr(0, pos) << kernel_decl;
        else
          output << kernel_decl;

        HandleOneKernelLine(kernel_code, false);
      } else
        output << line << '\n'; // output the original line

      return;
    }

    output << line << '\n';
  }

  void HandleOneKernelLine(const std::string& line,
                           bool handle_comment = true) {
    if (debug) dbgs() << "[K] " << line << " [K]\n";

    assert(code_partition == CP_KERNEL);

    auto aline = line;
    if (handle_comment) {
      bool skip_line = c_skip;
      // Strip any comments to avoid incorrect analysis of braces
      auto rline = std::regex_replace(line, std::regex("//.*$"), "");
      aline = HandleCComments(rline);
      if (skip_line && aline.empty()) {
        if (debug) dbgs() << " - comment\n";
        output << line << '\n';
        return;
      }
    }

    size_t end_pos = aline.size();
    for (size_t i = 0; i < aline.size(); ++i) {
      char c = aline[i];
      if (c == '{') {
        kernel_brace_count++;
      } else if (c == '}') {
        kernel_brace_count--;
        if (kernel_brace_count == 0) {
          end_pos = i;
          code_partition = CP_USER;
          break;
        }
      }
    }

    if (end_pos == aline.size()) {
      output << line << '\n'; // output the original line
      return;
    }

    // handle the dangling user code
    assert(kernel_brace_count == 0);

    output << aline.substr(0, end_pos) << '}';

    HandleOneUserLine(aline.substr(end_pos + 1));
  }

  void HandleOneChoreoLine(const std::string& line,
                           bool handle_comment = true) {
    if (debug) dbgs() << "[C] " << line << " [C]\n";

    assert(code_partition == CP_CHOREO);

    auto aline = line;
    if (handle_comment) {
      auto skip_line = c_skip;
      // Strip white spaces and "//" leading comments
      auto rline = std::regex_replace(line, std::regex("//.*"), "");
      aline = HandleCComments(rline);
      if (skip_line && aline.empty()) {
        if (debug) dbgs() << " - skipped\n";
        return;
      }
    }

    auto bline = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
    bool cur_cond = co_condition_stack.empty() || co_condition_stack.top();
    bool cur_skip = !co_skip_stack.empty() && co_skip_stack.top();
    if (isDirective(bline, "#ifdef")) {
      std::regex ifdefRegex("#ifdef\\s+(\\w+)");
      std::smatch match;
      if (std::regex_match(bline, match, ifdefRegex)) {
        bool condition = localDefines.find(match[1]) != localDefines.end();
        co_condition_stack.push(condition);
        co_skip_stack.push(co_skip_line);
        co_skip_line = co_skip_line || !condition;
        co_if_count++;
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#ifndef")) {
      std::regex ifndefRegex("#ifndef\\s+(\\w+)");
      std::smatch match;
      if (std::regex_match(bline, match, ifndefRegex)) {
        bool condition = localDefines.find(match[1]) == localDefines.end();
        co_condition_stack.push(condition);
        co_skip_stack.push(co_skip_line);
        co_skip_line = co_skip_line || !condition;
        co_if_count++;
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#define", false)) {
      if (cur_cond && !cur_skip) {
        std::regex defineRegex(
            R"(^\s*#define\s+(\w+)(?:\s+([^/]*?))?\s*(?://.*|/\*.*\*/)?\s*$)");
        std::smatch match;
        if (std::regex_match(bline, match, defineRegex)) {
          localDefines[match[1]] = match[2].matched ? match[2].str() : "1";
        } else {
          defineRegex = std::regex(R"(#define\s+(\w+)\(([^)]*)\)\s+(.*))");
          if (std::regex_match(bline, match, defineRegex)) {
            assert(match[2].matched && match[3].matched &&
                   "Expecting a function-like macro definition.");
            auto macro_func_name = match[1].str();
            auto macro_func_params = match[2].str();
            auto macro_func_body = match[3].str();

            if (localDefines.count(macro_func_name)) {
              errs() << "copp: in line " << line_num
                     << ": error: redefinition of macro function '"
                     << macro_func_name << "'\n";
              abort();
            } else {
              localDefines[macro_func_name] = macro_func_body;
              localDefinedFuncs[macro_func_name] =
                  std::make_tuple(macro_func_params, macro_func_body);
            }
          }
        }
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#undef")) {
      if (cur_cond && !cur_skip) {
        std::regex undefRegex("#undef\\s+(\\w+)");
        std::smatch match;
        if (std::regex_match(bline, match, undefRegex)) {
          localDefines.erase(match[1]);
          localDefinedFuncs.erase(match[1]);
        }
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#if")) {
      std::regex ifRegex("#if\\s+(.*)");
      std::smatch match;
      if (std::regex_match(bline, match, ifRegex)) {
        auto co_code = match[1].str();
        std::smatch co_code_match;
        std::regex co_code_regex(R"(!?defined\s*(?:\(\w+\)|\w+)\s*.*)");
        if (std::regex_match(co_code, co_code_match, co_code_regex)) {
          bool condition = EvaluateBooleanExpression(co_code, localDefines);
          co_condition_stack.push(condition);
          co_skip_stack.push(co_skip_line);
          co_skip_line = co_skip_line || !condition;
        } else {
          try {
            bool condition = std::stoi(match[1].str()) != 0;
            co_condition_stack.push(condition);
            co_skip_stack.push(co_skip_line);
            co_skip_line = co_skip_line || !condition;
          } catch (...) {
            co_condition_stack.push(false);
            co_skip_stack.push(co_skip_line);
            co_skip_line = true;
          }
        }
        co_if_count++;
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#elif")) {
      if (!co_condition_stack.empty())
        co_skip_line = co_skip_stack.top() || co_condition_stack.top();

      std::regex elifRegex("#elif\\s+(.*)");
      std::smatch match;
      if (std::regex_match(bline, match, elifRegex) && !co_skip_line) {
        auto co_code = match[1].str();
        std::regex co_code_regex(R"(!?defined\s*(?:\(\w+\)|\w+)\s*.*)");
        std::smatch co_code_match;
        if (std::regex_match(co_code, co_code_match, co_code_regex)) {
          bool condition = EvaluateBooleanExpression(co_code, localDefines);
          co_condition_stack.push(condition);
          co_skip_stack.push(co_skip_line);
          co_skip_line = co_skip_line || !condition;
        } else {
          try {
            bool condition = std::stoi(match[1].str()) != 0;
            co_condition_stack.push(condition);
            co_skip_stack.push(co_skip_line);
            co_skip_line = co_skip_line || !condition;
          } catch (...) {
            co_condition_stack.push(false);
            co_skip_stack.push(co_skip_line);
            co_skip_line = true;
          }
        }
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#else")) {
      if (!co_condition_stack.empty()) {
        bool currentCondition = co_condition_stack.top();
        co_condition_stack.top() = !currentCondition;
        co_skip_line = co_skip_stack.top() || !co_condition_stack.top();
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#endif")) {
      co_if_count--;
      if (co_if_count < 0) {
        errs() << "copp: in line " << line_num
               << ": error: redundant '#endif'\n";
        abort();
      }
      if (!co_condition_stack.empty()) {
        co_condition_stack.pop();
        co_skip_line = co_skip_stack.top();
        co_skip_stack.pop();
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (!co_skip_line) {
      size_t co_end = aline.size();
      for (size_t i = 0; i < aline.size(); ++i) {
        char c = aline[i];
        if (c == '{') {
          choreo_brace_count++;
        } else if (c == '}') {
          choreo_brace_count--;
          if (choreo_brace_count == 0) {
            co_end = i;
            break; // the followed code are not choreo code
          }
        }
      }

      if ((co_end == aline.size()) && (choreo_brace_count == 0)) {
        // has not entered the choreo code region
        auto sline = SubstituteLocalDefines(aline);
        if (!uc_skip_line) output << sline << '\n';
        return;
      }

      auto co_code = aline.substr(0, co_end);
      auto sline = SubstituteLocalDefines(co_code);
      bool changed = true;
      while (changed) { sline = SubstituteLocalMacroFuncs(sline, changed); }

      // output the choreo code
      if (!uc_skip_line) output << sline;

      if (choreo_brace_count == 0) {
        code_partition = CP_USER;
        localDefines.clear(); // Clear local defines

        if (!uc_skip_line) {
          output << "}"; // there could be host code followed
          // note user's C comments in this line is also stripped
          HandleOneUserLine(aline.substr(co_end + 1));
        } else
          output << "#line " << line_num + 1 << "\n";
      } else if (!uc_skip_line)
        output << '\n';
    } else {
      // skip the line
    }
  }

public:
  bool ExtractDeviceKernel(std::stringstream& cok_ss) {
    char temp_script_file_name[] = "/tmp/choreo_topscc_script_XXXXXX";

    int script_fd = mkstemp(temp_script_file_name);
    if (script_fd == -1) {
      errs() << "Cannot create temporary file.\n";
      return false;
    }
    close(script_fd);

    std::ofstream temp_script_file(temp_script_file_name);
    if (!temp_script_file.is_open()) {
      errs() << "Cannot open temporary file.\n";
      return false;
    }

    auto filename = RemoveDirectoryPrefix(
        RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
    build_path = CreateUniquePath();

    cc_file = build_path + "/__choreo_topscc_" + filename + ".cpp";
    pp_file = build_path + "/__choreo_topscc_" + filename + ".i";

    std::string line;
    bool has_include = false;
    for (auto code_line : include_lines) {
      if (isDirective(code_line, "#include", false)) {
        has_include = true;
        break;
      }
    }

    if (!has_include && cok_codes.empty()) {
      if (debug) dbgs() << "No #include found and no __cok__ region found\n";
      return true;
    }

    EmitScript(temp_script_file);
    temp_script_file.close();

    std::string cmd =
        "bash " + std::string(temp_script_file_name) + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret != 0) {
      if (debug) dbgs() << "Command failed: " << cmd << "\n";
      return true;
    }

    if (remove(temp_script_file_name) != 0) {
      errs() << "Cannot remove temporary file.\n";
      return false;
    }

    // scan the pp_file to extract code inside __cok__ { ... }
    std::ifstream pp_ifs(pp_file);
    if (!pp_ifs.is_open()) {
      errs() << "Cannot open preprocessed file: " << pp_file << "\n";
      return false;
    }
    std::string pp_code((std::istreambuf_iterator<char>(pp_ifs)),
                        std::istreambuf_iterator<char>());

    // find bundles between "__CLANG_OFFLOAD_BUNDLE____START__
    // tops-dtu-enflame-topsXXX" and
    // "__CLANG_OFFLOAD_BUNDLE____END__ tops-dtu-enflame-tops"
    auto bundles = find_bundles(pp_code);
    if (bundles.empty()) return true;
    std::vector<std::string> device_codes;
    for (auto& b : bundles) {
      std::string sub = pp_code.substr(b.start, b.end - b.start);
      auto ranges = extract_cok_sections(sub);
      for (auto& r : ranges)
        device_codes.push_back(sub.substr(r.start, r.end - r.start + 1));
    }

    cok_ss << "__real_cok__ {\n";
    for (auto& c : device_codes) { cok_ss << c << "\n"; }
    cok_ss << "}\n\n";
    return true;
  }

  bool Process(std::istream& input) {
    code_partition = CP_USER;
    std::string cur_line;
    std::string line_to_handle;
    while (std::getline(input, cur_line)) {
      // we skip the include of choreo.h, which is not needed for extracting
      // device kernels from preprocessed file
      if (isDirective(cur_line, "#include", false)) {
        std::regex ifdefRegex("#include\\s+\"(.*)\"");
        std::smatch match;
        if (std::regex_match(cur_line, match, ifdefRegex)) {
          std::string include_file = match[1];
          if (include_file != "choreo.h") include_lines.push_back(cur_line);
        }
      }
      if (code_partition == CP_KERNEL) cok_codes.push_back(cur_line);

      if (line_to_handle.back() == '\\') {
        line_to_handle.pop_back();
        line_to_handle += " " + trim(cur_line);
      } else
        line_to_handle = cur_line;

      size_t bs_pos = cur_line.find_last_of('\\');
      if (bs_pos != std::string::npos &&
          bs_pos == cur_line.find_last_not_of(' ')) {
        if (bs_pos != cur_line.size() - 1)
          errs() << ("copp: in line " + std::to_string(line_num) +
                     ": warning: backslash and newline separated by space\n");
        line_num++;
        line_to_handle =
            line_to_handle.substr(0, line_to_handle.find_last_not_of(' ') + 1);
        continue;
      }

      if (code_partition == CP_USER) {
        HandleOneUserLine(line_to_handle);
        line_num++;
      } else if (code_partition == CP_CHOREO) {
        HandleOneChoreoLine(line_to_handle);
        line_num++;
      } else if (code_partition == CP_KERNEL) {
        HandleOneKernelLine(line_to_handle);
        line_num++;
      } else
        choreo_unreachable("code partition is not known.");
    }

    if (co_if_count || uc_if_count) {
      errs() << "copp: in line " << line_num << ": error: missing '#endif'.\n";
      return false;
    }

    if (kernel_brace_count) {
      errs() << "copp: in line " << line_num
             << ": error: un-terminated '__cok__' code is detected.\n";
      return false;
    }

    if (choreo_brace_count) {
      errs() << "copp: in line " << line_num
             << ": error: un-terminated '__co__' function is detected.\n";
      return false;
    }

    return true;
  }
};

} // end namespace Choreo

#endif //__CHOREO_PRE_PROCESS_HPP__
