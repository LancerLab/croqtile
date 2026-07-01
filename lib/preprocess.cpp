#include "preprocess.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "fs_utils.hpp"
#include "io.hpp"
#include "options.hpp"
#include <filesystem>
#include <fstream>
#include <istream>
#include <regex>
#include <sstream>
#include <unordered_set>

using namespace Choreo;

Option<bool>
    applyDefinesInCO(OptionKind::Internal, "--apply-defines-in-co", "", true,
                     "Apply #define macros inside of Choreo Function.");
Option<bool>
    applyDefinesInCOK(OptionKind::Internal, "--apply-defines-in-cok", "", true,
                      "Apply #define macros inside of Choreo Function.");

Option<bool> debugPP(OptionKind::Internal, "--debug-pp", "", false,
                     "Debug the Preprocessor.");

static const std::string trim(const std::string& str) {
  size_t first = str.find_first_not_of(' ');
  if (first == std::string::npos) return "";

  size_t last = str.find_last_not_of(' ');
  return str.substr(first, last - first + 1);
}

Preprocess::Preprocess(std::ostream& o) : output(o), debug(debugPP) {
  // Replicate the target-specific macros
  for (auto& macro : CCtx().GetTarget().ChoreoMacros(CCtx().GetArch()))
    globalDefines.emplace(macro.first, macro.second);

  // command-line macros override
  for (auto& item : CCtx().GetCLMacros())
    globalDefines[item.first] = item.second;
}

const std::string Preprocess::SubStituteMacroFuncs(const std::string& line,
                                                   const FuncMap& funcs,
                                                   bool& changed) {
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

const std::string Preprocess::SubStituteDefines(const std::string& line,
                                                const DefineMap& defines,
                                                const FuncMap& funcs,
                                                std::vector<MacroSub>* subs) {
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
        if (it != defines.end() && func_it == funcs.end()) {
          if (subs && it->second.size() != current_token.size()) {
            int col = (int)(i - current_token.size()) + 1;
            subs->push_back(
                {col, (int)current_token.size(), (int)it->second.size()});
          }
          result += it->second;
        } else
          result += current_token;
        current_token.clear();
      }
      result += c;
    }
  }

  // last token
  if (!current_token.empty()) {
    auto it = defines.find(current_token);
    if (it != defines.end()) {
      if (subs && it->second.size() != current_token.size()) {
        int col = (int)(line.length() - current_token.size()) + 1;
        subs->push_back(
            {col, (int)current_token.size(), (int)it->second.size()});
      }
      result += it->second;
    } else
      result += current_token;
  }

  return result;
}

const std::string Preprocess::SubstituteGlobalDefines(const std::string& line) {
  return SubStituteDefines(line, globalDefines, globalDefinedFuncs);
}

const std::string
Preprocess::SubstituteLocalDefines(const std::string& line,
                                   std::vector<MacroSub>* subs) {
  return SubStituteDefines(line, localDefines, localDefinedFuncs, subs);
}

const std::string
Preprocess::SubstituteGlobalMacroFuncs(const std::string& line, bool& changed) {
  return SubStituteMacroFuncs(line, globalDefinedFuncs, changed);
}

const std::string Preprocess::SubstituteLocalMacroFuncs(const std::string& line,
                                                        bool& changed) {
  return SubStituteMacroFuncs(line, localDefinedFuncs, changed);
}

bool Preprocess::isDirective(const std::string& line,
                             const std::string& directive, bool blank) {
  auto d = directive;
  if (blank) d += " ";
  return line == directive || line.rfind(d, 0) == 0;
}

const std::string Preprocess::preprocessBooleanExpression(
    const std::string& expr, const DefineMap& defines,
    std::unordered_map<std::string, bool>& macroMap) {
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

bool Preprocess::EvaluateBooleanExpression(const std::string& condition_expr,
                                           const DefineMap& defines) {
  std::unordered_map<std::string, bool> macroMap;
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

namespace {

void skipWhitespace(const std::string& s, size_t& pos) {
  while (pos < s.size() && isspace(s[pos])) ++pos;
}

int64_t parseExpr(const std::string& s, size_t& pos, int minPrec);
int64_t parseUnary(const std::string& s, size_t& pos);
int64_t parsePrimary(const std::string& s, size_t& pos);

struct OpInfo {
  int prec;
  bool leftAssoc;
};

OpInfo getBinOpInfo(const std::string& s, size_t pos, std::string& op) {
  skipWhitespace(s, pos);
  if (pos >= s.size()) return {-1, true};

  char c = s[pos];
  char c2 = (pos + 1 < s.size()) ? s[pos + 1] : '\0';

  if (c == '|' && c2 == '|') {
    op = "||";
    return {1, true};
  }
  if (c == '&' && c2 == '&') {
    op = "&&";
    return {2, true};
  }
  if (c == '|' && c2 != '|') {
    op = "|";
    return {3, true};
  }
  if (c == '^') {
    op = "^";
    return {4, true};
  }
  if (c == '&' && c2 != '&') {
    op = "&";
    return {5, true};
  }
  if (c == '=' && c2 == '=') {
    op = "==";
    return {6, true};
  }
  if (c == '!' && c2 == '=') {
    op = "!=";
    return {6, true};
  }
  if (c == '<' && c2 == '=') {
    op = "<=";
    return {7, true};
  }
  if (c == '>' && c2 == '=') {
    op = ">=";
    return {7, true};
  }
  if (c == '<' && c2 != '<') {
    op = "<";
    return {7, true};
  }
  if (c == '>' && c2 != '>') {
    op = ">";
    return {7, true};
  }
  if (c == '<' && c2 == '<') {
    op = "<<";
    return {8, true};
  }
  if (c == '>' && c2 == '>') {
    op = ">>";
    return {8, true};
  }
  if (c == '+') {
    op = "+";
    return {9, true};
  }
  if (c == '-') {
    op = "-";
    return {9, true};
  }
  if (c == '*') {
    op = "*";
    return {10, true};
  }
  if (c == '/') {
    op = "/";
    return {10, true};
  }
  if (c == '%') {
    op = "%";
    return {10, true};
  }

  return {-1, true};
}

int64_t applyBinOp(const std::string& op, int64_t lhs, int64_t rhs) {
  if (op == "||") return (lhs || rhs) ? 1 : 0;
  if (op == "&&") return (lhs && rhs) ? 1 : 0;
  if (op == "|") return lhs | rhs;
  if (op == "^") return lhs ^ rhs;
  if (op == "&") return lhs & rhs;
  if (op == "==") return (lhs == rhs) ? 1 : 0;
  if (op == "!=") return (lhs != rhs) ? 1 : 0;
  if (op == "<") return (lhs < rhs) ? 1 : 0;
  if (op == ">") return (lhs > rhs) ? 1 : 0;
  if (op == "<=") return (lhs <= rhs) ? 1 : 0;
  if (op == ">=") return (lhs >= rhs) ? 1 : 0;
  if (op == "<<") return lhs << rhs;
  if (op == ">>") return lhs >> rhs;
  if (op == "+") return lhs + rhs;
  if (op == "-") return lhs - rhs;
  if (op == "*") return lhs * rhs;
  if (op == "/") return rhs != 0 ? lhs / rhs : 0;
  if (op == "%") return rhs != 0 ? lhs % rhs : 0;
  return 0;
}

int64_t parsePrimary(const std::string& s, size_t& pos) {
  skipWhitespace(s, pos);
  if (pos >= s.size()) return 0;

  if (s[pos] == '(') {
    ++pos;
    int64_t val = parseExpr(s, pos, 0);
    skipWhitespace(s, pos);
    if (pos < s.size() && s[pos] == ')') ++pos;
    return val;
  }

  if (isdigit(s[pos])) {
    int64_t val = 0;
    int base = 10;
    if (s[pos] == '0' && pos + 1 < s.size()) {
      char nc = s[pos + 1];
      if (nc == 'x' || nc == 'X') {
        base = 16;
        pos += 2;
      } else if (nc == 'b' || nc == 'B') {
        base = 2;
        pos += 2;
      } else if (isdigit(nc)) {
        base = 8;
        ++pos;
      }
    }
    while (pos < s.size()) {
      char c = s[pos];
      int digit = -1;
      if (c >= '0' && c <= '9')
        digit = c - '0';
      else if (c >= 'a' && c <= 'f')
        digit = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        digit = c - 'A' + 10;
      if (digit < 0 || digit >= base) break;
      val = val * base + digit;
      ++pos;
    }
    while (pos < s.size() &&
           (s[pos] == 'u' || s[pos] == 'U' || s[pos] == 'l' || s[pos] == 'L'))
      ++pos;
    return val;
  }

  if (isalpha(s[pos]) || s[pos] == '_') {
    while (pos < s.size() && (isalnum(s[pos]) || s[pos] == '_')) ++pos;
    return 0;
  }

  return 0;
}

int64_t parseUnary(const std::string& s, size_t& pos) {
  skipWhitespace(s, pos);
  if (pos >= s.size()) return 0;

  if (s[pos] == '!') {
    ++pos;
    return parseUnary(s, pos) ? 0 : 1;
  }
  if (s[pos] == '~') {
    ++pos;
    return ~parseUnary(s, pos);
  }
  if (s[pos] == '-') {
    ++pos;
    return -parseUnary(s, pos);
  }
  if (s[pos] == '+') {
    ++pos;
    return parseUnary(s, pos);
  }

  return parsePrimary(s, pos);
}

int64_t parseExpr(const std::string& s, size_t& pos, int minPrec) {
  int64_t lhs = parseUnary(s, pos);

  while (true) {
    skipWhitespace(s, pos);
    if (pos >= s.size()) break;

    std::string op;
    size_t savedPos = pos;
    OpInfo info = getBinOpInfo(s, pos, op);
    if (info.prec < minPrec) break;

    pos = savedPos + op.size();
    int nextMinPrec = info.leftAssoc ? info.prec + 1 : info.prec;
    int64_t rhs = parseExpr(s, pos, nextMinPrec);
    lhs = applyBinOp(op, lhs, rhs);
  }

  skipWhitespace(s, pos);
  if (pos < s.size() && s[pos] == '?') {
    ++pos;
    int64_t trueVal = parseExpr(s, pos, 0);
    skipWhitespace(s, pos);
    if (pos < s.size() && s[pos] == ':') ++pos;
    int64_t falseVal = parseExpr(s, pos, 0);
    lhs = lhs ? trueVal : falseVal;
  }

  return lhs;
}

std::string replaceDefinedOperator(const std::string& expr,
                                   const Preprocess::DefineMap& defines) {
  std::string result;
  size_t i = 0;
  while (i < expr.size()) {
    if (i + 7 <= expr.size() && expr.substr(i, 7) == "defined") {
      size_t j = i + 7;
      while (j < expr.size() && isspace(expr[j])) ++j;
      std::string macroName;
      if (j < expr.size() && expr[j] == '(') {
        ++j;
        while (j < expr.size() && isspace(expr[j])) ++j;
        size_t nameStart = j;
        while (j < expr.size() && (isalnum(expr[j]) || expr[j] == '_')) ++j;
        macroName = expr.substr(nameStart, j - nameStart);
        while (j < expr.size() && isspace(expr[j])) ++j;
        if (j < expr.size() && expr[j] == ')') ++j;
      } else {
        size_t nameStart = j;
        while (j < expr.size() && (isalnum(expr[j]) || expr[j] == '_')) ++j;
        macroName = expr.substr(nameStart, j - nameStart);
      }
      if (!macroName.empty()) {
        bool isDefined = defines.find(macroName) != defines.end();
        result += isDefined ? "1" : "0";
        i = j;
      } else {
        result += expr[i++];
      }
    } else if (isalpha(expr[i]) || expr[i] == '_') {
      size_t start = i;
      while (i < expr.size() && (isalnum(expr[i]) || expr[i] == '_')) ++i;
      std::string word = expr.substr(start, i - start);
      if (word == "defined") {
        result += word;
      } else {
        result += word;
      }
    } else {
      result += expr[i++];
    }
  }
  return result;
}

std::string substituteAndReplace(const std::string& raw_expr,
                                 const Preprocess::DefineMap& defines,
                                 const Preprocess::FuncMap& funcs) {
  std::string expr = replaceDefinedOperator(raw_expr, defines);

  std::string prev;
  int iterations = 0;
  while (expr != prev && iterations < 32) {
    prev = expr;
    std::string result;
    std::string token;
    for (size_t i = 0; i <= expr.size(); ++i) {
      char c = (i < expr.size()) ? expr[i] : '\0';
      if (isalnum(c) || c == '_') {
        token += c;
      } else {
        if (!token.empty()) {
          auto it = defines.find(token);
          auto fit = funcs.find(token);
          if (it != defines.end() && fit == funcs.end())
            result += it->second;
          else
            result += token;
          token.clear();
        }
        if (i < expr.size()) result += c;
      }
    }
    expr = result;
    ++iterations;
  }

  std::string final_result;
  std::string token;
  for (size_t i = 0; i <= expr.size(); ++i) {
    char c = (i < expr.size()) ? expr[i] : '\0';
    if (isalnum(c) || c == '_') {
      token += c;
    } else {
      if (!token.empty()) {
        bool allDigit = true;
        for (char tc : token)
          if (!isdigit(tc) &&
              !((tc >= 'a' && tc <= 'f') || (tc >= 'A' && tc <= 'F') ||
                tc == 'x' || tc == 'X'))
            allDigit = false;
        if (allDigit)
          final_result += token;
        else
          final_result += "0";
        token.clear();
      }
      if (i < expr.size()) final_result += c;
    }
  }

  return final_result;
}

} // anonymous namespace

bool Preprocess::EvaluateIfExpression(const std::string& raw_expr,
                                      const DefineMap& defines,
                                      const FuncMap& funcs) {
  std::string expr = substituteAndReplace(raw_expr, defines, funcs);
  size_t pos = 0;
  int64_t result = parseExpr(expr, pos, 0);
  return result != 0;
}

const std::string Preprocess::HandleCComments(const std::string& line) {
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

void Preprocess::HandleOneUserLine(const std::string& line) {
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
          R"(^\s*#define\s+(\w+)(?:\s+((?:[^/]|/(?![/*]))*))?\s*(?://.*|/\*.*\*/)?\s*$)");
      std::smatch match;
      if (std::regex_match(bline, match, defineRegex)) {
        std::string val = match[2].matched ? match[2].str() : "1";
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t'))
          val.pop_back();
        globalDefines[match[1]] = val.empty() ? "1" : val;
      } else {
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
      bool condition = EvaluateIfExpression(match[1].str(), globalDefines,
                                            globalDefinedFuncs);
      uc_condition_stack.push(condition);
      uc_skip_stack.push(uc_skip_line);
      uc_skip_line = uc_skip_line || !condition;
    }

    output << line << '\n';
    return;
  } else if (isDirective(bline, "#elif")) {
    if (!uc_condition_stack.empty())
      uc_skip_line = uc_skip_stack.top() || uc_condition_stack.top();

    std::regex ifRegex("#elif\\s+(.*)");
    std::smatch match;
    if (std::regex_match(bline, match, ifRegex)) {
      bool condition = EvaluateIfExpression(match[1].str(), globalDefines,
                                            globalDefinedFuncs);
      uc_condition_stack.push(condition);
      uc_skip_stack.push(uc_skip_line);
      uc_skip_line = uc_skip_line || !condition;
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
      errs() << "copp: in line " << line_num << ": error: redundant '#endif'\n";
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

void Preprocess::HandleOneKernelLine(const std::string& line,
                                     bool handle_comment) {
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

void Preprocess::HandleOneChoreoLine(const std::string& line,
                                     bool handle_comment) {
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
          R"(^\s*#define\s+(\w+)(?:\s+((?:[^/]|/(?![/*]))*))?\s*(?://.*|/\*.*\*/)?\s*$)");
      std::smatch match;
      if (std::regex_match(bline, match, defineRegex)) {
        std::string val = match[2].matched ? match[2].str() : "1";
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t'))
          val.pop_back();
        localDefines[match[1]] = val.empty() ? "1" : val;
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
      bool condition =
          EvaluateIfExpression(match[1].str(), localDefines, localDefinedFuncs);
      co_condition_stack.push(condition);
      co_skip_stack.push(co_skip_line);
      co_skip_line = co_skip_line || !condition;
      co_if_count++;
    }
    if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
  } else if (isDirective(bline, "#elif")) {
    if (!co_condition_stack.empty())
      co_skip_line = co_skip_stack.top() || co_condition_stack.top();

    std::regex elifRegex("#elif\\s+(.*)");
    std::smatch match;
    if (std::regex_match(bline, match, elifRegex) && !co_skip_line) {
      bool condition =
          EvaluateIfExpression(match[1].str(), localDefines, localDefinedFuncs);
      co_condition_stack.push(condition);
      co_skip_stack.push(co_skip_line);
      co_skip_line = co_skip_line || !condition;
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
      errs() << "copp: in line " << line_num << ": error: redundant '#endif'\n";
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
      std::vector<MacroSub> subs;
      auto sline = SubstituteLocalDefines(aline, &subs);
      if (!subs.empty()) CCtx().SetLineMacroSubs(line_num, std::move(subs));
      for (int iter = 0; iter < 32; ++iter) {
        auto prev = sline;
        sline = SubstituteLocalDefines(sline);
        if (sline == prev) break;
      }
      if (!uc_skip_line) output << sline << '\n';
      return;
    }

    std::vector<MacroSub> subs;
    auto co_code = aline.substr(0, co_end);
    auto sline = SubstituteLocalDefines(co_code, &subs);
    if (!subs.empty()) CCtx().SetLineMacroSubs(line_num, std::move(subs));
    for (int iter = 0; iter < 32; ++iter) {
      auto prev = sline;
      sline = SubstituteLocalDefines(sline);
      if (sline == prev) break;
    }
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

namespace {

std::string ResolveInclude(const std::string& filename,
                           const std::string& input_dir) {
  namespace fs = std::filesystem;
  auto candidate = fs::path(input_dir) / filename;
  if (fs::exists(candidate)) return candidate.string();
  for (auto& inc_path : CCtx().GetIncPaths()) {
    candidate = fs::path(inc_path) / filename;
    if (fs::exists(candidate)) return candidate.string();
  }
  return "";
}

std::string NormalizePath(const std::string& path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  auto p = fs::weakly_canonical(fs::path(path), ec);
  if (!ec) return p.string();
  p = fs::absolute(fs::path(path), ec);
  if (!ec) return p.string();
  return path;
}

// Line-based extraction of __cok__ block contents from a raw source file.
// This recursively follows quote includes to collect nested __cok__ blocks.
std::vector<std::string>
CollectCokLinesRecursive(const std::string& filepath,
                         std::unordered_set<std::string>& visited) {
  std::vector<std::string> result;
  auto normalized = NormalizePath(filepath);
  if (visited.count(normalized)) return result;
  visited.insert(normalized);

  std::ifstream ifs(normalized);
  if (!ifs.is_open()) return result;

  std::regex inc_re("^\\s*#include\\s+\"(.*)\"");
  auto current_dir =
      std::filesystem::path(normalized).parent_path().string();

  std::string line;
  int depth = 0;
  bool in_cok = false;
  while (std::getline(ifs, line)) {
    if (!in_cok) {
      std::smatch m;
      if (std::regex_match(line, m, inc_re)) {
        auto resolved = ResolveInclude(m[1].str(), current_dir);
        if (!resolved.empty()) {
          auto nested = CollectCokLinesRecursive(resolved, visited);
          result.insert(result.end(), nested.begin(), nested.end());
        }
      }
    }

    auto trimmed = line;
    size_t first = trimmed.find_first_not_of(" \t");
    if (first != std::string::npos) trimmed = trimmed.substr(first);

    if (!in_cok && trimmed.find("__cok__") == 0) {
      auto brace_pos = line.find('{');
      if (brace_pos != std::string::npos) {
        in_cok = true;
        depth = 1;
        auto after = line.substr(brace_pos + 1);
        if (after.find_first_not_of(" \t\r\n") != std::string::npos)
          result.push_back(after);
      }
      continue;
    }
    if (in_cok) {
      for (char c : line) {
        if (c == '{')
          depth++;
        else if (c == '}')
          depth--;
      }
      if (depth <= 0) {
        in_cok = false;
        auto close_pos = line.rfind('}');
        if (close_pos > 0) result.push_back(line.substr(0, close_pos));
      } else {
        result.push_back(line);
      }
    }
  }
  return result;
}

} // anonymous namespace

bool Preprocess::ExtractDeviceKernel(std::ostream& cok_ss) {
  std::vector<std::string> all_cok;

  auto input_file = OptionRegistry::GetInstance().GetInputFileName();
  auto input_dir = GetAbsPath(std::filesystem::current_path(), input_file);
  std::unordered_set<std::string> visited;

  std::regex inc_re("#include\\s+\"(.*)\"");
  for (auto& inc_line : include_lines) {
    std::smatch m;
    if (std::regex_match(inc_line, m, inc_re)) {
      auto resolved = ResolveInclude(m[1].str(), input_dir);
      if (!resolved.empty()) {
        auto inc_cok = CollectCokLinesRecursive(resolved, visited);
        all_cok.insert(all_cok.end(), inc_cok.begin(), inc_cok.end());
      }
    }
  }

  all_cok.insert(all_cok.end(), cok_codes.begin(), cok_codes.end());

  if (all_cok.empty()) return true;
  cok_ss << "__real_cok__ {\n";
  for (auto& c : all_cok) cok_ss << c << "\n";
  cok_ss << "}\n\n";
  return true;
}

bool Preprocess::Process(std::istream& input) {
  code_partition = CP_USER;
  std::string cur_line;
  std::string line_to_handle;
  int extra_line = 0;
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

    auto fisrt_pos = cur_line.find_first_of("//");
    // todo: more robust way to detect if in comment
    bool in_comment = c_skip || (fisrt_pos != std::string::npos &&
                                 fisrt_pos == cur_line.find_first_not_of(' '));

    if (line_to_handle.back() == '\\' && !in_comment) {
      line_to_handle.pop_back();
      line_to_handle += " " + trim(cur_line);
    } else
      line_to_handle = cur_line;

    size_t bs_pos = cur_line.find_last_of('\\');
    if (bs_pos != std::string::npos && !in_comment &&
        bs_pos == cur_line.find_last_not_of(' ')) {
      if (bs_pos != cur_line.size() - 1)
        errs() << ("copp: in line " + std::to_string(line_num) +
                   ": warning: backslash and newline separated by space\n");
      line_num++;
      line_to_handle =
          line_to_handle.substr(0, line_to_handle.find_last_not_of(' ') + 1);
      extra_line++;
      continue;
    }

    for (int i = 0; i < extra_line; ++i)
      output << "// #line " << line_num - extra_line + i << "\n";
    extra_line = 0;

    if (code_partition == CP_USER) {
      HandleOneUserLine(line_to_handle);
      line_num++;
    } else if (code_partition == CP_CHOREO) {
      HandleOneChoreoLine(line_to_handle);
      line_num++;
    } else if (code_partition == CP_KERNEL) {
      HandleOneKernelLine(line_to_handle);
      if (code_partition != CP_KERNEL && !cok_codes.empty())
        cok_codes.pop_back();
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
