#ifndef __CHOREO_PRE_PROCESS_HPP__
#define __CHOREO_PRE_PROCESS_HPP__

#include "aux.hpp"
#include "context.hpp"
#include "io.hpp"
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
  std::ostream& output;
  std::unordered_map<std::string, std::string> globalDefines;
  std::unordered_map<std::string, std::string> localDefines;

  int choreo_brace_count = 0;
  int kernel_brace_count = 0;

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
    default: break;
    }
    // command-line macros override
    for (auto& item : CCtx().GetCLMacros())
      globalDefines[item.first] = item.second;
  }

private:
  std::string SubstituteGlobalDefines(const std::string& line) {
    std::string result = line;
    for (const auto& [key, value] : globalDefines) {
      std::regex pattern("\\b" + key + "\\b");
      result = std::regex_replace(result, pattern, value);
    }
    return result;
  }

  std::string SubstituteLocalDefines(const std::string& line) {
    std::string result = line;
    for (const auto& [key, value] : localDefines) {
      std::regex pattern("\\b" + key + "\\b");
      result = std::regex_replace(result, pattern, value);
    }
    return result;
  }

  bool isDirective(const std::string& line, const std::string& directive) {
    return line == directive || line.rfind(directive + " ", 0) == 0;
  }

private:
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
        std::regex defineRegex("#define\\s+(\\w+)(?:\\s+(.*))?");
        std::smatch match;
        if (std::regex_match(bline, match, defineRegex))
          globalDefines[match[1]] = match[2].matched ? match[2].str() : "1";
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#undef")) {
      if (cur_cond && !cur_skip) {
        std::regex undefRegex("#undef\\s+(\\w+)");
        std::smatch match;
        if (std::regex_match(bline, match, undefRegex))
          globalDefines.erase(match[1]);
      }
      output << line << '\n';
      return;
    } else if (isDirective(bline, "#if")) {
      std::regex ifRegex("#if\\s+(.*)");
      std::smatch match;
      if (std::regex_match(bline, match, ifRegex)) {
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

    // Check if entering a __co__ function
    if (sline.find("__co__ ") != std::string::npos) {
      if (kernel_brace_count) {
        errs() << "copp: in line " << line_num
               << ": error: '__co__' function inside '__cok__' is illegal.\n";
        abort();
      }
      code_partition = CP_CHOREO;

      auto c_pos = sline.find_first_of('{');
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
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#define")) {
      if (cur_cond && !cur_skip) {
        std::regex defineRegex("#define\\s+(\\w+)(?:\\s+(.*))?");
        std::smatch match;
        if (std::regex_match(bline, match, defineRegex)) {
          localDefines[match[1]] = match[2].matched ? match[2].str() : "1";
        }
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#undef")) {
      if (cur_cond && !cur_skip) {
        std::regex undefRegex("#undef\\s+(\\w+)");
        std::smatch match;
        if (std::regex_match(bline, match, undefRegex)) {
          localDefines.erase(match[1]);
        }
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#if")) {
      std::regex ifRegex("#if\\s+(.*)");
      std::smatch match;
      if (std::regex_match(bline, match, ifRegex)) {
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
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#else")) {
      if (!co_condition_stack.empty()) {
        bool currentCondition = co_condition_stack.top();
        co_condition_stack.top() = !currentCondition;
        co_skip_line = co_skip_stack.top() || !co_condition_stack.top();
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (isDirective(bline, "#endif")) {
      if (!co_condition_stack.empty()) {
        co_condition_stack.pop();
        co_skip_line = co_skip_stack.top();
        co_skip_stack.pop();
      }
      if (!co_skip_line) output << "#line " << line_num + 1 << "\n";
    } else if (!co_skip_line) {
      size_t co_start = 0;
      size_t co_end = aline.size();
      for (size_t i = 0; i < aline.size(); ++i) {
        char c = aline[i];
        if (c == '{') {
          choreo_brace_count++;
          if (choreo_brace_count == 1) {
            // just entered
            co_start = i;
            localDefines = globalDefines;
          }
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
        output << line << '\n';
        return;
      }

      auto co_code = aline.substr(co_start, co_end - co_start);
      auto sline = SubstituteLocalDefines(co_code);

      // output the choreo code
      if (!uc_skip_line) output << aline.substr(0, co_start) << sline;

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
  bool Process(std::istream& input) {
    code_partition = CP_USER;
    std::string line;

    while (std::getline(input, line)) {
      if (code_partition == CP_USER) {
        HandleOneUserLine(line);
        line_num++;
      } else if (code_partition == CP_CHOREO) {
        HandleOneChoreoLine(line);
        line_num++;
      } else if (code_partition == CP_KERNEL) {
        HandleOneKernelLine(line);
        line_num++;
      } else
        choreo_unreachable("code partition is not known.");
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
