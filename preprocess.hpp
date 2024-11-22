#ifndef __CHOREO_PRE_PROCESS__
#define __CHOREO_PRE_PROCESS__

#include <iostream>
#include <istream>
#include <regex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Choreo {

class SimplePreprocessor {
private:
  std::unordered_map<std::string, std::string> globalDefines;
  std::unordered_map<std::string, std::string> localDefines;

  std::string substituteDefines(const std::string& line,
                                bool isInsideFunction) {
    if (!isInsideFunction) {
      // Do not substitute for lines outside a __co__ function
      return line;
    }

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

public:
  void process(std::istream& input, std::ostream& output) {
    bool skipBlock = false;
    std::stack<bool> conditionStack;
    std::stack<bool> skipStack;
    int braceCount = 0;
    bool insideFunction = false;
    std::string line;

    while (std::getline(input, line)) {
      std::string strippedLine =
          std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");

      // Output global defines directly if outside a __co__ function
      if (!insideFunction && isDirective(strippedLine, "#define")) {
        output << line << '\n';
        std::regex defineRegex("#define\\s+(\\w+)(?:\\s+(.*))?");
        std::smatch match;
        if (std::regex_match(strippedLine, match, defineRegex)) {
          globalDefines[match[1]] = match[2].matched ? match[2].str() : "1";
        }
        continue;
      }

      // Output undefines directly if outside a __co__ function
      if (!insideFunction && isDirective(strippedLine, "#undef")) {
        output << line << '\n';
        std::regex undefRegex("#undef\\s+(\\w+)");
        std::smatch match;
        if (std::regex_match(strippedLine, match, undefRegex)) {
          globalDefines.erase(match[1]);
        }
        continue;
      }

      // Check if entering a __co__ function
      if (strippedLine.find("__co__") != std::string::npos &&
          strippedLine.find("{") != std::string::npos) {
        insideFunction = true;
        braceCount = 1;               // Initial brace for the function
        localDefines = globalDefines; // Copy global defines to local scope
        output << substituteDefines(line, insideFunction) << '\n';
        continue;
      }

      // Check if exiting a __co__ function
      if (insideFunction) {
        braceCount += std::count(line.begin(), line.end(), '{');
        braceCount -= std::count(line.begin(), line.end(), '}');

        if (braceCount == 0) {
          insideFunction = false;
          output << substituteDefines(line, insideFunction) << '\n';
          continue;
        }
      }

      if (insideFunction) {
        if (isDirective(strippedLine, "#ifdef")) {
          std::regex ifdefRegex("#ifdef\\s+(\\w+)");
          std::smatch match;
          if (std::regex_match(strippedLine, match, ifdefRegex)) {
            bool condition = localDefines.find(match[1]) != localDefines.end();
            conditionStack.push(condition);
            skipStack.push(skipBlock);
            skipBlock = skipBlock || !condition;
          }
          continue;
        } else if (isDirective(strippedLine, "#ifndef")) {
          std::regex ifndefRegex("#ifndef\\s+(\\w+)");
          std::smatch match;
          if (std::regex_match(strippedLine, match, ifndefRegex)) {
            bool condition = localDefines.find(match[1]) == localDefines.end();
            conditionStack.push(condition);
            skipStack.push(skipBlock);
            skipBlock = skipBlock || !condition;
          }
          continue;
        } else if (isDirective(strippedLine, "#define")) {
          std::regex defineRegex("#define\\s+(\\w+)(?:\\s+(.*))?");
          std::smatch match;
          if (std::regex_match(strippedLine, match, defineRegex)) {
            localDefines[match[1]] = match[2].matched ? match[2].str() : "1";
          }
          continue;
        } else if (isDirective(strippedLine, "#undef")) {
          std::regex undefRegex("#undef\\s+(\\w+)");
          std::smatch match;
          if (std::regex_match(strippedLine, match, undefRegex)) {
            localDefines.erase(match[1]);
          }
          continue;
        } else if (isDirective(strippedLine, "#if")) {
          std::regex ifRegex("#if\\s+(.*)");
          std::smatch match;
          if (std::regex_match(strippedLine, match, ifRegex)) {
            try {
              bool condition = std::stoi(match[1].str()) != 0;
              conditionStack.push(condition);
              skipStack.push(skipBlock);
              skipBlock = skipBlock || !condition;
            } catch (...) {
              conditionStack.push(false);
              skipStack.push(skipBlock);
              skipBlock = true;
            }
          }
          continue;
        } else if (isDirective(strippedLine, "#else")) {
          if (!conditionStack.empty()) {
            bool currentCondition = conditionStack.top();
            conditionStack.top() = !currentCondition;
            skipBlock = skipStack.top() || !conditionStack.top();
          }
          continue;
        } else if (isDirective(strippedLine, "#endif")) {
          if (!conditionStack.empty()) {
            conditionStack.pop();
            skipBlock = skipStack.top();
            skipStack.pop();
          }
          continue;
        } else if (!skipBlock) {
          output << substituteDefines(line, insideFunction) << '\n';
        }
      } else {
        output << line << '\n';
      }
    }
  }
};

} // end namespace Choreo

#endif //__CHOREO_PRE_PROCESS__
