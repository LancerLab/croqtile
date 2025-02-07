#ifndef __CHOREO_COMMAND_LINE_HPP__
#define __CHOREO_COMMAND_LINE_HPP__

#include <string>
#include <unordered_map>

namespace Choreo {

class CommandLine {
private:
  int ret_code = 0;
  static std::unordered_map<std::string, std::string> macro_defs;

public:
  static const std::unordered_map<std::string, std::string>& GetMacros() {
    return macro_defs;
  }

public:
  bool Parse(int argc, char** argv);
  int ReturnCode() const { return ret_code; }
}; // CommandLine

} // end namespace Choreo

#endif // __CHOREO_COMMAND_LINE_HPP__
