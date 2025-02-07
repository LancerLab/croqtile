#ifndef __CHOREO_COMMAND_LINE_HPP__
#define __CHOREO_COMMAND_LINE_HPP__

#include <string>
#include <unordered_map>

namespace Choreo {

class CommandLine {
private:
  int ret_code = 0;

public:
  bool Parse(int argc, char** argv);
  int ReturnCode() const { return ret_code; }
}; // CommandLine

} // end namespace Choreo

#endif // __CHOREO_COMMAND_LINE_HPP__
