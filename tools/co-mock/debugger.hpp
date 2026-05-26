#ifndef CHOREO_CO_MOCK_DEBUGGER_HPP_
#define CHOREO_CO_MOCK_DEBUGGER_HPP_

#include "mock_memory.hpp"
#include <functional>
#include <iostream>
#include <set>
#include <string>

namespace Choreo {
namespace AST {
class Node;
}

namespace Mock {

class MockInterpreter;

class Debugger {
public:
  Debugger(MockMemory& mem, MockInterpreter& interp);

  // Called by the interpreter before executing each statement.
  // Returns false if the user chose to quit.
  bool OnStatement(AST::Node& stmt);

  bool IsActive() const { return active_; }
  void SetActive(bool v) { active_ = v; }

  void SetInputStream(std::istream* is) { input_ = is; }

  void EnterBlock() { ++current_depth_; }
  void LeaveBlock() { --current_depth_; }

private:
  enum StopReason { Step, Next, Breakpoint, Initial };
  enum Mode { Run, StepInto, StepOver };

  bool ShouldStop(AST::Node& stmt);
  bool CommandLoop(AST::Node& stmt);
  bool ProcessCommand(const std::string& input, AST::Node& stmt);

  void CmdHelp();
  void CmdList(AST::Node& stmt, int context_lines = 5);
  void CmdPrint(const std::string& var_name);
  void CmdInfo();
  void CmdInfoFutures();
  void CmdInfoMem();
  void CmdBreak(const std::string& arg);
  void CmdDelete(const std::string& arg);
  void CmdBreakpoints();

  void ShowLocation(AST::Node& stmt);

  MockMemory& mem_;
  MockInterpreter& interp_;

  bool active_ = false;
  Mode mode_ = StepInto;
  int step_depth_ = 0;
  int current_depth_ = 0;
  std::set<int> breakpoints_;
  std::istream* input_ = &std::cin;
};

} // namespace Mock
} // namespace Choreo

#endif // CHOREO_CO_MOCK_DEBUGGER_HPP_
