#include "debugger.hpp"
#include "ast.hpp"
#include "context.hpp"
#include "mock_interp.hpp"

#include <iostream>
#include <sstream>

namespace Choreo {
namespace Mock {

Debugger::Debugger(MockMemory& mem, MockInterpreter& interp)
    : mem_(mem), interp_(interp) {}

bool Debugger::OnStatement(AST::Node& stmt) {
  if (!active_) return true;
  if (ShouldStop(stmt)) return CommandLoop(stmt);
  return true;
}

bool Debugger::ShouldStop(AST::Node& stmt) {
  int line = stmt.LOC().begin.line;

  if (breakpoints_.count(line)) return true;

  switch (mode_) {
  case StepInto: return true;
  case StepOver: return current_depth_ <= step_depth_;
  case Run: return false;
  }
  return false;
}

void Debugger::ShowLocation(AST::Node& stmt) {
  int line = stmt.LOC().begin.line;
  auto src = CCtx().GetSourceLine(line);

  std::string bp_marker = breakpoints_.count(line) ? "B>" : "->";
  std::cout << bp_marker << " " << line << "  " << src << "\n";
}

bool Debugger::CommandLoop(AST::Node& stmt) {
  ShowLocation(stmt);

  bool scripted = (input_ != &std::cin);

  while (true) {
    if (!scripted) std::cout << "(co-mock) " << std::flush;
    std::string input;
    if (!std::getline(*input_, input)) {
      if (scripted) {
        // Script exhausted: switch to stdin for remaining execution
        input_ = &std::cin;
        mode_ = Run;
        return true;
      }
      std::cout << "\n";
      return false;
    }

    // Trim whitespace
    auto start = input.find_first_not_of(" \t");
    if (start == std::string::npos) {
      mode_ = StepInto;
      step_depth_ = current_depth_;
      return true;
    }
    input = input.substr(start);
    auto end = input.find_last_not_of(" \t");
    if (end != std::string::npos) input = input.substr(0, end + 1);

    if (scripted) std::cout << "(co-mock) " << input << "\n";

    if (!ProcessCommand(input, stmt)) return false;

    if (mode_ == Run || mode_ == StepInto || mode_ == StepOver) return true;
  }
}

bool Debugger::ProcessCommand(const std::string& input, AST::Node& stmt) {
  std::istringstream iss(input);
  std::string cmd;
  iss >> cmd;

  if (cmd == "h" || cmd == "help") {
    CmdHelp();
    mode_ = (Mode)-1; // stay in command loop
  } else if (cmd == "s" || cmd == "step") {
    mode_ = StepInto;
    step_depth_ = current_depth_;
  } else if (cmd == "n" || cmd == "next") {
    mode_ = StepOver;
    step_depth_ = current_depth_;
  } else if (cmd == "c" || cmd == "continue") {
    mode_ = Run;
  } else if (cmd == "p" || cmd == "print") {
    std::string rest;
    std::getline(iss >> std::ws, rest);
    if (rest.empty())
      std::cout << "Usage: p <variable>\n";
    else
      CmdPrint(rest);
    mode_ = (Mode)-1;
  } else if (cmd == "l" || cmd == "list") {
    CmdList(stmt);
    mode_ = (Mode)-1;
  } else if (cmd == "b" || cmd == "break") {
    std::string arg;
    iss >> arg;
    CmdBreak(arg);
    mode_ = (Mode)-1;
  } else if (cmd == "d" || cmd == "delete") {
    std::string arg;
    iss >> arg;
    CmdDelete(arg);
    mode_ = (Mode)-1;
  } else if (cmd == "info") {
    std::string what;
    iss >> what;
    if (what == "break" || what == "breakpoints" || what == "b")
      CmdBreakpoints();
    else if (what == "futures" || what == "fut" || what == "f")
      CmdInfoFutures();
    else if (what == "mem" || what == "memory" || what == "m")
      CmdInfoMem();
    else if (what == "locals" || what == "l")
      CmdInfo();
    else
      CmdInfo();
    mode_ = (Mode)-1;
  } else if (cmd == "q" || cmd == "quit") {
    return false;
  } else {
    // Try as a variable print
    if (mem_.Exists(cmd)) {
      CmdPrint(cmd);
    } else {
      std::cout << "Unknown command: '" << cmd
                << "'. Type 'h' for help.\n";
    }
    mode_ = (Mode)-1;
  }
  return true;
}

void Debugger::CmdHelp() {
  std::cout
      << "co-mock interactive debugger\n"
      << "\n"
      << "  s, step        Execute one statement (step into blocks)\n"
      << "  n, next        Execute one statement (step over blocks)\n"
      << "  c, continue    Run until next breakpoint or end\n"
      << "  p <var>        Print variable value\n"
      << "  l, list        Show source around current line\n"
      << "  b <line>       Set breakpoint at line number\n"
      << "  d <line>       Delete breakpoint at line number\n"
      << "  info           Show all variables in scope\n"
      << "  info futures   Show async DMA future status\n"
      << "  info mem       Show memory allocations\n"
      << "  info break     Show all breakpoints\n"
      << "  q, quit        Exit the debugger\n"
      << "  <Enter>        Repeat last step\n"
      << "  <varname>      Print variable (shorthand for 'p <var>')\n"
      << "\n";
}

void Debugger::CmdList(AST::Node& stmt, int context_lines) {
  int cur = stmt.LOC().begin.line;
  int from = std::max(1, cur - context_lines);
  int to = cur + context_lines;

  for (int i = from; i <= to; ++i) {
    auto src = CCtx().GetSourceLine(i);
    if (src.empty() && i > cur + 1) break;
    const char* marker = (i == cur) ? "->" : "  ";
    const char* bp = breakpoints_.count(i) ? "B" : " ";
    std::cout << bp << marker << " " << i << "\t" << src << "\n";
  }
}

void Debugger::CmdPrint(const std::string& var_name) {
  // Handle array element access: var[idx] or var.at(idx)
  std::string base = var_name;
  int idx = -1;

  auto bracket = var_name.find('[');
  if (bracket != std::string::npos) {
    auto close = var_name.find(']', bracket);
    if (close != std::string::npos) {
      base = var_name.substr(0, bracket);
      idx = std::stoi(var_name.substr(bracket + 1, close - bracket - 1));
    }
  }

  if (!mem_.Exists(base)) {
    std::cout << "Variable '" << base << "' not found.\n";
    return;
  }

  auto& val = mem_.Lookup(base);
  if (idx >= 0 && val.kind == Value::Pointer && val.alloc) {
    size_t elem_size = val.alloc->ElemSize();
    auto elem = val.ReadFromAlloc(idx * elem_size, val.alloc->elem_type);
    std::cout << var_name << " = " << elem.ToString() << "\n";
  } else {
    std::cout << base << " = " << val.ToString() << "\n";
    if (val.kind == Value::Pointer && val.alloc) {
      size_t total = val.alloc->TotalElements();
      size_t show = std::min(total, (size_t)16);
      std::cout << "  [";
      for (size_t i = 0; i < show; ++i) {
        if (i > 0) std::cout << ", ";
        auto elem =
            val.ReadFromAlloc(i * val.alloc->ElemSize(), val.alloc->elem_type);
        std::cout << elem.ToString();
      }
      if (show < total) std::cout << ", ...";
      std::cout << "]  (" << total << " elements)\n";
    }
  }
}

void Debugger::CmdInfo() {
  auto& scopes = mem_.AllScopes();
  for (int s = (int)scopes.size() - 1; s >= 0; --s) {
    if (scopes[s].empty()) continue;
    std::cout << "--- scope " << s << " ---\n";
    for (auto& [name, val] : scopes[s]) {
      // Skip internal anonymous variables
      if (name.find("anon_") == 0) continue;
      std::cout << "  " << name << " = " << val.ToString();
      if (val.kind == Value::Pointer && val.alloc) {
        size_t total = val.alloc->TotalElements();
        size_t show = std::min(total, (size_t)8);
        std::cout << "  [";
        for (size_t i = 0; i < show; ++i) {
          if (i > 0) std::cout << ", ";
          auto elem = val.ReadFromAlloc(i * val.alloc->ElemSize(),
                                        val.alloc->elem_type);
          std::cout << elem.ToString();
        }
        if (show < total) std::cout << ", ...";
        std::cout << "]";
      }
      std::cout << "\n";
    }
  }
}

void Debugger::CmdInfoFutures() {
  bool found = false;
  auto& scopes = mem_.AllScopes();
  for (int s = (int)scopes.size() - 1; s >= 0; --s) {
    for (auto& [name, val] : scopes[s]) {
      if (val.kind != Value::Future) continue;
      found = true;
      std::cout << "  " << name << ": " << val.ToString() << "\n";
    }
  }
  if (!found) std::cout << "No active futures.\n";
}

void Debugger::CmdInfoMem() {
  std::set<void*> seen;
  auto& scopes = mem_.AllScopes();
  for (int s = (int)scopes.size() - 1; s >= 0; --s) {
    for (auto& [name, val] : scopes[s]) {
      if (val.kind != Value::Pointer || !val.alloc) continue;
      if (seen.count(val.alloc->RawPtr())) continue;
      seen.insert(val.alloc->RawPtr());
      std::cout << "  " << name << ": " << STR(val.alloc->storage) << " "
                << STR(val.alloc->elem_type) << "[";
      for (size_t i = 0; i < val.alloc->shape.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << val.alloc->shape[i];
      }
      std::cout << "] (" << val.alloc->TotalBytes() << " bytes)\n";
    }
  }
  if (seen.empty()) std::cout << "No allocations.\n";
}

void Debugger::CmdBreak(const std::string& arg) {
  if (arg.empty()) {
    CmdBreakpoints();
    return;
  }
  try {
    int line = std::stoi(arg);
    breakpoints_.insert(line);
    std::cout << "Breakpoint set at line " << line << "\n";
  } catch (...) {
    std::cout << "Usage: b <line_number>\n";
  }
}

void Debugger::CmdDelete(const std::string& arg) {
  if (arg.empty()) {
    breakpoints_.clear();
    std::cout << "All breakpoints deleted.\n";
    return;
  }
  try {
    int line = std::stoi(arg);
    if (breakpoints_.erase(line))
      std::cout << "Breakpoint at line " << line << " deleted.\n";
    else
      std::cout << "No breakpoint at line " << line << ".\n";
  } catch (...) {
    std::cout << "Usage: d <line_number>\n";
  }
}

void Debugger::CmdBreakpoints() {
  if (breakpoints_.empty()) {
    std::cout << "No breakpoints set.\n";
    return;
  }
  std::cout << "Breakpoints:\n";
  for (int bp : breakpoints_) {
    auto src = CCtx().GetSourceLine(bp);
    std::cout << "  line " << bp << ": " << src << "\n";
  }
}

} // namespace Mock
} // namespace Choreo
