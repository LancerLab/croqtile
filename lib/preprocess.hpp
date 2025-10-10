#ifndef __CHOREO_PRE_PROCESS_HPP__
#define __CHOREO_PRE_PROCESS_HPP__

#include <iostream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Choreo {

struct Bundle {
  size_t start;
  size_t end;
};

struct Range {
  size_t start{}, end{};
};

class SimplePreprocessor {
public:
  SimplePreprocessor(std::ostream& o);

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

  bool debug = false;

private:
  std::vector<Bundle> find_bundles(const std::string& code);
  std::vector<Range> extract_cok_sections(const std::string& code,
                                          size_t base_offset = 0);

  void EmitScript(std::ostream& os);
  std::string SubStituteMacroFuncs(const std::string& line,
                                   const FuncMap& funcs, bool& changed);

  std::string SubStituteDefines(const std::string& line,
                                const DefineMap& defines, const FuncMap& funcs);
  std::string SubstituteGlobalDefines(const std::string& line);
  std::string SubstituteLocalDefines(const std::string& line);
  std::string SubstituteGlobalMacroFuncs(const std::string& line,
                                         bool& changed);
  std::string SubstituteLocalMacroFuncs(const std::string& line, bool& changed);
  bool isDirective(const std::string& line, const std::string& directive,
                   bool blank = true);
  std::string
  preprocessBooleanExpression(const std::string& expr, const DefineMap& defines,
                              std::unordered_map<std::string, bool>& macroMap);
  bool EvaluateBooleanExpression(const std::string& condition_expr,
                                 const DefineMap& defines);
  const std::string HandleCComments(const std::string& line);
  void HandleOneUserLine(const std::string& line);
  void HandleOneKernelLine(const std::string& line, bool handle_comment = true);
  void HandleOneChoreoLine(const std::string& line, bool handle_comment = true);

public:
  bool ExtractDeviceKernel(std::stringstream& cok_ss);
  bool Process(std::istream& input);
};

} // end namespace Choreo

#endif //__CHOREO_PRE_PROCESS_HPP__
