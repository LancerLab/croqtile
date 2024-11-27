#ifndef __CHOREO_CODEGEN_FACTOR_HPP__
#define __CHOREO_CODEGEN_FACTOR_HPP__

#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ast.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "factor_script.inc"
#include "types.hpp"

namespace Choreo {

namespace Factor {

// The example illustrates the a dimension detail:
//
// - 'hp0.shape()[1]' is the host dimension name of 2nd dim of hp0's shape
// - 'hp0' is the first host parameter, therefore its 'param_index == 0'
// - it refers the 2nd dim, so the 'dim_index == 1'
//
struct DimensionDetail {
  std::string hd_name; // host dimension name of a shape.
  size_t param_index;  // index of the parameter that gives the shape with the
                       // dimension
  size_t dim_index;    // dimension index inside a shape
};

struct FactorCodeGen : public CodeGenerator {
private:
  std::string factor_pname;               // choreo-factor program name
  std::string factor_fname;               // choreo-factor function name
  std::vector<std::string> factor_fnames; // all choreo-factor functions
  std::string indent;

  using FunctionStream = std::map<std::string, std::ostringstream>;
  // ochestrate multiple streams
  std::ostringstream ds; // buffer stream of the forward declarations
  std::ostringstream ks; // buffer stream of the kernel code (user provided)
  std::ostringstream fs; // buffer stream of the factor code (generated)
  std::ostringstream hs; // buffer stream of the host code
  std::ostringstream cs; // buffer stream of the user host code
  std::ostringstream alloc_in_fs; // buffer "alloc" statements in factor code
  std::string host_code;
  std::string factor_code;

  // std::string::size_type alloc_pos;
  // std::string alloc_indent;
  std::stack<std::ostringstream> alloc_fs_stack;
  std::stack<int> alloc_pos_stack;
  std::stack<std::string> alloc_indent_stack;

  // backend (factor) compile environment related
  std::string build_path;      // path for the script to build factor code
  std::string host_cpp_name;   // choreo entry and user host function
  std::string kernel_cpp_name; // __cok__ function as a cpp file
  std::string factor_cpp_name; // __co__ translated to factor code
  std::string factor_bin_name; // compiled factor binary
  bool compile_with_dynshape =
      false; // if factor compile requires dynshape support

  const std::string named_dim_ref_prefix = "__choreo_nd_ref_";

  bool void_return = false;
  size_t hp_count = 0; // host parameter count

  int parallel_level = 0;
  bool cross_compile = false;
  bool use_kernel_template = false;
  bool factor_host_unbraced = false;

  ptr<FunctionType> fty = nullptr;

  ValBind::BindInfo<std::string> bind_info;
  std::map<std::string, std::stack<std::vector<std::string>>> cur_bounded_vars;
  std::vector<std::unordered_set<std::string>> loop_vars; // the loop variables
  std::vector<RtMemUsageCheckInfo> rt_mem_usage_check_list;

  // map from a symbolic shape dimension to the associated runtime name
  std::map<std::string, DimensionDetail> dims_info;
  std::map<std::string, std::string> idnm_rts; // name in .co to symbolic name

  ptr<CodeGenInfo> cgi;

  StringifyTable factor_symbols;

  const FutureBufferInfo& FBInfo() const {
    return FCtx(fname).GetFutureBufferInfo();
  }

  size_t factor_host_arity = 0;
  size_t factor_device_arity = 0;

public:
  FactorCodeGen(const ptr<SymbolTable>& symtab,
                const std::vector<RtMemUsageCheckInfo>& list,
                const ptr<CodeGenInfo>& ci, bool cross_compile,
                bool use_kernel_template)
      : CodeGenerator("codegen", symtab), cross_compile(cross_compile),
        use_kernel_template(use_kernel_template), rt_mem_usage_check_list(list),
        cgi(ci) {
    factor_pname =
        "__choreo_" +
        RemoveDirectoryPrefix(RemoveSuffix(
            OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  }

  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::Return&) override;

private:
  bool ContainsLoopVar(const std::string&) const;

  void EmitScript();
  void EmitFactorSource();

  void EmitFixedHostHead();
  void EmitFixedFactorHead();
  void EmitHostFuncDecl(std::ostringstream&, const std::string&);
  void EmitHostRuntimeCheck(std::ostream&);
  void EmitHostRuntimeMemUsageCheck(std::ostream&);
  void EmitHostFunction(std::ostream&);

  const std::string ExprSTR(AST::ptr<AST::Node>, bool = true) const;
  std::string GenHostParamName() { return "hp" + std::to_string(hp_count++); }
  const std::string ReplaceRuntimeNames(const std::string&,
                                        const std::string& = "",
                                        bool host_code = true) const;
  std::string ReplaceFactorDynDimName(const std::string&);
  std::optional<std::string> ReplaceDynDimRef(const std::string&);
  const std::string ValueSTR(const ValueItem&) const;

  // common utils
  void IncrementIndent() { this->indent += "  "; }
  void DecrementIndent() {
    if (this->indent.size() >= 2)
      this->indent = this->indent.substr(0, this->indent.size() - 2);
  }

  // TODO: determine what to clear!
  void ResetChoreoFunctionStates() {
    hp_count = 0; // reset the count of stub parameter
    factor_host_unbraced = true;
    dims_info.clear();
    idnm_rts.clear();
    indent.clear();

    // Reset buffers;
    fs.str("");
    fs.clear();
    alloc_in_fs.clear();

    // recalculate the arities
    factor_host_arity = 0;
    factor_device_arity = 0;
    for (auto& item : GetFactorHostInParams()) {
      (void)item;
      factor_host_arity++;
    }
    for (auto& item : GetFactorDeviceInParams()) {
      (void)item;
      factor_device_arity++;
    }
    fty = nullptr;
    void_return = false;
  }

  // in factor, there exists choreo-host/factor-host/factor-device functions.
  // There wrappers make the parameter clear.
  FilterRange<SymbolDetail> GetChoreoParameters() {
    return cgi->GetParameters(fname);
  }

  FilterRange<SymbolDetail> GetFactorHostInParams() {
    return cgi->GetDeviceAllocIns(fname);
  }

  FilterRange<SymbolDetail> GetFactorDeviceInParams() {
    return cgi->GetDevicePassIns(fname);
  }

  size_t GetFactorHostInArity() { return factor_host_arity; }
  size_t GetFactorDeviceInArity() { return factor_device_arity; }

  std::optional<std::string> GetChoreoHostReturnTypeString() const;

}; // end FactorCodeGen

} // end namespace Factor

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_FACTOR_HPP__
