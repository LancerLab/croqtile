#ifndef CHOREO_CODEGEN_FACTOR_HPP_
#define CHOREO_CODEGEN_FACTOR_HPP_

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
  std::string factor_fname; // choreo-factor function name
  std::string fname; // function name in source code, also the C++ function of
                     // choreo entry

  std::string indent;

  // ochestrate multiple streams
  std::ostringstream ks; // buffer stream of the kernel code (user provided)
  std::ostringstream fs; // buffer stream of the factor code (generated)
  std::ostringstream hs; // buffer stream of the host code (user provided)
  std::ostringstream alloc_in_fs; // buffer "alloc" statements in factor code

  std::string::size_type alloc_pos;
  std::string alloc_indent;

  // backend (factor) compile environment related
  std::string host_filename;
  std::string build_path; // path for the script to build factor code
  bool compile_with_dynshape =
      false; // if factor compile requires dynshape support

  const std::string named_dim_ref_prefix = "__choreo_nd_ref_";

  bool void_return = false;
  size_t hp_count = 0; // host parameter count

  int parallel_level = 0;
  bool cross_compile = false;
  bool use_kernel_template = false;

  ValBind::BindInfo<std::string> bind_info;
  std::map<std::string, std::stack<std::vector<std::string>>> cur_bounded_vars;
  std::vector<std::unordered_set<std::string>> loop_vars; // the loop variables
  std::vector<RtMemUsageCheckInfo> rt_mem_usage_check_list;

  // map from a symbolic shape dimension to the associated runtime name
  std::map<std::string, DimensionDetail> dims_info;
  std::map<std::string, std::string> idnm_rts; // name in .co to symbolic name

  ptr<FutureBufferMap> fut_buf; // map a future to its associated buffer
  ptr<CodeGenInfo> cgi;

  StringifyTable factor_symbols;

public:
  FactorCodeGen(const ptr<SymbolTable>& symtab,
                const std::vector<RtMemUsageCheckInfo>& list,
                const ptr<FutureBufferMap>& fb, const ptr<CodeGenInfo>& ci,
                bool cross_compile, bool use_kernel_template)
      : CodeGenerator("codegen", symtab), cross_compile(cross_compile),
        use_kernel_template(use_kernel_template),
        rt_mem_usage_check_list(list), fut_buf(fb), cgi(ci) {}

  void OutputScript(const ptr<FunctionType>&);

  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  // bool Visit(AST::Node&) override;
  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::Boolean&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::MultiDimSpans&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::IntTuple&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::IntIndex&) override;
  bool Visit(AST::DataType&) override;
  bool Visit(AST::Identifier&) override;
  bool Visit(AST::Parameter&) override;
  bool Visit(AST::ParamList&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::Memory&) override;
  bool Visit(AST::SpanAs&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::Program&) override;

private:
  bool ContainsLoopVar(const std::string&) const;

  void EmitHostHead(std::ostream&);
  void EmitHostFuncDecl(std::ostringstream&, const FunctionType&,
                        const std::string&);
  void EmitRuntimeCheck(std::ostream&);
  void EmitRuntimeMemUsageCheck(std::ostream&);
  void EmitHostFuncBody(std::ostream&, const FunctionType&,
                        const std::string& fname);

  const std::string ExprSTR(AST::ptr<AST::Node>) const;
  std::string GenHostParamName() { return "hp" + std::to_string(hp_count++); }
  std::string ReplaceRuntimeNames(const std::string&, const std::string& = "",
                                  bool host_code = true);
  std::string ReplaceFactorDynDimName(const std::string&);
  std::optional<std::string> ReplaceDynDimRef(const std::string&);

  // common utils
  void IncrementIndent() { this->indent += "  "; }
  void DecrementIndent() {
    if (this->indent.size() >= 2)
      this->indent = this->indent.substr(0, this->indent.size() - 2);
  }

  // TODO: determine what to clear!
  void ClearChoreoFunctionStates() {
    hp_count = 0; // reset the count of stub parameter
    dims_info.clear();
    idnm_rts.clear();
    indent.clear();

    // Reset buffers;
    fs.str("");
    fs.clear();
    alloc_in_fs.clear();
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
};

} // end namespace Factor

} // end namespace Choreo

#endif // CHOREO_CODEGEN_FACTOR_HPP_
