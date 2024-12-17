#ifndef __CHOREO_CODEGEN_TOPSCC_HPP__
#define __CHOREO_CODEGEN_TOPSCC_HPP__

#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#include "ast.hpp"
// #include "choreo_topscc_header.inc"
#include "codegen.hpp"
// #include "topscc_script.inc"
#include "types.hpp"

namespace Choreo {

namespace Topscc {

struct SymbolicDimensionInfo {
  std::string hsd_expr; // (host) expression of the symbolic dimension
  size_t param_index;   // index of the parameter that gives the shape with the
                        // dimension
  size_t dim_index;     // dimension index inside a shape
};

// information about choreo symbolic dimensions
using SDimsInfo = std::map<std::string, SymbolicDimensionInfo>;

enum CodeSegment {
  CS_UNKNOWN,
  CS_USER,
  CS_COK,
  CS_CO,
};

struct TopsccCodeGen : public CodeGenerator {
private:
  std::map<std::string, std::vector<RtMemUsageCheckInfo>> muc;
  ptr<CodeGenInfo> cgi;

public:
  TopsccCodeGen(
      const std::map<std::string, std::vector<RtMemUsageCheckInfo>>& m,
      const ptr<CodeGenInfo>& ci)
      : CodeGenerator("codegen", CCtx().GetGlobalSymbolTable()), muc(m),
        cgi(ci) {
    cu_name = "__choreo_" + OptionRegistry::GetInstance().GetInputName();
    cmp_dir = CreateUniquePath();
  }

  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::MultiNodes&) override { return true; };
  bool Visit(AST::MultiValues&) override { return true; };
  bool Visit(AST::IntLiteral&) override { return true; };
  bool Visit(AST::Expr&) override { return true; };
  bool Visit(AST::MultiDimSpans&) override { return true; };
  bool Visit(AST::NamedTypeDecl&) override { return true; };
  bool Visit(AST::NamedVariableDecl&) override { return true; };
  bool Visit(AST::IntTuple&) override { return true; };
  bool Visit(AST::Assignment&) override { return true; };
  bool Visit(AST::IntIndex&) override { return true; };
  bool Visit(AST::DataType&) override { return true; };
  bool Visit(AST::Identifier&) override { return true; };
  bool Visit(AST::Parameter&) override { return true; };
  bool Visit(AST::ParamList&) override { return true; };
  bool Visit(AST::ParallelBy&) override { return true; };
  bool Visit(AST::WhereBind&) override { return true; };
  bool Visit(AST::WithIn&) override { return true; };
  bool Visit(AST::WithBlock&) override { return true; };
  bool Visit(AST::Memory&) override { return true; };
  bool Visit(AST::DMA&) override { return true; };
  bool Visit(AST::ChunkAt&) override { return true; };
  bool Visit(AST::Wait&) override { return true; };
  bool Visit(AST::Call&) override { return true; };
  bool Visit(AST::Rotate&) override { return true; };
  bool Visit(AST::Select&) override { return true; };
  bool Visit(AST::Return&) override { return true; };
  bool Visit(AST::LoopRange&) override { return true; };
  bool Visit(AST::ForeachBlock&) override { return true; };
  bool Visit(AST::Program&) override { return true; };

  bool Visit(AST::CppSourceCode& n) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::FunctionDecl&) override;

private:
  CodeSegment cs = CS_UNKNOWN;
  std::vector<std::string> code_segments; // multiple code segment

  std::string cmp_dir; // work directory
  std::string cu_name; // compilation unit name

  std::string device_fn; // current device function name

  std::string h_indent; // host indentation
  std::string d_indent; // device indentation

  int parallel_level = 0;

  size_t host_param_count = 0; // host parameter count

  ptr<FunctionType> fty = nullptr; // current function type
  bool void_return = false;

  SDimsInfo symbolic_dimensions;

  std::ostringstream ds; // device stream
  std::ostringstream hs; // host stream

private:
  void EmitFixedHostHead();
  void EmitFixedDeviceHead();

  void EmitHostFuncDecl(std::ostringstream&);
  void EmitDeviceFuncDecl(std::ostringstream&);

  void EmitSource();

private:
  void IncrHostIndent() { h_indent += "  "; }
  void IncrDeviceIndent() { d_indent += "  "; }
  void DecrHostIndent() {
    if (h_indent.size() < 2)
      choreo_unreachable("the indent can not be decreased.");
    h_indent = h_indent.substr(0, h_indent.size() - 2);
  }
  void DecrDeviceIndent() {
    if (d_indent.size() < 2)
      choreo_unreachable("the indent can not be decreased.");
    d_indent = d_indent.substr(0, d_indent.size() - 2);
  }

private:
  void ResetChoreoFunctionStates() {
    host_param_count = 0; // reset the count of host parameter
    symbolic_dimensions.clear();
    fty = nullptr;
    void_return = false;
  }

  std::string GenHostParamName() {
    return "hp" + std::to_string(host_param_count++);
  }

  FilterRange<SymbolDetail> GetDeviceFuncIns() {
    return cgi->GetDevicePassIns(fname);
  }

  FilterRange<SymbolDetail> GetChoreoFuncIns() {
    return cgi->GetParameters(fname);
  }
};

} // namespace Topscc

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_TOPSCC_HPP__
