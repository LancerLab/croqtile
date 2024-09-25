#ifndef CHOREO_CODEGEN_FACTOR_HPP_
#define CHOREO_CODEGEN_FACTOR_HPP_

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <map>
#include <vector>
#include <utility>
#include <unordered_set>

#include "ast.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "factor_script.inc"
#include "types.hpp"

namespace Choreo {

namespace Factor {

struct FactorCodeGen : public CodeGenerator {
 private:
  std::string current_fn = "";
  std::string entry_fn = "";
  std::string indent = "";
  std::string bin_fn;  // temporal filename of factor binary
  // buffer the kernel code
  std::ostringstream ks;
  // buffer the factor code
  std::ostringstream fs;
  // buffer the host code
  std::ostringstream hs;
  std::string host_fn;
  std::string target_fn;
  std::string build_path;
  // buffer of "alloc" statements in factor code
  std::ostringstream alloc_in_fs;
  std::string::size_type alloc_pos;
  std::string alloc_indent;
  // output variable name
  std::string output_v;

  bool void_return = false;
  int parallel_factor = 1;
  size_t sp_count = 0;
  int parallel_level = 0;
  bool dyn_shaped = false;
  bool cross_compile = false;

  ValBind::BindInfo<std::string> bind_info;
  std::vector<AST::ptr<AST::Parameter>> *cur_params = nullptr;
  AST::ptr<AST::DataType> current_output = nullptr;
  std::map<std::string, std::vector<std::string>> cur_bounded_vars;
  std::vector<std::unordered_set<std::string>> loop_vars;  // the loop variables
  std::vector<RtMemUsageCheckInfo> rt_mem_usage_check_list;
  // runtime host parameter names
  std::vector<std::string> host_params;
  // parameters: the name (of factor data) and associated size expression
  std::vector<std::pair<std::string, std::string>> param_map;

  // mapping from a symbolic shape dimensions to the associated runtime name
  std::map<std::string, std::string>
      rts_nmap;  // symbolic name to the runtime name
  std::map<std::string, size_t>
      rts_pidx;  // shape index in parameter list for the runtime shape name
  std::map<std::string, size_t>
      rts_nidx;  // dim index in shape for the runtime shape name
  StringifyTable factor_symbols;

 public:
  FactorCodeGen(std::ostream &os, const ptr<SymbolTable> &symtab, bool cross_compile)
      : CodeGenerator("codegen", os, symtab), cross_compile(cross_compile) {}
  FactorCodeGen(std::ostream &os, const ptr<SymbolTable> &symtab,
                const std::vector<RtMemUsageCheckInfo> &list, bool cross_compile)
      : CodeGenerator("codegen", os, symtab), rt_mem_usage_check_list(list),
        cross_compile(cross_compile) {}

  void ResetBuffers() {
    ks.clear();
    fs.clear();
    hs.clear();
  }

  void OutputScript(FunctionType *, const std::string &, const std::string &,
                    const std::string &, const Shape &);

  bool BeforeVisitImpl(AST::Node &) override;
  bool AfterVisitImpl(AST::Node &) override;

  // bool Visit(AST::Node&) override;
  bool Visit(AST::MultiNodes &) override;
  bool Visit(AST::MultiValues &) override;
  bool Visit(AST::IntLiteral &) override;
  bool Visit(AST::Boolean &) override;
  bool Visit(AST::Expr &) override;
  bool Visit(AST::MultiDimSpans &) override;
  bool Visit(AST::NamedTypeDecl &) override;
  bool Visit(AST::NamedVariableDecl &) override;
  bool Visit(AST::IntTuple &) override;
  bool Visit(AST::Assignment &) override;
  bool Visit(AST::IntIndex &) override;
  bool Visit(AST::DataType &) override;
  bool Visit(AST::Identifier &) override;
  bool Visit(AST::Parameter &) override;
  bool Visit(AST::ParamList &) override;
  bool Visit(AST::ParallelBy &) override;
  bool Visit(AST::WhereBind &) override;
  bool Visit(AST::WithIn &) override;
  bool Visit(AST::WithBlock &) override;
  bool Visit(AST::Memory &) override;
  bool Visit(AST::SpanAs &) override;
  bool Visit(AST::DMA &) override;
  bool Visit(AST::ChunkAt &) override;
  bool Visit(AST::Wait &) override;
  bool Visit(AST::Call &) override;
  bool Visit(AST::Swap &) override;
  bool Visit(AST::Select &) override;
  bool Visit(AST::Return &) override;
  bool Visit(AST::LoopRange &) override;
  bool Visit(AST::ForeachBlock &) override;
  bool Visit(AST::FunctionDecl &) override;
  bool Visit(AST::ChoreoFunction &) override;
  bool Visit(AST::CppSourceCode &) override;
  bool Visit(AST::Program &) override;

 private:
  bool ContainsLoopVar(const std::string &) const;

  void EmitHostHead(std::ostream &);
  void EmitHostFuncDecl(std::ostream &, const Type &, const std::string &,
                        bool = false);
  void EmitRuntimeCheck(std::ostream &, const Type &);
  void EmitRuntimeMemUsageCheck(std::ostream &, const Type &);
  void EmitHostFuncBody(std::ostream &, const Type &, const std::string &fname,
                        const std::string &o_sz, const std::string &o_ty,
                        const Shape &s);

  std::string GenHostParamName() { return "hp" + std::to_string(sp_count++); }
  std::string ReplaceRuntimeNames(const std::string &, const std::string & = "",
                                  bool host_code = true);
  std::string ReplaceDynDimName(const std::string &);
  // common utils
  void incrementIndent() { this->indent += "  "; }

  void decrementIndent() {
    if (this->indent.size() >= 2)
      this->indent = this->indent.substr(0, this->indent.size() - 2);
  }
};

}  // end namespace Factor

}  // end namespace Choreo

#endif // CHOREO_CODEGEN_FACTOR_HPP_
