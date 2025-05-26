#ifndef __CHOREO_CODEGEN_CUDA_HPP__
#define __CHOREO_CODEGEN_CUDA_HPP__

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
#include "codegen.hpp"
#include "types.hpp"

namespace Choreo {

namespace CUDA {

inline constexpr const char* backpatch_filename =
    "__choreo_kernel_file_name_that_will_be_back_patched_soon_ok_enough_i_am_"
    "bored__";

struct CUDACodeGen : public CodeGenerator {
private:
  std::string current_fn = "";
  std::string entry_fn = "";
  std::string indent = "";
  std::string bin_fn; // temporal filename of factor binary
  std::string return_string;
  std::string host_fn;
  std::string target_fn;
  std::string build_path;
  // buffer of "alloc" statements in factor code
  std::ostringstream alloc_in_fs;
  std::string::size_type alloc_pos;
  std::string alloc_indent;
  // buffer the kernel code
  std::ostringstream ks;
  // buffer the factor code
  std::ostringstream fs;
  // buffer the host code
  std::ostringstream hs;
  // output variable name
  std::string output_v;

  int parallel_cuda = 1;
  bool void_return = false;
  bool host_enclosed = false;
  // name suffix of factor function parameters
  size_t sp_count = 0;
  int parallel_level = 0;
  bool dyn_shaped = false;
  bool cross_compile = false;

  std::vector<AST::ptr<AST::Parameter>>* cur_params = nullptr;
  AST::ptr<AST::DataType> current_output = nullptr;
  std::map<std::string, std::stack<std::vector<std::string>>> cur_bounded_vars;
  ValBind::BindInfo<std::string> bind_info;
  std::vector<std::unordered_set<std::string>> loop_vars; // the loop variables
  // runtime host parameter names
  std::vector<std::string> host_params;
  // parameters: the name (of factor data) and associated size expression
  std::vector<std::pair<std::string, std::string>> param_map;

  // mapping from a symbolic shape dimensions to the associated runtime name
  std::map<std::string, std::string>
      rts_nmap; // symbolic name to the runtime name
  std::map<std::string, size_t>
      rts_pidx; // shape index in parameter list for the runtime shape name
  std::map<std::string, size_t>
      rts_nidx; // dim index in shape for the runtime shape name
  StringifyTable cuda_symbols;

public:
  CUDACodeGen()
      : CodeGenerator("codegen", CCtx().GetGlobalSymbolTable()),
        cross_compile(CCtx().CrossCompile()) {}
  void ResetBuffers() {
    // ks.str("");
    // fs.str("");
    // hs.str("");
  }

  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;
  void OutputScript(const ptr<FunctionType>&, const std::string&,
                    const std::string&, const std::string&, const Shape&);

  bool Visit(AST::Assignment&) override;
  bool Visit(AST::Boolean&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::DataType&) override;
  bool Visit(AST::SpanAs&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::Identifier&) override;
  bool Visit(AST::IntIndex&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::IntTuple&) override;
  bool Visit(AST::Memory&) override;
  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::MultiDimSpans&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::Parameter&) override;
  bool Visit(AST::ParamList&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::Program&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WithBlock&) override;

private:
  bool ContainsLoopVar(const std::string&) const;
  void EmitHostHead(std::ostream&);
  void EmitHostFuncDecl(std::ostream&, const Type&, const std::string&,
                        bool = false);
  void EmitRuntimeCheck(std::ostream&, const Type&);
  void EmitHostFuncBody(std::ostream&, const Type&, const std::string& fname,
                        const std::string& o_sz, const std::string& o_ty,
                        const Shape& s);
  void EmitHostTail(std::ostream&);

  std::string GenHostParamName() { return "hp" + std::to_string(sp_count++); }
  std::string ReplaceRuntimeNames(const std::string&, const std::string& = "",
                                  bool host_code = true);
  std::string ReplaceDynDimName(const std::string&);

  // common utils
  inline void incrementIndent() { this->indent += "  "; }

  inline void decrementIndent() {
    if (this->indent.size() >= 2)
      this->indent = this->indent.substr(0, this->indent.size() - 2);
  }
};

} // end namespace CUDA

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_CUDA_HPP__
