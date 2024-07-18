#ifndef __CHOREO_CODEGEN_HPP__
#define __CHOREO_CODEGEN_HPP__

#include "valbind.hpp"
#include "visitor.hpp"

namespace Choreo {

struct CodeGenerator : public VisitorWithSymTab {
  std::ostream &os;

  // some default method for the derived classes that do not want to override.
  bool BeforeVisitImpl(AST::Node &) override { return true; }
  bool AfterVisitImpl(AST::Node &) override { return true; }

  CodeGenerator(std::ostream &o, const ptr<SymbolTable> &symtab)
      : VisitorWithSymTab(symtab), os(o) {
    if (symtab == nullptr)
      choreo_unreachable("symbol table must be initialized.");
  }
};

struct FactorCodeGen : public CodeGenerator {
  // TODO: should the pointer be replaced?
  std::string current_fn = "";
  std::string entry_fn = "";
  std::string indent = "";
  std::vector<AST::ptr<AST::Parameter>> *cur_params = nullptr;
  AST::ptr<AST::DataType> current_output = nullptr;
  std::map<std::string, std::vector<std::string>> cur_bounded_vars;

  std::string bin_fn;  // temporal filename of factor binary
  int parallel_factor = 1;

  bool void_return = false;

  ValBind::BindInfo<std::string> bind_info;

  std::vector<std::unordered_set<std::string>> loop_vars;  // the loop variables
  bool ContainsLoopVar(const std::string &) const;

 private:
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

  // name suffix of factor function parameters
  size_t sp_count = 0;

  int parallel_level = 0;

  bool dyn_shaped = false;

  bool trace_visit = false;  // for debugging purpose only

  // mapping from a symbolic shape dimensions to the associated runtime name
  std::map<std::string, std::string>
      rts_nmap;  // symbolic name to the runtime name
  std::map<std::string, size_t>
      rts_pidx;  // shape index in parameter list for the runtime shape name
  std::map<std::string, size_t>
      rts_nidx;  // dim index in shape for the runtime shape name

  // runtime host parameter names
  std::vector<std::string> host_params;
  // parameters: the name (of factor data) and associated size expression
  std::vector<std::pair<std::string, std::string>> param_map;

  void EmitHostHead(std::ostream &);
  void EmitHostFuncDecl(std::ostream &, const Type &, const std::string &,
                        bool = false);
  void EmitRuntimeCheck(std::ostream &, const Type &);
  void EmitHostFuncBody(std::ostream &, const Type &, const std::string &fname,
                        const std::string &o_sz, const std::string &o_ty,
                        const Shape &s);

  std::string GenHostParamName() { return "hp" + std::to_string(sp_count++); }
  std::string ReplaceRuntimeNames(const std::string &, const std::string & = "",
                                  bool host_code = true);
  std::string ReplaceDynDimName(const std::string &);

 public:
  FactorCodeGen(std::ostream &os, const ptr<SymbolTable> &symtab)
      : CodeGenerator(os, symtab), trace_visit(std::getenv("TRACE_CODEGEN")) {}

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
  bool Visit(AST::DMA &) override;
  bool Visit(AST::ChunkAt &) override;
  bool Visit(AST::Wait &) override;
  bool Visit(AST::Call &) override;
  bool Visit(AST::Select &) override;
  bool Visit(AST::Return &) override;
  bool Visit(AST::ForeachBlock &) override;
  bool Visit(AST::FunctionDecl &) override;
  bool Visit(AST::ChoreoFunction &) override;
  bool Visit(AST::CppSourceCode &) override;
  bool Visit(AST::Program &) override;

  // common utils
  void incrementIndent() { this->indent += "  "; }

  void decrementIndent() {
    if (this->indent.size() >= 2)
      this->indent = this->indent.substr(0, this->indent.size() - 2);
  }
};

struct TopsccCodeGen : public CodeGenerator {
  // bool Visit(AST::Node&) override;

  bool Visit(AST::MultiNodes &) override { return true; };
  bool Visit(AST::MultiValues &) override { return true; };
  bool Visit(AST::IntLiteral &) override { return true; };
  bool Visit(AST::Expr &) override { return true; };
  bool Visit(AST::MultiDimSpans &) override { return true; };
  bool Visit(AST::NamedTypeDecl &) override { return true; };
  bool Visit(AST::NamedVariableDecl &) override { return true; };
  bool Visit(AST::IntTuple &) override { return true; };
  bool Visit(AST::Assignment &) override { return true; };
  bool Visit(AST::IntIndex &) override { return true; };
  bool Visit(AST::DataType &) override { return true; };
  bool Visit(AST::Identifier &) override { return true; };
  bool Visit(AST::Parameter &) override { return true; };
  bool Visit(AST::ParamList &) override { return true; };
  bool Visit(AST::ParallelBy &) override { return true; };
  bool Visit(AST::WhereBind &) override { return true; };
  bool Visit(AST::WithIn &) override { return true; };
  bool Visit(AST::WithBlock &) override { return true; };
  bool Visit(AST::Memory &) override { return true; };
  bool Visit(AST::DMA &) override { return true; };
  bool Visit(AST::ChunkAt &) override { return true; };
  bool Visit(AST::Wait &) override { return true; };
  bool Visit(AST::Call &) override { return true; };
  bool Visit(AST::Select &) override { return true; };
  bool Visit(AST::Return &) override { return true; };
  bool Visit(AST::ForeachBlock &) override { return true; };
  bool Visit(AST::FunctionDecl &) override { return true; };
  bool Visit(AST::ChoreoFunction &) override { return true; };
  bool Visit(AST::CppSourceCode &) override { return true; };
  bool Visit(AST::Program &) override { return true; };
};

}  // end namespace Choreo

#endif  // __CHOREO_CODEGEN_HPP__
