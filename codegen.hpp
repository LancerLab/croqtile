#ifndef __CHOREO_CODEGEN_HPP__
#define __CHOREO_CODEGEN_HPP__

#include <thread>

#include "valbind.hpp"
#include "visitor.hpp"
#include "MemUsageCheck.hpp"

namespace Choreo {

// utility macros define here
#define __TRACE_EACH_VISIT__(d)       \
  if (trace_visit) {                  \
    os << d.TypeNameString() << ": "; \
    os << "\n";                       \
  }


// Codegenerators for targets
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

 // TODO merge with is_dest_passing_style
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

  std::vector<RtMemUsageCheckInfo> rt_mem_usage_check_list;

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
  void EmitRuntimeMemUsageCheck(std::ostream &, const Type &);
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
  FactorCodeGen(std::ostream &os, const ptr<SymbolTable> &symtab,
                const std::vector<RtMemUsageCheckInfo> &list)
      : CodeGenerator(os, symtab), rt_mem_usage_check_list(list),
        trace_visit(std::getenv("TRACE_CODEGEN")) {}

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
  bool Visit(AST::LoopRange &) override;
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
  bool Visit(AST::LoopRange &) override { return true; };
  bool Visit(AST::ForeachBlock &) override { return true; };
  bool Visit(AST::FunctionDecl &) override { return true; };
  bool Visit(AST::ChoreoFunction &) override { return true; };
  bool Visit(AST::CppSourceCode &) override { return true; };
  bool Visit(AST::Program &) override { return true; };
};

struct CUDACodeGen : public CodeGenerator {

  std::string current_fn = "";
  std::string entry_fn = "";
  std::string indent = "";
  std::vector<AST::ptr<AST::Parameter>> *cur_params = nullptr;
  AST::ptr<AST::DataType> current_output = nullptr;
  std::map<std::string, std::vector<std::string>> cur_bounded_vars;

  std::string bin_fn;  // temporal filename of factor binary
  int parallel_cuda = 1;

  bool void_return = false;
  bool host_enclosed = false;
 // TODO merge with is_dest_passing_style
  std::string return_string;

  ValBind::BindInfo<std::string> bind_info;

  std::vector<std::unordered_set<std::string>> loop_vars;  // the loop variables
  bool ContainsLoopVar(const std::string &) const;
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

 public:
  CUDACodeGen(std::ostream &os, const ptr<SymbolTable> &symtab)
      : CodeGenerator(os, symtab), trace_visit(std::getenv("TRACE_CODEGEN")) {}
  void ResetBuffers() {
    ks.clear();
    fs.clear();
    hs.clear();
  }

  bool BeforeVisitImpl(AST::Node &) override;
  bool AfterVisitImpl(AST::Node &) override;
  void OutputScript(FunctionType *, 
                    const std::string &, 
                    const std::string &,
                    const std::string &, 
                    const Shape &);

  bool Visit(AST::Assignment &) override;
  bool Visit(AST::Boolean &) override;
  bool Visit(AST::Call &) override;
  bool Visit(AST::ChoreoFunction &) override;
  bool Visit(AST::ChunkAt &) override;
  bool Visit(AST::CppSourceCode &) override;
  bool Visit(AST::DataType &) override;
  bool Visit(AST::DMA &) override;
  bool Visit(AST::Expr &) override;
  bool Visit(AST::LoopRange &) override;
  bool Visit(AST::ForeachBlock &) override;
  bool Visit(AST::FunctionDecl &) override;
  bool Visit(AST::Identifier &) override;
  bool Visit(AST::IntIndex &) override;
  bool Visit(AST::IntLiteral &) override;
  bool Visit(AST::IntTuple &) override;
  bool Visit(AST::Memory &) override;
  bool Visit(AST::MultiNodes &) override;
  bool Visit(AST::MultiValues &) override;
  bool Visit(AST::MultiDimSpans &) override;
  bool Visit(AST::NamedTypeDecl &) override;
  bool Visit(AST::NamedVariableDecl &) override;
  bool Visit(AST::Parameter &) override;
  bool Visit(AST::ParamList &) override;
  bool Visit(AST::ParallelBy &) override;
  bool Visit(AST::Program &) override;
  bool Visit(AST::Return &) override;
  bool Visit(AST::Select &) override;
  bool Visit(AST::Wait &) override;
  bool Visit(AST::WhereBind &) override;
  bool Visit(AST::WithIn &) override;
  bool Visit(AST::WithBlock &) override;

  void EmitHostHead(std::ostream &);
  void EmitHostFuncDecl(std::ostream &, 
                        const Type &, const std::string &,
                        bool = false);
  void EmitRuntimeCheck(std::ostream &, const Type &);
  void EmitHostFuncBody(std::ostream &, 
                        const Type &, 
                        const std::string &fname,
                        const std::string &o_sz, 
                        const std::string &o_ty,
                        const Shape &s);
  void EmitHostTail(std::ostream &);
  std::string GenHostParamName() { return "hp" + std::to_string(sp_count++); }
  std::string ReplaceRuntimeNames(const std::string &, const std::string & = "",
                                  bool host_code = true);
  std::string ReplaceDynDimName(const std::string &);
  std::string EmitTo(Target target);

  // common utils
  void incrementIndent() { this->indent += "  "; }

  void decrementIndent() {
    if (this->indent.size() >= 2)
      this->indent = this->indent.substr(0, this->indent.size() - 2);
  }
};

/////////////////////////////////////////////////////////////
///  Util functions shared between targets
/////////////////////////////////////////////////////////////

inline constexpr const char *backpatch_filename =
    "__choreo_kernel_file_name_that_will_be_back_patched_soon_ok_enough_i_am_"
    "bored__";


inline static std::string create_unique_path() {
  // Get a high-resolution timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();

  // Convert timestamp to a more granular unit, like nanoseconds
  auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

  // Get the thread or process ID
  std::stringstream ss;
  ss << std::this_thread::get_id();
  std::string thread_id = ss.str();

  // Construct the path
  std::string path = "/tmp/" + std::to_string(nanoseconds) + "_" + thread_id;

  return path;
}

inline static void ReplaceInString(std::string &str, const std::string &from,
                                   const std::string &to) {
  if (from.empty()) return;

  size_t startPos = 0;
  while ((startPos = str.find(from, startPos)) != std::string::npos) {
    str.replace(startPos, from.length(), to);
    startPos += to.length();  // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}


static inline std::string HostTypeString(const Choreo::Type &ty,
                                         bool is_ret = false) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret)  // return by value
      return "choreo::spanned_data<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + ">";
    else  // pass by reference
      return "const choreo::spanned_view<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + "> &";
  }
  choreo_unreachable("unsupported host function type.");
  return "";
}


}  // end namespace Choreo

#endif  // __CHOREO_CODEGEN_HPP__
