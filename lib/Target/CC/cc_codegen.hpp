#ifndef __CHOREO_CODEGEN_CC_HPP__
#define __CHOREO_CODEGEN_CC_HPP__

#include <iostream>
#include <sstream>
#include <stack>

#include "assess.hpp"
#include "ast.hpp"
#include "codegen.hpp"
#include "codegen_utils.hpp"
#include "context.hpp"
#include "operator_info.hpp"
#include "types.hpp"

using namespace Choreo;

namespace Choreo {

namespace CC {

using ScopedSymbolMap = ::Choreo::ScopedSymbolMap;

struct CCCodeGen : public CodeGenerator {
private:
  ScopedSymbolMap ssm;

public:
  CCCodeGen() : CodeGenerator("codegen") {
    cpp_name = "__choreo_" + OptionRegistry::GetInstance().GetInputName();
    cmp_dir = CreateUniquePath();
  }

  bool BeforeVisitImpl(AST::Node&) override;
  bool InMidVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::MultiNodes&) override { return true; }
  bool Visit(AST::MultiValues&) override { return true; }
  bool Visit(AST::IntLiteral&) override { return true; }
  bool Visit(AST::FloatLiteral&) override { return true; }
  bool Visit(AST::Expr&) override { return true; }
  bool Visit(AST::MultiDimSpans&) override { return true; }
  bool Visit(AST::NamedTypeDecl&) override { return true; }
  bool Visit(AST::IntTuple&) override { return true; }
  bool Visit(AST::IntIndex&) override { return true; }
  bool Visit(AST::DataType&) override { return true; }
  bool Visit(AST::Identifier&) override { return true; }
  bool Visit(AST::Parameter&) override { return true; }
  bool Visit(AST::Memory&) override { return true; }
  bool Visit(AST::ChunkAt&) override { return true; }
  bool Visit(AST::Select&) override { return true; }
  bool Visit(AST::LoopRange&) override { return true; }
  bool Visit(AST::Program&) override { return true; }

  bool Visit(AST::ParamList&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::InThreadsBlock&) override;
  bool Visit(AST::IfElseBlock&) override;
  bool Visit(AST::WhileBlock&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::MMA&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Trigger&) override;
  bool Visit(AST::Break&) override;
  bool Visit(AST::Continue&) override;
  bool Visit(AST::Yield&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Synchronize&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::CppSourceCode& n) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::Return&) override;

private:
  std::string cmp_dir;
  std::string cpp_name;

  std::string indent;

  std::stack<ParallelLevel> levels;
  ParallelLevel Level() const { return levels.top(); }

  ptr<FunctionType> fty = nullptr;
  bool void_return = false;

  std::ostringstream os; // single output stream
  std::vector<std::string> code_segments;

  void IncrIndent() { indent += "  "; }
  void DecrIndent() {
    if (indent.size() >= 2) indent = indent.substr(0, indent.size() - 2);
  }

  std::ostringstream& IndStream() {
    os << indent;
    return os;
  }

  void EmitPreamble();
  void EmitSource();
  void EmitScript(std::ostream& out, const std::string& exe_fn = "");
  bool CompileWithScript(const std::string& action);

  void EmitFuncDecl();
  void EmitEntryAssertions();
  void BuildSiteAssertionMap();
  void EmitPreSiteAssertions(AST::Node& n);
  void EmitPostSiteAssertions(AST::Node& n);

  std::unordered_map<AST::Node*, std::vector<Assertion>> pre_site_assertions;
  std::unordered_map<AST::Node*, std::vector<Assertion>> post_site_assertions;

  const std::string ExprSTR(AST::ptr<AST::Node> n) const;
  const std::string CallSTR(AST::Call& n) const;
  const std::string OpValueSTR(const ValueItem& vi,
                               const std::string& parent_op, bool is_left_child,
                               bool ll_suffix = false) const;
  const std::string ValueSTR(const ValueItem& vi, bool ll_suffix = false) const;
  const std::string ValueSTR(const ValueList& vl,
                             const std::string& sep = ", ") const;
  const std::string TypeSTR(const Type& ty) const;

  void ResetFunctionStates() {
    fty = nullptr;
    void_return = false;
    indent = "";
    pre_site_assertions.clear();
    post_site_assertions.clear();
  }
};

} // namespace CC

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_CC_HPP__
