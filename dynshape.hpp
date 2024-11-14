#ifndef __CHOREO_CODEGEN_PREPARE_HPP__
#define __CHOREO_CODEGEN_PREPARE_HPP__

// This apply the type check and symbol table generation

#include "codegen.hpp"

namespace Choreo {

struct CodegenPrepare : public CodeGenerator {
private:
  size_t error_count = 0;

  ptr<CodeGenInfo> cgi;
  std::string fname; // current function name

private:
  bool BeforeVisitImpl(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      fname = f->name;
      cgi->storages[fname] = {};
    }
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) {
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(os << "Symbols in " << fname << ":\n");
      VST_DEBUG(for (auto& item : cgi->storages[fname]) {
        os << " |- " << item.name << ", ty: " << PSTR(item.type)
           << ", is_return: " << item.is_return << ", index: " << item.p_index
           << "\n";
      });
      fname = "";
    }
    return true;
  }

public:
  CodegenPrepare(const ptr<SymbolTable> s_tab, std::ostream& o = std::cout)
      : CodeGenerator("prepare", o, s_tab) {
    cgi = std::make_shared<CodeGenInfo>();
  }
  ~CodegenPrepare() {}

  const ptr<CodeGenInfo> GetASTInfo() { return cgi; }

  bool Visit(AST::MultiNodes&) { return true; }
  bool Visit(AST::MultiValues&) { return true; }
  bool Visit(AST::IntLiteral&) { return true; }
  bool Visit(AST::Boolean&) { return true; }
  bool Visit(AST::Expr&) { return true; }
  bool Visit(AST::MultiDimSpans&) { return true; }
  bool Visit(AST::NamedTypeDecl&) { return true; }
  bool Visit(AST::NamedVariableDecl& n) override {
    auto name = n.name_str;
    cgi->storages[fname].push_back(
        {InScopeName(name), GetSymbolType(name), false, -1});
    return true;
  }
  bool Visit(AST::IntTuple&) { return true; }
  bool Visit(AST::Assignment& n) override {
    auto name = n.name;
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value)) {
      cgi->storages[fname].push_back(
          {InScopeName(name), GetSymbolType(name), false, -1});
    }
    return true;
  }
  bool Visit(AST::IntIndex&) { return true; }
  bool Visit(AST::DataType&) { return true; }
  bool Visit(AST::Identifier&) { return true; }
  bool Visit(AST::Parameter&) { return true; }
  bool Visit(AST::ParamList& n) {
    int index = 0;
    for (auto param : n.values) {
      cgi->storages[fname].push_back(
          {InScopeName(param->sym->name), param->GetType(), false, index++});
    }
    return true;
  }
  bool Visit(AST::ParallelBy&) { return true; }
  bool Visit(AST::WhereBind&) { return true; }
  bool Visit(AST::WithIn&) { return true; }
  bool Visit(AST::WithBlock&) { return true; }
  bool Visit(AST::Memory&) { return true; }
  bool Visit(AST::SpanAs&) { return true; }
  bool Visit(AST::DMA&) { return true; }
  bool Visit(AST::ChunkAt&) { return true; }
  bool Visit(AST::Wait&) { return true; }
  bool Visit(AST::Call&) { return true; }
  bool Visit(AST::Rotate&) { return true; }
  bool Visit(AST::Select&) { return true; }

  bool Visit(AST::Return& n) override {
    auto id = GetIdentifier(*n.value);
    if (!id) return true;

    for (auto& item : cgi->storages[fname]) {
      if (item.name == InScopeName(id->name)) { item.is_return = true; }
    }
    return true;
  }
  bool Visit(AST::LoopRange&) { return true; }
  bool Visit(AST::ForeachBlock&) { return true; }
  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }

  bool HasError() { return false; }
};

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_PREPARE_HPP__
