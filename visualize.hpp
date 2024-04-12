#ifndef __CHOREO_VISUALIZE_DMA_HPP__
#define __CHOREO_VISUALIZE_DMA_HPP__

// This apply the type check and symbol table generation

#include "visitor.hpp"

namespace Choreo {

struct Visualizer : public Visitor {
 private:
  std::ostream &os;
  std::vector<AST::DMA *> statms;

 public:
  Visualizer(const ptr<SymbolTable> s_tab, std::ostream &o = std::cout)
      : Visitor(s_tab), os(o) {}
  ~Visualizer() {}

  // derived class must call this to incorporate with symbol table
  bool BeforeVisit(AST::Node &n) override {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      SSTab().EnterScope(f->name);
    } else if (isa<AST::ParallelBy>(&n)) {
      static size_t count = 0;
      SSTab().EnterScope("paraby_" + std::to_string(count++));
    } else if (isa<AST::WithBlock>(&n)) {
      static size_t count = 0;
      SSTab().EnterScope("within_" + std::to_string(count++));
    } else if (isa<AST::ForeachBlock>(&n)) {
      static size_t count = 0;
      SSTab().EnterScope("foreach_" + std::to_string(count++));
    }
    return true;
  }

  std::string InScopeName(const std::string &sym) {
    auto removeLastLevel = [](const std::string &input) -> std::string {
      size_t lastPos = input.rfind("::");
      if (lastPos == std::string::npos)
        return input;  // No "::" found, return the original string
      // Find the second-to-last "::" by searching up to the last found position
      size_t secondLastPos = input.rfind("::", lastPos - 1);
      if (secondLastPos == std::string::npos) return input;
      return input.substr(0,
                          secondLastPos + 2);  // Include the "::" in the result
    };
    std::string scope_name = SSTab().ScopeName();
    while (true) {
      std::string scoped_name = scope_name + sym;
      if (SymTab()->Exists(scoped_name)) return scoped_name;
      std::string stripped_scope = removeLastLevel(scope_name);
      if (stripped_scope == scope_name) break;
      scope_name = stripped_scope;
    }

    choreo_unreachable("unable to find symbol `" + sym +
                       "' in the symbol table.");
    return "";
  }

  virtual ptr<Type> GetSymbolType(const std::string &n) {
    return SymTab()->GetSymbol(InScopeName(n))->GetType();
  }

  bool Visit(AST::MultiNodes &) override { return true; }
  bool Visit(AST::MultiValues &) override { return true; }
  bool Visit(AST::IntLiteral &) override { return true; }
  bool Visit(AST::Expr &) override { return true; }
  bool Visit(AST::MultiDimSpans &) override { return true; }
  bool Visit(AST::NamedTypeDecl &) override { return true; }
  bool Visit(AST::NamedVariableDecl &) override { return true; }
  bool Visit(AST::IntTuple &) override { return true; }
  bool Visit(AST::Assignment &) override { return true; }
  bool Visit(AST::IntIndex &) override { return true; }
  bool Visit(AST::DataType &) override { return true; }
  bool Visit(AST::Identifier &) override { return true; }
  bool Visit(AST::Parameter &) override { return true; }
  bool Visit(AST::ParamList &) override { return true; }
  bool Visit(AST::ParallelBy &) override { return true; }
  bool Visit(AST::RequireBind &) override { return true; }
  bool Visit(AST::WithIn &) override { return true; }
  bool Visit(AST::WithBlock &) override { return true; }
  bool Visit(AST::Memory &) override { return true; }
  bool Visit(AST::DMA &n) override {
    statms.push_back(&n);
    auto *from = n.from.get();
    //auto *to = n.to.get();

    // TODO: collect any information for visualize
    if (auto ca = dyn_cast<AST::ChunkAt>(from)) {
      if (ca->positions) {
        for (auto pos : ca->positions->values) {
          auto id = dyn_cast<AST::Identifier>(pos);
          assert(id && "unhandled value.");
          std::cout << "name: " << id->name
                    << ", type: " << STR(*GetSymbolType(id->name)) << "\n";
        }
      }
      std::cout << "chuckat type: " << TYPE_STR(*from) << "\n";
    }
    return true;
  }
  bool Visit(AST::ChunkAt &) override { return true; }
  bool Visit(AST::Wait &) override { return true; }
  bool Visit(AST::Call &) override { return true; }
  bool Visit(AST::Return &) override { return true; }
  bool Visit(AST::ForeachBlock &) override { return true; }
  bool Visit(AST::FunctionDecl &) override { return true; }
  bool Visit(AST::ChoreoFunction &) override { return true; }
  bool Visit(AST::CppSourceCode &) override { return true; }
  bool Visit(AST::Program &) override { return true; }

 public:
  bool AfterVisit(AST::Node &n) {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
      SSTab().LeaveScope();
    }

    if (!isa<AST::Program>(&n)) return true;

    // TODO: post visiting AST::Program, render the picture
    return true;
  }
};

}  // end namespace Choreo

#endif  // __CHOREO_VISUALIZE_DMA_HPP__
