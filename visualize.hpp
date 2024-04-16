#ifndef __CHOREO_VISUALIZE_DMA_HPP__
#define __CHOREO_VISUALIZE_DMA_HPP__

// This apply the type check and symbol table generation

#include "visitor.hpp"

namespace Choreo {

struct Visualizer : public VisitorWithSymTab {
 private:
  std::ostream &os;
  std::vector<AST::DMA *> statms;

 public:
  Visualizer(const ptr<SymbolTable> s_tab, std::ostream &o = std::cout)
      : VisitorWithSymTab(s_tab), os(o) {}
  ~Visualizer() {}

  // derived class must call this to incorporate with symbol table

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
  bool BeforeVisitImpl(AST::Node &) override { return true; }
  bool AfterVisitImpl(AST::Node &n) override {
    if (!isa<AST::Program>(&n)) return true;

    // TODO: post visiting AST::Program, render the picture
    return true;
  }
};

}  // end namespace Choreo

#endif  // __CHOREO_VISUALIZE_DMA_HPP__
