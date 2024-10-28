#ifndef __CHOREO_LATE_NORM_HPP__
#define __CHOREO_LATE_NORM_HPP__

// This applies 'LATE_NORM' or 'canonicalization' of AST for shaped object

#include <iostream>
#include <tuple>

#include "symtab.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct LateNorm : public VisitorWithSymTab {
private:
  std::ostream& os;

private:
  bool changed = false;

  // for node hoisting
  using NodeInsertInfo =
      std::vector<std::tuple<int, ptr<AST::Node>, std::string>>;
  std::stack<AST::MultiNodes*> multi_nodes;
  int cur_dma_index = -1;
  std::map<AST::MultiNodes*, NodeInsertInfo> mnodes_insertions;

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) { os << n.TypeNameString() << ": " << STR(n) << "\n"; }
  }

public:
  // it does not require a symbol table
  LateNorm(const ptr<SymbolTable>& s_tab, std::ostream& o)
      : VisitorWithSymTab("latenorm", s_tab), os(o) {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (trace_visit) os << "before visiting " << n.TypeNameString() << "\n";

    if (auto m = dyn_cast<AST::MultiNodes>(&n)) {
      multi_nodes.push(m);
    } else if (auto d = dyn_cast<AST::DMA>(&n)) {
      cur_dma_index = multi_nodes.top()->GetIndex(d);
      assert(cur_dma_index != -1 && "unexpected node index.");
    }

    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (trace_visit) os << "after visiting " << n.TypeNameString() << "\n";
    return true;
  }

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);

    // insert the node at the given place
    assert(&n == multi_nodes.top());
    for (auto item : mnodes_insertions[&n]) {
      auto& index = std::get<0>(item);
      auto& pnode = std::get<1>(item);
      auto& sname = std::get<2>(item);

      n.values.insert(n.values.begin() + index, pnode);
      SymTab()->AddSymbol(SSTab().ScopedName(sname), pnode->GetType());
      VST_DEBUG(os << "Hoisted: " << PSTR(pnode)
                   << ", type: " << PSTR(pnode->GetType()) << "\n");
    }

    mnodes_insertions.erase(&n);
    multi_nodes.pop();
    cur_dma_index = -1;

    return true;
  }

  bool Visit(AST::MultiValues&) override { return true; }
  bool Visit(AST::IntLiteral&) override { return true; }
  bool Visit(AST::Boolean&) override { return true; }
  bool Visit(AST::Expr&) override { return true; }
  bool Visit(AST::MultiDimSpans&) override { return true; }
  bool Visit(AST::NamedTypeDecl&) override { return true; }
  bool Visit(AST::NamedVariableDecl&) override { return true; }
  bool Visit(AST::IntTuple&) override { return true; }
  bool Visit(AST::Assignment&) override { return true; }
  bool Visit(AST::IntIndex&) override { return true; }
  bool Visit(AST::DataType&) override { return true; }
  bool Visit(AST::Identifier&) override { return true; }
  bool Visit(AST::Parameter&) override { return true; }
  bool Visit(AST::ParamList&) override { return true; }
  bool Visit(AST::ParallelBy&) override { return true; }
  bool Visit(AST::WhereBind&) override { return true; }
  bool Visit(AST::WithIn&) override { return true; }
  bool Visit(AST::WithBlock&) override { return true; }
  bool Visit(AST::Memory&) override { return true; }
  bool Visit(AST::SpanAs&) override { return true; }

  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);

    if (n.operation == ".any") return true;
    if (!isa<AST::Memory>(n.to)) return true;

    // hoist the anonymous memory "=> local"
    auto ty = NodeType(n);
    auto shape = cast<FutureType>(ty)->GetShape();
    auto fty = GetSpannedType(NodeType(*cast<AST::ChunkAt>(n.from)->data));
    assert(fty);
    auto sty = MakeSpannedType(fty->ElementType(), shape,
                               cast<AST::Memory>(n.to)->Get());

    auto BUFFER_SUFFIX = "__buf__"; // hope user not name buffer this way
    // Note: Later passes only cares about the type. So it is possible to ignore
    // the syntax struct 'DataType'.
    auto anon_sym = (n.future.empty()) ? SymbolTable::GetAnonName()
                                       : n.future + BUFFER_SUFFIX;
    auto var = AST::Make<AST::NamedVariableDecl>(n.to->LOC(), anon_sym);
    var->SetType(sty);

    VST_DEBUG(os << "Replace: " << STR(n) << "\n");

    n.to = AST::Make<AST::ChunkAt>(
        n.to->LOC(), AST::Make<AST::Identifier>(n.to->LOC(), anon_sym));
    n.to->SetType(sty);

    VST_DEBUG(os << "with: " << STR(n) << ".\n");

    assert(cur_dma_index != -1);
    int index = cur_dma_index + mnodes_insertions[multi_nodes.top()].size();
    mnodes_insertions[multi_nodes.top()].emplace_back(
        std::make_tuple(index, var, anon_sym));

    return true;
  }

  bool Visit(AST::ChunkAt&) override { return true; }
  bool Visit(AST::Wait&) override { return true; }
  bool Visit(AST::Call&) override { return true; }
  bool Visit(AST::Swap&) override { return true; }
  bool Visit(AST::Select&) override { return true; }
  bool Visit(AST::Return&) override { return true; }
  bool Visit(AST::LoopRange&) override { return true; }
  bool Visit(AST::ForeachBlock&) override { return true; }
  bool Visit(AST::FunctionDecl&) override { return true; }
  bool Visit(AST::ChoreoFunction&) override { return true; }
  bool Visit(AST::CppSourceCode&) override { return true; }
  bool Visit(AST::Program&) override { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_LATE_NORM_HPP__
