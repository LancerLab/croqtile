#ifndef __CHOREO_TOPSCC_TRANS_HPP__
#define __CHOREO_TOPSCC_TRANS_HPP__

#include "visitor.hpp"

namespace Choreo {

struct TopsccTrans : public VisitorWithSymTab {
private:
  struct BufferInsertionInfo {
    int index = -1;
    ptr<AST::Node> node = nullptr;
    std::string sname; // buffer name
    std::string fname; // corresponding future name
  };
  using BufferInsertInfo = std::vector<BufferInsertionInfo>;
  std::stack<AST::MultiNodes*> multi_nodes;
  int cur_node_index = -1;
  std::map<AST::MultiNodes*, BufferInsertInfo> mnodes_insertions;

  FutureBufferInfo& FBInfo() { return FCtx(fname).GetFutureBufferInfo(); }

private:
  bool BeforeVisitImpl(AST::Node& n) {
    //    TraceEachVisit(n, "Before ");
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "Before " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    } else if (auto m = dyn_cast<AST::MultiNodes>(&n)) {
      multi_nodes.push(m);
      cur_node_index = -1;
    } else if (auto d = dyn_cast<AST::NamedVariableDecl>(&n)) {
      cur_node_index = multi_nodes.top()->GetIndex(d);
      assert(cur_node_index != -1 && "unexpected node index.");
    } else if (auto d = dyn_cast<AST::Assignment>(&n)) {
      cur_node_index = multi_nodes.top()->GetIndex(d);
      assert(cur_node_index != -1 && "unexpected node index.");
    }

    return true;
  }

  bool AfterVisitImpl(AST::Node& n) {
    //    TraceEachVisit(n, "After ");
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "After " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    }
    return true;
  }

public:
  TopsccTrans() : VisitorWithSymTab("otran", CCtx().GetGlobalSymbolTable()) {}
  ~TopsccTrans() {}

  void TraceEachVisit(AST::Node& n, const std::string& m = "") const {
    if (trace_visit) dbgs() << m << n.TypeNameString() << "\n";
  }

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);

    // insert the node at the given place
    assert(&n == multi_nodes.top());
    for (auto item : mnodes_insertions[&n]) {
      n.values.insert(n.values.begin() + item.index, item.node);
      auto sname = SSTab().ScopedName(item.sname);
      SymTab()->AddSymbol(sname, item.node->GetType());
      VST_DEBUG(dbgs() << "Hoisted: " << PSTR(item.node)
                       << ", type: " << PSTR(item.node->GetType()) << "\n");
      FBInfo()[item.fname].buffer = sname;
      VST_DEBUG(dbgs() << "Updated: " << STR(*FBInfo().find(item.fname))
                       << "\n");
    }

    mnodes_insertions.erase(&n);
    multi_nodes.pop();
    cur_node_index = -1;

    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) {
    TraceEachVisit(n);

    if (!AST::typeof<FutureType>(&n)) return true;
    if (!isa<AST::Select>(n.init_expr)) return true;

    auto sel = cast<AST::Select>(n.init_expr);
    auto buffer_list = AST::Make<AST::MultiValues>(n.LOC(), ", ");
    auto bty = cast<FutureType>(n.GetType())->GetSpannedType();
    for (auto& fid : sel->expr_list->AllValues()) {
      assert(GetIdentifier(*fid) && "expecting an identifier");
      auto fut_name = InScopeName(GetIdentifier(*fid)->name);
      assert(FBInfo().count(fut_name));
      auto buf_name = UnScopedName(FBInfo()[fut_name].buffer);
      auto bid = AST::Make<AST::Identifier>(fid->LOC(), buf_name);
      bid->SetType(bty);
      buffer_list->Append(bid);
    }
    auto buf_select =
        AST::Make<AST::Select>(n.LOC(), sel->select_factor, buffer_list);
    buf_select->SetType(bty);
    auto buf_name = SymbolTable::GetAnonName();
    auto buf_assign = AST::Make<AST::Assignment>(n.LOC(), buf_name, buf_select);
    buf_assign->SetType(bty);

    // The scope of the newly created buffer symbol is unknown. The map will be
    // modified later to reflect this.
    assert(cur_node_index != -1);
    int index = cur_node_index + mnodes_insertions[multi_nodes.top()].size();
    mnodes_insertions[multi_nodes.top()].push_back(
        {index, buf_assign, buf_name, InScopeName(n.name_str)});

    return true;
  }

  bool Visit(AST::Assignment& n) {
    TraceEachVisit(n);

    if (!AST::typeof<FutureType>(&n)) return true;
    if (!isa<AST::Select>(n.value)) return true;

    auto sel = cast<AST::Select>(n.value);

    auto buffer_list = AST::Make<AST::MultiValues>(n.LOC(), ", ");
    auto bty = cast<FutureType>(n.GetType())->GetSpannedType();
    for (auto& fid : sel->expr_list->AllValues()) {
      assert(GetIdentifier(*fid) && "expecting an identifier");
      auto fut_name = InScopeName(GetIdentifier(*fid)->name);
      assert(FBInfo().count(fut_name));
      auto buf_name = FBInfo()[fut_name].buffer;
      auto bid = AST::Make<AST::Identifier>(fid->LOC(), UnScopedName(buf_name));
      bid->SetType(bty);
      buffer_list->Append(bid);
    }
    auto buf_select =
        AST::Make<AST::Select>(n.LOC(), sel->select_factor, buffer_list);
    auto buf_name = SymbolTable::GetAnonName();
    SymTab()->AddSymbol(GetScope(n.name) + buf_name, bty);
    auto buf_assign = AST::Make<AST::Assignment>(n.LOC(), buf_name, buf_select);
    buf_assign->SetType(bty);
    buf_select->SetType(bty);

    FBInfo()[InScopeName(n.name)].buffer = InScopeName(buf_name);
    VST_DEBUG(dbgs() << "Updated: " << STR(*FBInfo().find(InScopeName(n.name)))
                     << "\n");

    return true;
  }

public:
  bool RunOnProgram(AST::Node& root) override {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    if (prt_visitor) dbgs() << "|- " << GetName() << NewL;
    root.accept(*this);
    if (prt_visitor) dbgs() << " |- TransformSwap" << NewL;

    if (HasError()) return false;

    if (abend_after) return false;

    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_TOPSCC_TRANS_HPP__
