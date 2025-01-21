#ifndef __CHOREO_FACTOR_TRANS_HPP__
#define __CHOREO_FACTOR_TRANS_HPP__

// It is necessary to do target specific transformation to make
// code generation possible. For Factor, it includes:
//
//  1. Generate buffers for future selections
//  1. Generated future selections for SWAP handling
//

#include "visitor.hpp"

namespace Choreo {

constexpr const char* SWAP_SFX_PRE = "_pre_swap__";
constexpr const char* SWAP_SFX_POS = "_post_swap__";

inline static std::string SymbolOfSameScope(const std::string& scopedName,
                                            const std::string& newSymbol) {
  // Find the last occurrence of "::"
  size_t pos = scopedName.rfind("::");

  if (pos == std::string::npos)
    choreo_unreachable("no scope is found for the symbol.");

  // Otherwise, append the new symbol in the current scope
  std::string newScopedName = scopedName.substr(0, pos + 2) + newSymbol;

  return newScopedName;
}

struct FactorTrans : public VisitorWithSymTab {
  // actually it forces to write two passes into a single visitor
  // TODO: should we make them two different passes?
  enum class Kind { T_NONE, T_SWAP, T_SELECT };

private:
  Kind kind = Kind::T_NONE;

  // For SWAP codegen
  std::stack<bool> replace_swap_names;

  std::vector<AST::Rotate*> cur_swaps;
  std::unordered_map<AST::Rotate*, std::unordered_map<std::string, std::string>>
      swap_pre;
  std::unordered_map<AST::Rotate*, std::unordered_map<std::string, std::string>>
      swap_post;

  void AddSwapPre(AST::Rotate* swap, const std::string& old,
                  const std::string& pre) {
    swap_pre[swap].emplace(old, pre);
    VST_DEBUG(dbgs() << "[SwapPre] " << PSTR(swap) << ": " << old << " -> "
                     << pre << "\n");
  }

  void AddSwapPost(AST::Rotate* swap, const std::string& old,
                   const std::string& post) {
    swap_post[swap].emplace(old, post);
    VST_DEBUG(dbgs() << "[SwapPost] " << PSTR(swap) << ": " << old << " -> "
                     << post << "\n");
  }

  const std::string NameToReplace(const std::string& name,
                                  const AST::Node& n) const {
    assert(!PrefixedWith(name, "::") && "expect a un-scoped name");

    auto res = name;
    auto sname = InScopeName(name);
    for (auto& item : swap_pre)
      if (item.second.count(sname)) {
        res = UnScopedName(item.second.at(sname));
        break;
      }

    if (res == name) {
      for (auto& item : swap_post)
        if (item.second.count(sname)) {
          res = UnScopedName(item.second.at(name));
          break;
        }
    }

    if (res != name)
      VST_DEBUG(dbgs() << "Replace '" << name << "' to be '" << res << "' in "
                       << STR(n) << ".\n");

    return res;
  }

  // for SELECT codegen
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
    TraceEachVisit(n, "Before ");
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "Before " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    } else if (auto f = dyn_cast<AST::ForeachBlock>(&n)) {
      if (kind == Kind::T_SWAP) {
        for (auto& stmt : f->stmts->AllSubs())
          if (auto swap = dyn_cast<AST::Rotate>(stmt))
            cur_swaps.push_back(swap.get());
        replace_swap_names.push(true);
      }
    } else if (auto m = dyn_cast<AST::MultiNodes>(&n)) {
      if (kind == Kind::T_SELECT) {
        multi_nodes.push(m);
        cur_node_index = -1;
      }
    } else if (auto d = dyn_cast<AST::NamedVariableDecl>(&n)) {
      if (kind == Kind::T_SELECT) {
        cur_node_index = multi_nodes.top()->GetIndex(d);
        assert(cur_node_index != -1 && "unexpected node index.");
      }
    } else if (auto d = dyn_cast<AST::Assignment>(&n)) {
      if (kind == Kind::T_SELECT) {
        cur_node_index = multi_nodes.top()->GetIndex(d);
        assert(cur_node_index != -1 && "unexpected node index.");
      }
    } else if (auto m = dyn_cast<AST::Select>(&n)) {
      if (kind == Kind::T_SWAP) {
        if (m->GetNote() == "gen") { replace_swap_names.push(false); }
      }
    }

    return true;
  }

  bool AfterVisitImpl(AST::Node& n) {
    TraceEachVisit(n, "After ");
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "After " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    } else if (isa<AST::ForeachBlock>(&n)) {
      if (kind == Kind::T_SWAP) {
        cur_swaps.clear();
        swap_pre.clear();
        swap_post.clear();
        assert(!replace_swap_names.empty());
        replace_swap_names.pop();
      }
    } else if (auto m = dyn_cast<AST::Select>(&n)) {
      if (kind == Kind::T_SWAP) {
        if (m->GetNote() == "gen") {
          assert(!replace_swap_names.empty());
          replace_swap_names.pop();
        }
      }
    }
    return true;
  }

public:
  FactorTrans() : VisitorWithSymTab("ftran", CCtx().GetGlobalSymbolTable()) {}
  ~FactorTrans() {}

  void SetKind(Kind k) { kind = k; }

  void TraceEachVisit(AST::Node& n, const std::string& m = "") const {
    if (trace_visit) dbgs() << m << n.TypeNameString() << "\n";
  }

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);

    if (kind != Kind::T_SELECT) return true;

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

  bool Visit(AST::MultiValues&) { return true; }
  bool Visit(AST::IntLiteral&) { return true; }
  bool Visit(AST::FloatLiteral&) { return true; }
  bool Visit(AST::Boolean&) { return true; }

  bool Visit(AST::Expr& n) {
    TraceEachVisit(n);

    if (kind == Kind::T_NONE) return true;
    if (n.op != "dataof") return true;

    auto id = dyn_cast<AST::Expr>(n.GetR())->GetSymbol();
    if (!id) return true;

    auto fut_name = InScopeName(id->name);
    if (!FBInfo().count(fut_name)) return true;

    VST_DEBUG(dbgs() << "Replace: " << STR(n) << "\nWith: ");

    auto rexp = cast<AST::Expr>(n.GetR());
    n.OverWrite(*rexp);
    n.GetSymbol()->name = UnScopedName(FBInfo()[fut_name].buffer);

    VST_DEBUG(dbgs() << STR(n) << "\n");

    return true;
  }

  bool Visit(AST::MultiDimSpans&) { return true; }
  bool Visit(AST::NamedTypeDecl&) { return true; }

  bool Visit(AST::NamedVariableDecl& n) {
    TraceEachVisit(n);

    if (kind != Kind::T_SELECT) return true;

    if (!AST::istypeof<FutureType>(&n)) return true;
    if (!isa<AST::Select>(n.init_expr)) return true;

    // do not care about the swap generated one
    if (SuffixedWith(n.name_str, SWAP_SFX_PRE)) return true;
    if (SuffixedWith(n.name_str, SWAP_SFX_POS)) return true;

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

  bool Visit(AST::IntTuple&) { return true; }

  bool Visit(AST::Assignment& n) {
    TraceEachVisit(n);

    if (kind != Kind::T_SELECT) return true;

    if (!AST::istypeof<FutureType>(&n)) return true;
    if (!isa<AST::Select>(n.value)) return true;

    // do not care about the swap generated one
    if (SuffixedWith(n.name, SWAP_SFX_PRE)) return true;
    if (SuffixedWith(n.name, SWAP_SFX_POS)) return true;

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

  bool Visit(AST::IntIndex&) { return true; }
  bool Visit(AST::DataType&) { return true; }

  bool Visit(AST::Identifier& n) {
    TraceEachVisit(n);

    if (replace_swap_names.empty()) return true;

    if (replace_swap_names.top()) { n.name = NameToReplace(n.name, n); }

    return true;
  }

  bool Visit(AST::Parameter&) { return true; }
  bool Visit(AST::ParamList&) { return true; }
  bool Visit(AST::ParallelBy&) { return true; }
  bool Visit(AST::WhereBind&) { return true; }
  bool Visit(AST::WithIn&) { return true; }
  bool Visit(AST::WithBlock&) { return true; }
  bool Visit(AST::Memory&) { return true; }
  bool Visit(AST::SpanAs&) { return true; }

  bool Visit(AST::DMA& n) {
    TraceEachVisit(n);

    if (kind != Kind::T_SWAP) return true;
    if (!AST::istypeof<FutureType>(&n)) return true;

    if (!n.future.empty()) {
      auto fut = n.future;
      n.future = NameToReplace(fut, n);
      if (fut != n.future) n.SetNote("use-fut");
    }

    return true;
  }

  bool Visit(AST::ChunkAt& n) {
    TraceEachVisit(n);

    if (kind != Kind::T_SWAP) return true;

    auto data_name = n.RefSymbol();
    n.data->name = NameToReplace(data_name, n);

    auto sname = InScopeName(n.data->name);
    // also replace future with the corresponding buffer
    if (FBInfo().count(sname))
      n.data->name = UnScopedName(FBInfo()[sname].buffer);

    return true;
  }

  bool Visit(AST::Wait&) { return true; }
  bool Visit(AST::Call&) { return true; }
  bool Visit(AST::Rotate& n) {
    TraceEachVisit(n);
    if (kind != Kind::T_SWAP) return true;
    if (swap_pre.count(&n)) swap_pre.erase(&n);
    return true;
  }
  bool Visit(AST::Select&) { return true; }
  bool Visit(AST::Return&) { return true; }
  bool Visit(AST::LoopRange&) { return true; }

  bool Visit(AST::ForeachBlock& n) {
    TraceEachVisit(n);

    if (kind != Kind::T_SWAP) return true;
    if (cur_swaps.empty()) return true;
    if (n.ranges->Count() > 1)
      choreo_unreachable("swapping inside multi-bounds is yet to support.");

    auto& ranges = n.GetRanges();
    auto iv_name = cast<AST::LoopRange>(ranges[0])->iv->name;
    auto lbound = cast<AST::LoopRange>(ranges[0])->lbound;
    if (lbound == nullptr) lbound = AST::Make<AST::IntLiteral>(n.LOC(), 0);

    // ((iv - lb) % 2 + 2) % 2
    auto Condition = AST::Make<AST::Expr>(
        n.LOC(), "%",
        AST::Make<AST::Expr>(
            n.LOC(), "+",
            AST::Make<AST::Expr>(
                n.LOC(), "%",
                AST::Make<AST::Expr>(
                    n.LOC(), "-", AST::Make<AST::Identifier>(n.LOC(), iv_name),
                    AST::Make<AST::Expr>(n.LOC(), lbound) /*end -*/),
                AST::Make<AST::Expr>(
                    n.LOC(), AST::Make<AST::IntLiteral>(n.LOC(), 2)) /*end %*/),
            AST::Make<AST::Expr>(
                n.LOC(), AST::Make<AST::IntLiteral>(n.LOC(), 2)) /*end +*/),
        AST::Make<AST::Expr>(n.LOC(),
                             AST::Make<AST::IntLiteral>(n.LOC(), 2)) /*end %*/);
    Condition->SetType(MakeIntegerType());

    std::vector<ptr<AST::Node>> new_stmts;

    // NOTE: must take care of the symbols and associated types
    for (auto swap : cur_swaps) {
      // generate selections on futures
      auto nty = NodeType(*swap->ValueAt(0));
      auto fty = cast<FutureType>(nty);
      auto sty = fty->GetSpannedType();
      auto lname = InScopeName(swap->IdAt(0)->name);
      auto rname = InScopeName(swap->IdAt(1)->name);
      auto lr_list = AST::Make<AST::MultiValues>(
          n.LOC(), ", ",
          AST::Make<AST::Identifier>(swap->IdAt(0)->LOC(), swap->IdAt(0)->name),
          AST::Make<AST::Identifier>(swap->IdAt(1)->LOC(),
                                     swap->IdAt(1)->name));
      auto rl_list = AST::Make<AST::MultiValues>(
          n.LOC(), ", ",
          AST::Make<AST::Identifier>(swap->IdAt(1)->LOC(), swap->IdAt(1)->name),
          AST::Make<AST::Identifier>(swap->IdAt(0)->LOC(),
                                     swap->IdAt(0)->name));
      auto true_on_lhs = AST::Make<AST::Select>(n.LOC(), Condition, lr_list);
      auto true_on_rhs = AST::Make<AST::Select>(n.LOC(), Condition, rl_list);
      true_on_lhs->SetType(nty);
      true_on_rhs->SetType(nty);
      true_on_lhs->SetNote("gen");
      true_on_rhs->SetNote("gen");
      auto lbs = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(lname + SWAP_SFX_PRE), true_on_lhs);
      auto las = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(lname + SWAP_SFX_POS), true_on_rhs);
      auto rbs = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(rname + SWAP_SFX_PRE), true_on_rhs);
      auto ras = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(rname + SWAP_SFX_POS), true_on_lhs);
      // mark it as generated
      lbs->SetType(nty);
      las->SetType(nty);
      rbs->SetType(nty);
      ras->SetType(nty);

      // now generate the buffer (associated with future) selections
      auto lbuf_name = FBInfo()[lname].buffer;
      auto rbuf_name = FBInfo()[rname].buffer;
      assert(!lbuf_name.empty());
      assert(!rbuf_name.empty());
      auto lbuf_id =
          AST::Make<AST::Identifier>(n.LOC(), UnScopedName(lbuf_name));
      auto rbuf_id =
          AST::Make<AST::Identifier>(n.LOC(), UnScopedName(rbuf_name));
      auto lr_buf_list =
          AST::Make<AST::MultiValues>(n.LOC(), ", ", lbuf_id, rbuf_id);
      auto rl_buf_list =
          AST::Make<AST::MultiValues>(n.LOC(), ", ", rbuf_id, lbuf_id);
      auto true_on_lbuf =
          AST::Make<AST::Select>(n.LOC(), Condition, lr_buf_list);
      auto true_on_rbuf =
          AST::Make<AST::Select>(n.LOC(), Condition, rl_buf_list);
      true_on_lbuf->SetType(sty);
      true_on_rbuf->SetType(sty);
      true_on_lbuf->SetNote("gen");
      true_on_rbuf->SetNote("gen");
      auto lbs_buf = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(lbuf_name + SWAP_SFX_PRE), true_on_lbuf);
      auto las_buf = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(lbuf_name + SWAP_SFX_POS), true_on_rbuf);
      auto rbs_buf = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(rbuf_name + SWAP_SFX_PRE), true_on_rbuf);
      auto ras_buf = AST::Make<AST::Assignment>(
          n.LOC(), UnScopedName(rbuf_name + SWAP_SFX_POS), true_on_lbuf);
      lbs_buf->SetType(sty);
      las_buf->SetType(sty);
      rbs_buf->SetType(sty);
      ras_buf->SetType(sty);

      // record the name mapping
      AddSwapPre(swap, lname, lname + SWAP_SFX_PRE);
      AddSwapPre(swap, rname, rname + SWAP_SFX_PRE);
      AddSwapPost(swap, lname, lname + SWAP_SFX_POS);
      AddSwapPost(swap, rname, rname + SWAP_SFX_POS);

      AddSwapPre(swap, lbuf_name, lbuf_name + SWAP_SFX_PRE);
      AddSwapPre(swap, rbuf_name, rbuf_name + SWAP_SFX_PRE);
      AddSwapPost(swap, lbuf_name, lbuf_name + SWAP_SFX_POS);
      AddSwapPost(swap, rbuf_name, rbuf_name + SWAP_SFX_POS);

      // modify the symbol table
      SymTab()->AddSymbol(lname + SWAP_SFX_PRE, nty);
      SymTab()->AddSymbol(rname + SWAP_SFX_PRE, nty);
      SymTab()->AddSymbol(lname + SWAP_SFX_POS, nty);
      SymTab()->AddSymbol(rname + SWAP_SFX_POS, nty);

      SymTab()->AddSymbol(lbuf_name + SWAP_SFX_PRE, sty);
      SymTab()->AddSymbol(rbuf_name + SWAP_SFX_PRE, sty);
      SymTab()->AddSymbol(lbuf_name + SWAP_SFX_POS, sty);
      SymTab()->AddSymbol(rbuf_name + SWAP_SFX_POS, sty);

      // modify the future buffer map
      FBInfo()[lname + SWAP_SFX_PRE].buffer = lbuf_name + SWAP_SFX_PRE;
      FBInfo()[rname + SWAP_SFX_PRE].buffer = rbuf_name + SWAP_SFX_PRE;
      FBInfo()[lname + SWAP_SFX_POS].buffer = lbuf_name + SWAP_SFX_POS;
      FBInfo()[rname + SWAP_SFX_POS].buffer = rbuf_name + SWAP_SFX_POS;

      VST_DEBUG(dbgs() << "Updated: "
                       << STR(*FBInfo().find(lname + SWAP_SFX_PRE)) << "\n");
      VST_DEBUG(dbgs() << "Updated: "
                       << STR(*FBInfo().find(rname + SWAP_SFX_PRE)) << "\n");
      VST_DEBUG(dbgs() << "Updated: "
                       << STR(*FBInfo().find(lname + SWAP_SFX_POS)) << "\n");
      VST_DEBUG(dbgs() << "Updated: "
                       << STR(*FBInfo().find(rname + SWAP_SFX_POS)) << "\n");

      // Add it into the new stmts
      new_stmts.push_back(lbs);
      new_stmts.push_back(las);
      new_stmts.push_back(rbs);
      new_stmts.push_back(ras);

      new_stmts.push_back(lbs_buf);
      new_stmts.push_back(las_buf);
      new_stmts.push_back(rbs_buf);
      new_stmts.push_back(ras_buf);
    }

    n.stmts->values.insert(n.stmts->values.begin(), new_stmts.begin(),
                           new_stmts.end());

    return true;
  }

  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }

  bool RunOnProgram(AST::Node& root) override {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    if (prt_visitor) dbgs() << "|- " << GetName() << NewL;

    if (prt_visitor) dbgs() << " |- TransformSelect" << NewL;
    SetKind(FactorTrans::Kind::T_SELECT);
    root.accept(*this);
    if (HasError()) return false;

    if (prt_visitor) dbgs() << " |- TransformSwap" << NewL;
    SetKind(FactorTrans::Kind::T_SWAP);
    root.accept(*this);
    if (HasError()) return false;

    if (abend_after) return false;

    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_FACTOR_TRANS_HPP__
