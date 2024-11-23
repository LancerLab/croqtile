#ifndef __CHOREO_LATE_NORM_HPP__
#define __CHOREO_LATE_NORM_HPP__

// This applies 'LATE_NORM' or 'canonicalization' of AST for shaped object

#include <iostream>
#include <tuple>

#include "symtab.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

// Auxillary structures for buffer generation
struct BufferInsertionInfo {
  int index = -1;
  ptr<AST::Node> node = nullptr;
  std::string sname;        // buffer name
  std::string fname;        // corresponding future name
  AST::Node* dma = nullptr; // if it requires dma to modification
};

using BufferInsertInfo = std::vector<BufferInsertionInfo>;

struct LateNormBase : public VisitorWithSymTab {
protected:
  std::stack<AST::MultiNodes*> multi_nodes;
  int cur_dma_index = -1;
  int cur_pb_index = -1;
  AST::MultiNodes* cur_pb_mn = nullptr;
  std::map<AST::MultiNodes*, BufferInsertInfo> mnodes_insertions;

  FutureBufferInfo& FBInfo() { return FCtx(fname).GetFutureBufferInfo(); }

  const std::string ProperBufferName(const std::string name) {
    auto BUFFER_SUFFIX =
        "__buf__"; // hope the users would not name symbols in this way
    return (name.empty()) ? SymbolTable::GetAnonName() : name + BUFFER_SUFFIX;
  }

protected:
  bool BeforeVisitImpl(AST::Node& n) {
    if (trace_visit) dbgs() << "before visiting " << n.TypeNameString() << "\n";
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "Before " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    } else if (auto m = dyn_cast<AST::MultiNodes>(&n)) {
      multi_nodes.push(m);
    } else if (auto m = dyn_cast<AST::ParallelBy>(&n)) {
      cur_pb_index = multi_nodes.top()->GetIndex(m);
      cur_pb_mn = multi_nodes.top();
      assert(cur_pb_index != -1 && "unexpected node index.");
    } else if (auto d = dyn_cast<AST::DMA>(&n)) {
      cur_dma_index = multi_nodes.top()->GetIndex(d);
      assert(cur_dma_index != -1 && "unexpected node index.");
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) {
    if (trace_visit) dbgs() << "after visiting " << n.TypeNameString() << "\n";

    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "After " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    } else if (isa<AST::ParallelBy>(&n)) {
      cur_pb_index = -1;
      cur_pb_mn = nullptr;
    }
    return true;
  }

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) { dbgs() << n.TypeNameString() << ": " << STR(n) << "\n"; }
  }

public:
  LateNormBase(const ptr<SymbolTable> s_tab, const std::string& pn)
      : VisitorWithSymTab(pn, s_tab) {}
  ~LateNormBase() {}

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);

    // insert the node at the given place
    assert(&n == multi_nodes.top());
    for (auto item : mnodes_insertions[&n]) {
      n.values.insert(n.values.begin() + item.index, item.node);

      auto ity = item.node->GetType();
      SSTab().DefineSymbol(item.sname, ity);
      auto sname = InScopeName(item.sname);
      VST_DEBUG(dbgs() << "Hoisted: " << PSTR(item.node)
                       << ", type: " << PSTR(ity) << "\n");

      // update the future-buffer information
      FBInfo()[item.fname].buffer = sname;
      VST_DEBUG(dbgs() << "Updated: " << STR(*FBInfo().find(item.fname))
                       << "\n");

      // if it requires to change the "=> local" to be real buffer name "=>
      // sname"
      if (item.dma) {
        auto dma = cast<AST::DMA>(item.dma);
        assert(isa<AST::Memory>(dma->to) &&
               "expect a storage type be updated.");

        // now replace the 'to' buffer inside DMA
        VST_DEBUG(dbgs() << "Replace: '" << PSTR(dma->to) << "' by ");

        dma->to = AST::Make<AST::ChunkAt>(
            dma->to->LOC(),
            AST::Make<AST::Identifier>(dma->to->LOC(), item.sname));
        dma->to->SetType(ity);

        VST_DEBUG(dbgs() << "with: " << PSTR(dma->to) << ".\n");
      }
    }

    mnodes_insertions.erase(&n);
    multi_nodes.pop();
    cur_dma_index = -1;

    return true;
  }

  bool Visit(AST::MultiValues&) override { return true; };
  bool Visit(AST::IntLiteral&) override { return true; };
  bool Visit(AST::Boolean&) override { return true; };
  bool Visit(AST::Expr&) override { return true; };
  bool Visit(AST::MultiDimSpans&) override { return true; };
  bool Visit(AST::NamedTypeDecl&) override { return true; };
  bool Visit(AST::NamedVariableDecl&) override { return true; };
  bool Visit(AST::IntTuple&) override { return true; };
  bool Visit(AST::Assignment&) override { return true; };
  bool Visit(AST::IntIndex&) override { return true; };
  bool Visit(AST::DataType&) override { return true; };
  bool Visit(AST::Identifier&) override { return true; };
  bool Visit(AST::Parameter&) override { return true; };
  bool Visit(AST::ParamList&) override { return true; };
  bool Visit(AST::ParallelBy&) override { return true; };
  bool Visit(AST::WhereBind&) override { return true; };
  bool Visit(AST::WithIn&) override { return true; };
  bool Visit(AST::WithBlock&) override { return true; };
  bool Visit(AST::Memory&) override { return true; };
  bool Visit(AST::SpanAs&) override { return true; };
  bool Visit(AST::DMA&) override { return true; };
  bool Visit(AST::ChunkAt&) override { return true; };
  bool Visit(AST::Wait&) override { return true; };
  bool Visit(AST::Call&) override { return true; };
  bool Visit(AST::Rotate&) override { return true; };
  bool Visit(AST::Select&) override { return true; };
  bool Visit(AST::Return&) override { return true; };
  bool Visit(AST::LoopRange&) override { return true; };
  bool Visit(AST::ForeachBlock&) override { return true; };
  bool Visit(AST::FunctionDecl&) override { return true; };
  bool Visit(AST::ChoreoFunction&) override { return true; };
  bool Visit(AST::CppSourceCode&) override { return true; };
  bool Visit(AST::Program&) override { return true; };
};

struct DummyBufferGen : public LateNormBase {
public:
  DummyBufferGen(const ptr<SymbolTable> s_tab)
      : LateNormBase(s_tab, "bufgen") {}
  ~DummyBufferGen() {}

public:
  bool Visit(AST::DMA& n) {
    // associate a future with its only buffer
    if (!n.future.empty() && (n.operation == ".any")) {
      auto future_name = InScopeName(n.future);
      auto& buf_info = FBInfo()[future_name];
      if (buf_info.buffer.empty()) {
        // the buffer does not exist
        auto sty = GetSpannedType(GetSymbolType(n.future));

        auto anon_sym = ProperBufferName(n.future);
        // Note: Later passes only cares about the type. So it is possible to
        // ignore the syntax struct 'DataType'.
        auto var = AST::Make<AST::NamedVariableDecl>(n.LOC(), anon_sym);
        var->SetType(sty);

        if (sty->GetStorage() == Storage::GLOBAL) {
          // it is a global, must not be inside parallel_by
          assert(cur_pb_index != -1);
          int index = cur_pb_index + mnodes_insertions[cur_pb_mn].size();
          mnodes_insertions[cur_pb_mn].push_back(
              {index, var, anon_sym, future_name, nullptr});
        } else {
          assert(cur_dma_index != -1);
          int index =
              cur_dma_index + mnodes_insertions[multi_nodes.top()].size();
          mnodes_insertions[multi_nodes.top()].push_back(
              {index, var, anon_sym, future_name, nullptr});
        }
      }
    }
    return true;
  }
};

struct BufferInfoCollect : public LateNormBase {
public:
  BufferInfoCollect(const ptr<SymbolTable> s_tab)
      : LateNormBase(s_tab, "latenorm") {}

  bool Visit(AST::DMA& n) {
    TraceEachVisit(n);
    // associate a future with its only buffer
    if (!n.future.empty() && (n.operation != ".any")) {
      assert(isa<AST::ChunkAt>(n.to));
      auto buf_name = n.ToSymbol();

      auto& buf_info = FBInfo()[InScopeName(n.future)];
      buf_info.buffer = InScopeName(buf_name);
      VST_DEBUG(dbgs() << "[MapFutureBuffer] " << n.future << " -> " << buf_name
                       << " - from: " << STR(buf_info.from_kind)
                       << ", to: " << STR(buf_info.to_kind) << ".\n");
    }
    return true;
  }
};

struct LateNorm : public LateNormBase {
public:
  // it requires a symbol table
  LateNorm(const ptr<SymbolTable> s_tab) : LateNormBase(s_tab, "latenorm") {}

  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);

    // update "=>local/shared/global", and generate buffer if necessary
    if (n.operation == ".any") return true;
    if (!isa<AST::Memory>(n.to)) return true;

    auto future_name = ((n.future.empty()) ? "" : InScopeName(n.future));
    if (n.future.empty() || FBInfo()[future_name].buffer.empty()) {
      // It requires to generate an anonymous buffer for "=> local/share/global"
      auto ty = NodeType(n);
      auto shape = cast<FutureType>(ty)->GetShape();
      auto fty = GetSpannedType(NodeType(*cast<AST::ChunkAt>(n.from)->data));
      assert(fty);
      auto sty = MakeSpannedType(fty->ElementType(), shape,
                                 cast<AST::Memory>(n.to)->Get());

      auto to_buffer_name = ProperBufferName(n.future);
      // Note: Later passes only cares about the type. So it is possible to
      // ignore the syntax struct 'DataType'.
      auto var = AST::Make<AST::NamedVariableDecl>(n.to->LOC(), to_buffer_name);
      var->SetType(sty);

      if (sty->GetStorage() == Storage::GLOBAL) {
        // it is a global, must not be inside parallel_by
        assert(cur_pb_index != -1);
        int index = cur_pb_index + mnodes_insertions[cur_pb_mn].size();
        mnodes_insertions[cur_pb_mn].push_back(
            {index, var, to_buffer_name, future_name, &n});
      } else {
        assert(cur_dma_index != -1);
        int index = cur_dma_index + mnodes_insertions[multi_nodes.top()].size();
        mnodes_insertions[multi_nodes.top()].push_back(
            {index, var, to_buffer_name, future_name, &n});
      }
    } else {
      auto to_buffer_name = UnScopedName(FBInfo()[future_name].buffer);

      // now replace the 'to' buffer inside DMA
      VST_DEBUG(dbgs() << "Replace: " << STR(n) << "\n");

      n.to = AST::Make<AST::ChunkAt>(
          n.to->LOC(), AST::Make<AST::Identifier>(n.to->LOC(), to_buffer_name));
      n.to->SetType(GetSymbolType(to_buffer_name));

      VST_DEBUG(dbgs() << "with: " << STR(n) << ".\n");
    }

    return true;
  }

  bool RunOnProgram(AST::Node& root) override {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    if (prt_visitor) dbgs() << "|- " << GetName() << NewL;

    DummyBufferGen bg(SymTab());
    bg.SetTraceVisit(trace_visit);
    bg.SetDebugVisit(debug_visit);

    if (prt_visitor) dbgs() << " |- " << bg.GetName() << NewL;
    root.accept(bg);
    if (HasError()) return false;

    if (prt_visitor) dbgs() << " |- " << GetName() << NewL;
    root.accept(*this);
    if (HasError()) return false;

    // after the transformations, recollect the future-buffer info
    BufferInfoCollect bic(SymTab());
    bic.SetTraceVisit(trace_visit);
    bic.SetDebugVisit(debug_visit);

    if (prt_visitor) dbgs() << " |- " << bic.GetName() << NewL;
    root.accept(bic);
    if (HasError()) return false;

    if (abend_after) return false;

    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_LATE_NORM_HPP__
