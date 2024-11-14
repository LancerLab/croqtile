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
  bool Visit(AST::Rotate&) override { return true; }
  bool Visit(AST::Select&) override { return true; }
  bool Visit(AST::Return&) override { return true; }
  bool Visit(AST::LoopRange&) override { return true; }
  bool Visit(AST::ForeachBlock&) override { return true; }
  bool Visit(AST::FunctionDecl&) override { return true; }
  bool Visit(AST::ChoreoFunction&) override { return true; }
  bool Visit(AST::CppSourceCode&) override { return true; }
  bool Visit(AST::Program&) override { return true; }
};

struct BufferInfoCollect : public VisitorWithSymTab {
private:
  std::ostream& os;
  size_t error_count = 0;

  std::string fname; // current function name
  ptr<FutureBufferMap> fut_buf = nullptr;

private:
  bool BeforeVisitImpl(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      assert(fut_buf->count(f->name) == 0);
      fut_buf->insert({f->name, {}});
      fname = f->name;
    } else if (auto dma = dyn_cast<AST::DMA>(&n)) {
      // associate a future with its only buffer
      if (!dma->future.empty() && (dma->operation != ".any")) {
        auto buf_name = cast<AST::ChunkAt>(dma->to)->RefSymbol();
        (*fut_buf)[fname].emplace(dma->future, buf_name);
        VST_DEBUG(os << "associate " << dma->future << " with " << buf_name
                     << "\n");
      }
    }
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) {
    if (isa<AST::ChoreoFunction>(&n)) { fname = ""; }
    return true;
  }

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) { os << n.TypeNameString() << ": " << STR(n) << "\n"; }
  }

public:
  BufferInfoCollect(const ptr<SymbolTable> s_tab, std::ostream& o = outs())
      : VisitorWithSymTab("bicol", s_tab), os(o) {
    fut_buf = std::make_shared<FutureBufferMap>();
  }
  ~BufferInfoCollect() {}

  const ptr<FutureBufferMap> FBInfo() { return fut_buf; }

  bool Visit(AST::MultiNodes&) { return true; }
  bool Visit(AST::MultiValues&) { return true; }
  bool Visit(AST::IntLiteral&) { return true; }
  bool Visit(AST::Boolean&) { return true; }
  bool Visit(AST::Expr&) { return true; }
  bool Visit(AST::MultiDimSpans&) { return true; }
  bool Visit(AST::NamedTypeDecl&) { return true; }
  bool Visit(AST::NamedVariableDecl&) { return true; }
  bool Visit(AST::IntTuple&) { return true; }
  bool Visit(AST::Assignment&) { return true; }
  bool Visit(AST::IntIndex&) { return true; }
  bool Visit(AST::DataType&) { return true; }
  bool Visit(AST::Identifier&) { return true; }
  bool Visit(AST::Parameter&) { return true; }
  bool Visit(AST::ParamList&) { return true; }
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
  bool Visit(AST::Return&) { return true; }
  bool Visit(AST::LoopRange&) { return true; }
  bool Visit(AST::ForeachBlock&) { return true; }
  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }

  bool HasError() { return false; }
};

struct BufferGenerate : public VisitorWithSymTab {
private:
  std::ostream& os;
  size_t error_count = 0;

  std::string fname; // current function name
  ptr<FutureBufferMap> fut_buf = nullptr;

  using NodeInsertInfo =
      std::vector<std::tuple<int, ptr<AST::Node>, std::string>>;
  std::stack<AST::MultiNodes*> multi_nodes;
  int cur_dma_index = -1;
  int cur_pb_index = -1;
  AST::MultiNodes* cur_pb_mn = nullptr;
  std::map<AST::MultiNodes*, NodeInsertInfo> mnodes_insertions;

private:
  bool BeforeVisitImpl(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      fname = f->name;
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
    if (isa<AST::ChoreoFunction>(&n)) {
      fname = "";
    } else if (isa<AST::ParallelBy>(&n)) {
      cur_pb_index = -1;
      cur_pb_mn = nullptr;
    }
    return true;
  }

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) { os << n.TypeNameString() << ": " << STR(n) << "\n"; }
  }

public:
  BufferGenerate(const ptr<SymbolTable> s_tab, const ptr<FutureBufferMap>& fb,
                 std::ostream& o = outs())
      : VisitorWithSymTab("bufgen", s_tab), os(o), fut_buf(fb) {}
  ~BufferGenerate() {}

  const ptr<FutureBufferMap> FBInfo() { return fut_buf; }

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
  bool Visit(AST::MultiValues&) { return true; }
  bool Visit(AST::IntLiteral&) { return true; }
  bool Visit(AST::Boolean&) { return true; }
  bool Visit(AST::Expr&) { return true; }
  bool Visit(AST::MultiDimSpans&) { return true; }
  bool Visit(AST::NamedTypeDecl&) { return true; }
  bool Visit(AST::NamedVariableDecl&) { return true; }
  bool Visit(AST::IntTuple&) { return true; }
  bool Visit(AST::Assignment&) { return true; }
  bool Visit(AST::IntIndex&) { return true; }
  bool Visit(AST::DataType&) { return true; }
  bool Visit(AST::Identifier&) { return true; }
  bool Visit(AST::Parameter&) { return true; }
  bool Visit(AST::ParamList&) { return true; }
  bool Visit(AST::ParallelBy&) { return true; }
  bool Visit(AST::WhereBind&) { return true; }
  bool Visit(AST::WithIn&) { return true; }
  bool Visit(AST::WithBlock&) { return true; }
  bool Visit(AST::Memory&) { return true; }
  bool Visit(AST::SpanAs&) { return true; }
  bool Visit(AST::DMA& n) {
    // associate a future with its only buffer
    if (!n.future.empty() && (n.operation == ".any")) {
      auto buf_name = n.future;
      if (!(*fut_buf)[fname].count(buf_name)) {
        // the buffer does not exist
        auto sty = GetSpannedType(GetSymbolType(buf_name));

        auto BUFFER_SUFFIX = "__buf__"; // hope user not name buffer this way
        // Note: Later passes only cares about the type. So it is possible to
        // ignore the syntax struct 'DataType'.
        auto anon_sym = (n.future.empty()) ? SymbolTable::GetAnonName()
                                           : n.future + BUFFER_SUFFIX;
        auto var = AST::Make<AST::NamedVariableDecl>(n.LOC(), anon_sym);
        var->SetType(sty);

        if (sty->GetStorage() == Storage::GLOBAL) {
          // it is a global, must not be inside parallel_by
          assert(cur_pb_index != -1);
          int index = cur_pb_index + mnodes_insertions[cur_pb_mn].size();
          mnodes_insertions[cur_pb_mn].emplace_back(
              std::make_tuple(index, var, anon_sym));
        } else {
          assert(cur_dma_index != -1);
          int index =
              cur_dma_index + mnodes_insertions[multi_nodes.top()].size();
          mnodes_insertions[multi_nodes.top()].emplace_back(
              std::make_tuple(index, var, anon_sym));
        }
        VST_DEBUG(os << "MapBuffer: " << buf_name << " -> " << anon_sym
                     << "\n");
        (*fut_buf)[fname][buf_name] = anon_sym;
      }
    }
    return true;
  }

  bool Visit(AST::ChunkAt&) { return true; }
  bool Visit(AST::Wait&) { return true; }
  bool Visit(AST::Call&) { return true; }
  bool Visit(AST::Rotate&) { return true; }
  bool Visit(AST::Select&) { return true; }
  bool Visit(AST::Return&) { return true; }
  bool Visit(AST::LoopRange&) { return true; }
  bool Visit(AST::ForeachBlock&) { return true; }
  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }

  bool HasError() { return false; }
};

} // end namespace Choreo

#endif // __CHOREO_LATE_NORM_HPP__
