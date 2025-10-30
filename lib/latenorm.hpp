#ifndef __CHOREO_LATE_NORM_HPP__
#define __CHOREO_LATE_NORM_HPP__

// This applies 'LATE_NORM' or 'canonicalization' of AST for shaped object

#include <iostream>
#include <tuple>

#include "symtab.hpp"
#include "types.hpp"
#include "visitor.hpp"

extern Choreo::Option<bool> use_hetero_tileflow;

namespace Choreo {

class WorkingList {
private:
  std::unordered_map<std::string, AST::DMA*> string_to_dma;

public:
  // Add a symbol to the symbol table
  // emittable = 'a'
  // type_symbol = 'a_type'
  // emitted = 'DRAMType(FloatType(32), {1, 2})'
  void AddDMA(AST::DMA& dma) { string_to_dma.emplace(dma.future, &dma); }

  // Retrieve typename of a symbol
  AST::DMA* GetDMA(const std::string& mnemonic) {
    if (string_to_dma.find(mnemonic) != string_to_dma.end())
      return string_to_dma.at(mnemonic);
    return nullptr;
  }

  void Reset() { string_to_dma.clear(); }
};

// Auxiliary structures for buffer generation
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
  int outer_pb_index = -1;
  AST::MultiNodes* cur_pb_mn = nullptr;
  AST::MultiNodes* outer_pb_mn = nullptr;
  WorkingList workinglist;
  std::map<AST::MultiNodes*, BufferInsertInfo> mnodes_insertions;

  // utils for heterogeneous scenario analysis
  int parallel_level = 0;
  int max_parallel_level = 0;

  FutureBufferInfo& FBInfo() { return FCtx(fname).GetFutureBufferInfo(); }

  const std::string ProperBufferName(const std::string name) {
    auto BUFFER_SUFFIX =
        "__buf__"; // hope the users would not name symbols in this way
    return (name.empty()) ? SymbolTable::GetAnonName() : name + BUFFER_SUFFIX;
  }

protected:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "before visiting " << n.TypeNameString() << "\n";
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "Before " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    } else if (auto m = dyn_cast<AST::MultiNodes>(&n)) {
      multi_nodes.push(m);
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      cur_pb_index = multi_nodes.top()->GetIndex(pb);
      cur_pb_mn = multi_nodes.top();
      if (pb->IsOuter()) {
        outer_pb_index = cur_pb_index;
        outer_pb_mn = cur_pb_mn;
      }
      assert(cur_pb_index != -1 && "unexpected node index.");
      // handle parallel-level
      parallel_level++;
    } else if (auto d = dyn_cast<AST::DMA>(&n)) {
      cur_dma_index = multi_nodes.top()->GetIndex(d);
      assert(cur_dma_index != -1 && "unexpected node index.");
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "after visiting " << n.TypeNameString() << "\n";

    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "After " << GetName() << " - " << STR(FBInfo())
                       << "\n");
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      cur_pb_index = -1;
      cur_pb_mn = nullptr;
      if (pb->IsOuter()) {
        outer_pb_index = -1;
        outer_pb_mn = nullptr;
      }
      // handle parallel-level
      parallel_level--;
    }
    return true;
  }

  inline bool IsHostSide() const {
    // if current stmt not enter parallel-by btw device and host
    // max_parallel_level is not set or set to 0 or has explicit distance to
    // inner-most
    return (parallel_level + 1 < max_parallel_level);
  }

  inline bool IsHostSymbol(const std::string& sym) const {
    int count = 0;
    size_t pos = 0;
    std::string target = "paraby";
    int host_side_parallel_lv_cnt = std::max(max_parallel_level - 2, 0);

    // find the target substring from the current position
    while ((pos = sym.find(target, pos)) != std::string::npos) {
      count++;
      pos += target.length(); // Move pos to the end of the found target
    }

    return (count <= host_side_parallel_lv_cnt);
  }

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) {
      dbgs() << n.TypeNameString() << ": ";
      n.InlinePrint(dbgs());
      dbgs() << "\n";
    }
  }

public:
  LateNormBase(const ptr<SymbolTable> s_tab, const std::string& pn,
               bool ugs = false)
      : VisitorWithSymTab(pn, s_tab, ugs) {}
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
};

// step 0 prepare:
// get max_parallel_level before handling;
struct ParallelLevelAnalysis : public VisitorWithSymTab {
  // utils for heterogeneous scenario analysis
  int parallel_level = 0;
  int max_parallel_level = 0;

public:
  ParallelLevelAnalysis(const ptr<SymbolTable> s_tab)
      : VisitorWithSymTab("parallel-level-analysis", s_tab) {}
  ~ParallelLevelAnalysis() {}

protected:
  bool BeforeVisitImpl(AST::Node& n) {
    if (isa<AST::ParallelBy>(&n)) {
      parallel_level++;
      max_parallel_level = parallel_level > max_parallel_level
                               ? parallel_level
                               : max_parallel_level;
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) {
    if (isa<AST::ParallelBy>(&n)) { parallel_level--; }
    return true;
  }

public:
  int getMaxParallelLv() { return max_parallel_level; }
};

struct HostSliceBufferGen : public LateNormBase {
  std::vector<std::string> buffer_list_tiled_by_host_iv = {};
  std::vector<ptr<AST::Node>> buffer_list_tiled_by_host_iv_ptr = {};
  std::vector<std::string> host_buffer_slice_list = {};
  AST::MultiNodes* host_device_border_mn;
  int host_device_border_index;
  std::string host_device_border_scope;

public:
  HostSliceBufferGen(const ptr<SymbolTable> s_tab, int max_plv)
      : LateNormBase(s_tab, "host-slice-buffer-gen") {
    max_parallel_level = max_plv;
  }
  ~HostSliceBufferGen() {}

public:
  // step 1: find out host-side IVs used at device-side, remember its id
  // impl'd within processHostIVForExpr
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
      parallel_level++;
      // if current parallel-by is the border between host-and-device
      // do staging for later handling.
      if (parallel_level == max_parallel_level - 1) {
        host_device_border_scope = SSTab().getParentScope();
        host_device_border_mn = cur_pb_mn;
        host_device_border_index = cur_pb_index;
      }
    } else if (auto d = dyn_cast<AST::DMA>(&n)) {
      cur_dma_index = multi_nodes.top()->GetIndex(d);
      assert(cur_dma_index != -1 && "unexpected node index.");
    } else if (auto n_ptr = dyn_cast<AST::Assignment>(&n)) {
      // since all chunkat is normalised at previous stage,
      // to generate a dummy buffer, we can find all IV usage (in chunkat)
      // at Assignment Node.
      AST::Assignment& an = *n_ptr;
      if (isa<AST::Expr>(an.value)) {
        processHostIVForExpr(an.value, an.GetName(), an);
      }
    }
    return true;
  }
  using LateNormBase::Visit;

  // step 2: find out which host-side buffer is used at device-side and tiled by
  // above mentioned host-side IVs.
  // step 3: create a buffer slice of the host-side buffer, and will be sent to
  // device-side; at this stage, data from host-side buffer to its buffer slice,
  // is essentially a pointer offset, but we need to add a dma stmt and let
  // backend handle it.
  bool Visit(AST::ChunkAt& n) {
    TraceEachVisit(n);
    for (auto tsi : n.AllOperations()) {
      for (auto& tile_factor : tsi->GetIndices()) {
        // from ChunkAt nodes, find out who uses host-side IV at device code
        if (std::find(buffer_list_tiled_by_host_iv.begin(),
                      buffer_list_tiled_by_host_iv.end(),
                      STR(tile_factor)) != buffer_list_tiled_by_host_iv.end()) {
          // get the host-side IV.
          auto tiler_node = buffer_list_tiled_by_host_iv_ptr[std::distance(
              buffer_list_tiled_by_host_iv.begin(),
              std::find(buffer_list_tiled_by_host_iv.begin(),
                        buffer_list_tiled_by_host_iv.end(), STR(tile_factor)))];
          // get the source buffer for this chunkat
          if (std::find(host_buffer_slice_list.begin(),
                        host_buffer_slice_list.end(),
                        n.data->name) == host_buffer_slice_list.end()) {
            host_buffer_slice_list.push_back(n.data->name);

            // formulate the updated shape/types for this optimisation
            auto ty = NodeType(n);
            auto ty_data = GetSymbolType(n.data->name);
            auto ty_tiler = NodeType(*tiler_node);

            auto data_shape =
                dyn_cast<SpannedType>(GetSymbolType(n.data->name))->GetShape();
            auto tiler_shape =
                dyn_cast<BoundedITupleType>(NodeType(*tiler_node))
                    ->GetUpperBounds();

            assert(data_shape.Rank() == tiler_shape.Rank());
            ValueList new_shape_values;
            for (size_t i = 0; i < data_shape.Rank(); ++i) {
              new_shape_values.push_back(data_shape.ValueAt(i) /
                                         tiler_shape.ValueAt(i));
            }
            auto chunkat_shape = Shape(data_shape.Rank(), new_shape_values);
            auto new_chunkat_ty = MakeSpannedType(
                GetBaseType(*ty_data), chunkat_shape, Storage::GLOBAL);

            // create new AST nodes, and rewrite
            // remember to settypes and update symbol table
            std::string slice_fut_postfix = "__future__";
            auto dnode_id = n.data; // ptr<Identifier>
            auto slice_fut_id = AST::Make<AST::Identifier>(
                dnode_id->loc, dnode_id->name + slice_fut_postfix);

            n.data = slice_fut_id;

            auto mv_node = AST::Make<AST::MultiValues>(dnode_id->loc);
            mv_node->Append(tiler_node);
            std::vector<ptr<AST::SpannedOperation>> nso;
            auto so = AST::Make<AST::SpannedOperation>(dnode_id->loc, mv_node);
            so->SetBlockShape(n.GetBlockShape());
            nso.push_back(so);

            auto ca_node =
                AST::Make<AST::ChunkAt>(dnode_id->loc, dnode_id, nullptr, nso);
            ca_node->SetType(new_chunkat_ty);

            auto dma_node = AST::Make<AST::DMA>(
                dnode_id->loc, ".copy", slice_fut_id->name, ca_node,
                AST::Make<AST::Memory>(dnode_id->loc, Storage::GLOBAL), false);
            dma_node->to->SetType(new_chunkat_ty);
            dma_node->from->SetType(new_chunkat_ty);
            dma_node->SetType(MakeFutureType(new_chunkat_ty, false));

            // NOTE: only built-in SymTab in scopedsymboltable will be persist
            // to next compilation pipeline stage. not scoped symbol table
            // (which wraps the builtin symtab) so we need to update by
            // SymTab(), not through SSTab()
            auto sname = host_device_border_scope + slice_fut_id->name;
            SymTab()->AddSymbol(sname, MakeFutureType(new_chunkat_ty, false));

            // update FBInfo(), to complete latter future-buffer
            // checks/handlings, correctly
            auto from_kind = DOK_CHUNK;
            auto to_kind = DOK_SYMBOL;
            FCtx(fname).GetFutureBufferInfo().emplace(
                sname, DMABufferInfo{"", from_kind, to_kind});

            host_device_border_mn->Insert(dma_node, host_device_border_index);
          }
        }
      }
    }
    return true;
  }

private:
  inline void dumpVector(
      std::vector<std::string>& _list) { // Capture everything by reference
    dbgs() << ">>>>>>>> DUMP Vector contents:\n";
    int index = 1;
    for (const auto& element : _list) {
      // Format and print each element with its index
      dbgs() << "[" << index++ << "] " << element << "\n";
    }
    dbgs() << ">>>>>>>> END Vector contents:\n";
  }

  template <typename T>
  inline void dumpStack(const std::stack<T>& s) {
    std::stack<T> tempStack = s; // Copy the original stack
    dbgs() << ">>>>>>>> Stack contents (from top to bottom):\n";
    int index = 1;
    while (!tempStack.empty()) {
      dbgs() << "Element #" << index++ << ": " << PSTR(tempStack.top()) << "\n";
      tempStack.pop(); // Pop elements from the copy to print
    }
    dbgs() << ">>>>>>>> END Stack contents:\n";
  }

  // NOTE: Recursively handle if a expr is a host-side IV ref'd at device-side
  // If so, we need to make it a unit IV, and set the data source handler
  // as a buffer slice of the original host-side buffer (must-be host-side
  // buffer, otherwise host-side IV is not permitted to use as a tiler for this
  // buffer) The buffer slice is just tiled (at host-side) by the host-side IV,
  // ONLY.
  void processHostIVForExpr(std::shared_ptr<AST::Node> expr_node_ptr,
                            std::string current_iv_name,
                            AST::Assignment& top_node) {
    // Ensure the expression pointer is valid
    if (!isa<AST::Expr>(expr_node_ptr)) return;
    auto expr_ptr = cast<AST::Expr>(expr_node_ptr);
    if (!expr_ptr) return;

    // If it's a unary expression, process the right operand (GetR)
    if (expr_ptr->IsReference()) {
      // guard checks for outliers
      if (ConvertibleToInt(NodeType(*expr_ptr))) return;
      if (isa<AST::IntLiteral>(expr_ptr)) return;
      if (isa<AST::FloatLiteral>(expr_ptr)) return;
      if (isa<AST::IntTuple>(expr_ptr)) return;
      if (isa<AST::SpanAs>(expr_ptr)) return;
      if (isa<AST::MultiDimSpans>(expr_ptr)) return;
      if (isa<AST::BoolLiteral>(expr_ptr)) return;
      if (IsHostSymbol(InScopeName(STR(expr_ptr->GetR()))) && !IsHostSide()) {
        if (std::find(buffer_list_tiled_by_host_iv.begin(),
                      buffer_list_tiled_by_host_iv.end(),
                      current_iv_name) == buffer_list_tiled_by_host_iv.end()) {
          buffer_list_tiled_by_host_iv.push_back(current_iv_name);
          buffer_list_tiled_by_host_iv_ptr.push_back(expr_ptr->GetR());
        }
        auto host_iv_ubs =
            dyn_cast<BoundedITupleType>(NodeType(*expr_ptr->GetR()))
                ->GetUpperBound(0);
        auto top_iv_ubs = dyn_cast<BoundedITupleType>(top_node.value->GetType())
                              ->GetUpperBound(0);
        auto new_host_iv_ty =
            MakeBoundedITupleType(Shape(1, top_iv_ubs / host_iv_ubs));

        auto new_node = AST::MakeIdExpr(expr_ptr->GetR()->loc, "_");
        new_node->Opts().SetVal(sbe::nu(0));
        auto unit_bound_ty = MakeBoundedITupleType(Shape(1, 1));

        new_node->SetType(unit_bound_ty);
        top_node.value->SetType(new_host_iv_ty);
        top_node.SetType(new_host_iv_ty);

        SymTab()->AddSymbol(SSTab().ScopeName() + "_", unit_bound_ty);
        SymTab()
            ->GetSymbol(InScopeName(top_node.GetName()))
            ->SetType(new_host_iv_ty);
        expr_ptr->SetR(new_node); // Replace GetR with a new node
      }
    } else if (expr_ptr->IsUnary()) {
      processHostIVForExpr(expr_ptr->GetR(), current_iv_name, top_node);
    } else if (expr_ptr->IsBinary()) {
      processHostIVForExpr(expr_ptr->GetL(), current_iv_name, top_node);
      processHostIVForExpr(expr_ptr->GetR(), current_iv_name, top_node);
    } else {
      return;
    }
  }
};

struct DummyBufferGen : public LateNormBase {
public:
  DummyBufferGen(const ptr<SymbolTable> s_tab)
      : LateNormBase(s_tab, "bufgen") {}
  ~DummyBufferGen() {}

public:
  using LateNormBase::Visit;

  bool Visit(AST::DMA& n) override {
    // associate a future with its only buffer
    if (!n.future.empty() && (n.operation == ".any")) {
      auto future_name = InScopeName(n.future);
      auto& buf_info = FBInfo()[future_name];
      if (buf_info.buffer.empty()) {
        // the buffer does not exist
        auto sym_type = GetSymbolType(n.future);
        if (isa<PlaceHolderType>(sym_type)) {
          Error1(n.LOC(),
                 "dma.any '" + n.future + "' is defined but never used!");
          return false;
        }
        auto sty = GetSpannedType(sym_type);
        assert(sty);
        auto anon_sym = ProperBufferName(n.future);
        // Note: Later passes only cares about the type. So it is possible to
        // ignore the syntax struct 'DataType'.
        auto var = AST::Make<AST::NamedVariableDecl>(n.LOC(), anon_sym);
        var->SetType(sty);

        if (sty->GetStorage() == Storage::GLOBAL) {
          // it is a global, must not be inside parallel_by
          // no outer PB exists
          assert(outer_pb_index != -1);
          int index = outer_pb_index + mnodes_insertions[outer_pb_mn].size();
          mnodes_insertions[outer_pb_mn].push_back(
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
      : LateNormBase(s_tab, "buffer-info-collect") {}

  using LateNormBase::Visit;

  bool Visit(AST::DMA& n) override {
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
  LateNorm() : LateNormBase(nullptr, "latenorm", true) {}

  using LateNormBase::Visit;

  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);

    // handle chained info, filling the DMA chain.
    workinglist.AddDMA(n);
    if (n.chained == true && !n.chain_from.empty()) {
      auto _chain_from_ptr = workinglist.GetDMA(n.chain_from);
      assert(_chain_from_ptr != nullptr &&
             "after primitive chained to non-exist future id\n");
      _chain_from_ptr->chained = true;
      _chain_from_ptr->chain_to = n.future;
    }

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
        if (outer_pb_index != -1) {
          // DMA dst is a global, must not be inside parallel_by
          int index = outer_pb_index + mnodes_insertions[outer_pb_mn].size();
          mnodes_insertions[outer_pb_mn].push_back(
              {index, var, to_buffer_name, future_name, &n});
        } else {
          // moving global dma
          assert(cur_dma_index != -1);
          int index =
              cur_dma_index + mnodes_insertions[multi_nodes.top()].size();
          mnodes_insertions[multi_nodes.top()].push_back(
              {index, var, to_buffer_name, future_name, &n});
        }
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

  bool RunOnProgramImpl(AST::Node& root) override {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    if (prt_visitor) dbgs() << "|- " << GetName() << NewL;

    if (use_hetero_tileflow) {
      ParallelLevelAnalysis pla(SymTab());
      root.accept(pla);
      if (prt_visitor) dbgs() << " |- " << pla.GetName() << NewL;
      if (pla.HasError()) return false;
      auto max_plv = pla.getMaxParallelLv();

      HostSliceBufferGen hsg(SymTab(), max_plv);
      hsg.SetTraceVisit(trace_visit);
      hsg.SetDebugVisit(debug_visit);
      if (prt_visitor) dbgs() << " |- " << hsg.GetName() << NewL;
      root.accept(hsg);
      if (hsg.HasError()) return false;
    }

    DummyBufferGen bg(SymTab());
    bg.SetTraceVisit(trace_visit);
    bg.SetDebugVisit(debug_visit);

    if (prt_visitor) dbgs() << " |- " << bg.GetName() << NewL;
    root.accept(bg);
    if (bg.HasError()) return false;

    if (prt_visitor) dbgs() << " |- " << GetName() << NewL;
    root.accept(*this);
    if (this->HasError()) return false;

    // after the transformations, recollect the future-buffer info
    BufferInfoCollect bic(SymTab());
    bic.SetTraceVisit(trace_visit);
    bic.SetDebugVisit(debug_visit);

    if (prt_visitor) dbgs() << " |- " << bic.GetName() << NewL;
    root.accept(bic);
    if (bic.HasError()) return false;
    if (abend_after) return false;

    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_LATE_NORM_HPP__
