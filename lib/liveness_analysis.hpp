#ifndef __CHOREO_LIVENESS_ANALYSIS_HPP__
#define __CHOREO_LIVENESS_ANALYSIS_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "loc.hpp"
#include "typeresolve.hpp"
#include "types.hpp"
#include "visitor.hpp"

#include <queue>
#include <sstream>
#include <unordered_set>

namespace Choreo {

// No changes will be made to AST
struct LivenessAnalyzer : public VisitorWithSymTab {
  // currently, only analyze the memory buffer.

  TypeConstraints type_equals{this};

  /*
  live_in(n)  = use(n) U (live_out(n) - def(n)) U
  (transitive_closure({bindings[x] for x in use(n)})). live_out(n) =
  live_in(next node). Since there is no branching in co.

  def of mem buffer
    can only be in NamedVariableDecl after norm.
    Assignment: reference not def!
  use of mem buffer(not shape)
    scalar:  x. Scalar cannot be assigned as the value in mem buffer.
    SpanAs:  use. only change the shape of mem buffer, as ref.
    ChunkAt: use. Inside DMA. Whether used as src or dst.
    Wait:    use. Wait the DMA. The buffer is reusable when the Wait is done.
    Call:    use. The buffer is used as parameter.
    Rotate:  use. Each item is future. The corresponding buffer is still in use.
    Select:  use. All the items are in use! Each item is buffer.
    Return:  use.
    Assignment:   use. rhs: spanas, select.
    FunctionDecl: use. But the buffers are Global buffers.
    NamedVariableDecl: use if it is a ref.

  实际上，内存的复用不必考虑 scope。暂定将整块 scratch pad 定义在 co 开始


  def in scope A, use in scope A::B
    then should add use in the last stmt of scope A::B.

  def in scope A
    use in scope A::B
    use in scope A
    then do nothing(the end of live range should be the nest stmt of the last
  use.)

  dst of DMA should be treated as def if it is the whole buffer(position is
  nullptr) else, should be treated as use. so, we need to know the last def of
  the current use!

  def in scope A
    def in scope A::B
    use in scope A
    because we may of may not enter scope A::B
    so should pick def in scope A as the begin of live range
    just find the def inside the current scope or outer scope!
  */

  // Certain types of nodes are treated as statements.
  using Stmt = AST::Node;
  Stmt* current_stmt = nullptr;

  struct ScopeEnd : public Stmt {
    ScopeEnd(const location& loc, AST::Node* s = nullptr)
        : Stmt(loc), sibling(s) {}
    AST::Node* sibling = nullptr;
    void Print(std::ostream& os,
               const std::string& prefix = {}) const override {
      os << prefix << "}\n";
    }
    void accept(Visitor& v) override {
      (void)v;
      return;
    }
  };
  std::vector<ptr<ScopeEnd>> scope_ends;
  using Event = std::pair<std::string, std::string>;
  std::unordered_map<const Stmt*, std::set<Event>> events_to_add;
  std::unordered_map<std::string, const Stmt*> scope2stmt;

  std::unordered_map<std::string, std::vector<Event>> var_events;

  size_t stmt_number = 0; // only preorder.
  // map from stmt to its index in stmts_preordered.
  std::unordered_map<const Stmt*, size_t> stmt2number;
  std::vector<const Stmt*> stmts_preordered;

  struct VisitOrder {
    size_t visit_begin = 0;
    size_t visit_end = 0;
  };
  size_t stmt_visit_order = 0; // contains both pre and post order.
  std::unordered_map<const Stmt*, VisitOrder> stmt2visit_order;

  std::stringstream stmts_with_indent;

  // map from AST::Node to the stmt that contains it.
  std::unordered_map<const AST::Node*, const Stmt*> node2stmt;

  struct NodeNumber {
    size_t pre_num = 0;
    // TODO: seems useless.
    // size_t post_num = 0;
  };
  // not the index of node! Double numbering for preOrder and postOrder.
  size_t node_number = 0;
  std::unordered_map<AST::Node*, NodeNumber> node_index;
  std::vector<AST::Node*> nodes_preordered;
  // TODO: seems useless.
  // std::vector<AST::Node*> nodes_postordered;

#if 0
  using BufSet = std::unordered_set<std::string>;
  using VarSet = std::unordered_set<std::string>;
  using BufNodes = std::unordered_set<AST::Node*>;
#else
  using BufSet = std::set<std::string>;
  using VarSet = std::set<std::string>;
  using BufNodes = std::set<AST::Node*>;
#endif

  BufSet buffers;
  BufSet shared_buffers;
  BufSet local_buffers;
  std::unordered_map<Storage, BufNodes> buf_nodes;

  // use and def of vars about memory buffer. var can be future or buffer.
  std::unordered_map<std::string, BufSet> var2buf;

  // one to one. Alias of buffer. Could happen in spanas, etc.
  std::unordered_map<std::string, std::string> Alias;

  // one to many. Bind var to other vars. Could happen in select, dma, etc.
  std::unordered_map<std::string, VarSet> Bindings;

  // record the binding info to do restoration in CalculateLiveInOut().
  std::unordered_map<const Stmt*, std::string> stmt2binding_restore;

  std::unordered_map<size_t, location> idx2loc;

  struct LivenessInfo {
    // TODO: optimize VarSet to use bitset.
    VarSet use;
    VarSet def;
    VarSet live_in;
    VarSet live_out;
    std::string name_if_alias = "";
    std::string name_if_binding = "";
    bool buffer_related = false;
  };
  std::unordered_map<const Stmt*, LivenessInfo> linfo;

  using BufInfo = std::pair<std::string, std::string>;
  std::unordered_map<std::string, std::set<BufInfo>> fut2buffers;

  struct Range {
    size_t start;
    size_t end;
    bool Overlaps(const Range& other) const {
      return start <= other.end && end >= other.start;
    }
  };

  LivenessAnalyzer()
      : VisitorWithSymTab("liveness", CCtx().GetGlobalSymbolTable()) {
    // TODO: delete the setting of debug_visit
    // debug_visit = true;
    if (trace_visit) debug_visit = true; // force debug when tracing
    if (debug_visit) type_equals.SetDebug(true);
  }
  ~LivenessAnalyzer() {}

  std::unordered_map<std::string, size_t> buf_sizes;
  std::unordered_map<std::string, std::vector<Range>> var_ranges;

  VarSet dma_any;
  // std::stack<size_t> iter_cnts;

  FutureBufferInfo& FBInfo() const { return FCtx(fname).GetFutureBufferInfo(); }

private:
  void DumpStmtBriefly(const Stmt& n, std::ostream& os, bool indent = false);
  VarSet GetAllSymbolicOperands(AST::Node* n) const;
  VarSet SetUnion(const VarSet& a, const VarSet& b) const;
  VarSet SetDiff(const VarSet& a, const VarSet& b) const;
  bool IsStmt(const AST::Node& n) const;
  bool IsRef(const AST::Node& n) const;
  std::string GetScopedName(const std::string& name) const;
  void AddUse(const Stmt* s, const std::string& var, bool is_future = false,
              bool add_extra_use = true);
  void AddDef(const Stmt* s, const std::string& var, bool is_buffer = false);
  void AddBufStmt(const Stmt* s, Storage sto);
  void AddAlias(const std::string& alias_var, const std::string& original_var);
  void RemoveAlias(const std::string& alias_var);
  void AddIsAlias(const Stmt* s, const std::string& alias_var);
  void AddIsBinding(const Stmt* s, const std::string& bind_res);
  void AddBinding(const std::string& bind_res, const std::string& bind_src);
  void RemoveBinding(const std::string& bind_res, const std::string& bind_src);
  void AddFut2Buffers(const std::string& fut, const std::string& src,
                      const std::string& dst);
  void CalculateLiveInOut();
  void calculateRanges();
  void HandleSelect(AST::Node& n, ptr<AST::Select> sel);

public:
  void TraceEachVisit(AST::Node& n, bool detail = false,
                      const std::string& m = "") const {
    if (!trace_visit) return;
    if (detail)
      dbgs() << m << STR(n) << "\n";
    else
      dbgs() << m << n.TypeNameString() << "\n";
  }
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::FloatLiteral&) override;
  bool Visit(AST::StringLiteral&) override;
  bool Visit(AST::Boolean&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::MultiDimSpans&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::IntTuple&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::IntIndex&) override;
  bool Visit(AST::DataType&) override;
  bool Visit(AST::Identifier&) override;
  bool Visit(AST::Parameter&) override;
  bool Visit(AST::ParamList&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::Memory&) override;
  bool Visit(AST::SpanAs&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::IncrementBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::Program&) override;

  bool HasError() override;
};

} // end namespace Choreo

#endif // __CHOREO_LIVENESS_ANALYSIS_HPP__