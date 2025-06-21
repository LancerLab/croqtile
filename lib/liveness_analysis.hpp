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

  In fact, memory reuse does not have to take scope into account.
  The whole scratch pad is tentatively defined at the beginning of co

  def in scope A
    use in scope A::B
  then should add use in the last stmt of scope A::B.

  def in scope A
    use in scope A::B
    use in scope A
  add use in the last stmt of scope A::B.
  but there is another use in scope A later.
  so the live range is still [def point, the second use]

  dst of DMA is treated as use for now.
  TODO: be treated as def if it is the whole buffer(position is nullptr)
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
        : Stmt(loc), scope_start(s) {}
    const AST::Node* scope_start = nullptr;

    ptr<Node> CloneImpl() const override {
      choreo_unreachable("unexpected clone.");
      return nullptr;
    }

    void Print(std::ostream& os, const std::string& prefix = {},
               bool = false) const override {
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
  // map from stmt to its index in preorder_stmts.
  std::unordered_map<const Stmt*, size_t> stmt2number;
  std::vector<const Stmt*> preorder_stmts;
  std::unordered_map<const Stmt*, std::string> stmt2str;

  std::stringstream stmts_with_indent;

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

  // use and def of vars about memory buffer. var can be future or buffer.
  std::unordered_map<std::string, BufSet> var2buf;

  // one to one. Alias of buffer. Could happen in spanas, etc.
  std::unordered_map<std::string, std::string> Alias;

  // one to many. Bind var to other vars. Could happen in select, dma, etc.
  std::unordered_map<std::string, VarSet> Bindings;

  // record the binding info to do restoration in ComputeLiveInOut().
  std::unordered_map<const Stmt*, std::string> stmt2binding_restore;

  std::unordered_map<size_t, location> idx2loc;

  std::unordered_set<std::string> paraby_bounded_vars;

  size_t inthreads_async_level = 0;
  std::unordered_map<std::string, VarSet> async_inthreads_vars;
  bool visiting_synchronize = false;

  struct LivenessInfo {
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
  using StrUintMap = std::unordered_map<std::string, size_t>;

  struct Range {
    size_t start;
    size_t end;

    bool Overlaps(const Range& other) const {
      return start <= other.end && end >= other.start;
    }

    bool operator<(const Range& other) const {
      return std::tie(start, end) < std::tie(other.start, other.end);
    }
  };

  struct Ranges {
    std::vector<Range> ranges;

    bool Empty() const { return ranges.empty(); }

    void PushBack(const Range& range) { ranges.push_back(range); }

    void Merge() {
      if (ranges.empty()) return;
      std::sort(ranges.begin(), ranges.end());
      std::vector<Range> merged;
      merged.push_back(ranges[0]);
      for (size_t i = 1; i < ranges.size(); ++i) {
        Range& back = merged.back();
        if (back.Overlaps(ranges[i]))
          back.end = std::max(back.end, ranges[i].end);
        else
          merged.push_back(ranges[i]);
      }
      ranges = std::move(merged);
    }

    bool operator<(const Ranges& other) const {
      if (ranges.empty()) return !other.ranges.empty();
      if (other.ranges.empty()) return false;
      return ranges.front().start < other.ranges.front().start;
    }

    const std::vector<Range>& Values() const { return ranges; }
  };

  LivenessAnalyzer()
      : VisitorWithSymTab("liveness", CCtx().GetGlobalSymbolTable()) {
    if (trace_visit) debug_visit = true; // force debug when tracing
    // cause --liveness is enabled by default.
    if (disabled) CCtx().SetLivenessAnalysis(false);
  }
  ~LivenessAnalyzer() {}

  std::unordered_map<std::string, Ranges> var_ranges;

  VarSet dma_any;

public:
  static VarSet SetUnion(const VarSet& a, const VarSet& b);
  static VarSet SetDiff(const VarSet& a, const VarSet& b);
  static bool IsRef(const AST::Node& n);

private:
  VarSet GetAllSymbolicOperands(const AST::Node* n) const;
  void DumpStmtBriefly(const Stmt& n, std::ostream& os, bool indent,
                       bool only_else = false);
  bool HasStmt(const AST::Node& n) const;
  std::string GetScopedName(const std::string& name) const;
  void AddUse(const Stmt* s, const std::string& var, bool add_extra_use = true);
  void AddUse(const Stmt* s, const VarSet& vars, bool add_extra_use = true);
  void AddDef(const Stmt* s, const std::string& var,
              bool is_buffer_or_future = false);
  void AddIsAlias(const Stmt* s, const std::string& alias_var);
  void AddAlias(const std::string& alias_var, const std::string& original_var);
  void RemoveAlias(const std::string& alias_var);
  void AddIsBinding(const Stmt* s, const std::string& bind_res);
  void AddBinding(const std::string& bind_res, const std::string& bind_src);
  void RemoveBinding(const std::string& bind_res, const std::string& bind_src);
  void AddFut2Buffers(const std::string& fut, const BufInfo& buf_info);
  void AddAsyncInthreadsVar(const std::string& scope_name,
                            const std::string& var);
  void ComputeLiveInOut();
  void ComputeLiveRange();
  void HandleSelect(AST::Node& n, ptr<AST::Select> sel);
  // handle stmt in Before/AfterVisitImpl
  void HandleStmtInBefore(AST::Node& n);
  void HandleStmtInMid(AST::Node& n);
  void HandleStmtInAfter(AST::Node& n);
  std::string SSTR(const Stmt* stmt) const;

  template <typename... MapTypes>
  VarSet TransitiveClosure(const VarSet& vars, const MapTypes&... maps);

public:
  const std::string STMTS_STR() const { return stmts_with_indent.str(); }
  const std::unordered_map<std::string, Ranges>& VarRanges() const {
    return var_ranges;
  }

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
  bool InMidVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Synchronize&) override;
  bool Visit(AST::Trigger&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::InThreadsBlock&) override;
  bool Visit(AST::IfElseBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;

  bool HasError() override;
};

} // end namespace Choreo

#endif // __CHOREO_LIVENESS_ANALYSIS_HPP__
