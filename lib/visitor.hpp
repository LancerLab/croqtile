#ifndef __CHOREO_VISITOR_HPP__
#define __CHOREO_VISITOR_HPP__

#include <cstring>
#include <unistd.h>
#include <unordered_set>

#include "ast.hpp"
#include "infra_utils.hpp"
#include "loc.hpp"

namespace Choreo {

// Debug macro used for all Visitors
#define VST_DEBUG(X)                                                           \
  do {                                                                         \
    if (DebugIsEnabled()) { X; }                                               \
  } while (false)

struct Visitor {
  static constexpr const char* SprT = "*******************";
  static constexpr const char* NewL = "\n";

  virtual bool BeforeVisit(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) { // per-function print
      if (print_ahead) {
        dbgs() << NewL << SprT << " Before " << name << ": " << f->name
               << " (Begin) " << SprT << NewL;
        n.Print(dbgs(), "", prt_node_ty);
        dbgs() << NewL;
        dbgs() << SprT << " Before " << name << ": " << f->name << " (End) "
               << SprT << NewL;
      }
    }
    return true;
  }

  virtual bool InMidVisit(AST::Node&) = 0; // not used by all

  virtual bool AfterVisit(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) { // by function print
      if (print_after) {
        dbgs() << NewL << SprT << " After " << name << ": " << f->name
               << " (Begin) " << SprT << NewL;
        n.Print(dbgs(), "", prt_node_ty);
        dbgs() << NewL;
        dbgs() << SprT << " After " << name << ": " << f->name << " (End) "
               << SprT << NewL;
      }
      if (dsyms_after) {
        dbgs() << NewL << SprT << " Symbol Table Dump (After " << name << ") "
               << SprT << NewL;
        if (SymTab()) SymTab()->Print(dbgs());
        dbgs() << SprT << SprT << NewL;
      }
    }
    return true;
  }

  // For any visitor, it should implement all the necessary steps
  virtual bool Visit(AST::MultiNodes&) = 0;
  virtual bool Visit(AST::MultiValues&) = 0;
  virtual bool Visit(AST::NoValue&) = 0;
  virtual bool Visit(AST::IntLiteral&) = 0;
  virtual bool Visit(AST::FloatLiteral&) = 0;
  virtual bool Visit(AST::StringLiteral&) = 0;
  virtual bool Visit(AST::BoolLiteral&) = 0;
  virtual bool Visit(AST::Expr&) = 0;
  virtual bool Visit(AST::CastExpr&) = 0;
  virtual bool Visit(AST::AttributeExpr&) = 0;
  virtual bool Visit(AST::MultiDimSpans&) = 0;
  virtual bool Visit(AST::NamedTypeDecl&) = 0;
  virtual bool Visit(AST::NamedVariableDecl&) = 0;
  virtual bool Visit(AST::IntTuple&) = 0;
  virtual bool Visit(AST::DataAccess&) = 0;
  virtual bool Visit(AST::Assignment&) = 0;
  virtual bool Visit(AST::IntIndex&) = 0;
  virtual bool Visit(AST::DataType&) = 0;
  virtual bool Visit(AST::Identifier&) = 0;
  virtual bool Visit(AST::Parameter&) = 0;
  virtual bool Visit(AST::ParamList&) = 0;
  virtual bool Visit(AST::ParallelBy&) = 0;
  virtual bool Visit(AST::WhereBind&) = 0;
  virtual bool Visit(AST::WithIn&) = 0;
  virtual bool Visit(AST::WithBlock&) = 0;
  virtual bool Visit(AST::Memory&) = 0;
  virtual bool Visit(AST::SpanAs&) = 0;
  virtual bool Visit(AST::DMA&) = 0;
  virtual bool Visit(AST::MMA&) = 0;
  virtual bool Visit(AST::ChunkAt&) = 0;
  virtual bool Visit(AST::Wait&) = 0;
  virtual bool Visit(AST::Trigger&) = 0;
  virtual bool Visit(AST::Break&) = 0;
  virtual bool Visit(AST::Continue&) = 0;
  virtual bool Visit(AST::Call&) = 0;
  virtual bool Visit(AST::Rotate&) = 0;
  virtual bool Visit(AST::Synchronize&) = 0;
  virtual bool Visit(AST::Select&) = 0;
  virtual bool Visit(AST::Return&) = 0;
  virtual bool Visit(AST::LoopRange&) = 0;
  virtual bool Visit(AST::ForeachBlock&) = 0;
  virtual bool Visit(AST::InThreadsBlock&) = 0;
  virtual bool Visit(AST::WhileBlock&) = 0;
  virtual bool Visit(AST::IfElseBlock&) = 0;
  virtual bool Visit(AST::IncrementBlock&) = 0;
  virtual bool Visit(AST::FunctionDecl&) = 0;
  virtual bool Visit(AST::ChoreoFunction&) = 0;
  virtual bool Visit(AST::CppSourceCode&) = 0;
  virtual bool Visit(AST::DeviceFunctionDecl&) = 0;
  virtual bool Visit(AST::Program&) = 0;

protected:
  // scoped variable handling
  ScopedSymbolTable scoped_symtab;
  bool use_global_symtab = false;

  std::string lvl_pfx;

protected:
  std::string name;
  bool trace_visit = false;
  bool debug_visit = false;
  bool print_ahead = false;
  bool print_after = false;
  bool dsyms_after = false;
  bool abend_after = false;
  bool prt_visitor = false;
  bool prt_node_ty = false;
  bool resolve_fns = false;
  bool disabled = false;
  mutable size_t error_count = 0;

  static std::unordered_set<std::string> AllVisitors;

  bool DebugIsEnabled() const { return debug_visit; }
  bool TraceIsEnabled() const { return trace_visit; }

public:
  Visitor(const std::string& n, const ptr<SymbolTable>& s_tab = nullptr,
          bool ugs = false)
      : scoped_symtab(s_tab), use_global_symtab(ugs), name(ToUpper(n)) {
    if (name.empty()) choreo_unreachable("a visitor must be named.");

    if (AllVisitors.count(name))
      choreo_unreachable("found visitor with same name: \"" + n + "\".");

    // to be deprecated. currently it is only used for quick debug
    if (std::getenv("TRACE")) {
      auto trace = ToUpper(std::string(std::getenv("TRACE")));
      if (ContainsExact(trace, name)) trace_visit = true;
    }

    if (std::getenv("CHOREO_TRACE_VISITOR")) {
      auto trace = ToUpper(std::string(std::getenv("CHOREO_TRACE_VISITOR")));
      if (ContainsExact(trace, name)) trace_visit = true;
    }

    if (std::getenv("CHOREO_DEBUG_VISITOR")) {
      auto debug = ToUpper(std::string(std::getenv("CHOREO_DEBUG_VISITOR")));
      if (ContainsExact(debug, name)) debug_visit = true;
    }

    if (std::getenv("CHOREO_PRINT_BEFORE")) {
      auto before = ToUpper(std::string(std::getenv("CHOREO_PRINT_BEFORE")));
      if (ContainsExact(before, "ALLPASSES")) print_ahead = true;
      if (ContainsExact(before, name)) print_ahead = true;
    }

    if (std::getenv("CHOREO_PRINT_AFTER")) {
      auto after = ToUpper(std::string(std::getenv("CHOREO_PRINT_AFTER")));
      if (ContainsExact(after, "ALLPASSES")) print_after = true;
      if (ContainsExact(after, name)) print_after = true;
    }

    if (std::getenv("CHOREO_DUMP_SYMTAB_AFTER")) {
      auto dump = ToUpper(std::string(std::getenv("CHOREO_DUMP_SYMTAB_AFTER")));
      if (ContainsExact(dump, name)) dsyms_after = true;
    }

    if (std::getenv("CHOREO_STOP_AFTER_PASS")) {
      auto abend = ToUpper(std::string(std::getenv("CHOREO_STOP_AFTER_PASS")));
      if (ContainsExact(abend, name)) abend_after = true;
    }

    if (std::getenv("CHOREO_DISABLE_VISIT")) {
      auto disable = ToUpper(std::string(std::getenv("CHOREO_DISABLE_VISIT")));
      if (ContainsExact(disable, name)) disabled = true;
    }

    if (std::getenv("CHOREO_PRINT_PASSES")) prt_visitor = true;

    if (std::getenv("CHOREO_PRINT_NODETYPE")) prt_node_ty = true;

    if (std::getenv("CHOREO_ANALYZE_DEVICE_FUNCTIONS")) resolve_fns = true;
  }

  virtual ~Visitor() {}

  virtual void SetTraceVisit(bool t) { trace_visit = t; }
  virtual void SetDebugVisit(bool d) { debug_visit = d; }

  // simple reference to the symbol table
  virtual ScopedSymbolTable& SSTab() { return scoped_symtab; }
  virtual const ScopedSymbolTable& SSTab() const { return scoped_symtab; }

  virtual const ptr<SymbolTable> SymTab() const {
    if (auto st = scoped_symtab.GlobalSymbolTable()) return st;
    choreo_unreachable("Retrieving an invalid symbol table.");
    return nullptr;
  }

  virtual const std::string& GetName() { return name; }

  virtual void SetLevelPrefix(const std::string& pfx) { lvl_pfx = pfx; }
  const std::string LevelPrefix() const { return lvl_pfx; }

  virtual bool RunOnProgram(AST::Node& root) = 0;
  virtual bool IsAllowed(AST::Node&) const { return true; }

  virtual bool RunOnProgramImpl(AST::Node& root) {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    root.accept(*this);

    if (HasError() || abend_after) return false;

    return true;
  }

public:
  // The node type can only be used when symbol table is constructed or under
  // construction
  virtual ptr<Type> NodeType(const AST::Node& n) const {
    if (auto id = dyn_cast<AST::Identifier>(&n))
      return GetSymbolType(id->name);
    else if (auto expr = dyn_cast<AST::Expr>(&n)) {
      if (auto id = expr->GetSymbol()) {
        return GetSymbolType(id->name);
      } else if (expr->op == "dataof" || expr->op == "mdataof") {
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (!GetSymbolType(id->name)) {
            // TODO: make NodeType be used properly
            // choreo_unreachable("\"dataof\" operation refers undefined symbol
            // '" + id->name + "'.");
            return nullptr;
          }
          return GetSymbolType(id->name +
                               (expr->op == "mdataof" ? ".mdata" : ".data"));
        }
      } else if (expr->op == "spanof") {
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (!GetSymbolType(id->name)) // make sure the symbol exists
            choreo_unreachable(
                "\"spanof\" operation refers undefined symbol '" + id->name +
                "'.");
          return GetSymbolType(id->name + ".span");
        }
      }
    }
    return n.GetType();
  }

protected:
  size_t CountMultiValues(const ptr<AST::MultiValues>& mv) {
    size_t count = 0;
    for (auto v : mv->AllValues()) {
      auto vty = NodeType(*v);
      assert((vty->Dims() != 0) && IsValidRank(vty->Dims()));
      count += vty->Dims();
    }
    return count;
  }

  // simple dependence runner
  template <typename VisitorType>
  std::unique_ptr<VisitorType> GetResult(AST::Node& root) const {
    auto v = std::make_unique<VisitorType>();
    v->SetLevelPrefix(LevelPrefix());
    if (DebugIsEnabled()) v->SetDebugVisit(true);
    if (TraceIsEnabled()) v->SetTraceVisit(true);
    v->RunOnProgram(root);
    return v;
  }

public:
  static bool shell_supports_colors() {
    const char* term = getenv("TERM");
    return term &&
           (strcmp(term, "xterm-256color") == 0 || strcmp(term, "xterm") == 0);
  }

  static bool should_use_colors() {
    return isatty(fileno(stdout)) && shell_supports_colors();
  }

public:
  virtual const ptr<Type> GetSymbolType(const std::string& n) const {
    return scoped_symtab.LookupSymbol(n);
  }

private:
  static constexpr const char* color_red = "\033[31m";
  static constexpr const char* color_yellow = "\033[33m";
  static constexpr const char* color_blue = "\033[34m";
  static constexpr const char* color_reset = "\033[0m";

protected:
  void ShowSourceLocation(const location& l) const {
    if (!CCtx().ShowSourceLocation()) return;

    // Retrieve the line that caused the error
    std::string error_line = CCtx().GetSourceLine(l.begin.line);
    if (!error_line.empty()) {
      errs() << "  " << error_line << "\n"; // Print the source line

      // Print caret (^) under the error position
      errs() << "  ";
      for (int i = 1; i < l.begin.column; ++i)
        errs() << " "; // Align the caret with the exact error position

      errs() << "^" << "\n";
    }
  }

public:
  void Error(const location& loc, const std::string& message) const {
    errs() << loc << ": " << (should_use_colors() ? color_red : "")
           << "error: " << (should_use_colors() ? color_reset : "");
    errs() << message << "\n";
    ShowSourceLocation(loc);
  }

  void Warning(const location& loc, const std::string& message) const {
    if (CCtx().InhibitWarning()) return;
    if (CCtx().WarningAsError()) {
      Error1(loc, message);
      return;
    }
    errs() << loc << ": " << (should_use_colors() ? color_yellow : "")
           << "warning: " << (should_use_colors() ? color_reset : "");
    errs() << message << "\n";
    ShowSourceLocation(loc);
  }

  void Note(const location& loc, const std::string& message) const {
    if (CCtx().InhibitWarning()) return;
    errs() << loc << ": " << ((should_use_colors()) ? color_blue : "")
           << "info: " << (should_use_colors() ? color_reset : "");
    errs() << message << "\n";
    ShowSourceLocation(loc);
  }

  // short-hand: emit error with the error count incremented by 1
  void Error1(const location& loc, const std::string& message) const {
    Error(loc, message);
    error_count++;
  }

  virtual int Status() { return error_count; }
  virtual bool HasError() const {
    if (error_count > 0) {
      dbgs() << "Totally " << error_count << " errors have been detected.\n";
      return true;
    }
    return false;
  }
};

// A visitor with simple symbol auto scoping functionality
struct VisitorWithScope : public Visitor {
protected:
  // for the derived classes
  virtual bool BeforeVisitImpl(AST::Node&) = 0;
  virtual bool AfterVisitImpl(AST::Node&) = 0;
  virtual bool InMidVisitImpl(AST::Node&) { return true; }

  // Tricky: sometimes it requires action before entering the scope
  virtual bool BeforeBeforeVisit(AST::Node&) { return true; }

  // special to within: map 'with' to its 'with-matchers'
  std::unordered_map<std::string, std::vector<std::string>> within_map;
  // similar for parallel-by
  std::unordered_map<std::string, std::vector<std::string>> pb_map;
  // all bounded variable mapping, including within and parallel-by
  std::unordered_map<std::string, std::vector<std::string>> bv_map;

  std::string fname; // current function name

private:
  int pb_count = 0; // counting for parallel_by
  int wi_count = 0; // counting for with_in
  int fe_count = 0; // counting for foreach
  int it_count = 0; // counting for inthreads
  int wl_count = 0; // counting for while
  int ie_count = 0; // counting for ifelse

  void Reset() {
    pb_count = 0;
    wi_count = 0;
    fe_count = 0;
    it_count = 0;
    wl_count = 0;
  }

public:
  bool BeforeVisit(AST::Node& n) final {
    BeforeBeforeVisit(n);
    Visitor::BeforeVisit(n);
    if (isa<AST::Program>(&n)) {
      Reset();
      SSTab().EnterScope(""); // global scope
      // fixed symbol definition
      SSTab().DefineSymbol("__choreo_no_tiling__",
                           MakeBoundedITupleType(MultiBounds(1, 1)));
      SSTab().DefineSymbol("@__choreo_no_tiling__", MakeIntegerType());
      SSTab().DefineSymbol("__choreo_parent_dim__", MakeIntegerType());
    } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      SSTab().EnterScope(f->name);
      fname = f->name;
    } else if (auto p = dyn_cast<AST::ParallelBy>(&n)) {
      SSTab().EnterScope("paraby_" + std::to_string(pb_count++));
      std::string scope_name = scoped_symtab.ScopeName();
      std::vector<std::string> matchers;
      // map the parallel variable to its matchers
      if (p->HasSubPVs()) {
        for (auto v : p->AllSubPVs())
          matchers.push_back(scope_name + cast<AST::Identifier>(v)->name);
      } else
        matchers.push_back(scope_name + p->BPV()->name); // only map to itself
      pb_map.emplace(scope_name + p->BPV()->name, matchers);
      bv_map.emplace(scope_name + p->BPV()->name, matchers);
      // map the sub parallel variables to their matchers
      for (auto v : p->AllSubPVs()) {
        auto name = scope_name + cast<AST::Identifier>(v)->name;
        std::vector<std::string> matchers;
        matchers.push_back(name);
        pb_map.emplace(name, matchers);
        bv_map.emplace(name, matchers);
      }
    } else if (isa<AST::WithBlock>(&n)) {
      SSTab().EnterScope("within_" + std::to_string(wi_count++));
    } else if (isa<AST::ForeachBlock>(&n)) {
      SSTab().EnterScope("foreach_" + std::to_string(fe_count++));
    } else if (isa<AST::InThreadsBlock>(&n)) {
      SSTab().EnterScope("inthreads_" + std::to_string(it_count++));
    } else if (isa<AST::WhileBlock>(&n)) {
      SSTab().EnterScope("while_" + std::to_string(wl_count++));
    } else if (isa<AST::IfElseBlock>(&n)) {
      SSTab().EnterScope("cond_if_" + std::to_string(ie_count++));
    } else if (isa<AST::IncrementBlock>(&n)) {
      SSTab().EnterScope("increment_" + std::to_string(fe_count++));
    } else if (auto w = dyn_cast<AST::WithIn>(&n)) {
      std::string scope_name = scoped_symtab.ScopeName();
      if (w->with) {
        std::vector<std::string> matchers;
        if (w->with_matchers) {
          for (auto v : w->GetMatchers())
            matchers.push_back(scope_name + cast<AST::Identifier>(v)->name);
        } else
          matchers.push_back(scope_name + w->with->name); // only map to itself
        within_map.emplace(scope_name + w->with->name, matchers);
        bv_map.emplace(scope_name + w->with->name, matchers);
      }
      if (w->with_matchers) {
        for (auto v : w->GetMatchers()) {
          auto sname = scope_name + cast<AST::Identifier>(v)->name;
          within_map.emplace(
              sname, std::vector<std::string>{sname}); // always map to itself
          bv_map.emplace(sname, std::vector<std::string>{sname});
        }
      }
    }
    return BeforeVisitImpl(n); // derived class to customize
  }

  bool InMidVisit(AST::Node& n) final {
    // this is very specialized code
    if (isa<AST::IfElseBlock>(&n)) {
      SSTab().LeaveScope();
      SSTab().EnterScope("cond_else" + std::to_string(ie_count));
    }
    return InMidVisitImpl(n);
  }

  bool AfterVisit(AST::Node& n) final {
    AfterVisitImpl(n); // derived class to customize
    if (isa<AST::Program>(&n)) {
      Reset();
      SSTab().LeaveScope();
      assert(SSTab().ScopeDepth() == 0 && "internal error: scope is not zero.");
    } else if (isa<AST::ChoreoFunction>(&n)) {
      fname = "";
      SSTab().LeaveScope();
    } else if (isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
               isa<AST::ForeachBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
               isa<AST::WhileBlock>(&n) || isa<AST::IfElseBlock>(&n) ||
               isa<AST::IncrementBlock>(&n)) {
      SSTab().LeaveScope();
    }

    return Visitor::AfterVisit(n);
  }

  const std::string& CurrentFunctionName() const { return fname; }

public:
  VisitorWithScope(const std::string& n,
                   const ptr<SymbolTable>& s_tab = nullptr, bool ugs = false)
      : Visitor(n, s_tab, ugs) {
    Reset();
  }
  ~VisitorWithScope() {}

  virtual const std::string InScopeName(const std::string& sym) const {
    return scoped_symtab.InScopeName(sym);
  }

  virtual const std::string UnScopedName(const std::string& name) const {
    return scoped_symtab.UnScopedName(name);
  }

  virtual const std::string GetScope(const std::string& name) const {
    return scoped_symtab.GetScope(name);
  }

  bool RunOnProgram(AST::Node& root) final {
    if (!IsAllowed(root) || disabled) return true;
    if (use_global_symtab)
      scoped_symtab.UpdateGlobal(CCtx().GetGlobalSymbolTable());
    if (prt_visitor) dbgs() << LevelPrefix() << "|- " << GetName() << NewL;
    return RunOnProgramImpl(root);
  }

public:
  // provide the defaults
  bool Visit(AST::MultiNodes&) override { return true; }
  bool Visit(AST::MultiValues&) override { return true; }
  bool Visit(AST::NoValue&) override { return true; }
  bool Visit(AST::IntLiteral&) override { return true; }
  bool Visit(AST::FloatLiteral&) override { return true; }
  bool Visit(AST::StringLiteral&) override { return true; }
  bool Visit(AST::BoolLiteral&) override { return true; }
  bool Visit(AST::Expr&) override { return true; }
  bool Visit(AST::CastExpr&) override { return true; }
  bool Visit(AST::AttributeExpr&) override { return true; }
  bool Visit(AST::MultiDimSpans&) override { return true; }
  bool Visit(AST::NamedTypeDecl&) override { return true; }
  bool Visit(AST::NamedVariableDecl&) override { return true; }
  bool Visit(AST::IntTuple&) override { return true; }
  bool Visit(AST::DataAccess&) override { return true; }
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
  bool Visit(AST::DMA&) override { return true; }
  bool Visit(AST::MMA&) override { return true; }
  bool Visit(AST::ChunkAt&) override { return true; }
  bool Visit(AST::Wait&) override { return true; }
  bool Visit(AST::Trigger&) override { return true; }
  bool Visit(AST::Break&) override { return true; }
  bool Visit(AST::Continue&) override { return true; }
  bool Visit(AST::Call&) override { return true; }
  bool Visit(AST::Rotate&) override { return true; }
  bool Visit(AST::Synchronize&) override { return true; }
  bool Visit(AST::Select&) override { return true; }
  bool Visit(AST::Return&) override { return true; }
  bool Visit(AST::LoopRange&) override { return true; }
  bool Visit(AST::ForeachBlock&) override { return true; }
  bool Visit(AST::InThreadsBlock&) override { return true; }
  bool Visit(AST::WhileBlock&) override { return true; }
  bool Visit(AST::IfElseBlock&) override { return true; }
  bool Visit(AST::IncrementBlock&) override { return true; }
  bool Visit(AST::FunctionDecl&) override { return true; }
  bool Visit(AST::ChoreoFunction&) override { return true; }
  bool Visit(AST::CppSourceCode&) override { return true; }
  bool Visit(AST::DeviceFunctionDecl&) override { return true; }
  bool Visit(AST::Program&) override { return true; }
};

// This accepts static symbol table and provide symbol lookup capability
// Caution: must be used when symbol table does not change.
struct VisitorWithSymTab : public VisitorWithScope {
protected:
  const std::string InScopeName(const std::string& sym) const final {
    auto removeLastLevel = [](const std::string& input) -> std::string {
      size_t lastPos = input.rfind("::");
      if (lastPos == std::string::npos) {
        return input; // No "::" found, return the original string
      }

      // Find the second-to-last "::" by searching up to the last found
      // position
      size_t secondLastPos = input.rfind("::", lastPos - 1);
      if (secondLastPos == std::string::npos) return input;
      return input.substr(0,
                          secondLastPos + 2); // Include the "::" in the result
    };
    std::string scope_name = scoped_symtab.ScopeName();
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

  const std::string UnScopedName(const std::string& name) const override {
    return scoped_symtab.UnScopedName(name);
  }

public:
  // use the immutable symbol table directly
  const ptr<Type> GetSymbolType(const std::string& n) const override {
    return SymTab()->GetSymbol(InScopeName(n))->GetType();
  }

  virtual ptr<Type> GetScopedSymbolType(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "expected a scoped symbol.");
    return SymTab()->GetSymbol(n)->GetType();
  }

public:
  VisitorWithSymTab(const std::string& n, const ptr<SymbolTable>& s_tab,
                    bool ugs = false)
      : VisitorWithScope(n, s_tab, ugs) {}
  // utilize the global symbol table
  VisitorWithSymTab(const std::string& n)
      : VisitorWithScope(n, nullptr, true) {}
  ~VisitorWithSymTab() {}
};

struct TracedVisitorWithSymTab : public VisitorWithSymTab {
public:
  TracedVisitorWithSymTab(const std::string& n, const ptr<SymbolTable>& s_tab)
      : VisitorWithSymTab(n, s_tab) {}
  TracedVisitorWithSymTab(const std::string& n) : VisitorWithSymTab(n) {}
  ~TracedVisitorWithSymTab() {}

public:
  virtual void TraceEachVisit(AST::Node& n, bool show_detail = false,
                              const std::string& prefix = "") {
    if (!trace_visit) return;
    if (show_detail)
      dbgs() << prefix << STR(n) << "\n";
    else
      dbgs() << prefix << n.TypeNameString() << "\n";
  }

  // delegate to VisitNode
  bool Visit(AST::MultiNodes& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::MultiValues& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::NoValue& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::IntLiteral& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::FloatLiteral& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::StringLiteral& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::BoolLiteral& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Expr& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::CastExpr& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::AttributeExpr& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::MultiDimSpans& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::NamedTypeDecl& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::NamedVariableDecl& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::IntTuple& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::DataAccess& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Assignment& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::IntIndex& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::DataType& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Identifier& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Parameter& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::ParamList& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::ParallelBy& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::WhereBind& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::WithIn& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::WithBlock& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Memory& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::SpanAs& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::DMA& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::MMA& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::ChunkAt& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Wait& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Trigger& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Break& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Call& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Rotate& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Synchronize& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Select& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Return& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::LoopRange& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::ForeachBlock& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::InThreadsBlock& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::WhileBlock& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::IfElseBlock& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::IncrementBlock& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::FunctionDecl& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::ChoreoFunction& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::CppSourceCode& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }
  bool Visit(AST::Program& n) final {
    TraceEachVisit(n);
    return VisitNode(n);
  }

public:
  // provide default
  virtual bool VisitNode(AST::MultiNodes&) { return true; }
  virtual bool VisitNode(AST::MultiValues&) { return true; }
  virtual bool VisitNode(AST::NoValue&) { return true; }
  virtual bool VisitNode(AST::IntLiteral&) { return true; }
  virtual bool VisitNode(AST::FloatLiteral&) { return true; }
  virtual bool VisitNode(AST::StringLiteral&) { return true; }
  virtual bool VisitNode(AST::BoolLiteral&) { return true; }
  virtual bool VisitNode(AST::Expr&) { return true; }
  virtual bool VisitNode(AST::CastExpr&) { return true; }
  virtual bool VisitNode(AST::AttributeExpr&) { return true; }
  virtual bool VisitNode(AST::MultiDimSpans&) { return true; }
  virtual bool VisitNode(AST::NamedTypeDecl&) { return true; }
  virtual bool VisitNode(AST::NamedVariableDecl&) { return true; }
  virtual bool VisitNode(AST::IntTuple&) { return true; }
  virtual bool VisitNode(AST::DataAccess&) { return true; }
  virtual bool VisitNode(AST::Assignment&) { return true; }
  virtual bool VisitNode(AST::IntIndex&) { return true; }
  virtual bool VisitNode(AST::DataType&) { return true; }
  virtual bool VisitNode(AST::Identifier&) { return true; }
  virtual bool VisitNode(AST::Parameter&) { return true; }
  virtual bool VisitNode(AST::ParamList&) { return true; }
  virtual bool VisitNode(AST::ParallelBy&) { return true; }
  virtual bool VisitNode(AST::WhereBind&) { return true; }
  virtual bool VisitNode(AST::WithIn&) { return true; }
  virtual bool VisitNode(AST::WithBlock&) { return true; }
  virtual bool VisitNode(AST::Memory&) { return true; }
  virtual bool VisitNode(AST::SpanAs&) { return true; }
  virtual bool VisitNode(AST::DMA&) { return true; }
  virtual bool VisitNode(AST::MMA&) { return true; }
  virtual bool VisitNode(AST::ChunkAt&) { return true; }
  virtual bool VisitNode(AST::Wait&) { return true; }
  virtual bool VisitNode(AST::Trigger&) { return true; }
  virtual bool VisitNode(AST::Break&) { return true; }
  virtual bool VisitNode(AST::Call&) { return true; }
  virtual bool VisitNode(AST::Rotate&) { return true; }
  virtual bool VisitNode(AST::Synchronize&) { return true; }
  virtual bool VisitNode(AST::Select&) { return true; }
  virtual bool VisitNode(AST::Return&) { return true; }
  virtual bool VisitNode(AST::LoopRange&) { return true; }
  virtual bool VisitNode(AST::ForeachBlock&) { return true; }
  virtual bool VisitNode(AST::InThreadsBlock&) { return true; }
  virtual bool VisitNode(AST::WhileBlock&) { return true; }
  virtual bool VisitNode(AST::IfElseBlock&) { return true; }
  virtual bool VisitNode(AST::IncrementBlock&) { return true; }
  virtual bool VisitNode(AST::FunctionDecl&) { return true; }
  virtual bool VisitNode(AST::ChoreoFunction&) { return true; }
  virtual bool VisitNode(AST::CppSourceCode&) { return true; }
  virtual bool VisitNode(AST::Program&) { return true; }
};

struct LoopVisitor : public VisitorWithSymTab {
protected:
  ptr<Loop> cur_loop;

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) { dbgs() << n.TypeNameString() << "\n"; }
  }

  virtual bool AfterBeforeVisitImpl(AST::Node&) { return true; }
  virtual bool BeforeAfterVisitImpl(AST::Node&) { return true; }

  bool BeforeVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "before visiting " << n.TypeNameString() << "\n";
    if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
      cur_loop = fb->loop;
      assert(cur_loop != nullptr && "internal error: loop is null.");
    }
    AfterBeforeVisitImpl(n);
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "after visiting " << n.TypeNameString() << "\n";
    BeforeAfterVisitImpl(n);
    if (isa<AST::ForeachBlock>(&n)) { cur_loop = cur_loop->GetParentLoop(); }
    return true;
  }

  bool InLoop() { return cur_loop != nullptr; }
  std::string LoopName() { return cur_loop ? cur_loop->LoopName() : ""; }

public:
  LoopVisitor(const ptr<SymbolTable> s_tab, const std::string& pn)
      : VisitorWithSymTab(pn, s_tab), cur_loop(nullptr) {}
  LoopVisitor(const std::string& pn)
      : VisitorWithSymTab(pn), cur_loop(nullptr) {}
};

class VisitorGroup : public Visitor {
private:
  std::vector<Visitor*> members;

public:
  VisitorGroup();

  template <typename... Visitors>
  VisitorGroup(const std::string& n, Visitors&... visitors)
      : Visitor(n, nullptr, false) {
    (members.push_back(&visitors), ...); // C++17 fold expression
  }

  //  VisitorGroup(const std::string& n) :  Visitor(n, nullptr, false) {}
  bool RunOnProgram(AST::Node& root) final {
    if (!IsAllowed(root) || disabled) return true;
    if (prt_visitor) dbgs() << LevelPrefix() << "|-+- " << GetName() << NewL;

    // Run in sequence
    for (auto& v : members) {
      v->SetLevelPrefix(LevelPrefix() + "  ");
      if (DebugIsEnabled()) v->SetDebugVisit(true);
      if (TraceIsEnabled()) v->SetTraceVisit(true);
      if (!v->RunOnProgram(root)) return false;
    }

    if (abend_after) return false;

    return true;
  }

private:
  // disable the interfaces
  bool InMidVisit(AST::Node&) final { return true; }
  bool Visit(AST::MultiNodes&) final { return true; }
  bool Visit(AST::MultiValues&) final { return true; }
  bool Visit(AST::NoValue&) final { return true; }
  bool Visit(AST::IntLiteral&) final { return true; }
  bool Visit(AST::FloatLiteral&) final { return true; }
  bool Visit(AST::StringLiteral&) final { return true; }
  bool Visit(AST::BoolLiteral&) final { return true; }
  bool Visit(AST::Expr&) final { return true; }
  bool Visit(AST::CastExpr&) final { return true; }
  bool Visit(AST::AttributeExpr&) final { return true; }
  bool Visit(AST::MultiDimSpans&) final { return true; }
  bool Visit(AST::NamedTypeDecl&) final { return true; }
  bool Visit(AST::NamedVariableDecl&) final { return true; }
  bool Visit(AST::IntTuple&) final { return true; }
  bool Visit(AST::DataAccess&) final { return true; }
  bool Visit(AST::Assignment&) final { return true; }
  bool Visit(AST::IntIndex&) final { return true; }
  bool Visit(AST::DataType&) final { return true; }
  bool Visit(AST::Identifier&) final { return true; }
  bool Visit(AST::Parameter&) final { return true; }
  bool Visit(AST::ParamList&) final { return true; }
  bool Visit(AST::ParallelBy&) final { return true; }
  bool Visit(AST::WhereBind&) final { return true; }
  bool Visit(AST::WithIn&) final { return true; }
  bool Visit(AST::WithBlock&) final { return true; }
  bool Visit(AST::Memory&) final { return true; }
  bool Visit(AST::SpanAs&) final { return true; }
  bool Visit(AST::DMA&) final { return true; }
  bool Visit(AST::MMA&) final { return true; }
  bool Visit(AST::ChunkAt&) final { return true; }
  bool Visit(AST::Wait&) final { return true; }
  bool Visit(AST::Trigger&) final { return true; }
  bool Visit(AST::Break&) final { return true; }
  bool Visit(AST::Continue&) final { return true; }
  bool Visit(AST::Call&) final { return true; }
  bool Visit(AST::Rotate&) final { return true; }
  bool Visit(AST::Synchronize&) final { return true; }
  bool Visit(AST::Select&) final { return true; }
  bool Visit(AST::Return&) final { return true; }
  bool Visit(AST::LoopRange&) final { return true; }
  bool Visit(AST::ForeachBlock&) final { return true; }
  bool Visit(AST::InThreadsBlock&) final { return true; }
  bool Visit(AST::WhileBlock&) final { return true; }
  bool Visit(AST::IfElseBlock&) final { return true; }
  bool Visit(AST::IncrementBlock&) final { return true; }
  bool Visit(AST::FunctionDecl&) final { return true; }
  bool Visit(AST::ChoreoFunction&) final { return true; }
  bool Visit(AST::CppSourceCode&) final { return true; }
  bool Visit(AST::DeviceFunctionDecl&) final { return true; }
  bool Visit(AST::Program&) final { return true; }
}; // VisitorGroup

inline const std::set<std::string>
ReferredSymbols(AST::Node* n, const VisitorWithScope* v = nullptr) {
  std::set<std::string> res;
  if (n == nullptr) {
    // nothing to do
  } else if (auto id = dyn_cast<AST::Identifier>(n)) {
    if (v)
      res.insert(v->InScopeName(id->name));
    else
      res.insert(id->name);
  } else if (auto expr = dyn_cast<AST::Expr>(n)) {
    auto r = ReferredSymbols(expr->GetR().get(), v);
    auto l = ReferredSymbols(expr->GetL().get(), v);
    auto c = ReferredSymbols(expr->GetC().get(), v);
    res.insert(r.begin(), r.end());
    res.insert(l.begin(), l.end());
    res.insert(c.begin(), c.end());
  } else if (auto mv = dyn_cast<AST::MultiValues>(n)) {
    for (auto sv : mv->AllValues()) {
      auto r = ReferredSymbols(sv.get(), v);
      res.insert(r.begin(), r.end());
    }
  } else if (auto ca = dyn_cast<AST::ChunkAt>(n)) {
    for (auto& sop : ca->AllOperations()) {
      if (auto t = dyn_cast<AST::SOP::Tiling>(sop)) {
        auto r = ReferredSymbols(t->GetTilingFactors().get(), v);
        res.insert(r.begin(), r.end());
      } else if (auto t = dyn_cast<AST::SOP::TileAt>(sop)) {
        auto r0 = ReferredSymbols(t->GetTilingFactors().get(), v);
        res.insert(r0.begin(), r0.end());
        auto r1 = ReferredSymbols(t->GetIndices().get(), v);
        res.insert(r1.begin(), r1.end());
      } else if (auto t = dyn_cast<AST::SOP::SubSpan>(sop)) {
        auto r0 = ReferredSymbols(t->GetSubSpan().get(), v);
        res.insert(r0.begin(), r0.end());
        auto r1 = ReferredSymbols(t->GetIndices().get(), v);
        res.insert(r1.begin(), r1.end());
        auto r2 = ReferredSymbols(t->GetStrides().get(), v);
        res.insert(r2.begin(), r2.end());
      } else if (auto t = dyn_cast<AST::SOP::ModSpan>(sop)) {
        auto r0 = ReferredSymbols(t->GetSubSpan().get(), v);
        res.insert(r0.begin(), r0.end());
        auto r1 = ReferredSymbols(t->GetIndices().get(), v);
        res.insert(r1.begin(), r1.end());
        auto r2 = ReferredSymbols(t->GetStrides().get(), v);
        res.insert(r2.begin(), r2.end());
      } else if (auto t = dyn_cast<AST::SOP::View>(sop)) {
        auto r0 = ReferredSymbols(t->GetSubSpan().get(), v);
        res.insert(r0.begin(), r0.end());
        auto r1 = ReferredSymbols(t->GetOffsets().get(), v);
        res.insert(r1.begin(), r1.end());
        auto r2 = ReferredSymbols(t->GetStrides().get(), v);
        res.insert(r2.begin(), r2.end());
      } else if (auto t = dyn_cast<AST::SOP::Reshape>(sop)) {
        auto r0 = ReferredSymbols(t->GetNewSpan().get(), v);
        res.insert(r0.begin(), r0.end());
      }
      if (v)
        res.insert(v->InScopeName(ca->RefSymbol()));
      else
        res.insert(ca->RefSymbol());
    }
  }

  return res;
}

} // end namespace Choreo

#endif // __CHOREO_VISITOR_HPP__
