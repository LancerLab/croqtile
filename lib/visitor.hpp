#ifndef __CHOREO_VISITOR_HPP__
#define __CHOREO_VISITOR_HPP__

#include <cstring>
#include <iostream>
#include <optional>
#include <unistd.h>
#include <unordered_set>

#include "ast.hpp"
#include "location.hh"

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
        dbgs() << STR(n) << NewL;
        dbgs() << SprT << " Before " << name << ": " << f->name << " (End) "
               << SprT << NewL;
      }
    }
    return true;
  }

  virtual bool AfterVisit(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) { // by function print
      if (print_after) {
        dbgs() << NewL << SprT << " After " << name << ": " << f->name
               << " (Begin) " << SprT << NewL;
        dbgs() << STR(n) << NewL;
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
  virtual bool Visit(AST::IntLiteral&) = 0;
  virtual bool Visit(AST::Boolean&) = 0;
  virtual bool Visit(AST::Expr&) = 0;
  virtual bool Visit(AST::MultiDimSpans&) = 0;
  virtual bool Visit(AST::NamedTypeDecl&) = 0;
  virtual bool Visit(AST::NamedVariableDecl&) = 0;
  virtual bool Visit(AST::IntTuple&) = 0;
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
  virtual bool Visit(AST::ChunkAt&) = 0;
  virtual bool Visit(AST::Wait&) = 0;
  virtual bool Visit(AST::Call&) = 0;
  virtual bool Visit(AST::Rotate&) = 0;
  virtual bool Visit(AST::Select&) = 0;
  virtual bool Visit(AST::Return&) = 0;
  virtual bool Visit(AST::LoopRange&) = 0;
  virtual bool Visit(AST::ForeachBlock&) = 0;
  virtual bool Visit(AST::IncrementBlock&) = 0;
  virtual bool Visit(AST::FunctionDecl&) = 0;
  virtual bool Visit(AST::ChoreoFunction&) = 0;
  virtual bool Visit(AST::CppSourceCode&) = 0;
  virtual bool Visit(AST::Program&) = 0;

protected:
  // scoped variable handling
  ScopedSymbolTable scoped_symtab;

protected:
  std::string name;
  bool trace_visit = false;
  bool debug_visit = false;
  bool print_ahead = false;
  bool print_after = false;
  bool dsyms_after = false;
  bool abend_after = false;
  bool prt_visitor = false;
  size_t error_count = 0;

  static std::unordered_set<std::string> AllVisitors;

  bool DebugIsEnabled() const { return debug_visit; }

public:
  Visitor(const std::string& n, const ptr<SymbolTable>& s_tab = nullptr)
      : scoped_symtab(s_tab), name(ToUpper(n)) {
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

    if (std::getenv("CHOREO_PRINT_PASSES")) prt_visitor = true;
  }

  virtual ~Visitor() {}

  virtual void SetTraceVisit(bool t) { trace_visit = t; }
  virtual void SetDebugVisit(bool d) { debug_visit = d; }

  // simple reference to the symbol table
  virtual ScopedSymbolTable& SSTab() { return scoped_symtab; }

  virtual const ptr<SymbolTable> SymTab() const {
    if (auto st = scoped_symtab.GlobalSymbolTable()) return st;
    choreo_unreachable("Retrieving an invalid symbol table.");
    return nullptr;
  }

  virtual const std::string& GetName() { return name; }

  virtual bool RunOnProgram(AST::Node& root) {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    if (prt_visitor) dbgs() << "|- " << GetName() << NewL;

    root.accept(*this);

    if (HasError() || abend_after) return false;

    return true;
  }

public:
  virtual ptr<Type> NodeType(const AST::Node& n) const {
    if (auto id = dyn_cast<AST::Identifier>(&n))
      return GetSymbolType(id->name);
    else if (auto expr = dyn_cast<AST::Expr>(&n)) {
      if (auto id = expr->GetSymbol()) {
        return GetSymbolType(id->name);
      } else if (expr->op == "dataof") {
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (!GetSymbolType(id->name)) {
            // choreo_unreachable("\"dataof\" operation refers undefined symbol
            // '" +  id->name + "'.");
            return nullptr;
          }
          return GetSymbolType(id->name + ".data");
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
  virtual ptr<Type> GetSymbolType(const std::string& n) const {
    return scoped_symtab.LookupSymbol(n);
  }

private:
  static constexpr const char* color_red = "\033[31m";
  static constexpr const char* color_yellow = "\033[33m";
  static constexpr const char* color_reset = "\033[0m";

public:
  void Error(const location& loc, const std::string& message) {
    errs() << loc << ": " << ((should_use_colors()) ? color_red : "")
           << "error: " << ((should_use_colors()) ? color_reset : "");
    errs() << message << "\n";
  }

  void Warning(const location& loc, const std::string& message) {
    errs() << loc << ": " << ((should_use_colors()) ? color_yellow : "")
           << "warning: " << ((should_use_colors()) ? color_reset : "");
    errs() << message << "\n";
  }

  void Note(const location& loc, const std::string& message) {
    errs() << loc << ": note: " << message << std::endl;
  }

  virtual int Status() { return error_count; }
  virtual bool HasError() { return error_count != 0; }
};

// A visitor with simple symbol auto scoping functionality
struct VisitorWithScope : public Visitor {
protected:
  // for the derived classes
  virtual bool BeforeVisitImpl(AST::Node& n) = 0;
  virtual bool AfterVisitImpl(AST::Node& n) = 0;

  // Tricky: sometimes it requires action before entering the scope
  virtual bool BeforeBeforeVisit(AST::Node&) { return true; }

  // special to within: map 'with' to its 'with-matchers'
  std::unordered_map<std::string, std::vector<std::string>> within_map;

  std::string fname; // current function name

private:
  int pb_count = 0; // counting for parallel_by
  int wi_count = 0; // counting for with_in
  int fe_count = 0; // counting for foreach

  void Reset() {
    pb_count = 0;
    wi_count = 0;
    fe_count = 0;
  }

public:
  bool BeforeVisit(AST::Node& n) final {
    BeforeBeforeVisit(n);
    Visitor::BeforeVisit(n);
    if (isa<AST::Program>(&n)) {
      Reset();
      SSTab().EnterScope(""); // global scope
    } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      SSTab().EnterScope(f->name);
      fname = f->name;
    } else if (isa<AST::ParallelBy>(&n)) {
      SSTab().EnterScope("paraby_" + std::to_string(pb_count++));
    } else if (isa<AST::WithBlock>(&n)) {
      SSTab().EnterScope("within_" + std::to_string(wi_count++));
    } else if (isa<AST::ForeachBlock>(&n)) {
      SSTab().EnterScope("foreach_" + std::to_string(fe_count++));
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
      }
      if (w->with_matchers) {
        for (auto v : w->GetMatchers()) {
          auto sname = scope_name + cast<AST::Identifier>(v)->name;
          within_map.emplace(
              sname, std::vector<std::string>{sname}); // always map to itself
        }
      }
    }
    return BeforeVisitImpl(n); // derived class to customize
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
               isa<AST::ForeachBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
      SSTab().LeaveScope();
    }

    return Visitor::AfterVisit(n);
  }

  const std::string& CurrentFunctionName() const { return fname; }

public:
  VisitorWithScope(const std::string& n,
                   const ptr<SymbolTable>& s_tab = nullptr)
      : Visitor(n, s_tab) {
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
  ptr<Type> GetSymbolType(const std::string& n) const override {
    return SymTab()->GetSymbol(InScopeName(n))->GetType();
  }

public:
  VisitorWithSymTab(const std::string& n, const ptr<SymbolTable>& s_tab)
      : VisitorWithScope(n, s_tab) {}
  ~VisitorWithSymTab() {}

  // provide default
  bool Visit(AST::MultiNodes&) override { return true; };
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
  bool Visit(AST::IncrementBlock&) override { return true; };
  bool Visit(AST::FunctionDecl&) override { return true; };
  bool Visit(AST::ChoreoFunction&) override { return true; };
  bool Visit(AST::CppSourceCode&) override { return true; };
  bool Visit(AST::Program&) override { return true; };
};

} // end namespace Choreo

#endif // __CHOREO_VISITOR_HPP__
