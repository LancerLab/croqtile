#ifndef __CHOREO_VISITOR_HPP__
#define __CHOREO_VISITOR_HPP__

#include <unistd.h>

#include <cstring>
#include <iostream>
#include <optional>
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
  // virtual bool Visit(AST::Node&) = 0;
  virtual bool BeforeVisit(AST::Node&) { return true; }
  virtual bool AfterVisit(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) { // by function print
      const char* Sep = "*******************";
      if (print_after) {
        std::cout << "\n"
                  << Sep << " After " << name << ": " << f->name << " (Begin) "
                  << Sep << "\n"
                  << STR(n) << "\n"
                  << Sep << " After " << name << ": " << f->name << " (End) "
                  << Sep << "\n";
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
  virtual bool Visit(AST::Swap&) = 0;
  virtual bool Visit(AST::Select&) = 0;
  virtual bool Visit(AST::Return&) = 0;
  virtual bool Visit(AST::LoopRange&) = 0;
  virtual bool Visit(AST::ForeachBlock&) = 0;
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
      if (trace.find(name) != std::string::npos) trace_visit = true;
    }

    if (std::getenv("CHOREO_TRACE_VISIT")) {
      auto trace = ToUpper(std::string(std::getenv("CHOREO_TRACE_TRACE")));
      if (trace.find(name) != std::string::npos) trace_visit = true;
    }

    if (std::getenv("CHOREO_DEBUG_VISITOR")) {
      auto debug = ToUpper(std::string(std::getenv("CHOREO_DEBUG_VISITOR")));
      if (debug.find(name) != std::string::npos) debug_visit = true;
    }

    if (std::getenv("CHOREO_PRINT_BEFORE")) {
      auto before = ToUpper(std::string(std::getenv("CHOREO_PRINT_BEFORE")));
      if (before.find("ALLPASSES") != std::string::npos) print_after = true;
      if (before.find(name) != std::string::npos) print_ahead = true;
    }

    if (std::getenv("CHOREO_PRINT_AFTER")) {
      auto after = ToUpper(std::string(std::getenv("CHOREO_PRINT_AFTER")));
      if (after.find("ALLPASSES") != std::string::npos) print_after = true;
      if (after.find(name) != std::string::npos) print_after = true;
    }
  }

  virtual ~Visitor() {}

  // simple reference to the symbol table
  virtual ScopedSymbolTable& SSTab() { return scoped_symtab; }

  virtual const ptr<SymbolTable> SymTab() const {
    if (auto st = scoped_symtab.GlobalSymbolTable()) return st;
    choreo_unreachable("Retrieving an invalid symbol table.");
    return nullptr;
  }

  virtual const std::string& GetName() { return name; }

public:
  virtual ptr<Type> NodeType(const AST::Node& n) const {
    if (auto id = dyn_cast<AST::Identifier>(&n))
      return GetSymbolType(id->name);
    else if (auto expr = dyn_cast<AST::Expr>(&n)) {
      if (auto ref = expr->GetReference()) {
        if (auto id = dyn_cast<AST::Identifier>(ref))
          return GetSymbolType(id->name);
      } else if (expr->op == "dataof") {
        if (auto ref = cast<AST::Expr>(expr->GetR())->GetReference()) {
          auto id = cast<AST::Identifier>(ref);
          if (!GetSymbolType(id->name)) // make sure the symbol exists
            return nullptr;
          return GetSymbolType(id->name + ".data");
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

public:
  void Error(const location& loc, const std::string& message) {
    static const char* red = "\033[31m";
    static const char* reset = "\033[0m";

    std::cerr << loc << ": ";

    if (should_use_colors())
      std::cerr << red << "error: " << reset;
    else
      std::cerr << "error: ";

    std::cerr << message << std::endl;
  }

  void Warning(const location& loc, const std::string& message) {
    static const char* yellow = "\033[33m";
    static const char* reset = "\033[0m";

    std::cerr << loc << ": ";

    if (should_use_colors())
      std::cerr << yellow << "warning: " << reset;
    else
      std::cerr << "warning: ";

    std::cerr << message << std::endl;
  }

  void Note(const location& loc, const std::string& message) {
    std::cerr << loc << ": note: " << message << std::endl;
  }
};

// A visitor with simple symbol auto scoping functionality
struct VisitorWithScope : public Visitor {
protected:
  // for the derived classes
  virtual bool BeforeVisitImpl(AST::Node& n) = 0;
  virtual bool AfterVisitImpl(AST::Node& n) = 0;

  // special to within: map 'with' to its 'with-matchers'
  std::unordered_map<std::string, std::vector<std::string>> within_map;

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
    if (isa<AST::Program>(&n)) {
      Reset();
      SSTab().EnterScope(""); // global scope
    } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      SSTab().EnterScope(f->name);
    } else if (isa<AST::ParallelBy>(&n)) {
      SSTab().EnterScope("paraby_" + std::to_string(pb_count++));
    } else if (isa<AST::WithBlock>(&n)) {
      SSTab().EnterScope("within_" + std::to_string(wi_count++));
    } else if (isa<AST::ForeachBlock>(&n)) {
      SSTab().EnterScope("foreach_" + std::to_string(fe_count++));
    } else if (auto w = dyn_cast<AST::WithIn>(&n)) {
      if (w->with && w->with_matchers) {
        std::vector<std::string> matchers;
        for (auto v : w->with_matchers->AllValues())
          matchers.push_back(cast<AST::Identifier>(v)->name);
        within_map.emplace(w->with->name, matchers);
      }
    }
    BeforeVisitImpl(n); // derived class to customize
    return true;
  }

  bool AfterVisit(AST::Node& n) final {
    AfterVisitImpl(n); // derived class to customize
    if (isa<AST::Program>(&n)) {
      Reset();
      SSTab().LeaveScope();
      assert(SSTab().ScopeDepth() == 0 && "internal error: scope is not zero.");
    } else if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
               isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
      SSTab().LeaveScope();
    }

    Visitor::AfterVisit(n);
    return true;
  }

public:
  VisitorWithScope(const std::string& n,
                   const ptr<SymbolTable>& s_tab = nullptr)
      : Visitor(n, s_tab) {
    Reset();
  }
  ~VisitorWithScope() {}
};

// This accepts static symbol table and provide symbol lookup capability
// Caution: must be used when symbol table does not change.
struct VisitorWithSymTab : public VisitorWithScope {
protected:
  virtual std::string InScopeName(const std::string& sym) const {
    auto removeLastLevel = [](const std::string& input) -> std::string {
      size_t lastPos = input.rfind("::");
      if (lastPos == std::string::npos)
        return input; // No "::" found, return the original string
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

public:
  // use the immutable symbol table directly
  ptr<Type> GetSymbolType(const std::string& n) const override {
    return SymTab()->GetSymbol(InScopeName(n))->GetType();
  }

public:
  VisitorWithSymTab(const std::string& n, const ptr<SymbolTable>& s_tab)
      : VisitorWithScope(n, s_tab) {}
  ~VisitorWithSymTab() {}
};

} // end namespace Choreo

#endif // __CHOREO_VISITOR_HPP__
