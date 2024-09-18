#ifndef __CHOREO_SYMREPLACE_HPP__
#define __CHOREO_SYMREPLACE_HPP__

#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>

#include "ast.hpp"
#include "symtab.hpp"
#include "types.hpp"
#include "visitor.hpp"

#include "extern/ginac/ginac-1.8.7/install/include/ginac/ginac.h"

namespace Choreo {

namespace sym_replace {

#define DEBUG(X)                                                               \
  do {                                                                         \
    if (0) { X; }                                                              \
  } while (false)

} // end namespace sym_replace

#define __TRACE_EACH_VISIT__                                                   \
  if (trace_visit) {                                                           \
    os << n.TypeNameString() << ": ";                                          \
    n.Print(os);                                                               \
    os << "\n";                                                                \
  }

// Symbolize all expression nodes.  For expression nodes with the same symbolic
// meaning, they are replaced with a unified form to facilitate value numbering
// and shape infering.
class SymReplace : public VisitorWithSymTab {
private:
  static const char* cyan;
  static const char* blue;
  static const char* reset;
  static const char* pass_name;

public:
  using SymExpr = GiNaC::ex;
  using Symbol = GiNaC::symbol;
  using SymValno = size_t;
public:
  bool trace = false;
  std::ostream& os;
  // for debugging purpose only
  bool trace_visit = false;
  bool cannot_proceed = false;
  size_t error_count = 0;

  explicit SymReplace(const ptr<SymbolTable> s_tab, bool t, std::ostream& o)
      : VisitorWithSymTab(s_tab), trace(t), os(o),
        trace_visit(std::getenv("TRACE_SYMREPL")) {}

  bool HasError() {
    if (error_count)
      os << "Totally " << error_count << " errors have been detected.\n";
    return error_count != 0;
  }

  // termial expr node.
  std::vector<ptr<AST::Node>> expr_nodes;
  // from node to InScope Name.
  std::map<ptr<AST::Node>, std::string> nd2sn;

  std::map<std::string, Symbol> name_symbol_map;
  std::map<std::string, SymExpr> name_sym_expr_map;
  // the next valid symbolic value number.
  // sym_valno of 0 indcates that the node is ignored.
  SymValno sym_valno = 1;
  // from expr to symbolic value number.
  std::map<ptr<AST::Node>, SymValno> expr_sym_valno_map;
  // from symbolic value number to symbolic expression.
  std::map<SymValno, SymExpr> sym_valno_sym_expr_map;

  std::map<SymValno, ptr<AST::Node>> sym_valno_expr_map;

  bool InsertNameSymbolMap(std::string name, const Symbol& sym);
  const Symbol& GetSymbolFromName(std::string name) const;

  bool InsertNameSymExprMap(std::string sname, const SymExpr& sym_expr);
  const SymExpr& GetSymExprFromName(std::string name) const;

  bool InsertExprSymValnoMap(ptr<AST::Node> n, const SymValno sym_valno);
  SymValno GetSymValnoFromExpr(ptr<AST::Node> n) const;

  // If `sym_valno` is existed, return `false`.
  bool InsertSymValnoSymExprMap(SymValno sym_valno, const SymExpr& sym_expr);
  const SymExpr& GetSymExprFromSymValno(SymValno sym_valno) const;

  inline bool IsEqualSymExpr(const SymExpr& sym_expr0,
                             const SymExpr& sym_expr1) const {
    return sym_expr0.expand().is_equal(sym_expr1.expand());
  }

  // insert pair to map: ptr<AST::Node> -> in scope name.
  // eg. int a = (x+y) + z.
  //     expr node: (x+y) + z -> ::foo::a
  void InsertNdSnSymMap(ptr<AST::Node> n, const std::string& name,
                        bool should_scoped = true);

  // helper function.
  inline std::string NameWithScope(const std::string& name) {
    return SSTab().InScopeName(name);
  }

  // reset all status and clear data structures.
  inline void Reset() {
    sym_valno = 0;
    name_symbol_map.clear();
    name_sym_expr_map.clear();
    expr_sym_valno_map.clear();
    sym_valno_sym_expr_map.clear();
    sym_valno_expr_map.clear();
    nd2sn.clear();
    expr_nodes.clear();
  }

  // only do (node -> scoped name and symbol) mapping recursively.
  void InitializeNode(ptr<AST::Node> n);

  SymValno GetValidSymValno(const SymExpr& sym_expr);

  // Symbolic division is commutative!
  // That is, x / y * y => x, but the result may be wrong in integer
  // programming. So, `x` / `y` will be treated as `x / y`, slash becomes part
  // of the name.

  // unary expr: |s.span|
  SymExpr StringifyOpFromSymExpr(ptr<AST::Node> n, const std::string& op,
                                 const SymExpr& sym_expr_r);
  // binary expr: cdiv(x, y), ...
  SymExpr StringifyOpFromSymExpr(ptr<AST::Node> n, const std::string& op,
                                 const SymExpr& sym_expr_l,
                                 const SymExpr& sym_expr_r);
  // binary expr: x * y, ...
  SymExpr StringifyOpFromSymExpr(ptr<AST::Node> n, const SymExpr& sym_expr_l,
                                 const std::string& op,
                                 const SymExpr& sym_expr_r);
  // ternary expr: x<y ? x : y
  SymExpr StringifyOpFromSymExpr(ptr<AST::Node> n, const SymExpr& sym_expr_c,
                                 const std::string& op0,
                                 const SymExpr& sym_expr_l,
                                 const std::string& op1,
                                 const SymExpr& sym_expr_r);

  // symbolize AST::Expr recursively.
  void SymbolizeExprNode(ptr<AST::Node> n);

  inline void AnalyseThenOptimizeExpr(ptr<AST::Node> n) {
    DEBUG(os << "AnalyseThenOptimizeExpr: " << PSTR(n) << "\n");
    expr_nodes.push_back(n);
    InitializeNode(n);
    SymbolizeExprNode(n);
  }

  inline void DumpTermimalExprs() const {
    DEBUG(
      os << "Terminal Expr Nodes:\n";
      for (auto& n : expr_nodes)
        os << "\t" << PSTR(n) << "\n";
    );
  }

  inline void DumpNameSymbolMap() const {
    DEBUG(
      os << "Name Symbol Map:\n";
      for (auto& [name, sym] : name_symbol_map)
        os << "\t" << name << " " << sym << "\n";
    );
  }

  inline void DumpNameSymExprMap() const {
    DEBUG(
      os << "Name SymExpr Map:\n";
      for (auto& [name, sym_expr] : name_sym_expr_map)
        os << "\t" << name << " " << sym_expr << "\n";
    );
  }

  inline void DumpExprSymValnoMap() const {
    DEBUG(
      os << "Expr SymbolValno Map:\n";
      for (auto& [expr, sym_valno] : expr_sym_valno_map)
        os << "\t" << PSTR(expr) << " " << sym_valno << "\n";
    );
  }

  inline void DumpSymValnoSymExprMap() const {
    DEBUG(
      os << "SymValno SymExpr Map:\n";
      for (auto& [sym_valno, sym_expr] : sym_valno_sym_expr_map)
        os << "\t" << sym_valno << " " << sym_expr << "\n";
    );
  }

  inline void DumpExprNodesWithSymExprAndSymValno() const {
    DEBUG(
      os << "ExprNodes With SymExpr And SymValno:\n";
      for (auto& expr : expr_nodes) {
        auto sym_valno = GetSymValnoFromExpr(expr);
        os << "\t" << PSTR(expr);
        os << "\n\t\t\t\t" << sym_valno << " == ";
        if (sym_valno != 0)
          os << GetSymExprFromSymValno(sym_valno);
        else
          os << "NONE";
        os << "\n";
      }
    );
  }

  // use symbolic information to replace expr nodes equivalently.
  void EquivalentlyReplaceExprNodes();

public:
  bool BeforeVisitImpl(AST::Node& n) override {
    DEBUG(
      if (auto cf = dyn_cast<AST::ChoreoFunction>(&n))
        os << "symbolic replacing starts for function "
           << cf->name << "\n";
    );
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      DumpTermimalExprs();
      DumpNameSymbolMap();
      DumpNameSymExprMap();
      DumpExprSymValnoMap();
      DumpSymValnoSymExprMap();
      DumpExprNodesWithSymExprAndSymValno();
      EquivalentlyReplaceExprNodes();
      Reset();
    }
    return true;
  }

public:
  bool Visit(AST::MultiNodes& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::MultiValues& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    for (auto value : n.AllValues()) {
      if (isa<AST::Expr>(value))
        AnalyseThenOptimizeExpr(value);
      else
        InitializeNode(value);
    }

    return true;
  }

  bool Visit(AST::IntLiteral& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Boolean& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Expr& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::MultiDimSpans& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::NamedTypeDecl& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    if (n.init_expr) {
      SSTab().DefineSymbol(n.name_str, n.GetType());
      assert(isa<AST::Expr>(n.init_expr));
      AnalyseThenOptimizeExpr(n.init_expr);
      InsertNdSnSymMap(n.init_expr, n.name_str);
    }

    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    if (n.init_expr && !isa<AST::Select>(n.init_expr)) {
      SSTab().DefineSymbol(n.name_str, n.init_expr->GetType());
      assert(isa<AST::Expr>(n.init_expr));
      AnalyseThenOptimizeExpr(n.init_expr);
      InsertNdSnSymMap(n.init_expr, n.name_str);
      // int x = a + b;
      // then ::foo::x -> (::foo::a+::foo::b).
      InsertNameSymExprMap(
          NameWithScope(n.name_str),
          GetSymExprFromSymValno(GetSymValnoFromExpr(n.init_expr)));
    } else {
      SSTab().DefineSymbol(n.name_str, n.GetType());
    }

    return true;
  }

  bool Visit(AST::IntTuple& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Assignment& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    if (!SSTab().IsDeclared(n.name))
      SSTab().DefineSymbol(n.name, n.value->GetType());

    if (isa<AST::Expr>(n.value))
      AnalyseThenOptimizeExpr(n.value);

    return true;
  };

  bool Visit(AST::IntIndex& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::DataType& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Identifier& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    if (!SSTab().IsDeclared(n.name))
      if (!SSTab().DeclaredInScope(n.name))
        SSTab().DefineSymbol(n.name, MakeIntegerType());

    return true;
  }

  bool Visit(AST::Parameter& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::ParamList& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::ParallelBy& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    // TODO
    // SSTab().DefineSymbol("@" + n.biv, MakeMDSpanType(s));
    SSTab().DefineSymbol(n.biv, n.GetType());

    return true;
  };

  bool Visit(AST::WhereBind& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::WithIn& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    // @xx == some int literal should be done in valno pass.
    // TODO(wsj): associate with LoopRange node, work later!

    return true;
  }

  bool Visit(AST::WithBlock& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::Memory& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::SpanAs& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::DMA& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    // TODO: define symbol

    return true;
  }

  bool Visit(AST::ChunkAt& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Wait& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Call& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::Select& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::Return& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::LoopRange& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::ForeachBlock& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::FunctionDecl& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    for (ptr<AST::Parameter> p : n.params->values)
      if (p->HasSymbol())
        InsertNdSnSymMap(p->sym, p->sym->name);

    return true;
  };

  bool Visit(AST::ChoreoFunction& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::CppSourceCode& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };
  bool Visit(AST::Program& n) override {
    __TRACE_EACH_VISIT__;

    if (cannot_proceed) return true;

    return true;
  };
};

} // end namespace Choreo

#endif // __CHOREO_SYMREPLACE_HPP__
