#ifndef __CHOREO_SYMREPLACE_HPP__
#define __CHOREO_SYMREPLACE_HPP__

#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>

#include "ast.hpp"
// #include "extern/ginac/ginac-1.8.7/install/include/ginac/ginac.h"
#include "ginac/ginac.h"
#include "symtab.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

// Symbolize all expression nodes.  For expression nodes with the same symbolic
// meaning, they are replaced with a unified form to facilitate value numbering
// and shape infering.
class SymReplace : public VisitorWithScope {
private:
  static const char* cyan;
  static const char* blue;
  static const char* reset;
  static const char* pass_name;

  void TraceEachVisit(const AST::Node& n, bool show_node = false) const {
    if (trace_visit) {
      dbgs() << n.TypeNameString();
      if (show_node) dbgs() << ": " << STR(n);
      dbgs() << "\n";
    }
  }

public:
  using SymExpr = GiNaC::ex;
  using Symbol = GiNaC::symbol;
  using SymValno = size_t;

public:
  // for debugging purpose only
  bool cannot_proceed = false;

  explicit SymReplace() : VisitorWithScope("symrepl") {}

  // terminal expr node.
  std::vector<ptr<AST::Node>> expr_nodes;
  // from node to InScope Name.
  std::map<ptr<AST::Node>, std::string> nd2sn;

  std::map<std::string, Symbol> name_symbol_map;
  std::map<std::string, SymExpr> name_sym_expr_map;
  // the next valid symbolic value number.
  // sym_valno 0 indicates that the node is ignored.
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
    sym_valno = 1;
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
    VST_DEBUG(dbgs() << "AnalyseThenOptimizeExpr: " << PSTR(n) << "\n");
    expr_nodes.push_back(n);
    InitializeNode(n);
    SymbolizeExprNode(n);
  }

  inline void DumpTerminalExprs() const {
    VST_DEBUG({
      dbgs() << "Terminal Expr Nodes:\n";
      for (auto& n : expr_nodes) dbgs() << "\t" << PSTR(n) << "\n";
    });
  }

  inline void DumpNameSymbolMap() const {
    VST_DEBUG({
      dbgs() << "Name Symbol Map:\n";
      for (auto& [name, sym] : name_symbol_map)
        dbgs() << "\t" << name << " " << sym << "\n";
    });
  }

  inline void DumpNameSymExprMap() const {
    VST_DEBUG(dbgs() << "Name SymExpr Map:\n";
              for (auto& [name, sym_expr] : name_sym_expr_map) dbgs()
              << "\t" << name << " " << sym_expr << "\n";);
  }

  inline void DumpExprSymValnoMap() const {
    VST_DEBUG({
      dbgs() << "Expr SymbolValno Map:\n";
      for (auto& [expr, sym_valno] : expr_sym_valno_map)
        dbgs() << "\t" << PSTR(expr) << " " << sym_valno << "\n";
    });
  }

  inline void DumpSymValnoSymExprMap() const {
    VST_DEBUG({
      dbgs() << "SymValno SymExpr Map:\n";
      for (auto& [sym_valno, sym_expr] : sym_valno_sym_expr_map)
        dbgs() << "\t" << sym_valno << " " << sym_expr << "\n";
    });
  }

  inline void DumpExprNodesWithSymExprAndSymValno() const {
    VST_DEBUG({
      dbgs() << "ExprNodes With SymExpr And SymValno:\n";
      for (auto& expr : expr_nodes) {
        auto sym_valno = GetSymValnoFromExpr(expr);
        dbgs() << "\t" << PSTR(expr);
        dbgs() << "\n\t\t\t\t" << sym_valno << " == ";
        if (sym_valno != 0)
          dbgs() << GetSymExprFromSymValno(sym_valno);
        else
          dbgs() << "NONE";
        dbgs() << "\n";
      }
    });
  }

  // use symbolic information to replace expr nodes equivalently.
  void EquivalentlyReplaceExprNodes();

public:
  bool BeforeVisitImpl(AST::Node& n) override {
    VST_DEBUG({
      if (auto cf = dyn_cast<AST::ChoreoFunction>(&n))
        dbgs() << "symbolic replacing starts for function " << cf->name << "\n";
    });
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      DumpTerminalExprs();
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
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::MultiValues& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    for (auto value : n.AllValues()) {
      if (auto e = dyn_cast<AST::Expr>(value)) {
        if (e->op != "elemof") AnalyseThenOptimizeExpr(value);
      } else
        InitializeNode(value);
    }

    return true;
  }

  bool Visit(AST::IntLiteral& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::FloatLiteral& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::StringLiteral& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::BoolLiteral& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::Expr& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::MultiDimSpans& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::NamedTypeDecl& n) override {
    TraceEachVisit(n);
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
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    if (n.init_expr && !isa<AST::Select>(n.init_expr)) {
      SSTab().DefineSymbol(n.name_str, n.init_expr->GetType());
      assert(isa<AST::Expr>(n.init_expr));
      AnalyseThenOptimizeExpr(n.init_expr);
      InsertNdSnSymMap(n.init_expr, n.name_str);
      // int x = a + b;
      // then ::foo::x -> (::foo::a+::foo::b).
      if (GetSymValnoFromExpr(n.init_expr) == 0) return true;
      InsertNameSymExprMap(
          NameWithScope(n.name_str),
          GetSymExprFromSymValno(GetSymValnoFromExpr(n.init_expr)));
    } else {
      SSTab().DefineSymbol(n.name_str, n.GetType());
    }

    return true;
  }

  bool Visit(AST::IntTuple& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::DataAccess& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Assignment& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    if (n.AssignToDataElement()) return true;

    if (!SSTab().IsDeclared(n.GetName()))
      SSTab().DefineSymbol(n.GetName(), n.value->GetType());

    if (isa<AST::Expr>(n.value)) AnalyseThenOptimizeExpr(n.value);

    return true;
  }

  bool Visit(AST::IntIndex& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::DataType& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Identifier& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    if (!SSTab().IsDeclared(n.name))
      if (!SSTab().DeclaredInScope(n.name))
        SSTab().DefineSymbol(n.name, MakeIntegerType());

    return true;
  }

  bool Visit(AST::Parameter& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::ParamList& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::ParallelBy& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    // TODO
    // SSTab().DefineSymbol("@" + n.bpv, MakeMDSpanType(s));
    SSTab().DefineSymbol(n.BPV()->name, MakeUnknownBoundedIntegerType());
    for (auto sym : n.AllSubPVs())
      SSTab().DefineSymbol(cast<AST::Identifier>(sym)->name,
                           MakeUnknownBoundedIntegerType());

    return true;
  }

  bool Visit(AST::WhereBind& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::WithIn& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    // @xx == some int literal should be done in valno pass.
    // TODO(wsj): associate with LoopRange node, work later!

    return true;
  }

  bool Visit(AST::WithBlock& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Memory& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::SpanAs& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    // TODO: define symbol

    return true;
  }

  bool Visit(AST::ChunkAt& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Wait& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Trigger& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Call& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Rotate& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Synchronize& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Select& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Return& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::LoopRange& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::ForeachBlock& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::InThreadsBlock& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::IfElseBlock& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::IncrementBlock& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::FunctionDecl& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    for (ptr<AST::Parameter> p : n.params->values)
      if (p->HasSymbol()) InsertNdSnSymMap(p->sym, p->sym->name);

    return true;
  }

  bool Visit(AST::ChoreoFunction& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::CppSourceCode& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Program& n) override {
    TraceEachVisit(n);
    if (cannot_proceed) return true;

    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_SYMREPLACE_HPP__
