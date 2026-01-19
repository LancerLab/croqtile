#include "sym_replace.hpp"

#include <sstream>

using namespace Choreo;

namespace {

inline std::string ExSTR(const SymReplace::SymExpr& sym_expr) {
  std::ostringstream oss;
  oss << sym_expr;
  return oss.str();
}

} // end of anonymous namespace

bool SymReplace::InsertNameSymbolMap(std::string name, const Symbol& sym) {
  assert(!name_symbol_map.count(name));
  VST_DEBUG(dbgs() << "insert name symbol map: " << name << " === " << sym
                   << "\n");
  name_symbol_map.emplace(name, sym);
  return true;
}

const SymReplace::Symbol&
SymReplace::GetSymbolFromName(std::string name) const {
  assert(name_symbol_map.count(name));
  return name_symbol_map.at(name);
}

bool SymReplace::InsertNameSymExprMap(std::string sname,
                                      const SymExpr& sym_expr) {
  assert(!name_sym_expr_map.count(sname));
  VST_DEBUG(dbgs() << "insert name symbol expr map: " << sname
                   << " === " << sym_expr << "\n");
  name_sym_expr_map.emplace(sname, sym_expr);
  return true;
}

const SymReplace::SymExpr&
SymReplace::GetSymExprFromName(std::string name) const {
  assert(name_sym_expr_map.count(name));
  return name_sym_expr_map.at(name);
}

bool SymReplace::InsertExprSymValnoMap(const ptr<AST::Node> n,
                                       const SymValno sym_valno) {
  assert(!expr_sym_valno_map.count(n));
  expr_sym_valno_map.emplace(n, sym_valno);
  VST_DEBUG(dbgs() << "insert expr with symvalno: [" << PSTR(n) << ", "
                   << sym_valno << "]\n\t" << n << "\n");
  return true;
}

SymReplace::SymValno
SymReplace::GetSymValnoFromExpr(const ptr<AST::Node> n) const {
  assert(expr_sym_valno_map.count(n));
  return expr_sym_valno_map.at(n);
}

bool SymReplace::InsertSymValnoSymExprMap(SymValno sym_valno,
                                          const SymExpr& sym_expr) {
  if (sym_valno_sym_expr_map.count(sym_valno)) return false;
  sym_valno_sym_expr_map.emplace(sym_valno, sym_expr);
  VST_DEBUG(dbgs() << "insert symvalno with symexpr: [" << sym_valno << ", "
                   << sym_expr << "]\n");
  return true;
}

const SymReplace::SymExpr&
SymReplace::GetSymExprFromSymValno(SymValno sym_valno) const {
  assert(sym_valno_sym_expr_map.count(sym_valno));
  return sym_valno_sym_expr_map.at(sym_valno);
}

void SymReplace::InsertNdSnSymMap(ptr<AST::Node> n, const std::string& name,
                                  bool should_scoped /* = true */) {
  VST_DEBUG(dbgs() << "trying to add nd2sn: " << PSTR(n) << " with " << name
                   << "\n");
  std::string res_name;
  if (should_scoped) {
    if (SSTab().IsDeclared(name))
      res_name = SSTab().InScopeName(name);
    else
      res_name = SSTab().ScopedName(name);
  } else {
    res_name = name;
  }
  nd2sn.emplace(n, res_name);
  if (!name_symbol_map.count(res_name)) {
    Symbol sym(res_name, res_name);
    InsertNameSymbolMap(res_name, sym);
  }
  VST_DEBUG(dbgs() << "add nd2sn: " << PSTR(n) << " with " << nd2sn.at(n)
                   << "\n");
}

void SymReplace::InitializeNode(ptr<AST::Node> n) {
  if (n == nullptr) return;

  VST_DEBUG(dbgs() << "Now initialize node: " << PSTR(n) << "\n");
  if (auto e = dyn_cast<AST::Expr>(n)) {
    InitializeNode(e->GetC());
    InitializeNode(e->GetL());
    InitializeNode(e->GetR());

    ptr<AST::Node> l = e->GetL();
    ptr<AST::Node> r = e->GetR();

    // Other types of AST nodes are processed in the form of operands.
    if (e->op == "ref") {
      if (nd2sn.count(r)) InsertNdSnSymMap(n, nd2sn.at(r), false);
    } else if (e->op == "ubound") {
      auto id = dyn_cast<AST::Identifier>(r);
      assert(id);
      InsertNdSnSymMap(r, id->name);
    } else if (e->op == "dimof" || e->op == "getith") {
      // TODO(wsj): corresponding to testcase foo5::y3
      ptr<AST::Node>& dim = r;
      auto ii = dyn_cast<AST::IntIndex>(dim);
      assert(ii);
      if (!nd2sn.count(l)) return;
      std::string name =
          "(" + nd2sn.at(l) + ")" + ((e->op == "getith") ? "[" : "(");
      if (auto num = dyn_cast<AST::IntLiteral>(ii->value))
        name += std::to_string(num->Val());
      else if (isa<AST::Identifier>(ii->value))
        name += nd2sn.at(ii->value);
      else
        choreo_unreachable("The node type of the value of AST::IntIndex is not "
                           "supported in SymReplace yet.");
      name += ((e->op == "getith") ? "]" : ")");
      InsertNdSnSymMap(n, name, false);
    } else {
    }
  } else if (auto id = dyn_cast<AST::Identifier>(n)) {
    InsertNdSnSymMap(n, id->name);
  } else if (isa<AST::IntLiteral>(n)) {
    // construct integer symbol directly when SymbolizeExprNode().
  } else if (isa<AST::FloatLiteral>(n) || isa<AST::StringLiteral>(n)) {
    // currently, float-point number is only used as literal.
    // no need to symbolize it.
  } else if (isa<AST::LoopRange>(n)) {
  } else if (isa<AST::IntTuple>(n)) {
    // since there may be AST::Expr in AST::MultiValues,
    // deal with AST::IntTuple in SymbolizeExprNode() rather than here.
  } else if (isa<AST::MultiDimSpans>(n)) {
    // concat is here.
  } else if (auto ii = dyn_cast<AST::IntIndex>(n)) {
    if (isa<AST::Identifier>(ii->value)) InitializeNode(ii->value);
  } else if (isa<AST::SpanAs>(n)) {
  } else if (auto b = dyn_cast<AST::BoolLiteral>(n)) {
    InsertNdSnSymMap(n, PSTR(b), false);
  } else if (isa<AST::ChunkAt>(n)) {
  } else if (isa<AST::Call>(n) || isa<AST::DataAccess>(n)) {
    InsertNdSnSymMap(n, PSTR(n), false);
  } else {
    choreo_unreachable("The node of type " + PSTR(n->GetType()) +
                       " is not supported in SymReplace yet.\n\t" +
                       n->TypeNameString() + "\n");
  }
}

SymReplace::SymValno SymReplace::GetValidSymValno(const SymExpr& sym_expr) {
  for (auto& [sv, se] : sym_valno_sym_expr_map)
    if (IsEqualSymExpr(sym_expr, se)) return sv;
  auto expanded_sym_expr = sym_expr.expand();
  InsertSymValnoSymExprMap(sym_valno, expanded_sym_expr);
  ++sym_valno;
  return sym_valno - 1;
}

SymReplace::SymExpr
SymReplace::StringifyOpFromSymExpr(ptr<AST::Node> n, const std::string& op,
                                   const SymExpr& sym_expr_r) {
  std::string symbol_name;
  // calling expand before STR is necessary!
  // so that the new symbol is in canonical format.
  std::string sym_expr_r_str = ExSTR(sym_expr_r.expand());
  if (op == "!")
    symbol_name = "!" + sym_expr_r_str;
  else if (op == "dataof")
    symbol_name = sym_expr_r_str + ".data";
  else if (op == "mdataof")
    symbol_name = sym_expr_r_str + ".mdata";
  else if (op == "sizeof")
    symbol_name = "|" + sym_expr_r_str + "|";
  else if (op == "ubound")
    symbol_name = "#" + sym_expr_r_str;
  else if (op == "~")
    symbol_name = "~" + sym_expr_r_str;
  else
    choreo_unreachable("The operator " + op +
                       " is not supported in SymReplace yet.");
  InsertNdSnSymMap(n, symbol_name, false);
  return SymExpr(GetSymbolFromName(symbol_name));
}

SymReplace::SymExpr
SymReplace::StringifyOpFromSymExpr(ptr<AST::Node> n, const SymExpr& sym_expr_l,
                                   const std::string& op,
                                   const SymExpr& sym_expr_r) {
  std::string symbol_name;
  std::string sym_expr_l_str = ExSTR(sym_expr_l.expand());
  std::string sym_expr_r_str = ExSTR(sym_expr_r.expand());
  symbol_name = sym_expr_l_str + " " + op + " " + sym_expr_r_str;
  InsertNdSnSymMap(n, symbol_name, false);
  return SymExpr(GetSymbolFromName(symbol_name));
}

SymReplace::SymExpr
SymReplace::StringifyOpFromSymExpr(ptr<AST::Node> n, const std::string& op,
                                   const SymExpr& sym_expr_l,
                                   const SymExpr& sym_expr_r) {
  std::string symbol_name;
  std::string sym_expr_l_str = ExSTR(sym_expr_l.expand());
  std::string sym_expr_r_str = ExSTR(sym_expr_r.expand());
  symbol_name = op + "(" + sym_expr_l_str + ", " + sym_expr_r_str + ")";
  InsertNdSnSymMap(n, symbol_name, false);
  return SymExpr(GetSymbolFromName(symbol_name));
}

SymReplace::SymExpr SymReplace::StringifyOpFromSymExpr(
    ptr<AST::Node> n, const SymExpr& sym_expr_c, const std::string& op0,
    const SymExpr& sym_expr_l, const std::string& op1,
    const SymExpr& sym_expr_r) {
  std::string symbol_name;
  std::string sym_expr_c_str = ExSTR(sym_expr_c.expand());
  std::string sym_expr_l_str = ExSTR(sym_expr_l.expand());
  std::string sym_expr_r_str = ExSTR(sym_expr_r.expand());
  symbol_name = sym_expr_c_str + " " + op0 + " " + sym_expr_l_str + " " + op1 +
                " " + sym_expr_r_str;
  InsertNdSnSymMap(n, symbol_name, false);
  return SymExpr(GetSymbolFromName(symbol_name));
}

void SymReplace::SymbolizeExprNode(ptr<AST::Node> n) {
  if (n == nullptr) return;
  VST_DEBUG(dbgs() << "Now symbolize for node: " << PSTR(n)
                   << " with addr: " << n.get() << "\n");
  auto e = dyn_cast<AST::Expr>(n);
  if (!e) return;
  VST_DEBUG(dbgs() << "\tthe node is expr!\n");

  auto C = e->GetC();
  auto L = e->GetL();
  auto R = e->GetR();
  auto op = e->op;

  SymbolizeExprNode(C);
  SymbolizeExprNode(L);
  SymbolizeExprNode(R);

  SymReplace::SymExpr res;

  auto dottedNames = [this](std::string& n, const AST::MultiValues& mvs) {
    for (auto v : mvs.AllValues()) {
      if (isa<AST::Expr>(v))
        n += ExSTR(GetSymExprFromSymValno(GetSymValnoFromExpr(v)).expand());
      else {
        assert(nd2sn.count(v));
        n += nd2sn.at(v);
      }
      n += ",";
    }
  };

  auto getSymEx = [this](const std::string& n) {
    if (name_sym_expr_map.count(n)) return GetSymExprFromName(n);
    return SymExpr(GetSymbolFromName(n));
  };

  if (e->IsReference()) {
    if (isa<AST::Identifier>(R)) {
      assert(nd2sn.count(R));
      std::string sname = nd2sn.at(R);
      res = getSymEx(sname);
    } else if (auto num = dyn_cast<AST::IntLiteral>(R)) {
      res = SymReplace::SymExpr(num->Val());
    } else if (auto f = dyn_cast<AST::FloatLiteral>(R)) {
      if (f->IsFloat32())
        res = SymReplace::SymExpr(f->Val_f32());
      else if (f->IsFloat64())
        res = SymReplace::SymExpr(f->Val_f64());
      else
        choreo_unreachable("unsupport float-point type: " + PSTR(f->GetType()));
    } else if (auto it = dyn_cast<AST::IntTuple>(R)) {
      std::string sname = "{";
      dottedNames(sname, *it->GetValues());
      sname.back() = '}';
      InsertNdSnSymMap(n, sname, false);
      InsertNdSnSymMap(R, sname, false);
      res = getSymEx(sname);
    } else if (auto sa = dyn_cast<AST::SpanAs>(R)) {
      auto mvs = cast<AST::MultiValues>(sa->list);
      std::string sname = sa->nid->name + ".span_as[";
      dottedNames(sname, *mvs);
      sname.back() = ']';
      InsertNdSnSymMap(n, sname, false);
      InsertNdSnSymMap(R, sname, false);
      res = getSymEx(sname);
    } else if (auto mds = dyn_cast<AST::MultiDimSpans>(R)) {
      if (mds->list == nullptr) {
        choreo_unreachable("The list of AST::MultiDimSpans here should not be "
                           "nullptr in SymReplace pass!");
      } else {
        if (auto mvs = dyn_cast<AST::MultiValues>(mds->list)) {
          std::string sname = "[";
          dottedNames(sname, *mvs);
          sname.back() = ']';
          InsertNdSnSymMap(n, sname, false);
          InsertNdSnSymMap(R, sname, false);
          res = getSymEx(sname);
        } else if (auto expr = dyn_cast<AST::Expr>(mds->list)) {
          (void)expr;
          // TODO: currently ignore concat op in mdspan.
          InsertExprSymValnoMap(n, 0);
          return;
        }
      }
    } else if (auto b = dyn_cast<AST::BoolLiteral>(R)) {
      (void)b;
      res = SymExpr(GetSymbolFromName(nd2sn.at(R)));
    } else if (auto ca = dyn_cast<AST::ChunkAt>(R)) {
      (void)ca;
      InsertExprSymValnoMap(n, 0);
    } else if (isa<AST::StringLiteral>(R)) {
      InsertExprSymValnoMap(n, 0);
    } else if (isa<AST::Call>(R) || isa<AST::DataAccess>(R)) {
      InsertExprSymValnoMap(n, 0);
    } else {
      choreo_unreachable("The ref node(" + R->TypeNameString() +
                         ") is not supported in SymReplace yet.");
    }
  } else if (e->IsUnary()) {
    if (op == "!" || op == "dataof" || op == "mdataof" || op == "sizeof" ||
        op == "~") {
      // construct new SymExpr from Symbol.
      res = StringifyOpFromSymExpr(
          n, op, GetSymExprFromSymValno(GetSymValnoFromExpr(R)));
    } else if (op == "++" || op == "--" || op == "addrof") {
      InsertExprSymValnoMap(n, 0);
    } else if (op == "ubound") {
      // The rhs is AST::Identifier, which is not AST::Expr.
      auto id = dyn_cast<AST::Identifier>(R);
      assert(id && nd2sn.count(R));
      SymExpr sym_expr_r;
      if (auto sname = nd2sn.at(R); name_sym_expr_map.count(sname))
        sym_expr_r = GetSymExprFromName(sname);
      else
        sym_expr_r = SymExpr(GetSymbolFromName(sname));
      res = StringifyOpFromSymExpr(n, op, sym_expr_r);
    } else if (op == "cast") {
      res = sym_valno_sym_expr_map.at(expr_sym_valno_map.at(R));
    } else {
      choreo_unreachable("The operator " + e->op +
                         " is not supported in SymReplace yet.");
    }
  } else if (e->IsBinary()) {
    if (op == "dimof" || op == "getith") {
      // Handle special cases: the operand of n is not AST::Expr.
      auto ii = dyn_cast<AST::IntIndex>(R);
      assert(ii);
      // new symbol: `lhs.span(1)`.
      // TODO(wsj): workaround for issue-31.co.
      if (!nd2sn.count(n)) {
        std::string name =
            "(" + nd2sn.at(L) + ")" + ((e->op == "getith") ? "[" : "(");
        if (auto num = dyn_cast<AST::IntLiteral>(ii->value))
          name += std::to_string(num->Val());
        else if (isa<AST::Identifier>(ii->value))
          name += nd2sn.at(ii->value);
        else
          choreo_unreachable(
              "The node type of the value of AST::IntIndex is not "
              "supported in SymReplace yet.");
        name += ((e->op == "getith") ? "]" : ")");
        InsertNdSnSymMap(n, name, false);
      }
      assert(name_symbol_map.count(nd2sn.at(n)));
      res = SymExpr(GetSymbolFromName(nd2sn.at(n)));
    } else {
      auto sym_valno_l = GetSymValnoFromExpr(L);
      auto sym_valno_r = GetSymValnoFromExpr(R);

      if (sym_valno_l == 0 || sym_valno_r == 0) {
        InsertExprSymValnoMap(n, 0);
      } else {
        auto sym_expr_l = GetSymExprFromSymValno(sym_valno_l);
        auto sym_expr_r = GetSymExprFromSymValno(sym_valno_r);
        if (op == "+") {
          res = SymExpr(sym_expr_l + sym_expr_r);
        } else if (op == "-") {
          res = SymExpr(sym_expr_l - sym_expr_r);
        } else if (op == "*") {
          res = SymExpr(sym_expr_l * sym_expr_r);
        } else if (op == "/" || op == "%") {
          res = StringifyOpFromSymExpr(n, sym_expr_l, op, sym_expr_r);
        } else if (op == "cdiv") {
          res = StringifyOpFromSymExpr(n, op, sym_expr_l, sym_expr_r);
        } else if (op == "#" || op == "#+" || op == "#-") {
          res = StringifyOpFromSymExpr(n, op, sym_expr_l, sym_expr_r);
        } else if (op == "||" || op == "&&") {
          res = StringifyOpFromSymExpr(n, sym_expr_l, op, sym_expr_r);
        } else if (op == "<" || op == ">" || op == "==" || op == "!=" ||
                   op == "<=" || op == ">=") {
          // unify the op to the less than form.
          // since we do not allow ++ or --, the transformation is valid.
          if (op == ">")
            res = StringifyOpFromSymExpr(n, sym_expr_r, "<", sym_expr_l);
          else if (op == ">=")
            res = StringifyOpFromSymExpr(n, sym_expr_r, "<=", sym_expr_l);
          else
            res = StringifyOpFromSymExpr(n, sym_expr_l, op, sym_expr_r);
        } else if (op == "&" || op == "|" || op == "^") {
          res = StringifyOpFromSymExpr(n, sym_expr_l, op, sym_expr_r);
        } else if (op == "<<" || op == ">>") {
          res = StringifyOpFromSymExpr(n, sym_expr_l, op, sym_expr_r);
        } else {
          choreo_unreachable("The operator " + e->op +
                             " is not supported in SymReplace yet.");
        }
      }
    }
  } else if (e->IsTernary()) {
    // similarly, make a new symbol.
    if (op == "?") {
      auto sym_expr_c = GetSymExprFromSymValno(GetSymValnoFromExpr(C));
      auto sym_expr_l = GetSymExprFromSymValno(GetSymValnoFromExpr(L));
      auto sym_expr_r = GetSymExprFromSymValno(GetSymValnoFromExpr(R));
      res = StringifyOpFromSymExpr(n, sym_expr_c, "?", sym_expr_l, ":",
                                   sym_expr_r);
    } else {
      choreo_unreachable("The operator " + e->op +
                         " is not supported in SymReplace yet.");
    }
  } else {
    choreo_unreachable("The form of expr " + PSTR(e) +
                       " is not supported in SymReplace yet.");
  }
  VST_DEBUG(dbgs() << "res symbolic expr is " << res << "\n");
  if (expr_sym_valno_map.count(n) && expr_sym_valno_map.at(n) == 0) {
    VST_DEBUG(dbgs() << "IGNORE node: " + PSTR(n) << "\n");
    return;
  }
  // TODO(wsj): workaround! `named_spanned_decls` in parser.yy leads to insert
  // the node into expr_sym_valno_map repeatedly! Some other
  // situations(tests/parse/spanned_decl.co) may lead to the same result.
  if (expr_sym_valno_map.count(n)) {
    VST_DEBUG(dbgs() << "WORKAROUND: RETURN due to repeatedly inserting node " +
                            PSTR(n) + " into expr_sym_valno_map.");
    return;
  }
  // TODO(wsj): workaround for issue-31.co.
  if (!nd2sn.count(n)) InsertNdSnSymMap(n, ExSTR(res), false);
  InsertExprSymValnoMap(n, GetValidSymValno(res));
  InsertSymValnoSymExprMap(GetSymValnoFromExpr(n), res);
  VST_DEBUG(dbgs() << "symbolize for node " << PSTR(n) << " is done\n");
}

void SymReplace::EquivalentlyReplaceExprNodes() {
  for (auto orig_node : expr_nodes) {
    auto sym_valno = GetSymValnoFromExpr(orig_node);
    if (sym_valno == 0) continue;
    if (!sym_valno_expr_map.count(sym_valno)) {
      sym_valno_expr_map.emplace(sym_valno, orig_node);
      continue;
    }
    // replace using symbolic infomation
    ptr<AST::Node> new_node = sym_valno_expr_map.at(sym_valno);
    auto new_expr = dyn_cast<AST::Expr>(new_node);
    assert(new_expr);
    auto orig_expr = dyn_cast<AST::Expr>(orig_node);
    assert(orig_expr);

    if (PSTR(orig_node) == PSTR(new_node)) {
      // unnecessary to replace: totally the same.
      // because STR() contains parenthesis to indicate priority.
      continue;
    }

    if (orig_expr->IsReference() && new_expr->IsReference()) {
      continue;
      // auto r = new_expr->GetR();
      // eg. int c = 4; int foo = c + 5;
      // The replacement of `c` with 4 is not necessary.
    }

    VST_DEBUG({
      if (should_use_colors()) {
        dbgs() << cyan << pass_name << " " << blue << "REPLACE " << reset
               << PSTR(orig_node) << blue << " WITH " << reset << PSTR(new_node)
               << "\n";
      } else {
        dbgs() << pass_name << " "
               << "REPLACE " << PSTR(orig_node) << " WITH " << PSTR(new_node)
               << "\n";
      }
    });

    auto C = new_expr->GetC();
    auto L = new_expr->GetL();
    auto R = new_expr->GetR();

    auto cty = (C) ? C->GetType() : nullptr;
    auto lty = (L) ? L->GetType() : nullptr;
    auto rty = (R) ? R->GetType() : nullptr;

    if (auto ii = dyn_cast<AST::IntIndex>(R))
      orig_expr->SetR(AST::Make<AST::IntIndex>(*ii));
    else if (auto e = dyn_cast<AST::Expr>(R))
      orig_expr->SetR(AST::Make<AST::Expr>(*e));
    else if (auto id = dyn_cast<AST::Identifier>(R))
      orig_expr->SetR(AST::Make<AST::Identifier>(*id));
    else if (auto sa = dyn_cast<AST::SpanAs>(R))
      orig_expr->SetR(AST::Make<AST::SpanAs>(*sa));
    else if (auto il = dyn_cast<AST::IntLiteral>(R))
      orig_expr->SetR(AST::Make<AST::IntLiteral>(*il));
    else if (auto b = dyn_cast<AST::BoolLiteral>(R))
      orig_expr->SetR(AST::Make<AST::BoolLiteral>(*b));
    else
      choreo_unreachable("The node of type " + PSTR(R->GetType()) +
                         " is not supported in SymReplace yet.");

    if (new_expr->IsUnary()) {
      orig_expr->ResetL();
    } else if (new_expr->IsBinary()) {
      if (auto ii = dyn_cast<AST::IntIndex>(L)) {
        orig_expr->SetL(AST::Make<AST::IntIndex>(*ii));
      } else if (auto e = dyn_cast<AST::Expr>(L)) {
        orig_expr->SetL(AST::Make<AST::Expr>(*e));
      } else {
        choreo_unreachable("The node of type " + PSTR(L->GetType()) +
                           " is not supported in SymReplace yet.");
      }
    } else if (new_expr->IsTernary()) {
      orig_expr->SetC(AST::Make<AST::Expr>(*C));
      if (auto ii = dyn_cast<AST::IntIndex>(L)) {
        orig_expr->SetL(AST::Make<AST::IntIndex>(*ii));
      } else if (auto e = dyn_cast<AST::Expr>(L)) {
        orig_expr->SetL(AST::Make<AST::Expr>(*e));
      } else {
        choreo_unreachable("The node of type " + PSTR(L->GetType()) +
                           " is not supported in SymReplace yet.");
      }
    }
    orig_expr->op = new_expr->op;
    orig_expr->SetForm(new_expr->GetForm());

    if (cty) orig_expr->GetC()->SetType(cty);
    if (lty) orig_expr->GetL()->SetType(lty);
    if (rty) orig_expr->GetR()->SetType(rty);
  }
}

const char* SymReplace::cyan = "\033[33m";
const char* SymReplace::blue = "\033[34m";
const char* SymReplace::reset = "\033[0m";
const char* SymReplace::pass_name = "[SymReplace]";
