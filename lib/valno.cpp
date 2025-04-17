#include "shapeinfer.hpp"

using namespace Choreo;

namespace {

std::vector<int> CollectValueNumbers(const std::string& input) {
  std::vector<int> res;
  std::regex valuePattern("#(-?\\d+)");
  auto begin = std::sregex_iterator(input.begin(), input.end(), valuePattern);
  auto end = std::sregex_iterator();

  for (auto i = begin; i != end; ++i) {
    std::smatch match = *i;
    std::string matchStr = match.str(1); // Capture the number part of the match
    res.push_back(std::stoi(matchStr));
  }
  return res;
}

std::vector<int> GetOperandsValNo(const std::string& input) {
  std::vector<int> operands;
  size_t startPos = 0;
  size_t colonPos;

  // Loop through all colon-separated parts of the string
  while ((colonPos = input.find(':', startPos)) != std::string::npos) {
    std::string part = input.substr(startPos, colonPos - startPos);
    std::istringstream iss(part.substr(1));
    int number;
    // Try to extract a number from the current part
    if (iss >> number) operands.push_back(number);
    startPos = colonPos + 1;
  }

  // Handle the last part after the last colon
  if (startPos + 1 < input.length()) {
    std::istringstream iss(input.substr(startPos + 1));
    int number;
    if (iss >> number) operands.push_back(number);
  }

  return operands;
}

} // namespace

const std::string
ValueNumbering::VNSymbolName(const AST::Identifier& id) const {
  auto sig = id.name;
  auto pty = visitor->NodeType(id);
  if (isa<SpannedType>(pty) || GeneralFutureType(pty)) {
    sig = RemoveSuffix(sig, ".span") +
          ".span"; // only cares about value inside the mdspan
  } else if (isa<BoundedType>(pty)) {
    sig = "@" + sig; // only cares about the upper bound
  }
  return sig;
}

void ValueNumbering::EnterScope() {
  std::string indent = ScopeIndent();
  assert(!indent.empty() && "unexpected empty indent.");
  indent.pop_back();

  expressionValueNumbers.push_back({});
  valueNumberExpressions.push_back({});
  nodeValueNumbers.push_back({});

  if (trace)
    if (visitor->SSTab().ScopeDepth() > 1)
      dbgs() << indent << "scope-" << visitor->SSTab().ScopeDepth() - 1
             << " {\n";
}

void ValueNumbering::LeaveScope() {
  if (visitor->SSTab().ScopeDepth() <= 1) return;

  std::string sname = std::to_string(visitor->SSTab().ScopeDepth() - 1);

  assert(!expressionValueNumbers.empty() && !valueNumberExpressions.empty());

  expressionValueNumbers.pop_back();
  valueNumberExpressions.pop_back();
  nodeValueNumbers.pop_back();

  // reset value number when leaving the function scope
  if (expressionValueNumbers.empty() || expressionValueNumbers.size() == 1) {
    nextValueNumber = 0;
    bind_info.Clear();
  }

  std::string indent = ScopeIndent();
  assert(!indent.empty() && "unexpected empty indent.");
  indent.pop_back();

  if (trace) dbgs() << indent << "} // end scope-" << sname << "\n";
}

// It binds a expression sigature with an existing value number.  use it
// carefully.
void ValueNumbering::AssociateSignatureWithValueNumber(const std::string& sig,
                                                       int valno) {
  assert(InternalHasValNoExpr(valno) && "invalid value number is provided.");
  if (InternalHasExprValNo(sig))
    assert((InternalGetExprValNo(sig) == valno) &&
           "must associate signature with different value number.");

  InternalUpdateExprValNo(sig, valno);

  if (trace)
    dbgs() << ScopeIndent() << "Alias \"" << sig << "\" -> #" << valno << "\n";
}

void ValueNumbering::AssociateSignatureWithInvalidValueNumber(
    const std::string& sig) {
  assert(!InternalHasExprValNo(sig) && "signature does exists.");

  InternalUpdateExprValNo(sig, GetInvalidValueNumber());

  if (trace)
    dbgs() << ScopeIndent() << "Alias \"" << sig << "\" -> #<invalid>\n";
}

void ValueNumbering::RebindSignatureWithValueNumber(const std::string& sig,
                                                    int valno) {
  bool changed = false;
  for (auto evn = expressionValueNumbers.rbegin();
       evn != expressionValueNumbers.rend(); ++evn) {
    if (!evn->count(sig)) continue;

    assert(!ValidVN((*evn)[sig]) &&
           "expecting rebind of an invalid value number.");
    (*evn)[sig] = valno;
    if (!changed && trace)
      dbgs() << ScopeIndent() << "Alias(Rebind) \"" << sig << "\" -> #" << valno
             << "\n";
    changed = true;
  }

  if (!changed) choreo_unreachable("failed to rebind valno for `" + sig + "'.");
}

std::string ValueNumbering::SignBinaryCompositeValues(const location& loc,
                                                      const std::string& op,
                                                      const std::string& l_sig,
                                                      const std::string& r_sig,
                                                      bool verbose) {
  // handle concatenation
  if (op == "concat") {
    auto GetSignature = [this](const std::string& sig) {
      if (CountElementsInSignature(sig) > 1)
        return sig;
      else
        return "#" + std::to_string(GetValueNumberOfSignature(sig));
    };
    return GetSignature(l_sig) + "," + GetSignature(r_sig);
  }

  std::string lhs, rhs;
  // specially handle ituple/mdspan + integer: broadcast integer

  auto l_elem_cnt = CountElementsInSignature(l_sig);
  auto r_elem_cnt = CountElementsInSignature(r_sig);

  if (l_elem_cnt == 1 && r_elem_cnt == 1) {
    int l_valno = GetValueNumberOfSignature(l_sig);
    lhs = "#" + std::to_string(l_valno);
    int r_valno = GetValueNumberOfSignature(r_sig);
    rhs = "#" + std::to_string(r_valno);
  } else if (l_elem_cnt == 1 && r_elem_cnt > 1) {
    int valno = GetValueNumberOfSignature(l_sig);
    lhs = "#" + std::to_string(valno);
    for (int i = 1; i < r_elem_cnt; ++i) lhs += ",#" + std::to_string(valno);
    rhs = r_sig;
  } else if (l_elem_cnt > 1 && r_elem_cnt == 1) {
    int valno = GetValueNumberOfSignature(r_sig);
    rhs = "#" + std::to_string(valno);
    for (int i = 1; i < l_elem_cnt; ++i) rhs += ",#" + std::to_string(valno);
    lhs = l_sig;
  } else {
    lhs = l_sig;
    rhs = r_sig;
  }

  assert(CountElementsInSignature(lhs) == CountElementsInSignature(rhs));

  auto l_vns = CollectValueNumbers(lhs);
  auto r_vns = CollectValueNumbers(rhs);
  assert(l_vns.size() == r_vns.size());

  std::vector<int> signatures;
  for (size_t i = 0; i < l_vns.size(); ++i) {
    auto l_sig = GetSignatureFromValueNumber(l_vns[i]);
    auto r_sig = GetSignatureFromValueNumber(r_vns[i]);
    auto opt_sig = TryToSimplifyBinary(loc, op, l_sig, r_sig, verbose);
    if (opt_sig)
      signatures.push_back(GetOrInsertValueNumberFromSignature(*opt_sig));
    else {
      std::ostringstream oss;
      oss << op << ":#" << GetValueNumberOfSignature(l_sig) << ":#"
          << GetValueNumberOfSignature(r_sig);
      signatures.push_back(GetOrInsertValueNumberFromSignature(oss.str()));
    }
  }
  assert(signatures.size() > 0);

  std::ostringstream oss;
  oss << "#" << signatures[0];
  for (size_t i = 1; i < signatures.size(); ++i) oss << ",#" << signatures[i];
  return oss.str();
}

std::optional<std::string>
ValueNumbering::TryToSimplifyBinary(const location& loc, const std::string& op,
                                    const std::string& lhs,
                                    const std::string& rhs, bool verbose) {
  if (op == "concat") return std::nullopt;
  auto l_cv = RemovePrefixOrNull("const_", lhs);
  auto r_cv = RemovePrefixOrNull("const_", rhs);
  if (l_cv && r_cv) {
    std::string res = "const_";
    using Var = std::variant<float, double, long long>;
    auto parseNumeric = [](const std::string& s) -> Var {
      if (auto d = Str2Double(s)) return *d;
      if (auto f = Str2Float(s)) return *f;
      return std::stoll(s);
    };

    auto applyOp = [&](auto&& L, auto&& R, auto f) -> Var {
      return std::visit(
          [&](auto a, auto b) -> Var {
            using CT = std::common_type_t<decltype(a), decltype(b)>;
            return static_cast<CT>(f(a, b));
          },
          L, R);
    };

    Var L = parseNumeric(*l_cv);
    Var R = parseNumeric(*r_cv);

    auto isZero = [](const Var& v) {
      if (std::holds_alternative<double>(v)) return std::get<double>(v) == 0.0;
      if (std::holds_alternative<float>(v)) return std::get<float>(v) == 0.0f;
      if (std::holds_alternative<long long>(v))
        return std::get<long long>(v) == 0LL;
      else
        choreo_unreachable("unexpected type for zero check.");
    };

    std::unordered_map<std::string, std::function<Var(const Var&, const Var&)>>
        op_table;
    op_table["+"] = [&](auto&& A, auto&& B) {
      return applyOp(A, B, [](auto a, auto b) { return a + b; });
    };
    op_table["-"] = [&](auto&& A, auto&& B) {
      return applyOp(A, B, [](auto a, auto b) { return a - b; });
    };
    op_table["*"] = [&](auto&& A, auto&& B) {
      return applyOp(A, B, [](auto a, auto b) { return a * b; });
    };
    op_table["/"] = [&](auto&& A, auto&& B) {
      if (isZero(B)) {
        dbgs() << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / "
               << rhs << "\n";
        choreo_unreachable("divide by zero is found in shape evaluation.");
      }
      return applyOp(A, B, [](auto a, auto b) { return a / b; });
    };
    op_table["%"] = [&](auto&& A, auto&& B) {
      if (std::holds_alternative<long long>(A) &&
          std::holds_alternative<long long>(B)) {
        if (isZero(B)) {
          dbgs() << ScopeIndent() << "<ERROR> mod by zero: " << lhs << " % "
                 << rhs << "\n";
          choreo_unreachable("mod by zero is found in shape evaluation.");
        }
        return std::get<long long>(A) % std::get<long long>(B);
      } else {
        dbgs() << ScopeIndent()
               << "<ERROR> mod with float-point is not support: " << lhs
               << " % " << rhs << "\n";
        choreo_unreachable(
            "mod with float-point is found in shape evaluation.");
      }
    };
    op_table["<"] = [&](auto&& A, auto&& B) {
      return Var{static_cast<long long>(
          std::visit([&](auto a, auto b) { return a < b; }, A, B))};
    };
    op_table[">"] = [&](auto&& A, auto&& B) {
      return Var{static_cast<long long>(
          std::visit([&](auto a, auto b) { return a > b; }, A, B))};
    };
    op_table["=="] = [&](auto&& A, auto&& B) {
      // TODO: do we need EPSILON for float-point?
      return Var{static_cast<long long>(
          std::visit([&](auto a, auto b) { return a == b; }, A, B))};
    };
    op_table["!="] = [&](auto&& A, auto&& B) {
      // TODO: do we need EPSILON for float-point?
      return Var{static_cast<long long>(
          std::visit([&](auto a, auto b) { return a != b; }, A, B))};
    };
    op_table["<="] = [&](auto&& A, auto&& B) {
      // TODO: do we need EPSILON for float-point?
      return Var{static_cast<long long>(
          std::visit([&](auto a, auto b) { return a <= b; }, A, B))};
    };
    op_table[">="] = [&](auto&& A, auto&& B) {
      // TODO: do we need EPSILON for float-point?
      return Var{static_cast<long long>(
          std::visit([&](auto a, auto b) { return a >= b; }, A, B))};
    };
    op_table["cdiv"] = [&](auto&& A, auto&& B) {
      if (std::holds_alternative<long long>(A) &&
          std::holds_alternative<long long>(B)) {
        if (isZero(B)) {
          dbgs() << ScopeIndent() << "<ERROR> cdiv by zero: " << lhs << " % "
                 << rhs << "\n";
          choreo_unreachable("cdiv by zero is found in shape evaluation.");
        }
        return (std::get<long long>(A) + std::get<long long>(B) - 1ll) /
               std::get<long long>(B);
      } else {
        dbgs() << ScopeIndent()
               << "<ERROR> cdiv with float-point is not support: " << lhs
               << " % " << rhs << "\n";
        choreo_unreachable(
            "cdiv with float-point is found in shape evaluation.");
      }
    };
    op_table["#"] = [&](auto&& A, auto&& B) {
      if (std::holds_alternative<long long>(A) &&
          std::holds_alternative<long long>(B)) {
        return std::get<long long>(A) * std::get<long long>(B);
      } else {
        dbgs() << ScopeIndent()
               << "<ERROR> # with float-point is not support: " << lhs << " % "
               << rhs << "\n";
        choreo_unreachable("# with float-point is found in shape evaluation.");
      }
    };

    if (op_table.count(op)) {
      Var raw = op_table.at(op)(L, R);
      std::visit(
          [&](auto x) -> std::string {
            using T = decltype(x);
            if constexpr (std::is_same_v<T, long long>) {
              if (op == "<" || op == ">" || op == "==" || op == "!=" ||
                  op == "<=" || op == ">=") {
                return res = x ? "true" : "false";
              }
              return res += std::to_string(x);
            } else if constexpr (std::is_same_v<T, float>) {
              return res += std::to_string(x) + "f";
            } else if constexpr (std::is_same_v<T, double>) {
              return res += std::to_string(x);
            } else {
              choreo_unreachable("unexpected type for simplification.");
            }
          },
          raw);
    } else {
      Error(loc,
            "simplification of operation `" + op + "' is not yet supported.");
      return std::nullopt;
    }
    if (trace && verbose)
      dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
             << rhs << " to '" << res << "'\n";
    return res;
  }

  if (op == "/") {
    // a/a == 1
    if (GetValueNumberOfSignature(lhs) == GetValueNumberOfSignature(rhs)) {
      std::string res = "const_1";
      if (trace && verbose)
        dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
               << rhs << " to '" << res << "'\n";
      return res;
    }
    // a/1 = a
    if (rhs == "const_1") return lhs;
    // useful simplification: a/(a/b) = b
    if (!PrefixedWith(lhs, "#") /*not multiple values*/) {
      int rvn = GetValueNumberOfSignature(rhs);
      auto bind_set = GetBindSet(rvn);
      bind_set.insert(rvn); // always add self
      for (auto div_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(div_vn);
        if (!PrefixedWith(rhs, "/:")) continue;
        auto div = GetOperandsValNo(sig);
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(lhs) == div[0]) {
          auto res = GetSignatureFromValueNumber(div[1]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
  } else if (op == "-") {
    // a-a == 1
    if (GetValueNumberOfSignature(lhs) == GetValueNumberOfSignature(rhs)) {
      std::string res = "const_0";
      if (trace && verbose)
        dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
               << rhs << " to '" << res << "'\n";
      return res;
    }
  } else if (op == "+") {
    // useful simplification: a-b+b = a
    if (!PrefixedWith(rhs, "#") /*not multiple values*/) {
      int lvn = GetValueNumberOfSignature(lhs);
      auto bind_set = GetBindSet(lvn);
      bind_set.insert(lvn); // always add self
      for (auto minus_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(minus_vn);
        if (!PrefixedWith(lhs, "-:")) continue;
        auto minus = GetOperandsValNo(sig);
        assert(minus.size() == 2);
        if (GetValueNumberOfSignature(rhs) == minus[1]) {
          auto res = GetSignatureFromValueNumber(minus[0]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
  } else if (op == "#") {
    // suppose `a` and `b` are bounded vars
    // `#a` is 4, `#b` is `N/#a` where `N` is dynamic dim
    // if `xx.chunkat(a#b)`, then the result shape should be 1
    // that is, N / (#a * #b) = N / N = 1
    // so, `a#b` should be simplified to a bounded var whose ubound is `N`
    if (!PrefixedWith(lhs, "#") /*not multiple values*/) {
      int rvn = GetValueNumberOfSignature(rhs);
      auto bind_set = GetBindSet(rvn);
      bind_set.insert(rvn); // always add self
      for (auto div_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(div_vn);
        if (!PrefixedWith(rhs, "/:")) continue;
        auto div = GetOperandsValNo(sig);
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(lhs) == div[1]) {
          auto res = GetSignatureFromValueNumber(div[0]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
    if (!PrefixedWith(rhs, "#")) {
      int lvn = GetValueNumberOfSignature(lhs);
      auto bind_set = GetBindSet(lvn);
      bind_set.insert(lvn); // always add self
      for (auto div_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(div_vn);
        if (!PrefixedWith(lhs, "/:")) continue;
        auto div = GetOperandsValNo(sig);
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(rhs) == div[1]) {
          auto res = GetSignatureFromValueNumber(div[0]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<std::string>
ValueNumbering::SignBoundedOperation(const location& loc, const std::string& op,
                                     const AST::Node& lhs, const AST::Node& rhs,
                                     bool verbose) {
  std::optional<std::string> res;
  if (op == "+" || op == "-" || op == "*" || op == "/" || op == "%") {
    if (isa<BoundedType>(lhs.GetType()) && (lhs.GetType()->Dims() == 1) &&
        (isa<IntegerType>(rhs.GetType())))
      res = GetSignatureForNode(lhs);
    else if (isa<BoundedType>(rhs.GetType()) && (rhs.GetType()->Dims() == 1) &&
             (isa<IntegerType>(lhs.GetType())))
      res = GetSignatureForNode(rhs);
    else if (isa<BoundedITupleType>(lhs.GetType()) &&
             (isa<ITupleType>(rhs.GetType())))
      res = GetSignatureForNode(lhs);
    else if (isa<BoundedITupleType>(rhs.GetType()) &&
             (isa<ITupleType>(lhs.GetType())))
      res = GetSignatureForNode(rhs);
    else
      choreo_unreachable("operation '" + op + "' is not permitted for '" +
                         STR(lhs) + "(" + PSTR(lhs.GetType()) + ")' and '" +
                         STR(rhs) + "(" + PSTR(rhs.GetType()) + ")'.");
  } else if (op == "#") {
    if (IsActualBoundedIntegerType(lhs.GetType()) &&
        IsActualBoundedIntegerType(rhs.GetType())) {
      auto lbound = GetSingleUpperBound(lhs.GetType());
      auto rbound = GetSingleUpperBound(rhs.GetType());
      auto lsig = ValueItemAsString(lbound);
      auto rsig = ValueItemAsString(rbound);
      if (isa<int>(&lbound))
        lsig = "const_" + lsig;
      else
        lsig = SignatureOfSymbol(visitor->SSTab().InScopeName("@" + lsig));
      if (isa<int>(&rbound))
        rsig = "const_" + rsig;
      else
        rsig = SignatureOfSymbol(visitor->SSTab().InScopeName("@" + rsig));
      res = TryToSimplifyBinary(loc, op, lsig, rsig, verbose);
      if (!res) {
        auto lvn = GetOrInsertValueNumberFromSignature(lsig);
        auto rvn = GetOrInsertValueNumberFromSignature(rsig);
        res = op + ":#" + std::to_string(lvn) + ":#" + std::to_string(rvn);
      }
    } else
      choreo_unreachable("operation is not permitted.");
  } else
    choreo_unreachable("operation is not supported for bounded variables.");

  if (trace && verbose)
    dbgs() << ScopeIndent() << "<Bounded> '" << STR(lhs) << " " << op << " "
           << STR(rhs) << "' ubound: '" << *res << "'\n";
  return res;
}

std::optional<std::string>
ValueNumbering::GenerateSpecialNodeSignature(const AST::Node& node) {
  if (auto* n = dyn_cast<AST::Expr>(&node))
    if (n->IsArith()) {
      if (isa<BoundedType>(n->GetL()->GetType()) ||
          isa<BoundedType>(n->GetR()->GetType())) {
        return SignBoundedOperation(n->LOC(), n->op, *n->GetL(), *n->GetR(),
                                    trace);
      }
    }

  return std::nullopt;
}

std::optional<std::string>
ValueNumbering::TryToSimplifyNodeSignature(const AST::Node& node) {
  if (isa<AST::Identifier>(&node)) {
    return std::nullopt;
  } else if (auto* n = dyn_cast<AST::Expr>(&node)) {
    // Applies the algebraic simplification
    std::map<std::string, std::function<std::optional<std::string>()>>
        alg_simp = {
            {"+",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "+",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));

               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " + "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"-",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "-",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));

               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " - "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"*",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "*",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " * "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"/",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "/",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " / "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"%",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "%",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " % "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"cdiv",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "cdiv",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " cdiv "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"#",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "#",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " # "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"||",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"&&",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"!",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"?",
             [this, &n]() -> std::optional<std::string> {
               const auto& cond = n->GetC();
               auto res_cond = TryToSimplifyBinary(
                   cond->LOC(), cond->op, GetSignatureForNode(*cond->GetL()),
                   GetSignatureForNode(*cond->GetR()));
               if (res_cond) {
                 std::string res;
                 if (res_cond == "true")
                   res = GetSignatureForNode(*n->GetL());
                 else
                   res = GetSignatureForNode(*n->GetR());
                 if (trace) {
                   dbgs() << ScopeIndent() << "<Simplify> '"
                          << GenerateNodeSignature(*n->GetC(), false) << " ? "
                          << GenerateNodeSignature(*n->GetL(), false) << " : "
                          << GenerateNodeSignature(*n->GetR(), false)
                          << "' to '" << res << "'\n";
                 }
                 return res;
               }
               return std::nullopt;
             }},
            {"<",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), "<",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " < "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {">",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(n->LOC(), ">",
                                              GetSignatureForNode(*n->GetL()),
                                              GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false) << " > "
                        << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                        << res.value() << "'\n";
               return res;
             }},
            {"==",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(
                   n->LOC(), "==", GetSignatureForNode(*n->GetL()),
                   GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false)
                        << " == " << GenerateNodeSignature(*n->GetR(), false)
                        << "' to '" << res.value() << "'\n";
               return res;
             }},
            {"!=",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(
                   n->LOC(), "!=", GetSignatureForNode(*n->GetL()),
                   GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false)
                        << " != " << GenerateNodeSignature(*n->GetR(), false)
                        << "' to '" << res.value() << "'\n";
               return res;
             }},
            {"<=",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(
                   n->LOC(), "<=", GetSignatureForNode(*n->GetL()),
                   GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false)
                        << " <= " << GenerateNodeSignature(*n->GetR(), false)
                        << "' to '" << res.value() << "'\n";
               return res;
             }},
            {">=",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyBinary(
                   n->LOC(), ">=", GetSignatureForNode(*n->GetL()),
                   GetSignatureForNode(*n->GetR()));
               if (res && trace)
                 dbgs() << ScopeIndent() << "<Simplify> '"
                        << GenerateNodeSignature(*n->GetL(), false)
                        << " >= " << GenerateNodeSignature(*n->GetR(), false)
                        << "' to '" << res.value() << "'\n";
               return res;
             }},
            {"dataof",
             [this, &n]() -> std::optional<std::string> {
               // care about the span the data has
               return GetSignatureForNode(*n->GetR());
             }},
            {"sizeof",
             [this, &n]() -> std::optional<std::string> {
               auto s = GetShape(visitor->NodeType(*n->GetR()));
               if (s.IsValid() && !s.IsDynamic())
                 return "const_" + s.GetElementCountExpression();
               // TODO: associate span.size valno with any span in value
               // numbering
               return std::nullopt;
             }},
            {"ubound",
             [this, &n]() -> std::optional<std::string> {
               if (auto id = dyn_cast<AST::Identifier>(n->GetR())) {
                 return visitor->SSTab().NameInScopeOrNull(
                     "@" + cast<AST::Identifier>(id)->name);
               } else
                 choreo_unreachable("upper bound expression is unexpected.");
             }},
            {"getith",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt;
             }},
            {"elemof",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt;
             }},
            {"dimof", // calculate the dim of a given mdspan index
             [this, &n]() -> std::optional<std::string> {
               std::string base_sig;
               if (isa<BoundedType>(n->GetL()->GetType())) {
                 auto id = cast<AST::Expr>(n->GetL())->GetSymbol();
                 assert(id != nullptr && "not an identifier.");
                 base_sig = SignatureOfSymbol(
                     visitor->SSTab().InScopeName("@" + id->name));
               } else
                 base_sig = GetSignatureForNode(*n->GetL());
               auto cv = RemovePrefixOrNull("index_const_",
                                            GetSignatureForNode(*n->GetR()));
               assert(cv && "indexing of mdspan can not be evaluated.");
               assert((std::stoi(*cv) < CountElementsInSignature(base_sig)) &&
                      "out of bound in 'dimof'.");

               if (CountElementsInSignature(base_sig) == 1) return base_sig;

               return base_sig + "(" + *cv + ")";
             }},
            {"ref", // it is a reference to another node
             [this, &n]() -> std::optional<std::string> {
               auto expr = GetSignatureForNode(*n->GetR());

               return expr;
             }},
        };
    if ((n->GetForm() == AST::Expr::Binary) && (n->op != "dimof") &&
        (n->op != "elemof") &&
        ((CountElementsInSignature(GetSignatureForNode(*n->GetR())) > 1) ||
         (CountElementsInSignature(GetSignatureForNode(*n->GetL())) > 1)))
      return SignBinaryCompositeValues(n->LOC(), n->op,
                                       GetSignatureForNode(*n->GetL()),
                                       GetSignatureForNode(*n->GetR()));
    // Try to simplify immediately
    auto it = alg_simp.find(n->op);
    if (it != alg_simp.end())
      return it->second(); // Execute the lambda function if found
    else {
      choreo_unreachable(
          ("No handler for operation `" + n->op + "'").c_str()); // Default case
    }
  }
  return std::nullopt;
}

std::string ValueNumbering::GenerateNodeSignature(const AST::Node& node,
                                                  bool optimiz) {
  // handle bounded variables
  if (auto ssig = GenerateSpecialNodeSignature(node)) return *ssig;

  if (optimiz) {
    auto sns = TryToSimplifyNodeSignature(node);
    if (sns) return *sns;
  }

  if (auto* n = dyn_cast<AST::IntLiteral>(&node)) {
    if (IsUnKnownInteger(n->value)) return "?";
    return "const_" + std::to_string(n->value);
  } else if (auto* n = dyn_cast<AST::FloatLiteral>(&node)) {
    if (n->IsFloat32()) {
      auto f32 = n->Val_f32();
      if (IsUnKnownFloatPoint(f32)) return "?";
      return "const_" + std::to_string(f32) + "f";
    } else if (n->IsFloat64()) {
      auto f64 = n->Val_f64();
      if (IsUnKnownFloatPoint(f64)) return "?";
      return "const_" + std::to_string(f64);
    } else {
      choreo_unreachable("unexpected float point type.");
    }
  } else if (auto* n = dyn_cast<AST::Boolean>(&node)) {
    return n->value;
  } else if (auto* v = dyn_cast<AST::Identifier>(&node)) {
    auto sname = VNSymbolName(*v);
    if (auto name_in_scope = visitor->SSTab().NameInScopeOrNull(sname)) {
      if (HasValueNumberOfSignature(*name_in_scope)) return *name_in_scope;
      // error: the name exists but does not have a value number
      Error(node.LOC(), "symbol `" + *name_in_scope +
                            "' is not associated with a value number.");
      choreo_unreachable();
    }
    // or else, it is a new name definition
    return visitor->SSTab().ScopedName(sname);
  } else if (auto* b = dyn_cast<AST::Expr>(&node)) {
    auto signature = b->op;

    if (b->GetC()) {
      int valno = GetValueNumberForNode(*b->GetC());
      assert(ValidVN(valno) && "invalid value number.");
      signature += ":#" + std::to_string(valno);
    }
    if (b->GetL()) {
      int valno = GetValueNumberForNode(*b->GetL());
      assert(ValidVN(valno) && "invalid value number.");
      signature += ":#" + std::to_string(valno);
    }

    assert(b->GetR() && "expression is invalid.");
    int valno = GetValueNumberForNode(*b->GetR());
    if (!ValidVN(valno)) {
      // a reference node may have no valNo
      assert((b->GetForm() == AST::Expr::Reference) && "invalid value number.");
      return "";
    }
    return signature + ":#" + std::to_string(valno);
  } else if (auto* b = dyn_cast<AST::MultiValues>(&node)) {
    assert(b->values.size() > 0 && "must have values inside.");

    // when it contains a single value, return the reference.
    if (b->values.size() == 1) return GetSignatureForNode(*b->values[0]);

    auto NodeSignature = [this](AST::Node& n) {
      int vn = GetValueNumberForNode(n);
      auto sig = GetSignatureFromValueNumber(vn);
      return "#" + std::to_string(vn);
    };
    std::string signature = NodeSignature(*b->values[0]);
    for (size_t i = 1; i < b->values.size(); ++i)
      signature += "," + NodeSignature(*b->values[i]);
    return signature;
  } else if (auto* it = dyn_cast<AST::IntTuple>(&node)) {
    return GenerateNodeSignature(*(it->GetValues()));
  } else if (auto* mds = dyn_cast<AST::MultiDimSpans>(&node)) {
    return GenerateNodeSignature(*(mds->list));
  } else if (auto* sa = dyn_cast<AST::SpanAs>(&node)) {
    return GenerateNodeSignature(*(sa->list));
  } else if (auto* b = dyn_cast<AST::ParamList>(&node)) {
    std::string signature;
    if (b->values.size() > 0) {
      signature = "p:#" + std::to_string(GetValueNumberForNode(*b->values[0]));
      for (size_t i = 1; i < b->values.size(); ++i)
        signature +=
            ",#" + std::to_string(GetValueNumberForNode(*b->values[i]));
    }
    return signature;
  } else if (auto* n = dyn_cast<AST::IntIndex>(&node)) {
    return "index_" + GenerateNodeSignature(*n->value);
  } else if (auto* s = dyn_cast<AST::Select>(&node)) {
    // only care about span
    auto vn = GetValueNumberForNode(*s->expr_list->ValueAt(0));
    return GetSignatureFromValueNumber(vn);
  }

  if (trace)
    Warning(node.LOC(), "invalid signature for expression `" + AST::STR(node) +
                            "': " + node.TypeNameString() + ".");

  return ""; // invalid value
}

bool ValueNumbering::HasValueNumberForNode(const AST::Node& n) {
  // the node has been visited before
  if (InternalHasNodeValNo(&n)) return true;

  std::string signature = GenerateNodeSignature(n);
  if (signature == "") return false;

  return HasValueNumberOfSignature(signature);
}

// Note:
// the valno related to a node could either be:
//   1. integer value of the expression
//   2. associated span value of the expression
//   3. associated upper bound value of the expression
int ValueNumbering::GetValueNumberForNode(const AST::Node& n) {
  // if it is an visited/numbered node
  if (InternalHasNodeValNo(&n)) return InternalGetNodeValNo(&n);

  // workaround
  // TODO(wsj) reference with bounded var and spanned var
  if (auto id = dyn_cast<AST::Identifier>(&n)) {
    // Must consider about the scope of any identifier reference
    auto sname = VNSymbolName(*id);
    auto pty = visitor->NodeType(n);
    if (auto name_in_scope = visitor->SSTab().NameInScopeOrNull(sname))
      return GetValueNumberOfSignature(*name_in_scope);
    else
      choreo_unreachable("symbol `" + id->name + "` with name: " + sname +
                         " is not valued.");
  }

  std::string signature = GenerateNodeSignature(n);
  if (signature == "")
    Error(n.LOC(),
          "failed to generate signature for expression `" + AST::STR(n) + "'.");

  return GetValueNumberOfSignature(signature);
}

int ValueNumbering::GenerateValueNumberForNode(const AST::Node& n) {
  std::string signature = GenerateNodeSignature(n);
  if (signature == "")
    Error(n.LOC(),
          "failed to generate signature for expression `" + AST::STR(n) + "'.");

  // Duplicated computation: different expression encounters the same signature
  if (HasValueNumberOfSignature(signature))
    return GetValueNumberOfSignature(signature);

  int valNo = GenerateValueNumberFromSignature(signature);

  // cache the value number
  assert(!InternalHasNodeValNo(&n));
  InternalUpdateNodeValNo(&n, valNo);

  return valNo;
}

int ValueNumbering::GetValueNumberOfSignature(const std::string& signature) {
  if (signature == "") choreo_unreachable("invalid signature provided.");

  if (signature == "?") return UnknownValue();

  // Check if this expression has been encountered before
  if (InternalHasExprValNo(signature))
    return InternalGetExprValNo(signature); // Return existing value number

  choreo_unreachable("failed to get value number of signature \"" + signature +
                     "\".");

  return GetInvalidValueNumber();
}

void ValueNumbering::BindValueNumbers(int vn0, int vn1) {
  assert(ValidVN(vn0) && ValidVN(vn1) && "invalid value number is provided.");

  AddBind(vn0, vn1);

  if (trace)
    dbgs() << ScopeIndent() << "<Bind> VN #" << vn0 << " <-> VN #" << vn1
           << "\n";
}

bool ValueNumbering::HasValueNumberOfSignature(const std::string& signature) {
  return InternalHasExprValNo(signature);
}

bool ValueNumbering::HasValidValueNumberOfSignature(
    const std::string& signature) {
  if (InternalHasExprValNo(signature))
    return ValidVN(InternalGetExprValNo(signature));

  return false;
}

int ValueNumbering::GetOrInsertValueNumberFromSignature(
    const std::string& signature) {
  if (PrefixedWith(signature, "#") &&
      (CountElementsInSignature(signature) == 1)) {
    // works for input like "#1"
    auto res = RemovePrefixOrNull("#", signature);
    assert(res);
    return std::stoi(res.value());
  }
  if (HasValueNumberOfSignature(signature))
    return GetValueNumberOfSignature(signature);
  return GenerateValueNumberFromSignature(signature);
}

int ValueNumbering::GenerateValueNumberFromSignature(
    const std::string& signature) {
  if (signature == "?") return UnknownValue();

  if (HasValueNumberOfSignature(signature))
    choreo_unreachable("signature \"" + signature + "\" has already existed.");

  int valNo = nextValueNumber++;

  InternalUpdateExprValNo(signature, valNo);
  InternalUpdateValNoExpr(valNo, signature);

  if (trace)
    dbgs() << ScopeIndent() << "New VN #" << valNo << ": '" << signature
           << "'\n";

  return valNo;
}

std::string ValueNumbering::ScopeIndent() {
  std::string indent;
  for (size_t i = 0; i <= visitor->SSTab().ScopeDepth(); ++i) indent += " ";
  return indent;
}

void ValueNumbering::Error(const location& loc, const std::string& message) {
  visitor->Error(loc, message);
}

void ValueNumbering::Warning(const location& loc, const std::string& message) {
  visitor->Warning(loc, message);
}
