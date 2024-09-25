#include "valno.hpp"

using namespace Choreo;

namespace {
std::vector<int> CollectValueNumbers(const std::string& input) {
  std::vector<int> res;
  std::regex valuePattern("#(-?\\d+)");
  auto begin = std::sregex_iterator(input.begin(), input.end(), valuePattern);
  auto end = std::sregex_iterator();

  for (auto i = begin; i != end; ++i) {
    std::smatch match = *i;
    std::string matchStr =
        match.str(1);  // Capture the number part of the match
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

}  // namespace

void ValueNumbering::EnterScope(const std::string& name) {
  std::string indent = ScopeIndent();
  visitor->SSTab().EnterScope(name);

  if (expressionValueNumbers.empty()) {
    expressionValueNumbers.push_back({});
    nodeValueNumbers.push_back({});
  } else {
    expressionValueNumbers.push_back(expressionValueNumbers.back());
    nodeValueNumbers.push_back(nodeValueNumbers.back());
  }

  if (valueNumberExpressions.empty())
    valueNumberExpressions.push_back({});
  else
    valueNumberExpressions.push_back(valueNumberExpressions.back());

  if (trace)
    if (visitor->SSTab().ScopeDepth() > 1)
      os << indent << "scope-" << visitor->SSTab().ScopeDepth() - 1 << " {\n";
}

void ValueNumbering::LeaveScope() {
  if (visitor->SSTab().ScopeDepth() <= 1) return;

  std::string sname = std::to_string(visitor->SSTab().ScopeDepth() - 1);
  visitor->SSTab().LeaveScope();

  assert(!expressionValueNumbers.empty() && !valueNumberExpressions.empty());

  expressionValueNumbers.pop_back();
  valueNumberExpressions.pop_back();
  nodeValueNumbers.pop_back();

  // reset value number when leaving the function scope
  if (expressionValueNumbers.empty() || expressionValueNumbers.size() == 1) {
    nextValueNumber = 0;
    bind_info.Clear();
  }

  if (trace) os << ScopeIndent() << "} // end scope-" << sname << "\n";
}

// It binds a expression sigature with an existing value number.  use it
// carefully.
void ValueNumbering::AssociateSignatureWithValueNumber(const std::string& sig,
                                                       int valno) {
  if (expressionValueNumbers.back().count(sig)) {
    // signature exists
    assert((expressionValueNumbers.back()[sig] == valno) &&
           "must associate signature with different value number.");
  }

  assert(valueNumberExpressions.back().count(valno) &&
         "invalid value number provided.");

  expressionValueNumbers.back()[sig] = valno;

  if (trace)
    os << ScopeIndent() << "Alias \"" << sig << "\" -> #" << valno << "\n";
}

void ValueNumbering::AssociateSignatureWithInvalidValueNumber(
    const std::string& sig) {
  assert(!expressionValueNumbers.back().count(sig) && "signature exists.");
  expressionValueNumbers.back()[sig] = GetInvalidValueNumber();

  if (trace) os << ScopeIndent() << "Alias \"" << sig << "\" -> #<invalid>\n";
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
      os << ScopeIndent() << "Alias(Rebind) \"" << sig << "\" -> #" << valno
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

  assert((CountElementsInSignature(l_sig) > 1) ||
         (CountElementsInSignature(r_sig) > 1));

  std::string lhs, rhs;
  // specially handle ituple/mdspan + integer: broadcast integer
  if (CountElementsInSignature(l_sig) == 1) {
    int elem_count = CountElementsInSignature(r_sig);
    assert(elem_count > 1);
    int valno = GetValueNumberOfSignature(l_sig);
    lhs = "#" + std::to_string(valno);
    for (int i = 1; i < elem_count; ++i) lhs += ",#" + std::to_string(valno);
    rhs = r_sig;
  } else if (CountElementsInSignature(r_sig) == 1) {
    int elem_count = CountElementsInSignature(l_sig);
    assert(elem_count > 1);
    int valno = GetValueNumberOfSignature(r_sig);
    rhs = "#" + std::to_string(valno);
    for (int i = 1; i < elem_count; ++i) rhs += ",#" + std::to_string(valno);
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

std::optional<std::string> ValueNumbering::TryToSimplifyBinary(
    const location& loc, const std::string& op, const std::string& lhs,
    const std::string& rhs, bool verbose) {
  if (op == "concat") return std::nullopt;
  auto l_cv = RemovePrefixOrNull("const_", lhs);
  auto r_cv = RemovePrefixOrNull("const_", rhs);
  if (l_cv && r_cv) {
    std::string res = "const_";
    if (op == "+")
      res += std::to_string(std::stoi(*l_cv) + std::stoi(*r_cv));
    else if (op == "-")
      res += std::to_string(std::stoi(*l_cv) - std::stoi(*r_cv));
    else if (op == "*")
      res += std::to_string(std::stoi(*l_cv) * std::stoi(*r_cv));
    else if (op == "/") {
      int div_end = std::stoi(*r_cv);
      if (div_end == 0) {
        os << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / " << rhs
           << "\n";
        choreo_unreachable("divide by zero is found in shape evaluation.");
      }
      res += std::to_string(std::stoi(*l_cv) / div_end);
    } else if (op == "%") {
      int div_end = std::stoi(*r_cv);
      if (div_end == 0) {
        os << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / " << rhs
           << "\n";
        choreo_unreachable("divide by zero is found in shape evaluation.");
      }
      res += std::to_string(std::stoi(*l_cv) % std::stoi(*r_cv));
    } else if (op == "<") {
      res = std::stoi(*l_cv) < std::stoi(*r_cv) ? "true" : "false";
    } else if (op == ">") {
      res = std::stoi(*l_cv) > std::stoi(*r_cv) ? "true" : "false";
    } else if (op == "==") {
      res = std::stoi(*l_cv) == std::stoi(*r_cv) ? "true" : "false";
    } else if (op == "!=") {
      res = std::stoi(*l_cv) != std::stoi(*r_cv) ? "true" : "false";
    } else if (op == "<=") {
      res = std::stoi(*l_cv) <= std::stoi(*r_cv) ? "true" : "false";
    } else if (op == ">=") {
      res = std::stoi(*l_cv) >= std::stoi(*r_cv) ? "true" : "false";
    } else if (op == "cdiv") {
      int div_end = std::stoi(*r_cv);
      if (div_end == 0) {
        os << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / " << rhs
           << "\n";
        choreo_unreachable("divide by zero is found in shape evaluation.");
      }
      res += std::to_string((std::stoi(*l_cv) + std::stoi(*r_cv) - 1) /
                            std::stoi(*r_cv));
    } else {
      Error(loc,
            "simplification of operation `" + op + "' is not yet supported.");
      return std::nullopt;
    }
    if (trace && verbose)
      os << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " " << rhs
         << " to '" << res << "'\n";
    return res;
  }

  // useful simplification: a/(a/b) = b
  if ((op == "/") && !PrefixedWith(lhs, "#") /*not multiple values*/) {
    int rvn = GetValueNumberOfSignature(rhs);
    auto bind_set = GetBindSet(rvn);
    bind_set.insert(rvn);  // always add self
    for (auto div_vn : bind_set) {
      auto sig = GetSignatureFromValueNumber(div_vn);
      if (!PrefixedWith(rhs, "/:")) continue;
      auto div = GetOperandsValNo(sig);
      assert(div.size() == 2);
      if (GetValueNumberOfSignature(lhs) == div[0]) {
        auto res = GetSignatureFromValueNumber(div[1]);

        if (trace && verbose)
          os << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
             << rhs << " to '" << res << "'\n";

        return res;
      }
    }
  }

  return std::nullopt;
}

std::optional<std::string> ValueNumbering::TryToSimplifyNodeSignature(
    AST::Node& node) {
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
                    << GenerateNodeSignature(*n->GetL(), false) << " cdiv "
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
                   os << ScopeIndent() << "<Simplify> '"
                      << GenerateNodeSignature(*n->GetC(), false) << " ? "
                      << GenerateNodeSignature(*n->GetL(), false) << " : "
                      << GenerateNodeSignature(*n->GetR(), false) << "' to '"
                      << res << "'\n";
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
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
                 os << ScopeIndent() << "<Simplify> '"
                    << GenerateNodeSignature(*n->GetL(), false)
                    << " >= " << GenerateNodeSignature(*n->GetR(), false)
                    << "' to '" << res.value() << "'\n";
               return res;
             }},
            {"dataof",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"sizeof",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"ubound",
             [this, &n]() -> std::optional<std::string> {
               if (auto id = dyn_cast<AST::Identifier>(n->GetR())) {
                 return visitor->SSTab().NameInScopeOrNull(
                     "@" + cast<AST::Identifier>(id)->name);
               } else
                 choreo_unreachable("upper bound expression is unexpected.");
             }},
            {"dimof",  // calculate the dim of a given mdspan index
             [this, &n]() -> std::optional<std::string> {
               std::string base_sig;
               if (IsBoundedType(n->GetL()->GetType())) {
                 auto id = cast<AST::Expr>(n->GetL())->GetSymbol();
                 assert(id != nullptr && "not an identifier.");
                 base_sig = SignatureOfSymbol(
                     visitor->SSTab().InScopeName("@" + id->name));
               } else
                 base_sig = GetSignatureForNode(*n->GetL());
               auto cv = RemovePrefixOrNull("index_const_",
                                            GetSignatureForNode(*n->GetR()));
               assert(cv && "indexing of mdspan can not be evaluated.");
               return base_sig + "(" + *cv + ")";
             }},
            {"ref",  // it is a reference to another node
             [this, &n]() -> std::optional<std::string> {
               auto expr = GetSignatureForNode(*n->GetR());

               return expr;
             }},
        };
    if ((n->GetForm() == AST::Expr::Binary) && (n->op != "dimof") &&
        ((CountElementsInSignature(GetSignatureForNode(*n->GetR())) > 1) ||
         (CountElementsInSignature(GetSignatureForNode(*n->GetL())) > 1)))
      return SignBinaryCompositeValues(n->LOC(), n->op,
                                       GetSignatureForNode(*n->GetL()),
                                       GetSignatureForNode(*n->GetR()));
    // Try to simplify immediately
    auto it = alg_simp.find(n->op);
    if (it != alg_simp.end())
      return it->second();  // Execute the lambda function if found
    else {
      choreo_unreachable(("No handler for operation `" + n->op + "'")
                             .c_str());  // Default case
    }
  }
  return std::nullopt;
}

std::string ValueNumbering::GenerateNodeSignature(AST::Node& node,
                                                  bool optimiz) {
  if (optimiz) {
    auto sns = TryToSimplifyNodeSignature(node);
    if (sns) return *sns;
  }

  if (auto* n = dyn_cast<AST::IntLiteral>(&node)) {
    if (IsUnKnownInteger(n->value)) return "?";
    return "const_" + std::to_string(n->value);
  } else if (auto* n = dyn_cast<AST::Boolean>(&node)) {
    return n->value;
  } else if (auto* v = dyn_cast<AST::Identifier>(&node)) {
    if (auto name_in_scope = visitor->SSTab().NameInScopeOrNull(v->name)) {
      if (HasValueNumberOfSignature(*name_in_scope)) return *name_in_scope;
      // error: the name exists but does not have a value number
      Error(node.LOC(), "symbol `" + *name_in_scope +
                            "' is not associated with a value number.");
      choreo_unreachable();
    }
    // or else, it is a new name definition
    return visitor->SSTab().ScopedName(v->name);
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
    auto vn = GetValueNumberForNode(*s->span_expr_list->ValueAt(0));
    return GetSignatureFromValueNumber(vn);
  }

  if (trace)
    Warning(node.LOC(), "invalid signature for expression `" + AST::STR(node) +
                            "': " + node.TypeNameString() + ".");

  return "";  // invalid value
}

bool ValueNumbering::HasValueNumberForNode(AST::Node& n) {
  // the node has been visited before
  if (nodeValueNumbers.back().count(&n)) return true;

  std::string signature = GenerateNodeSignature(n);
  if (signature == "") return false;

  return HasValueNumberOfSignature(signature);
}

int ValueNumbering::GetValueNumberForNode(AST::Node& n) {
  // if it is an visited/numbered node
  if (nodeValueNumbers.back().count(&n)) return (nodeValueNumbers.back())[&n];

  // workaround
  // TODO(wsj) reference with bounded var and spanned var
  if (auto id = dyn_cast<AST::Identifier>(&n)) {
    // Must consider about the scope of any identifier reference
    auto name = id->name;
    auto pty = visitor->SSTab().LookupSymbol(name);
    if (isa<SpannedType>(pty) || isa<FutureType>(pty)) {
      name += ".span";
    } else if (isa<BoundedITupleType>(pty)) {
      name = "@" + name;
    }
    if (auto name_in_scope = visitor->SSTab().NameInScopeOrNull(name))
      return GetValueNumberOfSignature(*name_in_scope);
    else
      choreo_unreachable("symbol `" + id->name + "` with name: " + name +
                         " is not valued.");
  }

  std::string signature = GenerateNodeSignature(n);
  if (signature == "")
    Error(n.LOC(),
          "failed to generate signature for expression `" + AST::STR(n) + "'.");

  return GetValueNumberOfSignature(signature);
}

int ValueNumbering::GenerateValueNumberForNode(AST::Node& n) {
  std::string signature = GenerateNodeSignature(n);
  if (signature == "")
    Error(n.LOC(),
          "failed to generate signature for expression `" + AST::STR(n) + "'.");

  // Duplicated computation: different expression encounters the same signature
  if (HasValueNumberOfSignature(signature))
    return GetValueNumberOfSignature(signature);

  int valNo = GenerateValueNumberFromSignature(signature);

  // cache the value number
  nodeValueNumbers.back().emplace(&n, valNo);

  return valNo;
}

int ValueNumbering::GetValueNumberOfSignature(const std::string& signature) {
  if (signature == "") choreo_unreachable("invalid signature provided.");

  if (signature == "?") return UnknownValue();

  // Check if this expression has been encountered before
  auto it = expressionValueNumbers.back().find(signature);
  if (it != expressionValueNumbers.back().end())
    return it->second;  // Return existing value number

  choreo_unreachable("failed to get value number of signature \"" + signature +
                     "\".");

  return GetInvalidValueNumber();
}

void ValueNumbering::BindValueNumbers(int vn0, int vn1) {
  assert(ValidVN(vn0) && ValidVN(vn1) && "invalid value number is provided.");

  AddBind(vn0, vn1);

  if (trace)
    os << ScopeIndent() << "<Bind> VN #" << vn0 << " <-> VN #" << vn1 << "\n";
}

bool ValueNumbering::HasValueNumberOfSignature(const std::string& signature) {
  // Check if this expression has been encountered before
  auto it = expressionValueNumbers.back().find(signature);
  if (it != expressionValueNumbers.back().end())
    return true;  // Return existing value number

  return false;
}

int ValueNumbering::GetOrInsertValueNumberFromSignature(
    const std::string& signature) {
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
  expressionValueNumbers.back()[signature] = valNo;
  valueNumberExpressions.back()[valNo] = signature;

  if (trace)
    os << ScopeIndent() << "New VN #" << valNo << ": '" << signature << "'\n";

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
