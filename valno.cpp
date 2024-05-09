#include "valno.hpp"

using namespace Choreo;

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
  if (expressionValueNumbers.empty() || expressionValueNumbers.size() == 1)
    nextValueNumber = 0;

  if (trace) os << ScopeIndent() << "} // end scope-" << sname << "\n";
}

// It binds a expression sigature with an existing value number.  use it
// carefully.
void ValueNumbering::AssociateSignatureWithValueNumber(const std::string& sig,
                                                       int valno) {
  assert(!expressionValueNumbers.back().count(sig) && "signature existed.");
  assert(valueNumberExpressions.back().count(valno) &&
         "invalid value number provided.");

  expressionValueNumbers.back()[sig] = valno;

  if (trace)
    os << ScopeIndent() << "Alias \"" << sig << "\" -> #" << valno << "\n";
}

std::optional<std::string> ValueNumbering::TryToSimplifyTernary(
    const location& loc, const std::string& op, const std::string& lhs,
    const std::string& rhs, bool verbose) {
  auto l_cv = PrefixedWith("const_", lhs);
  auto r_cv = PrefixedWith("const_", rhs);
  if (l_cv && r_cv) {
    std::string res = "const_";
    if (op == "+")
      res += std::to_string(std::stoi(*l_cv) + std::stoi(*r_cv));
    else if (op == "-")
      res += std::to_string(std::stoi(*l_cv) - std::stoi(*r_cv));
    else if (op == "*")
      res += std::to_string(std::stoi(*l_cv) * std::stoi(*r_cv));
    else if (op == "/")
      res += std::to_string(std::stoi(*l_cv) / std::stoi(*r_cv));
    else if (op == "%")
      res += std::to_string(std::stoi(*l_cv) % std::stoi(*r_cv));
    else {
      Error(loc,
            "simplification of operation `" + op + "' is not yet supported.");
      return std::nullopt;
    }
    if (trace && verbose)
      os << ScopeIndent() << "<Simplify> '" << lhs << " / " << rhs << " to '"
         << res << "'\n";
    return res;
  }
  return std::nullopt;
}

std::optional<std::string> ValueNumbering::TryToSimplifyNodeSignature(
    AST::Node& node) {
  if (auto* b = dyn_cast<AST::Identifier>(&node)) {
    (void)b;
    return std::nullopt;
  } else if (auto* n = dyn_cast<AST::Expr>(&node)) {
    // This is the algebraic simplification
    std::map<std::string, std::function<std::optional<std::string>()>>
        alg_simp = {
            {"+",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyTernary(
                   n->LOC(), "+", GetSignatureForNode(*n->value_l),
                   GetSignatureForNode(*n->value_r));
               if (res && trace)
                 os << ScopeIndent() << "<Simplify> '"
                    << GenerateNodeSignature(*n->value_l, false) << " + "
                    << GenerateNodeSignature(*n->value_r, false) << "' to '"
                    << res.value() << "'\n";
               return res;
             }},
            {"-",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyTernary(
                   n->LOC(), "-", GetSignatureForNode(*n->value_l),
                   GetSignatureForNode(*n->value_r));
               if (res && trace)
                 os << ScopeIndent() << "<Simplify> '"
                    << GenerateNodeSignature(*n->value_l, false) << " - "
                    << GenerateNodeSignature(*n->value_r, false) << "' to '"
                    << res.value() << "'\n";
               return res;
             }},
            {"*",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyTernary(
                   n->LOC(), "*", GetSignatureForNode(*n->value_l),
                   GetSignatureForNode(*n->value_r));
               if (res && trace)
                 os << ScopeIndent() << "<Simplify> '"
                    << GenerateNodeSignature(*n->value_l, false) << " * "
                    << GenerateNodeSignature(*n->value_r, false) << "' to '"
                    << res.value() << "'\n";
               return res;
             }},
            {"/",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyTernary(
                   n->LOC(), "/", GetSignatureForNode(*n->value_l),
                   GetSignatureForNode(*n->value_r));
               if (res && trace)
                 os << ScopeIndent() << "<Simplify> '"
                    << GenerateNodeSignature(*n->value_l, false) << " / "
                    << GenerateNodeSignature(*n->value_r, false) << "' to '"
                    << res.value() << "'\n";
               return res;
             }},
            {"%",
             [this, &n]() -> std::optional<std::string> {
               auto res = TryToSimplifyTernary(
                   n->LOC(), "%", GetSignatureForNode(*n->value_l),
                   GetSignatureForNode(*n->value_r));
               if (res && trace)
                 os << ScopeIndent() << "<Simplify> '"
                    << GenerateNodeSignature(*n->value_l, false) << " % "
                    << GenerateNodeSignature(*n->value_r, false) << "' to '"
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
            {"$",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"<",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {">",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"==",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"!=",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"<=",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {">=",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"sizeof",
             [this, &n]() -> std::optional<std::string> {
               return std::nullopt; /*TODO*/
             }},
            {"ubound",
             [this, &n]() -> std::optional<std::string> {
               if (auto id = dyn_cast<AST::Identifier>(n->value_r)) {
                 return visitor->SSTab().NameInScope(
                     "@" + cast<AST::Identifier>(id)->name);
               } else
                 choreo_unreachable("upper bound expression is unexpected.");
             }},
            {"dimof",  // calculate the dim of a given mdspan index
             [this, &n]() -> std::optional<std::string> {
               auto base = GetSignatureForNode(*n->value_l);
               auto cv = PrefixedWith("index_const_",
                                      GetSignatureForNode(*n->value_r));
               assert(cv && "indexing of mdspan can not be evaluated.");
               return base + "(" + *cv + ")";
             }},
            {"ref",  // it is a reference to another node
             [this, &n]() -> std::optional<std::string> {
               auto expr = GetSignatureForNode(*n->value_r);

               return expr;
             }},
        };
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
    if (n->value == __UNKNOWN_INTVAL__) return "?";
    return "const_" + std::to_string(n->value);
  } else if (auto* v = dyn_cast<AST::Identifier>(&node)) {
    if (auto name_in_scope = visitor->SSTab().NameInScope(v->name)) {
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

    if (b->value_c) {
      int valno = GetValueNumberForNode(*b->value_c);
      assert(ValidVN(valno) && "invalid value number.");
      signature += ":#" + std::to_string(valno);
    }
    if (b->value_l) {
      int valno = GetValueNumberForNode(*b->value_l);
      assert(ValidVN(valno) && "invalid value number.");
      signature += ":#" + std::to_string(valno);
    }

    assert(b->value_r && "expression is invalid.");
    int valno = GetValueNumberForNode(*b->value_r);
    if (!ValidVN(valno)) {
      // a reference node may have no valNo
      assert((b->t == AST::Expr::Reference) && "invalid value number.");
      return "";
    }
    return signature + ":#" + std::to_string(valno);
  } else if (auto* b = dyn_cast<AST::MultiValues>(&node)) {
    assert(b->values.size() > 0 && "must have values inside.");

    // when it contains a single value, return the reference.
    if (b->values.size() == 1) return GetSignatureForNode(*b->values[0]);

    std::string signature =
        "#" + std::to_string(GetValueNumberForNode(*b->values[0]));
    for (size_t i = 1; i < b->values.size(); ++i)
      signature += ",#" + std::to_string(GetValueNumberForNode(*b->values[i]));
    return signature;
  } else if (auto* it = dyn_cast<AST::IntTuple>(&node)) {
    return GenerateNodeSignature(*(it->list));
  } else if (auto* mds = dyn_cast<AST::MultiDimSpans>(&node)) {
    return GenerateNodeSignature(*(mds->list));
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
  }

  if (trace)
    Warning(node.LOC(), "invalid signature for expression `" + AST::STR(node) +
                            "': " + node.NodeTypeString() + ".");

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

  if (auto id = dyn_cast<AST::Identifier>(&n)) {
    // Must consider about the scope of any identifier reference
    if (auto name = visitor->SSTab().NameInScope(id->name))
      return GetValueNumberOfSignature(*name);
    else
      choreo_unreachable("symbol `" + id->name + "' is not valued.");
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
          "failed to generate signature for nession `" + AST::STR(n) + "'.");

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

  return InvalidValueNumber();
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
