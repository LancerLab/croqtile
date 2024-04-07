#include "valno.hpp"

using namespace Choreo;

void ValueNumbering::EnterScope(const std::string& name) {
  std::string indent = ScopeIndent();
  visitor->EnterScope(name);

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

  if (trace) os << indent << "scope-" << visitor->ScopeDepth() << " {\n";
}

void ValueNumbering::LeaveScope() {
  std::string sname = std::to_string(visitor->ScopeDepth());
  visitor->LeaveScope();

  assert(!expressionValueNumbers.empty() && !valueNumberExpressions.empty());

  expressionValueNumbers.pop_back();
  valueNumberExpressions.pop_back();
  nodeValueNumbers.pop_back();

  // reset value number when leaving the function scope
  if (expressionValueNumbers.empty())
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

std::optional<std::string> ValueNumbering::TryToSimplifyNodeSignature(
    AST::Node& node) {
  if (auto* b = dyn_cast<AST::Identifier>(&node)) {
    (void)b;
    return std::nullopt;
  } else if (auto* n = dyn_cast<AST::Expr>(&node)) {
    // Try to simplify immediately
    std::map<std::string, std::function<std::optional<std::string>()>> actions =
        {
            {"+",
             [this, &n]() -> std::optional<std::string> {
               auto l_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_l));
               auto r_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_r));
               if (l_cv && r_cv) {
                 auto res = "const_" +
                            std::to_string(std::stoi(*l_cv) + std::stoi(*r_cv));
                 if (trace)
                   os << ScopeIndent() << "<Simplify> '"
                      << GenerateNodeSignature(*n->value_l, false) << " + "
                      << GenerateNodeSignature(*n->value_r, false) << "' to '"
                      << res << "'\n";
                 return res;
               } else
                 return std::nullopt;
             }},
            {"-",
             [this, &n]() -> std::optional<std::string> {
               auto l_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_l));
               auto r_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_r));
               if (l_cv && r_cv) {
                 auto res = "const_" +
                            std::to_string(std::stoi(*l_cv) - std::stoi(*r_cv));
                 if (trace)
                   os << ScopeIndent() << "<Simplify> '"
                      << GenerateNodeSignature(*n->value_l, false) << " - "
                      << GenerateNodeSignature(*n->value_r, false) << "' to '"
                      << res << "'\n";
                 return res;
               } else
                 return std::nullopt;
             }},
            {"*",
             [this, &n]() -> std::optional<std::string> {
               auto l_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_l));
               auto r_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_r));
               if (l_cv && r_cv) {
                 auto res = "const_" +
                            std::to_string(std::stoi(*l_cv) * std::stoi(*r_cv));
                 if (trace)
                   os << ScopeIndent() << "<Simplify> '"
                      << GenerateNodeSignature(*n->value_l, false) << " * "
                      << GenerateNodeSignature(*n->value_r, false) << "' to '"
                      << res << "'\n";
                 return res;
               } else
                 return std::nullopt;
             }},
            {"/",
             [this, &n]() -> std::optional<std::string> {
               auto l_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_l));
               auto r_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_r));
               if (l_cv && r_cv) {
                 auto res = "const_" +
                            std::to_string(std::stoi(*l_cv) / std::stoi(*r_cv));
                 if (trace)
                   os << ScopeIndent() << "<Simplify> '"
                      << GenerateNodeSignature(*n->value_l, false) << " / "
                      << GenerateNodeSignature(*n->value_r, false) << "' to '"
                      << res << "'\n";
                 return res;
               } else
                 return std::nullopt;
             }},
            {"%",
             [this, &n]() -> std::optional<std::string> {
               auto l_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_l));
               auto r_cv =
                   PrefixedWith("const_", GetSignatureForNode(*n->value_r));
               if (l_cv && r_cv) {
                 auto res = "const_" +
                            std::to_string(std::stoi(*l_cv) % std::stoi(*r_cv));
                 if (trace)
                   os << ScopeIndent() << "<Simplify> '"
                      << GenerateNodeSignature(*n->value_l, false) << " % "
                      << GenerateNodeSignature(*n->value_r, false) << "' to '"
                      << res << "'\n";
                 return res;
               } else
                 return std::nullopt;
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

#if 0
               // handle syntax suger "a {(0)}";
               if (!ref) return expr;
               int refNo = GetValueNumberOfSignature(ref.value());
               if (!ValidVN(refNo)) return expr;
               auto ref_expr = GetSignatureFromValueNumber(refNo);

               if (auto cv = PrefixedWith("index_const_",
                                          GetSignatureForNode(*n->value_r))) {
                 auto signature = ref_expr + "(" + *cv + ")";
                 if (trace)
                   os << ScopeIndent() << "<DeRef> fill '(" + *cv + ")' to be "
                      << signature << "\n";
                 return signature;
               }
#endif
               return expr;
             }},
        };
    auto it = actions.find(n->op);
    if (it != actions.end())
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
    if (n->value == __UNKNOWN_INTVAL__)
      return "?";
    return "const_" + std::to_string(n->value);
  } else if (auto* v = dyn_cast<AST::Identifier>(&node)) {
    if (auto name_in_scope = visitor->InScopeName(v->name)) {
      if (HasValueNumberOfSignature(name_in_scope.value()))
        return name_in_scope.value();
      // error: the name exists but does not have a value number
      Error(node.LOC(), "symbol `" + name_in_scope.value() +
                            "' is not associated with a value number.");
      choreo_unreachable();
    }
    // or else, it is a new name definition
    return visitor->ScopedName(v->name);
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
    if (auto name = visitor->InScopeName(id->name))
      return GetValueNumberOfSignature(name.value());
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
  if (signature == "?")
    return UnknownValue();

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
  for (size_t i = 0; i <= visitor->ScopeDepth(); ++i) indent += " ";
  return indent;
}

void ValueNumbering::Error(const location& loc, const std::string& message) {
  visitor->Error(loc, message);
}
void ValueNumbering::Warning(const location& loc, const std::string& message) {
  visitor->Warning(loc, message);
}
