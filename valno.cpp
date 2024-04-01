#include "valno.hpp"

using namespace Choreo;

void ValueNumbering::EnterScope(const std::string& name) {
  std::string indent = ScopeIndent();
  visitor->EnterScope(name);

  if (expressionValueNumbers.empty())
    expressionValueNumbers.push_back({});
  else
    expressionValueNumbers.push_back(expressionValueNumbers.back());

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
               auto mdspan = GetSignatureForNode(*n->value_l);
               auto cv = PrefixedWith("index_const_",
                                      GetSignatureForNode(*n->value_r));
               assert(cv && "mdspan index can not be evaluated.");
               return mdspan + "(" + *cv + ")";
             }},
            {"ref",  // it is a reference to another node
             [this, &n]() -> std::optional<std::string> {
               int valNo = GetValueNumberForNode(*n->value_r);
               if (!ValidVN(valNo)) return std::nullopt;

               auto expr = GetSignatureFromValueNumber(valNo);

               // handle syntax suger "a {(0)}";
               if (!ref) return expr;
               int refNo = GetValueNumberFromSignature(ref.value());
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
    return "const_" + std::to_string(n->value);
  } else if (auto* v = dyn_cast<AST::Identifier>(&node)) {
    auto scoped_name = visitor->ScopedName(v->name);
    int valno = GetOrInsertValueNumberFromSignature(scoped_name);
    assert(ValidVN(valno));
    return scoped_name;
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
  } else if (auto* b = dyn_cast<AST::MultiNodes>(&node)) {
    std::string signature;
    if (b->values.size() > 0 && ValidVN(GetValueNumberForNode(*b->values[0]))) {
      signature = "#" + std::to_string(GetValueNumberForNode(*b->values[0]));
      for (size_t i = 1; i < b->values.size(); ++i)
        signature +=
            ",#" + std::to_string(GetValueNumberForNode(*b->values[i]));
    }
    return signature;
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
  return "";
}

int ValueNumbering::GetValueNumberForNode(AST::Node& expr) {
  std::string signature = GenerateNodeSignature(expr);
  if (signature == "") return InvalidValueNumber();

  // std::cout << "Garfee: signature: " << signature << std::endl;
  return GetOrInsertValueNumberFromSignature(signature);
}

int ValueNumbering::GetValueNumberFromSignature(const std::string& signature) {
  // Check if this expression has been encountered before
  auto it = expressionValueNumbers.back().find(signature);
  if (it != expressionValueNumbers.back().end())
    return it->second;  // Return existing value number

  return InvalidValueNumber();
}

int ValueNumbering::GetOrInsertValueNumberFromSignature(
    const std::string& signature) {
  int valNo = GetValueNumberFromSignature(signature);
  if (ValidVN(valNo)) return valNo;

  // If not, assign a new value number
  valNo = nextValueNumber++;
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
