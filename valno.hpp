#ifndef __CHOREO_VALUE_NUMBERING_HPP__
#define __CHOREO_VALUE_NUMBERING_HPP__

#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "ast.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

class ValueNumbering {
 private:
  std::vector<std::unordered_map<std::string, int>> variableScopes;
  std::unordered_map<std::string, int> expressionValueNumbers;
  std::unordered_map<int, std::string> valueNumberExpressions;

  int nextValueNumber = 0;

  bool trace = false;
  std::ostream& os;

 public:
  explicit ValueNumbering(bool t, std::ostream& o) : trace(t), os(o) {
    variableScopes.emplace_back();
  }

  void EnterScope() {
    if (trace)
      os << ScopeIndent() << "scope-" << variableScopes.size() << " {\n";

    variableScopes.emplace_back();
  }

  void LeaveScope() {
    if (!variableScopes.empty()) variableScopes.pop_back();

    if (trace)
      os << ScopeIndent() << "} // end scope-" << variableScopes.size()
         << "\n";
  }

  int GetValueNumberForVariable(const std::string& varName) {
    // Look for the variable in the current scope and up through outer scopes
    for (auto it = variableScopes.rbegin(); it != variableScopes.rend(); ++it) {
      if (it->find(varName) != it->end()) {
        return it->at(varName);
      }
    }

    int valueNumber = nextValueNumber++;
    variableScopes.back()[varName] = valueNumber;
    return valueNumber;
  }

  void AssignValueNumberForVariable(const std::string& varName, int valno) {
    variableScopes.back()[varName] = valno;

    if (trace)
      os << ScopeIndent() << "alias VN: \"" << varName << "\" : #" << valno
         << "\n";
  }

  std::optional<std::string> TrySimplifyExpression(AST::Node& node) {
    if (auto* b = dyn_cast<AST::Identifier>(&node)) {
      return std::nullopt;
    } else if (auto* n = dyn_cast<AST::Expr>(&node)) {
      // Try to simplify immediately
      std::map<std::string, std::function<std::optional<std::string>()>>
          actions = {
              {"+",
               [this, &n]() -> std::optional<std::string> {
                 auto l_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_l));
                 auto r_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_r));
                 if (l_cv && r_cv) {
                   int val = std::stoi(*l_cv) + std::stoi(*r_cv);
                   return "const_" + std::to_string(val);
                 } else
                   return std::nullopt;
               }},
              {"-",
               [this, &n]() -> std::optional<std::string> {
                 auto l_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_l));
                 auto r_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_r));
                 if (l_cv && r_cv) {
                   int val = std::stoi(*l_cv) - std::stoi(*r_cv);
                   return "const_" + std::to_string(val);
                 } else
                   return std::nullopt;
               }},
              {"*",
               [this, &n]() -> std::optional<std::string> {
                 auto l_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_l));
                 auto r_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_r));
                 if (l_cv && r_cv) {
                   int val = std::stoi(*l_cv) * std::stoi(*r_cv);
                   return "const_" + std::to_string(val);
                 } else
                   return std::nullopt;
               }},
              {"/",
               [this, &n]() -> std::optional<std::string> {
                 auto l_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_l));
                 auto r_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_r));
                 if (l_cv && r_cv) {
                   int val = std::stoi(*l_cv) / std::stoi(*r_cv);
                   return "const_" + std::to_string(val);
                 } else
                   return std::nullopt;
               }},
              {"%",
               [this, &n]() -> std::optional<std::string> {
                 auto l_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_l));
                 auto r_cv = PrefixedWith(
                     "const_", GetValueNumberStringForExpression(*n->value_r));
                 if (l_cv && r_cv) {
                   int val = std::stoi(*l_cv) % std::stoi(*r_cv);
                   return "const_" + std::to_string(val);
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

  std::string GenerateExpressionSignature(AST::Node& node) {
    auto ss = TrySimplifyExpression(node);
    if (ss) return *ss;

    if (auto* n = dyn_cast<AST::IntLiteral>(&node)) {
      return "const_" + std::to_string(n->value);
    } else if (auto* v = dyn_cast<AST::Identifier>(&node)) {
      int valno = GetValueNumberForVariable(v->name);
      assert((valno != __INVALID_INTVAL__) && "invalid value number.");
      return std::to_string(valno);
    } else if (auto* b = dyn_cast<AST::Expr>(&node)) {
      auto signature = b->op;

      if (b->value_c) {
        int valno = GetValueNumberForExpression(*b->value_c);
        assert((valno != __INVALID_INTVAL__) && "invalid value number.");
        signature += ":#" + std::to_string(valno);
      }
      if (b->value_l) {
        int valno = GetValueNumberForExpression(*b->value_l);
        assert((valno != __INVALID_INTVAL__) && "invalid value number.");
        signature += ":#" + std::to_string(valno);
      }

      assert(b->value_r && "expr is invalid.");
      int valno = GetValueNumberForExpression(*b->value_r);
      assert((valno != __INVALID_INTVAL__) && "invalid value number.");
      return signature + ":#" + std::to_string(valno);
    } else if (auto* b = dyn_cast<AST::MultiNodes>(&node)) {
      std::string signature;
      if (b->values.size() > 0) {
        signature =
            "#" + std::to_string(GetValueNumberForExpression(*b->values[0]));
        for (size_t i = 1; i < b->values.size(); ++i)
          signature +=
              ",#" + std::to_string(GetValueNumberForExpression(*b->values[i]));
      }
      return signature;
    } else if (auto* b = dyn_cast<AST::ParamList>(&node)) {
      std::string signature;
      if (b->values.size() > 0) {
        signature =
            "p:#" + std::to_string(GetValueNumberForExpression(*b->values[0]));
        for (size_t i = 1; i < b->values.size(); ++i)
          signature +=
              ",#" + std::to_string(GetValueNumberForExpression(*b->values[i]));
      }
      return signature;
    }
    return "";
  }

  int GetValueNumberForExpression(AST::Node& expr) {
    std::string signature = GenerateExpressionSignature(expr);
    if (signature == "") return __INVALID_INTVAL__;

    // Check if this expression has been encountered before
    auto it = expressionValueNumbers.find(signature);
    if (it != expressionValueNumbers.end())
      return it->second;  // Return existing value number

    // If not, assign a new value number
    int valueNumber = nextValueNumber++;
    expressionValueNumbers[signature] = valueNumber;
    valueNumberExpressions[valueNumber] = signature;

    if (trace)
      os << ScopeIndent() << "(New) VN: \"" << signature << "\": #"
         << valueNumber << "\n";

    return valueNumber;
  }

  std::string GetExpressionStringForValueNumber(int vn) {
    return valueNumberExpressions.at(vn);
  }

  std::string GetValueNumberStringForExpression(AST::Node& expr) {
    return GetExpressionStringForValueNumber(GetValueNumberForExpression(expr));
  }

  void Print(std::ostream& os) {
    for (auto& item : expressionValueNumbers)
      os << "expr: \"" << item.first << "\", value_no: #" << item.second
         << "\n";
  }

 private:
  std::string ScopeIndent() {
    std::string indent;
    for (size_t i = 0; i < variableScopes.size() - 1; ++i) indent += " ";
    return indent;
  }
};

class ValueNumberingVisitor : public Visitor {
 private:
  ValueNumbering vn;
  std::unordered_map<AST::Node*, int> node_vn;

  int cur_vn = __INVALID_INTVAL__;
  int cur_mdspan_vn = __INVALID_INTVAL__;

 public:
  ValueNumberingVisitor(bool t = false, std::ostream& o = std::cout)
      : vn(t, o) {}

 public:
  void PrintValueNumbers(std::ostream& os) {
    os << "value numbers for choreo code:\n";
    vn.Print(os);
    os << "\n";
  }

 public:
  virtual bool BeforeVisit(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n))
      vn.EnterScope();
    return true;
  }

  virtual bool AfterVisit(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n))
      vn.LeaveScope();
    return true;
  }

 public:
  bool Visit(AST::MultiNodes& n) {
    int valno = vn.GetValueNumberForExpression(n);
    node_vn.emplace(&n, valno);
    cur_vn = valno;
    return true;
  }

  bool Visit(AST::IntLiteral& n) {
    int valno = vn.GetValueNumberForExpression(n);
    node_vn.emplace(&n, valno);
    cur_vn = valno;
    return true;
  }

  bool Visit(AST::SValList&) { return true; };

  bool Visit(AST::Expr& n) {
    int valno = vn.GetValueNumberForExpression(n);
    node_vn.emplace(&n, valno);
    cur_vn = valno;
    return true;
  }

  bool Visit(AST::MultiDimSpans& n) {
    if (n.list)
      cur_mdspan_vn = cur_vn;
    else
      cur_mdspan_vn = __UNKNOWN_INTVAL__;
    return true;
  }

  bool Visit(AST::NamedTypeDecl& n) {
    if (n.init_expr) {
      assert(cur_vn != __INVALID_INTVAL__);
      vn.AssignValueNumberForVariable(n.name_str, cur_vn);
      cur_vn = __INVALID_INTVAL__;
    }
    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) {
    if (n.initializer) {
      assert(cur_vn != __INVALID_INTVAL__);
      vn.AssignValueNumberForVariable(n.name_str, cur_vn);
      cur_vn = __INVALID_INTVAL__;
    }
    return true;
  }

  bool Visit(AST::IntTuple&) { return true; };
  bool Visit(AST::Assignment&) { return true; };
  bool Visit(AST::IntIndex&) { return true; };
  bool Visit(AST::NthBound&) { return true; };
  bool Visit(AST::IntIndexList&) { return true; };
  bool Visit(AST::DataType&) { return true; };  // defer the mdspan evaluation
  bool Visit(AST::Identifier&) { return true; };

  bool Visit(AST::Parameter& n) {
    if (n.type->isSpanned()) {
      assert(isa<AST::MultiDimSpans>(n.type->mdspan_type.get()) &&
             "Invalid mdspan.");
      auto span = cast<AST::MultiDimSpans>(n.type->mdspan_type.get());
      if (span->list) {
        assert((cur_mdspan_vn != __INVALID_INTVAL__) &&
               "unexpected value number for mdspan.");

        // Put alias names of mdspan into the value number table
        vn.AssignValueNumberForVariable(n.sym->name + ".span", cur_mdspan_vn);
        auto vn_str = vn.GetExpressionStringForValueNumber(cur_mdspan_vn);
        ProcessValueNumberString(vn_str, [this, &n](int valno, size_t index) {
          vn.AssignValueNumberForVariable(
              n.sym->name + ".span(" + std::to_string(index) + ")", valno);
        });

        auto vl = GenMDSpanValueFromVNString(vn_str);
        if (span->dim_count != __INVALID_VALUE__ && vl.Dims() != span->dim_count) {
          if (n.sym)
            Error(n.LOC(), "parameter `" + n.sym->name +"''s dimension is inconsistent.");
          else
            Error(n.LOC(), "dimension is inconsistent.");
        }
        n.type->SetType(MakeSpannedType(n.type->base_type, vl));
        cur_mdspan_vn = __INVALID_INTVAL__;
        cur_vn = __INVALID_INTVAL__;
      } else {
        assert((cur_mdspan_vn == __UNKNOWN_INTVAL__) &&
               "unexpected value number for mdspan.");
        cur_vn = __INVALID_INTVAL__;
        if (span->dim_count != __INVALID_VALUE__)
          n.type->SetType(MakeSpannedType(n.type->base_type, span->MakeValueList()));
      }
    }

    return true;
  }

  bool Visit(AST::ParamList&) { return true; };
  bool Visit(AST::ParallelBy&) { return true; };
  bool Visit(AST::RequireBind&) { return true; };
  bool Visit(AST::WithIn&) { return true; };
  bool Visit(AST::WithBlock&) { return true; };
  bool Visit(AST::Memory&) { return true; };
  bool Visit(AST::DMA&) { return true; };
  bool Visit(AST::ChunkAt&) { return true; };
  bool Visit(AST::Wait&) { return true; };
  bool Visit(AST::Call&) { return true; };
  bool Visit(AST::ForeachBlock&) { return true; };
  bool Visit(AST::FunctionDecl&) { return true; };

  bool Visit(AST::ChoreoFunction& n) { return true; }

  bool Visit(AST::CppSourceCode&) { return true; };
  bool Visit(AST::Program&) { return true; };

 private:
  void ProcessValueNumberString(const std::string& input,
                                std::function<void(int, size_t)> lambda) {
    std::regex valuePattern("#(\\d+)");
    auto begin = std::sregex_iterator(input.begin(), input.end(), valuePattern);
    auto end = std::sregex_iterator();

    size_t matchIndex = 0;
    for (auto i = begin; i != end; ++i, ++matchIndex) {
      std::smatch match = *i;
      std::string matchStr =
          match.str(1);  // Capture the number part of the match
      int number = std::stoi(matchStr);

      // Call the passed lambda function with the extracted string and its index
      lambda(number, matchIndex);
    }
  }

  MDSpanValue GenMDSpanValueFromVNString(const std::string& input) {
    ValueList result;
    std::istringstream stream(input);
    std::string component;

    auto handleElement = [&result](const std::string& str) {
      auto digit = PrefixedWith("const_", str);
      if (digit) {
        int val = std::stoi(*digit);
        result.push_back(val);
      } else
        result.push_back(str);
    };

    // assume earlier simplification makes value number expression only 1-level
    // of indirection
    while (std::getline(stream, component, ',')) {
      // Trim whitespace
      component.erase(remove_if(component.begin(), component.end(), isspace),
                      component.end());

      assert(!component.empty() && "unexpected component.");

      if (component[0] == '#') {
        // Look up value number in table and simplify
        int valNo = std::stoi(component.substr(1));
        handleElement(vn.GetExpressionStringForValueNumber(valNo));
      } else {
        // no value numbers
        handleElement(component);
      }
    }

    return {result.size(), result};
  }
};

}  // end namespace Choreo

#endif  // __CHOREO_VALUE_NUMBERING_HPP__
