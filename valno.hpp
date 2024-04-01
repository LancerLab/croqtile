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

inline constexpr size_t InvalidCount() { return __INVALID_VALUE__; }
inline constexpr int InvalidValueNumber() { return __INVALID_INTVAL__; }
inline bool ValidVN(int vn) { return vn != InvalidValueNumber(); }
inline void InvalidateVN(int& vn) { vn = InvalidValueNumber(); }
inline bool UnknownVN(int& vn) { return vn == __UNKNOWN_INTVAL__; }
inline void SetUnknownVN(int& vn) { vn = __UNKNOWN_INTVAL__; }

// Remove the prefix
inline std::string RemovePrefix(const std::string& str,
                                const std::string& prefix) {
  if (str.find(prefix) == 0) return str.substr(prefix.length());
  return str;
}

// retrieve the n-th element from the comma-seperated input string
inline std::optional<std::string> getNthElement(const std::string& input,
                                                int n) {
  std::istringstream iss(input);
  std::string token;
  int currentIndex = 0;

  // Iterate through the tokens in the string
  while (std::getline(iss, token, ',')) {
    if (currentIndex == n) {
      // We've found the token at the specified index
      return token;
    }
    currentIndex++;
  }

  // If the index is out of range, return an empty string
  return std::nullopt;
}

class ValueNumberingVisitor;

class ValueNumbering {
 private:
  ValueNumberingVisitor* visitor;
  std::vector<std::unordered_map<std::string, int>> expressionValueNumbers;
  std::vector<std::unordered_map<int, std::string>> valueNumberExpressions;

  int nextValueNumber = 0;

  bool trace = false;
  std::ostream& os;

  std::optional<std::string> ref = std::nullopt;

 public:
  explicit ValueNumbering(ValueNumberingVisitor* v, bool t, std::ostream& o)
      : visitor(v), trace(t), os(o) {}

  void EnterScope(const std::string &);
  void LeaveScope();

  void SetListReference(const std::string& r) { ref = r; }
  void ResetListReference() { ref.reset(); }

  // It binds a expression sigature with an existing value number.
  void AssociateSignatureWithValueNumber(const std::string& sig, int valno);

  std::optional<std::string> TryToSimplifyNodeSignature(AST::Node& node);

  std::string GenerateNodeSignature(AST::Node& node, bool optimiz = true);

  int GetValueNumberForNode(AST::Node& expr);

  int GetValueNumberFromSignature(const std::string& signature);

  int GetOrInsertValueNumberFromSignature(const std::string& signature);

  std::string GetSignatureFromValueNumber(int vn) {
    return valueNumberExpressions.back().at(vn);
  }

  std::string GetSignatureForNode(AST::Node& expr) {
    return GetSignatureFromValueNumber(GetValueNumberForNode(expr));
  }

  void Print(std::ostream& os) {
    int scope = 0;
    for (auto& stack : expressionValueNumbers) {
      os << scope++ << "\n";
      for (auto& item : stack)
        os << "expr: \"" << item.first << "\", value_no: #" << item.second
           << "\n";
    }
  }

 private:
  std::string ScopeIndent();
};

#define __TRACE_EACH_VISIT__          \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
    n.Print(os);                      \
    os << "\n";                       \
  }

class ValueNumberingVisitor : public Visitor {
 private:
  ValueNumbering vn;
  std::unordered_map<AST::Node*, int> node_vn;  // TODO: caching (note scope)

  int cur_vn = InvalidValueNumber();
  int cur_mdspan_vn = InvalidValueNumber();

 private:
  std::ostream& os;
  // for debugging purpose only
  bool trace_visit = false;

 public:
  ValueNumberingVisitor(bool t = false, std::ostream& o = std::cout)
      : vn(this, t, o), os(o), trace_visit(std::getenv("TRACE_VISIT")) {}

 public:
  void PrintValueNumbers(std::ostream& os) {
    os << "value numbers for choreo code:\n";
    vn.Print(os);
    os << "\n";
  }

 public:
  virtual bool BeforeVisit(AST::Node& n) override {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      vn.EnterScope(f->name);
    } else if (isa<AST::ParallelBy>(&n)) {
      static size_t count = 0;
      vn.EnterScope("paraby_" + std::to_string(count++));
    } else if (isa<AST::WithBlock>(&n)) {
      static size_t count = 0;
      vn.EnterScope("within_" + std::to_string(count++));
    } else if (isa<AST::ForeachBlock>(&n)) {
      static size_t count = 0;
      vn.EnterScope("foreach_" + std::to_string(count++));
    } else if (auto* b = dyn_cast<AST::MultiDimSpans>(&n)) {
      if (b->ref_name != "") vn.SetListReference(ScopedName(b->ref_name));
    }
    return true;
  }

  virtual bool AfterVisit(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
      vn.LeaveScope();
    } else if (isa<AST::MultiDimSpans>(&n)) {
      vn.ResetListReference();
    }
    return true;
  }

 public:
  bool Visit(AST::MultiNodes& n) {
    int valNo = vn.GetValueNumberForNode(n);
    node_vn.emplace(&n, valNo);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::IntLiteral& n) {
    __TRACE_EACH_VISIT__;
    int valNo = vn.GetValueNumberForNode(n);
    node_vn.emplace(&n, valNo);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::Expr& n) {
    __TRACE_EACH_VISIT__;
    int valNo = vn.GetValueNumberForNode(n);
    node_vn.emplace(&n, valNo);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::MultiDimSpans& n) {
    __TRACE_EACH_VISIT__;

    if (n.list) {
      // The MDSpanValue now can be deduced from the value number.
      // Update the type detail acoordingly.
      auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

      // set alias expressions with proper value numbers
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig](int valno, size_t index) {
            vn.AssociateSignatureWithValueNumber(
                vn_sig + "(" + std::to_string(index) + ")", valno);
          });

      auto vl = GenMDSpanValueFromVNString(vn_sig);
      n.SetTypeDetail(vl);

      if (n.Dims() != InvalidCount()) {
        if (vl.Dims() != n.Dims())
          Error(n.LOC(),
                "mdspan's dimension is inconsistent with its initialization "
                "expression.");
      } else
        n.SetDims(vl.Dims());

      // pass the value number over
      cur_mdspan_vn = cur_vn;
    } else
      SetUnknownVN(cur_mdspan_vn);  // failed to deduce the type detail

    InvalidateVN(cur_vn);
    return true;
  }

  bool Visit(AST::NamedTypeDecl& n) {
    __TRACE_EACH_VISIT__;

    if (n.init_expr) {
      assert(ValidVN(cur_mdspan_vn) &&
             "invalid value number for the named type.");
      vn.AssociateSignatureWithValueNumber(ScopedName(n.name_str), cur_mdspan_vn);

      InvalidateVN(cur_mdspan_vn);  // comsumes the mdspan
    }
    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) {
    __TRACE_EACH_VISIT__;
    if (n.initializer && ValidVN(cur_vn)) {
      vn.AssociateSignatureWithValueNumber(ScopedName(n.name_str), cur_vn);
      InvalidateVN(cur_vn);
    }
    return true;
  }

  bool Visit(AST::IntTuple& n) {
    __TRACE_EACH_VISIT__;
    if (auto i = dyn_cast<AST::MultiNodes>(n.list.get()))
      n.SetType(MakeITupleType(i->Count()));
    else
      n.SetType(MakeUninitITupleType());
    InvalidateVN(cur_vn);  // Currently cut off value numbering
    return true;
  }

  bool Visit(AST::Assignment& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };

  bool Visit(AST::IntIndex& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };

  bool Visit(AST::DataType& n) {
    __TRACE_EACH_VISIT__;
    return true;
  }

  bool Visit(AST::Identifier& n) {
    __TRACE_EACH_VISIT__;
    int valNo = vn.GetValueNumberForNode(n);
    node_vn.emplace(&n, valNo);
    cur_vn = valNo;
    return true;
  };

  bool Visit(AST::Parameter& n) {
    __TRACE_EACH_VISIT__;
    if (n.type->isSpanned()) {
      assert(isa<AST::MultiDimSpans>(n.type->mdspan_type.get()) &&
             "Invalid mdspan.");
      auto span = cast<AST::MultiDimSpans>(n.type->mdspan_type.get());
      if (span->list) {
        assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");

        // Put alias names of mdspan into the value number table
        vn.AssociateSignatureWithValueNumber(ScopedName(n.sym->name + ".span"),
                                             cur_mdspan_vn);

        InvalidateVN(cur_mdspan_vn);
      } else {
        assert(UnknownVN(cur_mdspan_vn) &&
               "unexpected value number for mdspan.");
        InvalidateVN(cur_vn);
      }
      n.type->SetType(
          MakeSpannedType(n.type->base_type, span->GetTypeDetail()));
      return true;
    }

    return true;
  }

  bool Visit(AST::ParamList& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::ParallelBy& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::RequireBind& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::WithIn& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::WithBlock& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::Memory& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::DMA& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::ChunkAt& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::Wait& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::Call& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::ForeachBlock& n) {
    if (trace_visit) os << n.NodeTypeString() << "\n";
    return true;
  };
  bool Visit(AST::FunctionDecl& n) {
    if (trace_visit) os << n.NodeTypeString() << "\n";
    return true;
  };

  bool Visit(AST::ChoreoFunction& n) {
    if (trace_visit) os << n.NodeTypeString() << "\n";
    return true;
  }

  bool Visit(AST::CppSourceCode& n) {
    if (trace_visit) os << n.NodeTypeString() << "\n";
    return true;
  };
  bool Visit(AST::Program& n) {
    if (trace_visit) os << n.NodeTypeString() << "\n";
    return true;
  };

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

  // TODO: should be recursive
  MDSpanValue GenMDSpanValueFromVNString(const std::string& input) {
    ValueList result;
    std::istringstream stream(input);
    std::string component;

    auto handleElement = [&result, this](const std::string& str) {
      auto digit = PrefixedWith("const_", str);
      if (digit) {
        int val = std::stoi(*digit);
        result.push_back(val);
        return;
      }

      if (str[1] == ':' &&
          ((str[0] == '+') || (str[0] == '-') || (str[0] == '*') ||
           (str[0] == '/') || (str[0] == '%'))) {
        std::vector<std::string> parts;
        std::string part;

        // Extract each part separated by ':'
        std::istringstream stream(str);
        while (std::getline(stream, part, ':')) {
          if (part[0] == '#') {
            auto sig =
                vn.GetSignatureFromValueNumber(std::stoi(part.substr(1)));
            parts.push_back(RemovePrefix(sig, "const_"));
          } else
            parts.push_back(part);
        }
        assert(parts.size() == 3);
        result.push_back(parts[1] + parts[0] + parts[2]);
        return;
      }
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
        handleElement(vn.GetSignatureFromValueNumber(valNo));
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
