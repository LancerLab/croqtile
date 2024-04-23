#ifndef __CHOREO_VALUE_NUMBERING_HPP__
#define __CHOREO_VALUE_NUMBERING_HPP__

#include <charconv>
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
inline constexpr int UnknownValue() { return -1; }
inline bool ValidVN(int vn) { return vn != InvalidValueNumber(); }
inline void InvalidateVN(int& vn) { vn = InvalidValueNumber(); }
inline bool UnknownVN(int vn) { return vn == UnknownValue(); }
inline void SetUnknownVN(int& vn) { vn = UnknownValue(); }

// Remove the prefix
inline std::string RemovePrefix(const std::string& str,
                                const std::string& prefix) {
  if (str.find(prefix) == 0) return str.substr(prefix.length());
  return str;
}

// retrieve the n-th element from the comma-seperated input string
inline std::optional<std::string> GetNthElement(const std::string& input,
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

inline int CountElementsInSignature(const std::string& input) {
  if (input.empty()) return 0;  // Return 0 if the string is empty
  int count = 1;
  for (char c : input) {
    if (c == ',') ++count;  // Increment for each comma found
  }
  return count;
}

class ShapeInference;
class ValueNumbering {
 private:
  ShapeInference* visitor;
  std::vector<std::unordered_map<std::string, int>> expressionValueNumbers;
  std::vector<std::unordered_map<int, std::string>> valueNumberExpressions;

  std::vector<std::unordered_map<AST::Node*, int>>
      nodeValueNumbers;  // cache to direct map node to value number

  int nextValueNumber = 0;

  bool trace = false;
  std::ostream& os;

  std::optional<std::string> ref = std::nullopt;

 public:
  explicit ValueNumbering(ShapeInference* v, bool t, std::ostream& o)
      : visitor(v), trace(t), os(o) {}

  void EnterScope(const std::string&);
  void LeaveScope();

  void SetListReference(const std::string& r) { ref = r; }
  void ResetListReference() { ref.reset(); }

  // It binds a expression sigature with an existing value number.
  void AssociateSignatureWithValueNumber(const std::string& sig, int valno);

  std::optional<std::string> TryToSimplifyNodeSignature(AST::Node& node);

  // Generate the signature for a node, simplify the signature when optimiz flag
  // is set.
  std::string GenerateNodeSignature(AST::Node& node, bool optimiz = true);

  // Directly get the value number. Abort when it fails.
  int GetValueNumberForNode(AST::Node&);

  // Generate the new value number. Abort when the value number exists.
  int GenerateValueNumberForNode(AST::Node&);

  // Check if the value number exists for the node
  bool HasValueNumberForNode(AST::Node&);

  // Directly get the value number from a signature. Abort when it fails.
  int GetValueNumberOfSignature(const std::string&);

  // Generate the new value number from a signature. Abort when the value number
  // exists.
  int GenerateValueNumberFromSignature(const std::string& signature);

  // Check if the value number exists for the signature
  bool HasValueNumberOfSignature(const std::string&);

  int GetOrInsertValueNumberFromSignature(const std::string& signature);

  // Retrieve the signature from a value number. About when fails.
  std::string GetSignatureFromValueNumber(int vn) {
    if (vn == UnknownValue()) return "?";

    if (valueNumberExpressions.back().count(vn) == 0)
      choreo_unreachable("value number " + std::to_string(vn) +
                         " does not exists in the value number table.");
    return (valueNumberExpressions.back())[vn];
  }

  std::string SignatureOfSymbol(std::string sym) {
    return GetSignatureFromValueNumber(GetValueNumberOfSignature(sym));
  }

  std::string GetSignatureForNode(AST::Node& n) {
    return GetSignatureFromValueNumber(GetValueNumberForNode(n));
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

  std::optional<std::string> TryToSimplifyTernary(const location&,
                                                  const std::string&,
                                                  const std::string&,
                                                  const std::string&,
                                                  bool = false);

 private:
  std::string ScopeIndent();

  void Error(const location& loc, const std::string& message);
  void Warning(const location& loc, const std::string& message);
};

#define __TRACE_EACH_VISIT__          \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
    n.Print(os);                      \
    os << "\n";                       \
  }

class ShapeInference : public Visitor {
 private:
  ValueNumbering vn;

  int cur_vn = InvalidValueNumber();
  int cur_ituple_vn = InvalidValueNumber();
  int cur_mdspan_vn = InvalidValueNumber();

 private:
  std::ostream& os;
  // for debugging purpose only
  bool trace_visit = false;

 public:
  ShapeInference(bool t = false, std::ostream& o = std::cout)
      : vn(this, t, o), os(o), trace_visit(std::getenv("TRACE_VALNO")) {}

 public:
  void PrintValueNumbers(std::ostream& os) {
    os << "value numbers for choreo code:\n";
    vn.Print(os);
    os << "\n";
  }

 public:
  virtual bool BeforeVisit(AST::Node& n) override {
    if (isa<AST::Program>(&n)) {
      vn.EnterScope("");  // global scope
    } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
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
      if (b->ref_name != "") {
        auto n = SSTab().NameInScope(b->ref_name);
        if (!n)
          choreo_unreachable(
              ("variable `" + b->ref_name + "' is not found in scopes.")
                  .c_str());
        vn.SetListReference(n.value());
      }
    } else if (auto* b = dyn_cast<AST::IntTuple>(&n)) {
      if (b->ref_name != "") {
        auto n = SSTab().NameInScope(b->ref_name);
        if (!n)
          choreo_unreachable(
              ("variable `" + b->ref_name + "' is not found in scopes.")
                  .c_str());
        vn.SetListReference(n.value());
      }
    }
    return true;
  }

  virtual bool AfterVisit(AST::Node& n) override {
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
        isa<AST::ForeachBlock>(&n)) {
      vn.LeaveScope();
    } else if (isa<AST::MultiDimSpans>(&n) || isa<AST::IntTuple>(&n)) {
      vn.ResetListReference();
    }
    return true;
  }

 public:
  bool Visit(AST::MultiNodes&) { return true; }

  bool Visit(AST::MultiValues& n) {
    int valNo = vn.GenerateValueNumberForNode(n);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::IntLiteral& n) {
    __TRACE_EACH_VISIT__;
    int valNo = vn.GenerateValueNumberForNode(n);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::Expr& n) {
    __TRACE_EACH_VISIT__;
    if (auto ref = n.GetReference()) {
      if (auto id = dyn_cast<AST::Identifier>(ref.get())) {
        if (SSTab().IsDeclared(id->name)) {
          cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(id->name));
          return true;
        }
      }
    }

    cur_vn = vn.GenerateValueNumberForNode(n);
    return true;
  }

  bool Visit(AST::MultiDimSpans& n) {
    __TRACE_EACH_VISIT__;

    if (n.list) {
      // The Shape now can be deduced from the value number.
      // Update the type detail acoordingly.
      auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

      // set alias expressions with proper value numbers
      if (CountElementsInSignature(vn_sig) > 1) {
        ProcessValueNumberString(
            vn_sig, [this, &vn_sig](int valno, size_t index) {
              if (UnknownVN(valno))
                return;  // do not associate it with vn of "?"
              vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                     std::to_string(index));
              auto elem_sig = vn_sig + "(" + std::to_string(index) + ")";
              if (!vn.HasValueNumberOfSignature(elem_sig))
                vn.AssociateSignatureWithValueNumber(
                    vn_sig + "(" + std::to_string(index) + ")", valno);
            });
      }

      auto vl = GenShapeFromSignature(vn_sig);
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
    } else if (n.dim_count > 0) {
      std::string unknown_spans = "#" + std::to_string(UnknownValue());
      for (size_t i = 1; i < n.dim_count; ++i)
        unknown_spans = unknown_spans + ",#" + std::to_string(UnknownValue());
      cur_mdspan_vn = vn.GetOrInsertValueNumberFromSignature(unknown_spans);
      n.SetTypeDetail(GenShapeFromSignature(unknown_spans));
    } else {
      SetUnknownVN(cur_mdspan_vn);  // failed to deduce the type detail
    }

    InvalidateVN(cur_vn);
    return true;
  }

  bool Visit(AST::NamedTypeDecl& n) {
    __TRACE_EACH_VISIT__;

    if (n.init_expr) {
      assert(ValidVN(cur_mdspan_vn) &&
             "invalid value number for the named type.");
      SSTab().DefineSymbol(n.name_str, n.GetType());

      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(n.name_str),
                                           cur_mdspan_vn);

      InvalidateVN(cur_mdspan_vn);  // comsumes the mdspan
    }
    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) {
    __TRACE_EACH_VISIT__;

    if (SSTab().DeclaredInScope(n.name_str)) {
      Error(n.LOC(), "ODR violation: symbol `" + n.name_str +
                         "' has been declared already.");
      return false;
    }

    Storage s = Storage::NONE;
    if (n.mem) s = n.mem->st;

    if (n.initializer) {
      if (ValidVN(cur_ituple_vn)) {
        // assert(!ValidVN(cur_vn) && "expected current value number.");
        // assert(!ValidVN(cur_mdspan_vn) && "expected current mdspan value
        // number.");
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(n.name_str),
                                             cur_ituple_vn);
        InvalidateVN(cur_ituple_vn);
        SSTab().DefineSymbol(n.name_str, MakeUninitITupleType());

      } else if (ValidVN(cur_vn)) {
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(n.name_str),
                                             cur_vn);
        InvalidateVN(cur_vn);
        SSTab().DefineSymbol(n.name_str, n.GetType());
      }
    } else {
      assert(n.type && "missed type annotation.");
      if (ValidVN(cur_mdspan_vn)) {
        vn.AssociateSignatureWithValueNumber(
            SSTab().ScopedName(n.name_str + ".span"), cur_mdspan_vn);
        auto mds_value = GenShapeFromSignature(
            vn.GetSignatureFromValueNumber(cur_mdspan_vn));
        SSTab().DefineSymbol(n.name_str,
                             MakeSpannedType(n.type->base_type, mds_value, s));
        SSTab().DefineSymbol(n.name_str + ".span", MakeMDSpanType(mds_value));
      } else if (ValidVN(cur_vn)) {
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(n.name_str),
                                             cur_vn);
        SSTab().DefineSymbol(n.name_str, n.GetType());
      } else
        choreo_unreachable();
    }

    return true;
  }

  bool Visit(AST::IntTuple& n) {
    __TRACE_EACH_VISIT__;
    auto i = n.list.get();
    cur_ituple_vn = cur_vn;
    n.SetType(MakeITupleType(i->Count()));

    auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

    if (CountElementsInSignature(vn_sig) > 1) {
      // set alias expressions with proper value numbers
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig](int valno, size_t index) {
            if (UnknownVN(valno)) return;  // do not associate it with vn of "?"
            vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                   std::to_string(index));
            vn.AssociateSignatureWithValueNumber(
                vn_sig + "(" + std::to_string(index) + ")", valno);
          });
    }
    InvalidateVN(cur_vn);  // Currently cut off value numbering
    return true;
  }

  bool Visit(AST::Assignment& n) {
    __TRACE_EACH_VISIT__;

    if (SSTab().IsDeclared(n.name)) {
      return true;
    }

    // this is the un-type-annotated declaration
    SSTab().DefineSymbol(n.name, n.value->GetType());

    if (ValidVN(cur_ituple_vn)) {
      assert(!ValidVN(cur_vn) && "expected current value number.");
      assert(!ValidVN(cur_mdspan_vn) &&
             "expected current mdspan value number.");
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(n.name),
                                           cur_ituple_vn);
      InvalidateVN(cur_ituple_vn);
    }

    return true;
  };

  bool Visit(AST::IntIndex& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };

  bool Visit(AST::DataType& n) {
    __TRACE_EACH_VISIT__;
    if (ValidVN(cur_mdspan_vn)) {
      cur_vn = cur_mdspan_vn;
    }

    return true;
  }

  bool Visit(AST::Identifier& n) {
    __TRACE_EACH_VISIT__;
    if (SSTab().IsDeclared(n.name)) {
      // it is a reference
      auto name = n.name;
      auto pty = SSTab().LookupSymbol(name);
      if (isa<SpannedType>(pty.get()) || isa<FutureType>(pty.get())) {
        name += ".span";
        assert(SSTab().IsDeclared(name) && "span symbol is not declared.");
      }
      assert(vn.HasValueNumberOfSignature(SSTab().InScopeName(name)) &&
             "value number has not been generated.");
      cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(name));
    } else {
      if (vn.HasValueNumberForNode(n)) {
        Error(n.LOC(), "value number has been generated for `" + n.name + "'.");
        return false;
      }
      cur_vn = vn.GenerateValueNumberForNode(n);
    }

    return true;
  }

  bool Visit(AST::Parameter& n) {
    __TRACE_EACH_VISIT__;

    if (n.type->isSpanned()) {
      assert(isa<AST::MultiDimSpans>(n.type->mdspan_type.get()) &&
             "Invalid mdspan.");
      auto span = cast<AST::MultiDimSpans>(n.type->mdspan_type.get());
      if (span->list) {
        assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");

        // Put alias names of mdspan into the value number table
        vn.AssociateSignatureWithValueNumber(
            SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
        n.type->SetType(
            MakeSpannedType(n.type->base_type, span->GetTypeDetail()));

      } else if (span->dim_count != __INVALID_VALUE__) {
        assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");
        // Put alias names of mdspan into the value number table
        vn.AssociateSignatureWithValueNumber(
            SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
        n.type->SetType(
            MakeSpannedType(n.type->base_type, span->GetTypeDetail()));
      } else {
        // the value number is unknown at compile time
        Error(n.LOC(), "The type can not be inference at compile time.");
        return false;
      }

      InvalidateVN(cur_vn);

      if (n.sym) {
        SSTab().DefineSymbol(
            n.sym->name + ".span",
            cast<SpannedType>(n.type->GetType())->GetMDSpanType());
        SSTab().DefineSymbol(n.sym->name, n.type->GetType());
      }

      return true;
    }

    if (n.sym && n.type->isScalar()) {
      // get the value number and make it defined
      vn.GetValueNumberOfSignature(SSTab().ScopedName(n.sym->name));
      if (n.sym) SSTab().DefineSymbol(n.sym->name, n.GetType());
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
    std::string bound = "const_" + std::to_string(n.bound);
    int valno = vn.GetOrInsertValueNumberFromSignature(bound);
    std::string iv_name =
        SSTab().ScopedName("@" + n.biv);  // upper-bound of bounded variable
    vn.AssociateSignatureWithValueNumber(iv_name, valno);
    Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(valno));
    n.SetType(MakeBoundedITupleType(s, "pv"));
    SSTab().DefineSymbol("@" + n.biv, MakeMDSpanType(s));
    return true;
  };

  bool Visit(AST::RequireBind& n) {
    __TRACE_EACH_VISIT__;
    return true;
  }

  bool Visit(AST::WithIn& n) {
    __TRACE_EACH_VISIT__;
    if (auto mds = dyn_cast<AST::MultiDimSpans>(n.in.get())) {
      assert(ValidVN(cur_mdspan_vn) &&
             "no valid value number generated for the mdspan.");
      if (n.with_matchers)
        if (n.with_matchers->Count() != mds->Dims()) {
          Error(n.LOC(), "inconsistent with-in values and bounds.");
          return false;
        }

      auto vn_sig = vn.GetSignatureFromValueNumber(cur_mdspan_vn);
      bool gen_alias = (CountElementsInSignature(vn_sig) > 1);
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig, &n, gen_alias](int valno, size_t index) {
            if (UnknownVN(valno)) return;  // do not associate it with vn of "?"
            if (n.with && gen_alias) {
              std::string name = SSTab().ScopedName(n.with->name) + "(" +
                                 std::to_string(index) + ")";
              vn.AssociateSignatureWithValueNumber(name, valno);
            }

            if (n.with_matchers) {
              auto sym =
                  cast<AST::Identifier>((n.with_matchers->values[index]).get());
              std::string name = SSTab().ScopedName("@" + sym->name);
              if (gen_alias) vn.AssociateSignatureWithValueNumber(name, valno);
              Shape s =
                  GenShapeFromSignature(vn.GetSignatureFromValueNumber(valno));
              sym->SetType(MakeBoundedITupleType(s));
              SSTab().DefineSymbol("@" + sym->name, MakeMDSpanType(s));
            }
          });

      if (n.with) {
        vn.AssociateSignatureWithValueNumber(
            SSTab().ScopedName("@" + n.with->name), cur_mdspan_vn);
        Shape s = GenShapeFromSignature(vn_sig);
        n.with->SetType(MakeBoundedITupleType(s));
        SSTab().DefineSymbol("@" + n.with->name, MakeMDSpanType(s));
      }
      InvalidateVN(cur_mdspan_vn);
    } else {
      // TODO
      choreo_unreachable("span expression is required to be supported.");
    }
    return true;
  }

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

    std::string f_span = n.future + ".span";
    assert(ValidVN(cur_vn) &&
           "unexpected current value number for future.span inference.");

    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(f_span), cur_vn);
    auto s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
    n.SetType(MakeFutureType(s));
    SSTab().DefineSymbol(n.future, n.GetType());
    SSTab().DefineSymbol(f_span,
                         MakeMDSpanType(s));  // this is implicit symbol

    return true;
  };

  bool Visit(AST::ChunkAt& n) {
    __TRACE_EACH_VISIT__;

    int ca_valno = InvalidValueNumber();

    auto pty = SSTab().LookupSymbol(n.data->name);
    assert(isa<SpannedType>(pty) && "unexpected data type.");

    if (!n.positions) {
      ca_valno = vn.GetValueNumberOfSignature(
          SSTab().InScopeName(n.data->name + ".span"));
      // it is just a symbol reference
    } else {
      std::string data_sig =
          vn.SignatureOfSymbol(SSTab().InScopeName(n.data->name + ".span"));
      int dim_count = CountElementsInSignature(data_sig);
      int dim_index = 0;

      std::string fs_signature;  // signature of the future.span
      auto AppendSignature = [this, &fs_signature, &n, &dim_index, dim_count](
                                 int dividend_vn, int divisor_vn) {
        // the signature without optimiz
        std::string res_sig = "/:#" + std::to_string(dividend_vn) + ":#" +
                              std::to_string(divisor_vn);

        if (auto quotient = vn.TryToSimplifyTernary(
                n.LOC(), "/", vn.GetSignatureFromValueNumber(dividend_vn),
                vn.GetSignatureFromValueNumber(divisor_vn), true))
          res_sig = quotient.value();

        // now generate the value number from the signature
        int res_valno = vn.GetOrInsertValueNumberFromSignature(res_sig);

        // and append the value number as
        if (!fs_signature.empty()) fs_signature += ",";
        fs_signature += "#" + std::to_string(res_valno);
      };

      for (auto pos : n.positions->values) {
        auto biv = cast<AST::Identifier>(pos.get());
        auto bound_name = SSTab().InScopeName("@" + biv->name);
        int bound_vn = vn.GetValueNumberOfSignature(bound_name);
        std::string bound_sn = vn.GetSignatureFromValueNumber(bound_vn);
        auto dim_ith = GetNthElement(data_sig, dim_index);
        if (!dim_ith) {
          Error(n.LOC(), "internal error: value number is not obtained.");
          return false;
        }
        assert(dim_ith.value()[0] == '#');
        int dim_valno = std::stoi(dim_ith.value().substr(1));

        if (CountElementsInSignature(bound_sn) <= 1) {
          // this is a simple bound
          AppendSignature(dim_valno, bound_vn);
        } else {
          // multiple bounds
          ProcessValueNumberString(
              bound_sn,
              [this, &dim_valno, &AppendSignature](int valno, size_t) {
                AppendSignature(dim_valno, valno);
              });
        }

        if (++dim_index > dim_count) {
          Error(n.LOC(), "dimensions inconsistence is found between `" +
                             n.data->name + "' and chunkat expression.");
          return false;
        }
      }
      ca_valno = vn.GetOrInsertValueNumberFromSignature(fs_signature);
    }

    // set the chunkat's type
    n.SetType(MakeSpannedType(
        cast<SpannedType>(pty)->f_type,
        GenShapeFromSignature(vn.GetSignatureFromValueNumber(ca_valno)),
        cast<SpannedType>(pty)->GetStorage()));

    int fs_valno = ca_valno;
    cur_vn = fs_valno;

    return true;
  }

  bool Visit(AST::Wait& n) {
    __TRACE_EACH_VISIT__;
    return true;
  }
  bool Visit(AST::Call& n) {
    __TRACE_EACH_VISIT__;
    return true;
  };
  bool Visit(AST::Return& n) {
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
    std::regex valuePattern("#(-?\\d+)");
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
  Shape GenShapeFromSignature(const std::string& input) {
    ValueList result;

    std::istringstream stream(input);
    std::string component;
    while (std::getline(stream, component, ',')) {
      // Trim whitespace
      component.erase(remove_if(component.begin(), component.end(), isspace),
                      component.end());

      assert(!component.empty() && "unexpected component.");

      auto expr = GenerateExpression(component);
      int int_val;
      auto [ptr, ec] =
          std::from_chars(expr.data(), expr.data() + expr.size(), int_val);
      if (ec == std::errc() && ptr == expr.data() + expr.size()) {
        result.emplace_back(int_val);
      } else
        result.emplace_back(expr);
    }

    return {result.size(), result};
  }

  std::string GenerateExpression(const std::string& sig) {
    if (auto digit = PrefixedWith("const_", sig)) return *digit;

    // a value number reference
    if (auto digit = PrefixedWith("#", sig))
      return GenerateExpression(
          vn.GetSignatureFromValueNumber(std::stoi(*digit)));

    // binary expressions
    if (sig[1] == ':' &&
        ((sig[0] == '+') || (sig[0] == '-') || (sig[0] == '*') ||
         (sig[0] == '/') || (sig[0] == '%'))) {
      std::istringstream stream(sig);
      std::vector<std::string> parts;
      std::string part;

      while (std::getline(stream, part, ':')) parts.push_back(part);
      assert(parts.size() == 3);
      return GenerateExpression(parts[1]) + parts[0] +
             GenerateExpression(parts[2]);
    }

    // this is a symbol
    return sig;
  }
};

}  // end namespace Choreo

#endif  // __CHOREO_VALUE_NUMBERING_HPP__
