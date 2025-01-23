#ifndef __CHOREO_VALUE_NUMBERING_HPP__
#define __CHOREO_VALUE_NUMBERING_HPP__

#include <charconv>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "ast.hpp"
#include "typeresolve.hpp"
#include "types.hpp"
#include "valbind.hpp"
#include "visitor.hpp"

namespace Choreo {

inline constexpr int UnknownValue() { return -1; }
inline bool ValidVN(int vn) { return IsValidValueNumber(vn); }
inline void InvalidateVN(int& vn) { vn = GetInvalidValueNumber(); }
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
  if (input.empty()) return 0; // Return 0 if the string is empty
  int count = 1;
  for (char c : input) {
    if (c == ',') ++count; // Increment for each comma found
  }
  return count;
}

class ShapeInference;
class ValueNumbering {
private:
  ShapeInference* visitor;
  std::vector<std::unordered_map<std::string, int>> expressionValueNumbers;
  std::vector<std::unordered_map<int, std::string>> valueNumberExpressions;
  ValBind::BindInfo<int> bind_info;

  std::vector<std::unordered_map<const AST::Node*, int>>
      nodeValueNumbers; // cache to direct map node to value number

  bool InternalHasExprValNo(const std::string& expr) {
    for (auto expr_valno = expressionValueNumbers.rbegin();
         expr_valno != expressionValueNumbers.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      return true;
    }
    return false;
  }

  int InternalGetExprValNo(const std::string& expr) {
    for (auto expr_valno = expressionValueNumbers.rbegin();
         expr_valno != expressionValueNumbers.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      return (*expr_valno)[expr];
    }
    choreo_unreachable("can not find valno of expression : " + expr + ".");
  }

  void InternalUpdateExprValNo(const std::string& expr, int val_no) {
    for (auto expr_valno = expressionValueNumbers.rbegin();
         expr_valno != expressionValueNumbers.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      (*expr_valno)[expr] = val_no;
      return;
    }
    assert(!expressionValueNumbers.empty() &&
           "empty expression value number map.");
    expressionValueNumbers.back()[expr] = val_no;
  }

  bool InternalHasValNoExpr(int vn) {
    for (auto valno_expr = valueNumberExpressions.rbegin();
         valno_expr != valueNumberExpressions.rend(); valno_expr++) {
      if (!valno_expr->count(vn)) continue;
      return true;
    }
    return false;
  }

  const std::string& InternalGetValNoExpr(int vn) {
    for (auto valno_expr = valueNumberExpressions.rbegin();
         valno_expr != valueNumberExpressions.rend(); valno_expr++) {
      if (!valno_expr->count(vn)) continue;
      return (*valno_expr)[vn];
    }
    choreo_unreachable(
        "can not find expression of valno: " + std::to_string(vn) + ".");
  }

  void InternalUpdateValNoExpr(int vn, const std::string& expr) {
    for (auto valno_expr = valueNumberExpressions.rbegin();
         valno_expr != valueNumberExpressions.rend(); valno_expr++) {
      if (!valno_expr->count(vn)) continue;
      (*valno_expr)[vn] = expr;
    }
    assert(!valueNumberExpressions.empty() &&
           "empty value number expression map.");
    valueNumberExpressions.back()[vn] = expr;
  }

  bool InternalHasNodeValNo(const AST::Node* node) {
    for (auto node_valno = nodeValueNumbers.rbegin();
         node_valno != nodeValueNumbers.rend(); node_valno++) {
      if (!node_valno->count(node)) continue;
      return true;
    }
    return false;
  }

  int InternalGetNodeValNo(const AST::Node* node) {
    for (auto node_valno = nodeValueNumbers.rbegin();
         node_valno != nodeValueNumbers.rend(); node_valno++) {
      if (!node_valno->count(node)) continue;
      return (*node_valno)[node];
    }
    choreo_unreachable("can not find valno of node: " + PSTR(node) + ".");
  }

  void InternalUpdateNodeValNo(const AST::Node* node, int vn) {
    for (auto node_valno = nodeValueNumbers.rbegin();
         node_valno != nodeValueNumbers.rend(); node_valno++) {
      if (!node_valno->count(node)) continue;
      (*node_valno)[node] = vn;
    }
    assert(!nodeValueNumbers.empty() && "empty node value number map.");
    nodeValueNumbers.back()[node] = vn;
  }

  int nextValueNumber = 0;

  bool trace = false;

  std::optional<std::string> ref = std::nullopt;

public:
  explicit ValueNumbering(ShapeInference* v, bool t) : visitor(v), trace(t) {}

  void EnterScope();
  void LeaveScope();

  void SetListReference(const std::string& r) { ref = r; }
  void ResetListReference() { ref.reset(); }

  // It binds a expression sigature with an existing value number.
  void AssociateSignatureWithValueNumber(const std::string& sig, int valno);
  void AssociateSignatureWithInvalidValueNumber(const std::string& sig);
  // rebind/modify the value number.
  // Caution: only used for scenario where the value number has not been
  // determined yet.
  void RebindSignatureWithValueNumber(const std::string& sig, int valno);

  std::optional<std::string> TryToSimplifyNodeSignature(const AST::Node& node);

  // Generate the signature for a node, simplify the signature when optimiz flag
  // is set.
  std::string GenerateNodeSignature(const AST::Node& node, bool optimiz = true);

  // special for bounded variables
  std::optional<std::string> GenerateSpecialNodeSignature(const AST::Node&);

  // Directly get the value number. Abort when it fails.
  int GetValueNumberForNode(const AST::Node&);

  // Generate the new value number. Abort when the value number exists.
  int GenerateValueNumberForNode(const AST::Node&);

  // Check if the value number exists for the node
  bool HasValueNumberForNode(const AST::Node&);

  // Directly get the value number from a signature. Abort when it fails.
  int GetValueNumberOfSignature(const std::string&);

  // Bind two value numbers
  void BindValueNumbers(int, int);

  // Bind two value numbers
  const ValBind::Binds<int>::Set& GetBindSet(int vn) {
    auto& ret = bind_info.GetSet(vn);
    return ret;
  }

  void AddBind(int vn0, int vn1) { bind_info.AddBind(vn0, vn1); }

  // Generate the new value number from a signature. Abort when the value number
  // exists.
  int GenerateValueNumberFromSignature(const std::string& signature);

  // Check if the value number exists for the signature
  bool HasValueNumberOfSignature(const std::string&);

  // Check if the value number exists and is valid for the signature
  bool HasValidValueNumberOfSignature(const std::string&);

  int GetOrInsertValueNumberFromSignature(const std::string& signature);

  // Symbol names related to the value numbering
  const std::string VNSymbolName(const AST::Identifier&) const;

  // Retrieve the signature from a value number. About when fails.
  std::string GetSignatureFromValueNumber(int vn) {
    if (vn == UnknownValue()) return "?";

    if (!InternalHasValNoExpr(vn))
      choreo_unreachable("value number " + std::to_string(vn) +
                         " does not exists in the value number table.");
    return InternalGetValNoExpr(vn);
  }

  std::string SignatureOfSymbol(std::string sym) {
    return GetSignatureFromValueNumber(GetValueNumberOfSignature(sym));
  }

  std::string GetSignatureForNode(const AST::Node& n) {
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

  std::optional<std::string>
  SignBoundedOperation(const location&, const std::string&, const AST::Node&,
                       const AST::Node&, bool verbose);

  std::optional<std::string>
  TryToSimplifyBinary(const location&, const std::string&, const std::string&,
                      const std::string&, bool = false);

  std::string SignBinaryCompositeValues(const location&, const std::string&,
                                        const std::string&, const std::string&,
                                        bool = false);

private:
  std::string ScopeIndent();

  void Error(const location& loc, const std::string& message);
  void Warning(const location& loc, const std::string& message);
};

class ShapeInference : public VisitorWithScope {
private:
  ValueNumbering vn;

  // valno rendered from current ast node
  int cur_vn = GetInvalidValueNumber();

  // implicit valno of spanned-type with ".span" annotation
  int cur_mdspan_vn = GetInvalidValueNumber();

  std::string cur_fn;
  // when values are consumed instead of generated
  bool gen_values = true;

  bool allow_named_dim = false; // named dimension (mdspan param only)

  TypeConstraints type_equals{this};

  OptimizedValues& SymVal(const std::string sym) {
    return FCtx(fname).GetSymbolValues(sym);
  }

private:
  // for debugging purpose only
  bool cannot_proceed = false;

  void TraceEachVisit(AST::Node& n, bool detail = false,
                      const std::string& m = "") const {
    if (!trace_visit) return;
    if (detail)
      dbgs() << m << STR(n) << "\n";
    else
      dbgs() << m << n.TypeNameString() << "\n";
  }

public:
  ShapeInference(bool t = false) : VisitorWithScope("valno"), vn(this, t) {
    type_equals.SetDebug(debug_visit);
  }

public:
  void PrintValueNumbers(std::ostream& os) {
    os << "value numbers for choreo code:\n";
    vn.Print(os);
    os << "\n";
  }

  bool HasError() override {
    if (error_count)
      dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return error_count != 0;
  }

public:
  virtual bool BeforeVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, false, "before ");

    if (isa<AST::Program>(&n)) {
      vn.EnterScope(); // global scope
    } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      vn.EnterScope();
      cur_fn = f->name;
      cannot_proceed = false; // recover state when starting a new function
      int valno = vn.GetOrInsertValueNumberFromSignature("const_1");
      vn.AssociateSignatureWithValueNumber(InScopeName("@__choreo_no_tiling__"),
                                           valno);
    } else if (isa<AST::ParallelBy>(&n)) {
      vn.EnterScope();
    } else if (isa<AST::WithBlock>(&n)) {
      vn.EnterScope();
    } else if (isa<AST::ForeachBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
      vn.EnterScope();
      gen_values = false; // disable valno on range expressions
    } else if (auto* b = dyn_cast<AST::MultiDimSpans>(&n)) {
      if (b->ref_name != "") {
        auto n = SSTab().NameInScopeOrNull(b->ref_name);
        if (!n)
          choreo_unreachable("variable `" + b->ref_name +
                             "' is not found in scopes.");
        vn.SetListReference(n.value());
      }
    } else if (auto* b = dyn_cast<AST::IntTuple>(&n)) {
      if (b->ref_name != "") {
        auto n = SSTab().NameInScopeOrNull(b->ref_name);
        if (!n)
          choreo_unreachable(
              ("variable `" + b->ref_name + "' is not found in scopes.")
                  .c_str());
        vn.SetListReference(n.value());
      }
    } else if (isa<AST::Wait>(&n) || isa<AST::Call>(&n) ||
               isa<AST::Rotate>(&n) || isa<AST::Select>(&n)) {
      gen_values = false;
    } else if (isa<AST::Parameter>(&n)) {
      allow_named_dim = true;
    }
    return true;
  }

  virtual bool AfterVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, false, "after ");
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n)) {
      vn.LeaveScope();
    } else if (isa<AST::ForeachBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
      vn.LeaveScope();
    } else if (isa<AST::MultiDimSpans>(&n) || isa<AST::IntTuple>(&n)) {
      vn.ResetListReference();
    } else if (isa<AST::Wait>(&n) || isa<AST::Call>(&n) ||
               isa<AST::Rotate>(&n) || isa<AST::Select>(&n)) {
      gen_values = true;
    } else if (isa<AST::Parameter>(&n)) {
      allow_named_dim = false;
    }

    return true;
  }

public:
  bool Visit(AST::MultiNodes& n) {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::MultiValues& n) {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    if (gen_values) {
      int valNo = vn.GenerateValueNumberForNode(n);
      cur_vn = valNo;
    } else
      InvalidateVN(cur_vn);
    return true;
  }

  bool Visit(AST::IntLiteral& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;
    int valNo = vn.GenerateValueNumberForNode(n);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::FloatLiteral& n) {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    int valNo = vn.GenerateValueNumberForNode(n);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::StringLiteral& n) {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    InvalidateVN(cur_vn);
    return true;
  }

  bool Visit(AST::Boolean& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;
    int valNo = vn.GenerateValueNumberForNode(n);
    cur_vn = valNo;
    return true;
  }

  bool Visit(AST::Expr& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;
    if (auto id = n.GetSymbol()) {
      auto name = vn.VNSymbolName(*id);
      if (SSTab().IsDeclared(name)) {
        if (vn.HasValueNumberOfSignature(SSTab().InScopeName(name))) {
          cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(name));
          if (isa<MDSpanType>(SSTab().LookupSymbol(name))) {
            cur_mdspan_vn = cur_vn;
            InvalidateVN(cur_vn);
          }
          if (ValidVN(cur_vn)) {
            n.s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
            if (ConvertibleToInt(NodeType(n))) {
              assert(n.s.DimCount() == 1);
              if (!n.s.IsDynamic()) {
                n.opt_vals.int_expr = n.s.ValueAt(0);
                VST_DEBUG(dbgs() << "[ExprVal] " << STR(n) << ": "
                                 << STR(n.s.ValueAt(0)) << "\n");
              }
            }
          }
        } else {
          // no value number is obtained
          InvalidateVN(cur_vn);
        }
        return true;
      }
    } else if (n.op == "dataof") {
      // fill the mdspan type of this node
      auto id = cast<AST::Expr>(n.GetR())->GetSymbol();
      assert(!SuffixedWith(id->name, ".span"));
      auto name = id->name + ".span";
      assert(SSTab().IsDeclared(name));
      cur_mdspan_vn = vn.GetValueNumberOfSignature(InScopeName(name));
      assert(ValidVN(cur_mdspan_vn));
      n.s =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_mdspan_vn));
      VST_DEBUG(dbgs() << "[ExprShape] Shape for " << STR(n) << ": " << STR(n.s)
                       << "\n");
      InvalidateVN(cur_vn); // a spanned data does not have a value number
      return true;
    }

    // the expression could be mdspan/ituple. record the information for later
    // type inference
    cur_vn = vn.GenerateValueNumberForNode(n);
    n.s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));

    if (ConvertibleToInt(NodeType(n))) {
      assert(n.s.DimCount() == 1);
      if (!n.s.IsDynamic()) {
        n.opt_vals.int_expr = n.s.ValueAt(0);
        VST_DEBUG(dbgs() << "[ExprVal] " << STR(n) << ": "
                         << STR(n.s.ValueAt(0)) << "\n");
      }
    }

    if (AST::istypeof<MDSpanType>(&n)) {
      cur_mdspan_vn = cur_vn;
      cast<MDSpanType>(n.GetType())
          ->SetShape(GenShapeFromSignature(
              vn.GetSignatureFromValueNumber(cur_mdspan_vn)));
      //      InvalidateVN(cur_vn);
    } else if (n.op == "#") {
      if (IsActualBoundedIntegerType(n.GetL()->GetType()) &&
          IsActualBoundedIntegerType(n.GetR()->GetType())) {
        assert(n.s.DimCount() == 1);
        n.SetType(MakeBoundedIntegerType(n.s.ValueAt(0)));
      }
    }

    return true;
  }

  bool Visit(AST::MultiDimSpans& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    if (n.list) {
      // The Shape now can be deduced from the value number.
      // Update the type detail acoordingly.
      auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

      // set alias expressions with proper value numbers
      if (CountElementsInSignature(vn_sig) > 1) {
        ProcessValueNumberString(
            vn_sig, [this, &vn_sig](int valno, size_t index) {
              if (UnknownVN(valno))
                return; // do not associate it with vn of "?"
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

      if (IsValidRank(n.Rank())) {
#if 0
        if (vl.Dims() != n.Rank())
          Error(n.LOC(),
                "mdspan's dimension is inconsistent with its initialization "
                "expression: " +
                    std::to_string(vl.Dims()) + " vs. " +
                    std::to_string(n.Rank()) + ".");
#endif
      } else
        n.SetRank(vl.Rank());

      // pass the value number over
      cur_mdspan_vn = cur_vn;
    } else if (n.Rank() > 0) {
      std::string unknown_spans = "#" + std::to_string(UnknownValue());
      for (size_t i = 1; i < n.Rank(); ++i)
        unknown_spans = unknown_spans + ",#" + std::to_string(UnknownValue());
      cur_mdspan_vn = vn.GetOrInsertValueNumberFromSignature(unknown_spans);
      n.SetTypeDetail(GenShapeFromSignature(unknown_spans));
    } else {
      SetUnknownVN(cur_mdspan_vn); // failed to deduce the type detail
    }

    InvalidateVN(cur_vn);
    return true;
  }

  bool Visit(AST::NamedTypeDecl& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    auto name = n.name_str;
    if (n.init_expr) {
      assert(ValidVN(cur_mdspan_vn) &&
             "invalid value number for the named type.");
      SSTab().DefineSymbol(name, n.GetType());

      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name),
                                           cur_mdspan_vn);

      InvalidateVN(cur_mdspan_vn); // comsumes the mdspan
    }
    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    auto name = n.name_str;
    if (SSTab().DeclaredInScope(name)) {
      Error(n.LOC(),
            "ODR violation: symbol `" + name + "' has been declared already.");
      error_count++;
      return false;
    }

    Storage sto = Storage::NONE;
    if (auto sel = dyn_cast<AST::Select>(n.init_expr)) {
      if (auto sty = dyn_cast<SpannedType>(sel->GetType())) {
        assert(!n.mem);
        sto = sty->GetStorage();
      }
    }

    if (n.mem) sto = n.mem->st;

    ptr<Type> nty = nullptr;
    if (n.init_expr) {
      nty = n.init_expr->GetType();
      if (GetSpannedType(NodeType(*n.init_expr))) {
        assert(ValidVN(cur_mdspan_vn) && "expecting a valid mdspan valno.");
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name + ".span"),
                                             cur_mdspan_vn);
      } else {
        if (!isa<PlaceHolderType>(nty)) {
          assert(ValidVN(cur_vn) &&
                 "cur_mdspan_vn and cur_vn must be exclusive.");
          vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name),
                                               cur_vn);
        }
      }
    } else {
      // obtain the types from declaration
      if (ValidVN(cur_mdspan_vn)) {
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name + ".span"),
                                             cur_mdspan_vn);
        auto mds_value = GenShapeFromSignature(
            vn.GetSignatureFromValueNumber(cur_mdspan_vn));
        nty = MakeSpannedType(n.type->base_type, mds_value, sto);
      } else if (ValidVN(cur_vn)) {
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_vn);
        nty = NodeType(*n.type);
      } else
        nty = NodeType(*n.type);
    }

    // fill-up the symbol table
    assert(nty);
    SSTab().DefineSymbol(name, nty);
    n.SetType(nty);

    if (isa<IntegerType>(nty) && ValidVN(cur_vn)) {
      auto shape =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
      assert(shape.DimCount() == 1);
      VST_DEBUG(dbgs() << "[SymVal] " << InScopeName(name) << ": "
                       << STR(shape.ValueAt(0)) << "\n");
      SymVal(InScopeName(name)).int_expr = shape.ValueAt(0);
    }

    if (isa<FutureType>(n.GetType()) || isa<SpannedType>(n.GetType()))
      SSTab().DefineSymbol(name + ".span", GetSpannedType(n.GetType()));

    InvalidateVN(cur_mdspan_vn); // stop propagation
    InvalidateVN(cur_vn);

    return true;
  }

  bool Visit(AST::IntTuple& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    auto mvals = n.GetValues();
    // cur_ituple_vn = cur_vn;
    n.SetType(MakeITupleType(mvals->Count()));

    auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

    if (CountElementsInSignature(vn_sig) > 1) {
      // set alias expressions with proper value numbers
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig](int valno, size_t index) {
            if (UnknownVN(valno)) return; // do not associate it with vn of "?"
            vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                   std::to_string(index));
            vn.AssociateSignatureWithValueNumber(
                vn_sig + "(" + std::to_string(index) + ")", valno);
          });
    }
    InvalidateVN(cur_vn); // Currently cut off value numbering
    return true;
  }

  bool Visit(AST::Assignment& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;
    if (SSTab().IsDeclared(n.name)) return true;

    // this is the un-type-annotated declaration
    auto nty = n.value->GetType();
    SSTab().DefineSymbol(n.name, nty);

    if (auto san = dyn_cast<AST::SpanAs>(n.value))
      assert((n.name == san->nid->name) &&
             "inconsistent span_as variable name.");

    auto name = n.name;
    if (auto sty = GetSpannedType(nty)) {
      name += ".span";
      SSTab().DefineSymbol(name, sty->GetMDSpanType());
      assert(ValidVN(cur_mdspan_vn) &&
             "expected a valid current value number.");
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name),
                                           cur_mdspan_vn);
      return true;
    }

    if (IsActualBoundedIntegerType(nty)) {
      name = "@" + name;
      SSTab().DefineSymbol(name, MakeIntegerType());
    }

    assert(ValidVN(cur_vn) && "expected a valid current value number.");
    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_vn);

    if (isa<IntegerType>(nty) && ValidVN(cur_vn)) {
      auto shape =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
      assert(shape.DimCount() == 1);
      VST_DEBUG(dbgs() << "[SymVal] " << SSTab().ScopedName(name) << ": "
                       << STR(shape.ValueAt(0)) << "\n");
      SymVal(SSTab().ScopedName(name)).int_expr = shape.ValueAt(0);
    }

    return true;
  }

  bool Visit(AST::IntIndex& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    cur_vn = vn.GenerateValueNumberForNode(n);

    return true;
  }

  bool Visit(AST::DataType& n) {
    TraceEachVisit(n);

    allow_named_dim = false;

    if (cannot_proceed) return true;

    if (ValidVN(cur_mdspan_vn)) { cur_vn = cur_mdspan_vn; }

    return true;
  }

  bool Visit(AST::Identifier& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    if (!gen_values) return false;

    auto name = vn.VNSymbolName(n);
    if (SSTab().IsDeclared(name)) {
      // it is a reference
      if (!vn.HasValueNumberOfSignature(SSTab().InScopeName(name)))
        choreo_unreachable("value number of `" + SSTab().InScopeName(name) +
                           "' has not been generated.");
      cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(name));
      return true;
    }

    if (allow_named_dim) { // for named dims in parameters
      if (!SSTab().DeclaredInScope(n.name)) {
        SSTab().DefineSymbol(n.name, MakeIntegerType());
        cur_vn =
            vn.GenerateValueNumberFromSignature(SSTab().InScopeName(n.name));
      } else {
        cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(n.name));
      }
      return true;
    }

    if (vn.HasValueNumberForNode(n)) {
      Error(n.LOC(), "value number has been generated for `" + n.name + "'.");
      error_count++;
      return false;
    }

    // sometime we need value a symbol (symbolic value)
    // TODO: improve it - only generate valno for integer types
    if (!ValidVN(cur_mdspan_vn)) cur_vn = vn.GenerateValueNumberForNode(n);

    return true;
  }

  bool Visit(AST::Parameter& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

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

      } else if (IsValidRank(span->Rank())) {
        assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");
        // Put alias names of mdspan into the value number table
        vn.AssociateSignatureWithValueNumber(
            SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
        n.type->SetType(
            MakeSpannedType(n.type->base_type, span->GetTypeDetail()));
      } else {
        // the value number is unknown at compile time
        Error(n.LOC(), "The type can not be inference at compile time.");
        error_count++;
        return false;
      }

      InvalidateVN(cur_mdspan_vn);
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
      assert(!ValidVN(cur_mdspan_vn) && "unexpected current mdspan value.");

      // get the value number and make it defined
      vn.GetValueNumberOfSignature(SSTab().ScopedName(n.sym->name));
      if (n.sym) SSTab().DefineSymbol(n.sym->name, n.GetType());

      InvalidateVN(cur_vn);
      return true;
    }

    return true;
  }

  bool Visit(AST::ParamList& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::ParallelBy& n) override {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
    n.SetType(MakeMDSpanType(s));

    std::string iv_name = SSTab().ScopedName("@" + n.biv->name);
    vn.AssociateSignatureWithValueNumber(iv_name, cur_vn);
    n.biv->SetType(MakeBoundedITupleType(s, "pv"));
    SSTab().DefineSymbol("@" + n.biv->name, MakeMDSpanType(s));
    SSTab().DefineSymbol(n.biv->name, n.biv->GetType());

    std::map<size_t, std::string> idx2dim;
    idx2dim[0] = "x";
    idx2dim[1] = "y";
    idx2dim[2] = "z";
    for (size_t i = 0; i < n.dims; ++i) {
      const auto& [sym, b] = n.GetIV(i);
      std::string bound;
      if (auto il = dyn_cast<AST::IntLiteral>(b))
        bound = "const_" + std::to_string(il->Val());
      else if (auto id = dyn_cast<AST::Identifier>(b))
        bound = id->name;
      else
        choreo_unreachable("unexpected type of parallelby bound item");
      int valno = vn.GetOrInsertValueNumberFromSignature(bound);
      std::string iv_name = SSTab().ScopedName("@" + sym->name);
      vn.AssociateSignatureWithValueNumber(iv_name, valno);
      Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(valno));
      sym->SetType(MakeBoundedITupleType(s, "pi:" + idx2dim[i]));
      SSTab().DefineSymbol("@" + sym->name, MakeMDSpanType(s));
      SSTab().DefineSymbol(sym->name, sym->GetType());
    }
    return true;
  };

  bool Visit(AST::WhereBind& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    assert(isa<AST::Identifier>(n.lhs) &&
           "non-id is not supported in where bind.");
    assert(isa<AST::Identifier>(n.rhs) &&
           "non-id is not supported in where bind.");

    auto l_id = cast<AST::Identifier>(n.lhs);
    auto r_id = cast<AST::Identifier>(n.rhs);
    auto l_vn =
        vn.GetValueNumberOfSignature(SSTab().ScopedName("@" + l_id->name));
    auto r_vn =
        vn.GetValueNumberOfSignature(SSTab().ScopedName("@" + r_id->name));

    // TODO: sometimes the lhs would have same valno with existing one, which is
    // allowed. However, for runtime valued bound, they may have different
    // bound. The problem here is how to judge if the upper bound of bounded
    // variables are actually illegal? (e.g, different static upper bound)
    vn.BindValueNumbers(l_vn, r_vn);
    return true;
  }

  bool Visit(AST::WithIn& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    if (auto mds = dyn_cast<AST::MultiDimSpans>(n.in)) {
      assert(ValidVN(cur_mdspan_vn) &&
             "no valid value number generated for the mdspan.");
      if (n.with_matchers)
        if (n.with_matchers->Count() != mds->Rank()) {
          Error(n.LOC(), "inconsistent with-in values and bounds.");
          error_count++;
          return false;
        }

    } else if (isa<AST::Expr>(n.in)) {
      if (auto id = AST::GetIdentifier(*n.in))
        cur_mdspan_vn =
            vn.GetValueNumberOfSignature(SSTab().InScopeName(id->name));
      else
        cur_mdspan_vn = cur_vn;
      assert(ValidVN(cur_mdspan_vn) && "no valid vn for with-in.");
      InvalidateVN(cur_vn);
    } else {
      choreo_unreachable("unexpected with-in statement.");
    }

    auto vn_sig = vn.GetSignatureFromValueNumber(cur_mdspan_vn);

    // requires the elements inside mdspan to be non-zero values
    bool found_zero = false;
    ProcessValueNumberString(
        vn_sig, [this, &vn_sig, &n, &found_zero](int valno, size_t) {
          auto sig = vn.GetSignatureFromValueNumber(valno);
          if (sig == "const_0") { found_zero = true; }
        });
    if (found_zero) {
      Error(
          n.LOC(),
          "zero value is deduced for the mdspan inside the with-in statement.");
      error_count++;
      cannot_proceed = true;
      Error(n.LOC(),
            "unable to apply shape inference for function '" + cur_fn + "'.");
      return false;
    }

    auto GenSignatureAndDoValno = [this, &vn_sig, &n](int valno, size_t index) {
      if (UnknownVN(valno)) return; // do not associate it with vn of "?"
      if (n.with) {
        std::string name = SSTab().ScopedName("@" + n.with->name) + "(" +
                           std::to_string(index) + ")";
        vn.AssociateSignatureWithValueNumber(name, valno);
      }

      if (n.with_matchers) {
        auto sym = cast<AST::Identifier>((*n.with_matchers)[index]);
        std::string name = SSTab().ScopedName("@" + sym->name);
        vn.AssociateSignatureWithValueNumber(name, valno);
        Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(valno));
        sym->SetType(MakeBoundedITupleType(s));
        SSTab().DefineSymbol("@" + sym->name, MakeMDSpanType(s));

        // because we use bounded integer var as identifier
        name = SSTab().ScopedName(sym->name);
        SSTab().DefineSymbol(sym->name, sym->GetType());
        // vn.GetOrInsertValueNumberFromSignature(name);
        // TODO(wsj): deal with expression contains bounded integers
      }
    };
    if (CountElementsInSignature(vn_sig) ==
        1) // support `with idx={m} in [xx] {}`
      GenSignatureAndDoValno(vn.GetValueNumberOfSignature(vn_sig), 0);
    else
      ProcessValueNumberString(vn_sig, GenSignatureAndDoValno);

    if (n.with) {
      vn.AssociateSignatureWithValueNumber(
          SSTab().ScopedName("@" + n.with->name), cur_mdspan_vn);
      Shape s = GenShapeFromSignature(vn_sig);
      n.with->SetType(MakeBoundedITupleType(s));
      SSTab().DefineSymbol("@" + n.with->name, MakeMDSpanType(s));
      SSTab().DefineSymbol(n.with->name, n.with->GetType());
    }
    InvalidateVN(cur_mdspan_vn);

    return true;
  }

  bool Visit(AST::WithBlock& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::Memory& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::SpanAs& n) {
    TraceEachVisit(n);
    assert(ValidVN(cur_vn) && "failed to get the list value.");

    auto pty = SSTab().LookupSymbol(n.id->name);
    assert((isa<SpannedType>(pty) || isa<FutureType>(pty)) &&
           "unexpected data type.");

    auto sty = GetSpannedType(pty);
    if (!sty) {
      Error(n.LOC(), "internal error: span_as operates on non-spanned type.");
      return false;
    }

    vn.AssociateSignatureWithValueNumber(
        SSTab().ScopedName(n.nid->name + ".span"), cur_vn);

    cur_mdspan_vn = cur_vn;

    auto shape = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
    auto nty = MakeSpannedType(sty->ElementType(), shape, sty->GetStorage());

    n.SetType(nty);

    return true;
  }

  bool Visit(AST::DMA& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    if (n.operation == ".any") {
      assert(!n.future.empty() && "unexpected: the future is empty.");
      SSTab().DefineSymbol(n.future, MakePlaceHolderFutureType());
      SSTab().DefineSymbol(n.future + ".span", MakePlaceHolderMDSpanType());
      vn.AssociateSignatureWithInvalidValueNumber(
          SSTab().ScopedName(n.future + ".span"));
      InvalidateVN(cur_vn);
      return true;
    }

    assert(ValidVN(cur_vn) &&
           "unexpected current value number for shape inference of dma.");

    if (auto pcfg = dyn_cast<PadConfig>(n.config)) {
      size_t size = pcfg->pad_high.size();
      std::vector<size_t> all_pads(size);
      std::fill_n(all_pads.begin(), size, 0);
      for (size_t i = 0; i < size; ++i)
        all_pads[i] += pcfg->pad_high[i] + pcfg->pad_low[i] + pcfg->pad_mid[i];
      // now generate signature for original signature plus padding values
      auto ElementSignature = [this](size_t n) {
        std::string cv = "const_" + std::to_string(n);
        return "#" + std::to_string(vn.GetOrInsertValueNumberFromSignature(cv));
      };
      std::string sig;
      if (size > 1) {
        sig = ElementSignature(all_pads[0]);
        for (size_t i = 1; i < size; ++i)
          sig += "," + ElementSignature(all_pads[i]);
      } else {
        sig = "const_" + std::to_string(all_pads[0]);
      }
      std::string add_sig = vn.SignBinaryCompositeValues(
          n.LOC(), "+", vn.GetSignatureFromValueNumber(cur_vn), sig);
      // update the cur_vn
      cur_vn = vn.GetOrInsertValueNumberFromSignature(add_sig);
    } else if (auto tcfg = dyn_cast<TransposeConfig>(n.config)) {
      // gen new vn if and only if n.to is AST::Memory
      if (isa<AST::Memory>(n.to)) {
        auto& dim_values = tcfg->dim_values;
        auto orig_sig = vn.GetSignatureFromValueNumber(cur_vn);
        auto shape_components = SplitStringByDelimiter(orig_sig);
        auto sig = shape_components[dim_values[0]];
        for (size_t i = 1; i < dim_values.size(); ++i)
          sig += "," + shape_components[dim_values[i]];
        cur_vn = vn.GetOrInsertValueNumberFromSignature(sig);
      }
    }

    // annotate the shape on AST for later type inference
    auto s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
    n.SetType(MakeShapedFutureType(s, n.async));

    if (n.future.empty()) {
      InvalidateVN(cur_vn);
      return true;
    }

    if (SSTab().IsDeclared(n.future)) {
      assert(
          cast<PlaceHolderType>(SSTab().LookupSymbol(n.future))->Category() ==
          TypeCategory::FUTURE);
      vn.RebindSignatureWithValueNumber(SSTab().InScopeName(n.future) + ".span",
                                        cur_vn);
      SSTab().ModifySymbolType(n.future, n.GetType());
      SSTab().ModifySymbolType(n.future + ".span", MakeMDSpanType(s));
    } else {
      std::string f_span = n.future + ".span";
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(f_span), cur_vn);
      SSTab().DefineSymbol(n.future, n.GetType());
      SSTab().DefineSymbol(f_span, MakeMDSpanType(s)); // implicit symbol
    }

    auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);
    // set alias expressions with proper value numbers
    if (CountElementsInSignature(vn_sig) > 1) {
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig](int valno, size_t index) {
            if (UnknownVN(valno)) return; // do not associate it with vn of "?"
            vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                   std::to_string(index));
            auto elem_sig = vn_sig + "(" + std::to_string(index) + ")";
            if (!vn.HasValueNumberOfSignature(elem_sig))
              vn.AssociateSignatureWithValueNumber(
                  vn_sig + "(" + std::to_string(index) + ")", valno);
          });
    }

    InvalidateVN(cur_vn);
    return true;
  }

  bool Visit(AST::ChunkAt& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    int ca_valno = GetInvalidValueNumber();

    auto pty = SSTab().LookupSymbol(n.data->name);
    assert((isa<SpannedType>(pty) || isa<FutureType>(pty)) &&
           "unexpected data type.");

    auto span_name = RemoveSuffix(n.data->name, ".data") + ".span";
    auto sty = GetSpannedType(pty);

    if (!n.positions) {
      // it is just a symbol reference
      ca_valno = vn.GetValueNumberOfSignature(SSTab().InScopeName(span_name));
      // set the chunkat's type
      n.SetType(MakeSpannedType(
          sty->f_type,
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(ca_valno)),
          sty->GetStorage()));

      cur_vn = ca_valno;
      return true;
    }

    std::string data_sig = vn.SignatureOfSymbol(SSTab().InScopeName(span_name));
    int dim_count = CountElementsInSignature(data_sig);
    int dim_index = 0;

    // we only expect signature in multi-sig format
    if ((dim_count == 1) && (!PrefixedWith(data_sig, "#"))) {
      data_sig = "#" + std::to_string(vn.GetValueNumberOfSignature(
                           SSTab().InScopeName(span_name)));
    }

    std::string fs_signature; // signature of the future.span
    auto AppendSignature = [this, &fs_signature, &n, &dim_index,
                            dim_count](int dividend_vn, int divisor_vn) {
      // the signature without optimiz
      std::string res_sig = "/:#" + std::to_string(dividend_vn) + ":#" +
                            std::to_string(divisor_vn);

      if (auto quotient = vn.TryToSimplifyBinary(
              n.LOC(), "/", vn.GetSignatureFromValueNumber(dividend_vn),
              vn.GetSignatureFromValueNumber(divisor_vn), true))
        res_sig = quotient.value();

      // now generate the value number from the signature
      int res_valno = vn.GetOrInsertValueNumberFromSignature(res_sig);

      // and append the value number as
      if (!fs_signature.empty()) fs_signature += ",";
      fs_signature += "#" + std::to_string(res_valno);
    };

    int index = -1;
    for (auto pos : n.positions->values) {
      ++index;
      auto biv = dyn_cast<AST::Identifier>(pos);
      if (!biv) {
        auto expr = cast<AST::Expr>(pos);
        assert(expr->op == "getith");
        biv = cast<AST::Expr>(expr->GetL())->GetSymbol();
      }
      assert(biv && "failed to obtain the identifier.");
      int bound_vn = GetInvalidValueNumber();
      if (n.bounds) {
        // when explicit bound exists
        auto bnode = n.bounds->ValueAt(index);
        if (isa<AST::IntLiteral>(AST::Ref(bnode)))
          bound_vn =
              vn.GetOrInsertValueNumberFromSignature("const_" + STR(*bnode));
        else
          bound_vn = vn.GenerateValueNumberForNode(*bnode);
      } else {
        auto bound_name = SSTab().InScopeName("@" + biv->name);
        bound_vn = vn.GetValueNumberOfSignature(bound_name);
      }
      std::string bound_sn = vn.GetSignatureFromValueNumber(bound_vn);

      // get the value number of i-th in multi-dim sigature
      auto GetDimValNO = [this, &n, &data_sig](int idx) {
        auto dim_ith = GetNthElement(data_sig, idx);
        if (!dim_ith) {
          Error(n.LOC(), "internal error: value number is not obtained.");
          error_count++;
          return GetInvalidValueNumber();
        }

        assert(dim_ith.value()[0] == '#' ||
               (dim_ith.value().substr(0, 6) == "const_"));

        int dim_valno = dim_ith.value()[0] == '#'
                            ? std::stoi(dim_ith.value().substr(1))
                            : vn.GetValueNumberOfSignature(dim_ith.value());
        return dim_valno;
      };

      size_t err_cnt = error_count;
      if (CountElementsInSignature(bound_sn) <= 1) {
        // this is a simple bound
        AppendSignature(GetDimValNO(dim_index), bound_vn);
        if (++dim_index > dim_count) {
          Error(n.LOC(), "dimensions inconsistence is found between `" +
                             n.data->name + "' and chunkat expression.");
          error_count++;
        }
      } else {
        // multiple bounds
        ProcessValueNumberString(bound_sn, [this, &GetDimValNO,
                                            &AppendSignature, &dim_index,
                                            &dim_count, &n](int valno, size_t) {
          AppendSignature(GetDimValNO(dim_index), valno);
          if (++dim_index > dim_count) {
            Error(n.LOC(), "dimensions inconsistence is found between `" +
                               n.data->name + "' and chunkat expression.");
            error_count++;
          }
        });
      }

      if (error_count != err_cnt) return false;
    }

    ca_valno = vn.GetOrInsertValueNumberFromSignature(fs_signature);

    // set the chunkat's type
    n.SetType(MakeSpannedType(
        sty->f_type,
        GenShapeFromSignature(vn.GetSignatureFromValueNumber(ca_valno)),
        sty->GetStorage()));

    cur_vn = ca_valno;

    return true;
  }

  bool Visit(AST::Wait& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Call& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    // value the scalars
    for (auto& s : n.arguments->AllValues()) {
      if (isa<IntegerType>(NodeType(*s))) {
        auto expr = cast<AST::Expr>(s);
        expr->s = GenShapeFromSignature(vn.GetSignatureForNode(*s));
        VST_DEBUG(dbgs() << "[ExprShape] Shape for " << PSTR(s) << ": "
                         << STR(expr->s) << "\n");
        assert(expr->s.DimCount() == 1);
        expr->opt_vals.int_expr = expr->s.ValueAt(0);
        VST_DEBUG(dbgs() << "[ExprVal] Value for " << PSTR(expr) << ": "
                         << STR(expr->s.ValueAt(0)) << "\n");
      }
    }

    return true;
  };

  bool Visit(AST::Rotate& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    auto rty = type_equals.ResolveEqualFutures(*n.ids, true);

    if (!rty) {
      Error(n.LOC(), "Failed to resolve future types.");
      error_count++;
      return false;
    }

    // do not care about placeholders
    if (isa<PlaceHolderType>(rty)) return true;

    int valno = GetOnlyValueNumberFromMultiValues(*n.ids);

    if (!ValidVN(valno)) {
      Error(n.LOC(), "failed to find a valid value number inside ROTATE.");
      error_count++;
      cannot_proceed = true;
      return false;
    }

    // now update the valnos
    UpdateValueNumberForMultiValues(*n.ids, valno);

    return true;
  };

  bool Visit(AST::Select& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    assert(!n.inDMA);
    if (auto sty = dyn_cast<SpannedType>(NodeType(n))) {
      auto s0 = cast<AST::Expr>(n.expr_list->ValueAt(0));
      auto s0ty = NodeType(*s0);
      if (s0ty && s0ty->HasSufficientInfo())
        n.SetType(s0ty);
      else {
        // handle dataof expr (TODO: any better idea?)
        if (!s0->s.IsValid()) {
          Error(n.LOC(), "Failed to decide the type of Select." + STR(n) +
                             ", type0: " + PSTR(s0ty));
          error_count++;
          return false;
        }
        auto nty = MakeSpannedType(sty->f_type, s0->s, sty->GetStorage());
        n.SetType(nty);
      }

      cur_mdspan_vn = vn.GenerateValueNumberForNode(n);
      InvalidateVN(cur_vn); // used for variable def
    } else if (GeneralFutureType(NodeType(n))) {
      InvalidateVN(cur_vn);

      auto fty = type_equals.ResolveEqualFutures(*n.expr_list, true);
      if (!fty) {
        Error(n.LOC(), "Failed to resolve future types.");
        error_count++;
        return false;
      }
      n.SetType(fty);
      if (isa<PlaceHolderType>(fty)) return true;

      cur_mdspan_vn = GetOnlyValueNumberFromMultiValues(*n.expr_list);
      if (!ValidVN(cur_mdspan_vn)) {
        Error(n.LOC(),
              "no valid value number is found for a SELECT expression.");
        error_count++;
        cannot_proceed = true;
        return false;
      }

      // now update the valnos
      UpdateValueNumberForMultiValues(*n.expr_list, cur_mdspan_vn);
    } else
      choreo_unreachable("unsupported type.");

    return true;
  };

  bool Visit(AST::Return& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::LoopRange& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::ForeachBlock& n) {
    TraceEachVisit(n);

    gen_values = true; // allow generate values for statements

    // invalidate any current value generated
    InvalidateVN(cur_mdspan_vn);
    InvalidateVN(cur_vn);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::IncrementBlock& n) {
    TraceEachVisit(n);

    gen_values = true; // allow generate values for statements

    // invalidate any current value generated
    InvalidateVN(cur_mdspan_vn);
    InvalidateVN(cur_vn);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::FunctionDecl& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::ChoreoFunction& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::CppSourceCode& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::Program& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    return true;
  };

private:
  // Given a multi-value signature, process each value
  void ProcessValueNumberString(const std::string& input,
                                std::function<void(int, size_t)> lambda) {
    std::regex valuePattern("#(-?\\d+)");
    auto begin = std::sregex_iterator(input.begin(), input.end(), valuePattern);
    auto end = std::sregex_iterator();

    size_t matchIndex = 0;
    for (auto i = begin; i != end; ++i, ++matchIndex) {
      std::smatch match = *i;
      std::string matchStr =
          match.str(1); // Capture the number part of the match
      int number = std::stoi(matchStr);

      // Call the passed lambda function with the extracted string and its
      // index
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
    if (auto digit = RemovePrefixOrNull("const_", sig)) return *digit;

    // a value number reference
    if (auto digit = RemovePrefixOrNull("#", sig))
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
      return "(" + GenerateExpression(parts[1]) + ")" + parts[0] + "(" +
             GenerateExpression(parts[2]) + ")";
    }

    // this is a symbol
    return sig;
  }

  int GetOnlyValueNumberFromMultiValues(const AST::MultiValues& mv) {
    int valno = GetInvalidValueNumber();
    for (auto& v : mv.AllValues()) {
      auto id = AST::GetIdentifier(*v);
      if (!id) choreo_unreachable("expect an identifier.\n");
      auto ln = SSTab().InScopeName(vn.VNSymbolName(*id));

      if (!ValidVN(valno)) {
        valno = vn.GetValueNumberOfSignature(ln);
        continue;
      }

      // Check for consistence between different values
      if (vn.HasValidValueNumberOfSignature(ln)) {
        if (valno != vn.GetValueNumberOfSignature(ln)) {
// currently some equivalence cannot be detected, drop the check
#if 0
          Error(mv.LOC(), "value number does not match.");
          cannot_proceed = true;
          return GetInvalidValueNumber();
#endif
        }
      }
    }
    return valno;
  }

  void UpdateValueNumberForMultiValues(const AST::MultiValues& mv, int valno) {
    for (auto& v : mv.AllValues()) {
      if (auto id = AST::GetIdentifier(*v)) {
        auto symbol = SSTab().InScopeName(vn.VNSymbolName(*id));
        // the VN is considered to be identical if none exist
        if (!ValidVN(vn.GetValueNumberOfSignature(symbol))) {
          vn.RebindSignatureWithValueNumber(symbol, valno);
          auto equals = type_equals.GetEquals(SSTab().InScopeName(id->name));
          for (auto& e : equals.value().get()) {
            auto asym = e + ".span";
            if (!vn.HasValidValueNumberOfSignature(asym))
              vn.RebindSignatureWithValueNumber(asym, valno);
            else
              assert(valno == vn.GetValueNumberOfSignature(asym));
          }
        }
      } else
        choreo_unreachable("expect an identifier.");
    }
  }
};

} // end namespace Choreo

#endif // __CHOREO_VALUE_NUMBERING_HPP__
