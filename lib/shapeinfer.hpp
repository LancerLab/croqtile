#ifndef __CHOREO_SHAPE_INFERENCE_HPP__
#define __CHOREO_SHAPE_INFERENCE_HPP__

#include "valno.hpp"

namespace Choreo {

class ShapeInference : public VisitorWithScope {
private:
  ValueNumbering vn;

  // valno rendered from current ast node
  int cur_vn = GetInvalidValueNumber();

  // implicit valno of spanned-type with ".span" annotation
  int cur_mdspan_vn = GetInvalidValueNumber();

  // implicit valno of upper-bound
  int cur_ub_vn = GetInvalidValueNumber();

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
    if (!trace_visit && !debug_visit) return;

    if (debug_visit) dbgs() << "[";

    if (detail)
      dbgs() << m << STR(n);
    else
      dbgs() << m << n.TypeNameString();

    if (debug_visit)
      dbgs() << "]\t "
             << (ValidVN(cur_vn) ? ("#" + std::to_string(cur_vn)) : "nil")
             << "(vn),\t "
             << (ValidVN(cur_mdspan_vn) ? ("#" + std::to_string(cur_mdspan_vn))
                                        : "nil")
             << "(mds),\t "
             << (ValidVN(cur_ub_vn) ? ("#" + std::to_string(cur_ub_vn)) : "nil")
             << "(ub)";

    dbgs() << "\n";
  }

  void InvalidateVisitorValNOs() {
    InvalidateVN(cur_vn);
    InvalidateVN(cur_mdspan_vn);
    InvalidateVN(cur_ub_vn);
  }

public:
  ShapeInference() : VisitorWithScope("valno"), vn(this) {
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
    } else if (isa<AST::ChoreoFunction>(&n)) {
      vn.EnterScope();
      cannot_proceed = false; // recover state when starting a new function
      int valno = vn.GetOrInsertValueNumberFromSignature("const_1");
      vn.AssociateSignatureWithValueNumber(InScopeName("@__choreo_no_tiling__"),
                                           valno);
      InvalidateVisitorValNOs();
    } else if (isa<AST::ParallelBy>(&n)) {
      vn.EnterScope();
    } else if (isa<AST::WithBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
               isa<AST::IfElseBlock>(&n)) {
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
               isa<AST::Rotate>(&n) || isa<AST::Select>(&n) ||
               isa<AST::Trigger>(&n) || isa<AST::DataAccess>(&n)) {
      gen_values = false;
    } else if (isa<AST::Parameter>(&n)) {
      allow_named_dim = true;
    } else if (isa<AST::MultiNodes>(&n))
      InvalidateVisitorValNOs();

    return true;
  }

  virtual bool InMidVisitImpl(AST::Node& n) override {
    if (isa<AST::IfElseBlock>(&n)) {
      vn.LeaveScope(); // must clear the vn inside if-scope
      vn.EnterScope();
    }
    return true;
  }

  virtual bool AfterVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, false, "after ");
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n)) {
      vn.LeaveScope();
    } else if (isa<AST::ForeachBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
               isa<AST::IfElseBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
      vn.LeaveScope();
    } else if (isa<AST::MultiDimSpans>(&n) || isa<AST::IntTuple>(&n)) {
      vn.ResetListReference();
    } else if (isa<AST::Wait>(&n) || isa<AST::Call>(&n) ||
               isa<AST::Rotate>(&n) || isa<AST::Select>(&n) ||
               isa<AST::Trigger>(&n) || isa<AST::DataAccess>(&n)) {
      gen_values = true;
    } else if (isa<AST::Parameter>(&n)) {
      allow_named_dim = false;
    }

    return true;
  }

public:
  // enable NodeType to retrive a scoped name
  ptr<Type> GetSymbolType(const std::string& n) const override {
    return SSTab().LookupSymbol(n);
  }

  ptr<Type> NodeType(const AST::Node& n) const override {
    if (auto id = dyn_cast<AST::Identifier>(&n)) {
      if (!SSTab().IsDeclared(id->name)) {
        return n.GetType();
      } else {
        return GetSymbolType(id->name);
      }
    } else if (auto expr = dyn_cast<AST::Expr>(&n)) {
      if (auto sym = expr->GetSymbol()) return GetSymbolType(sym->name);
      return expr->GetType();
    }
    return VisitorWithScope::NodeType(n);
  }

public:
  bool Visit(AST::MultiNodes& n) {
    TraceEachVisit(n);
    if (cannot_proceed) return true;
    return true;
  }

  bool Visit(AST::MultiValues& n) {
    TraceEachVisit(n);

    if (cannot_proceed || !CanBeValueNumbered(&n)) {
      InvalidateVN(cur_vn);
      return true;
    }

    if (gen_values) {
      int valNo = vn.GenerateValueNumberForNode(n);
      cur_vn = valNo;
      cur_mdspan_vn = cur_vn;
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
    n.SetType(MakeStringType());
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

    if (!CanBeValueNumbered(&n)) {
      InvalidateVN(cur_vn);
      return true;
    }

    if (auto id = n.GetSymbol()) {
      auto name = vn.VNSymbolName(*id);
      if (SSTab().IsDeclared(name)) {
        if (vn.HasValueNumberOfSignature(SSTab().InScopeName(name))) {
          cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(name));
          auto nty = NodeType(n);
          if (isa<MDSpanType>(SSTab().LookupSymbol(name))) {
            n.s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
            cur_mdspan_vn = cur_vn;
            InvalidateVN(cur_vn);
          }
          if (ValidVN(cur_vn)) {
            n.s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
            if (ConvertibleToInt(NodeType(n))) {
              assert(n.s.DimCount() == 1);
              if (!n.s.IsDynamic()) {
                n.SetOptValExpr(n.s.ValueAt(0));
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

    if (n.IsUBArith()) {
      n.SetType(MakeBoundedITupleType(n.s));
      vn.AssociateSignatureWithValueNumber(
          vn.GetSignatureFromValueNumber(cur_vn), cur_vn);
    }
    if (IsActualBoundedIntegerType(NodeType(n))) {
      cur_ub_vn = cur_vn;
      InvalidateVN(cur_vn);
    }

    if (ConvertibleToInt(NodeType(n))) {
      assert(n.s.DimCount() == 1);
      if (!n.s.IsDynamic()) {
        n.SetOptValExpr(n.s.ValueAt(0));
        VST_DEBUG(dbgs() << "[ExprVal] " << STR(n) << ": "
                         << STR(n.s.ValueAt(0)) << "\n");
      }
    }

    if (AST::istypeof<MDSpanType>(&n)) {
      cur_mdspan_vn = cur_vn;
      auto vn_sig = vn.GetSignatureFromValueNumber(cur_mdspan_vn);
      cast<MDSpanType>(n.GetType())->SetShape(GenShapeFromSignature(vn_sig));
      if (CountElementsInSignature(vn_sig) > 1) {
        // set alias expressions with proper value numbers
        ProcessValueNumberString(
            vn_sig, [this, &vn_sig](int valno, size_t index) {
              if (UnknownVN(valno))
                return; // do not associate it with vn of "?"
              vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                     std::to_string(index));
              vn.AssociateSignatureWithValueNumber(
                  vn_sig + "(" + std::to_string(index) + ")", valno);
            });
      }
      //      InvalidateVN(cur_vn);
    } else if (n.op == "#") {
      if (IsActualBoundedIntegerType(n.GetL()->GetType()) &&
          IsActualBoundedIntegerType(n.GetR()->GetType())) {
        assert(n.s.DimCount() == 1);
        n.SetType(MakeBoundedIntegerType(n.s.ValueAt(0)));
      }
    } else if (AST::istypeof<ITupleType>(&n)) {
      auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

      if (CountElementsInSignature(vn_sig) > 1) {
        // set alias expressions with proper value numbers
        ProcessValueNumberString(
            vn_sig, [this, &vn_sig](int valno, size_t index) {
              if (UnknownVN(valno))
                return; // do not associate it with vn of "?"
              vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                     std::to_string(index));
              vn.AssociateSignatureWithValueNumber(
                  vn_sig + "(" + std::to_string(index) + ")", valno);
            });
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

      // pass the list value number over
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
      DefineASymbol(name, n.GetType());

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

    if (!CanBeValueNumbered(&n)) {
      DefineASymbol(name, NodeType(n));
      return true; // mutables and events are not valno-able
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
      nty = NodeType(*n.init_expr);
      if (GetSpannedType(NodeType(*n.init_expr))) {
        assert(ValidVN(cur_mdspan_vn) && "expecting a valid mdspan valno.");
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name + ".span"),
                                             cur_mdspan_vn);
      } else if (IsActualBoundedIntegerType(nty)) {
        assert(ValidVN(cur_ub_vn));
        DefineASymbol("@" + name, MakeBoundedIntegerType(cur_ub_vn));
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName("@" + name),
                                             cur_ub_vn);
        Shape s =
            GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_ub_vn));
        nty = MakeBoundedITupleType(s);
        vn.AssociateSignatureWithValueNumber(name, cur_ub_vn);

        InvalidateVN(cur_ub_vn);
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
        if (n.IsArray())
          nty = MakeSpannedArrayType(n.type->base_type, mds_value,
                                     n.ArrayDimensions(), sto);
        else
          nty = MakeSpannedType(n.type->base_type, mds_value, sto);
      } else if (ValidVN(cur_vn)) {
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_vn);
        nty = NodeType(*n.type);
      } else
        nty = NodeType(*n.type);
    }

    // fill-up the symbol table
    assert(nty);
    DefineASymbol(name, nty);
    n.SetType(nty);

    // TODO(wsj): BooleanType? HalfType...?
    if ((isa<FloatType>(nty) || isa<DoubleType>(nty) ||
         isa<IntegerType>(nty)) &&
        ValidVN(cur_vn)) {
      auto shape =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
      assert(shape.DimCount() == 1);
      VST_DEBUG(dbgs() << "[SymVal] " << InScopeName(name) << ": "
                       << STR(shape.ValueAt(0)) << "\n");
      SymVal(InScopeName(name)).val_expr = shape.ValueAt(0);
    }

    if (isa<FutureType>(n.GetType()) || isa<SpannedType>(n.GetType()))
      DefineASymbol(name + ".span",
                    GetSpannedType(n.GetType())->GetMDSpanType());

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

  bool Visit(AST::DataAccess& n) {
    TraceEachVisit(n);
    return true;
  }

  bool Visit(AST::Assignment& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    if (IsMutable(*NodeType(n))) return true; // mutables are not valno-able

    if (SSTab().IsDeclared(n.GetName())) return true;

    // this is the un-type-annotated declaration
    auto nty = n.value->GetType();
    DefineASymbol(n.GetName(), nty);

    auto name = n.GetName();
    if (auto san = dyn_cast<AST::SpanAs>(n.value))
      assert((name == san->nid->name) && "inconsistent span_as variable name.");

    if (auto sty = GetSpannedType(nty)) {
      name += ".span";
      DefineASymbol(name, sty->GetMDSpanType());
      assert(ValidVN(cur_mdspan_vn) &&
             "expected a valid current value number.");
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name),
                                           cur_mdspan_vn);
      return true;
    }

    if (IsActualBoundedIntegerType(nty)) {
      name = "@" + name;
      assert(ValidVN(cur_ub_vn));
      DefineASymbol(name, MakeBoundedIntegerType(cur_ub_vn));
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_ub_vn);
      InvalidateVN(cur_ub_vn);
    } else {
      assert(ValidVN(cur_vn) && "expected a valid current value number.");
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_vn);
    }

    if (isa<IntegerType>(nty) && ValidVN(cur_vn)) {
      auto shape =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
      assert(shape.DimCount() == 1);
      VST_DEBUG(dbgs() << "[SymVal] " << SSTab().ScopedName(name) << ": "
                       << STR(shape.ValueAt(0)) << "\n");
      SymVal(SSTab().ScopedName(name)).val_expr = shape.ValueAt(0);
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
    if (SSTab().IsDeclared(n.name))
      if (!CanBeValueNumbered(&n)) { return false; }

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
        DefineASymbol(n.name, MakeIntegerType());
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

      if (n.sym) {
        DefineASymbol(n.sym->name + ".span",
                      cast<SpannedType>(n.type->GetType())->GetMDSpanType());
        DefineASymbol(n.sym->name, n.type->GetType());
      }

      InvalidateVisitorValNOs();
      return true;
    }

    if (n.sym && n.type->isScalar()) {
      assert(!ValidVN(cur_mdspan_vn) && "unexpected current mdspan value.");

      // get the value number and make it defined
      vn.GetValueNumberOfSignature(SSTab().ScopedName(n.sym->name));
      if (n.sym) DefineASymbol(n.sym->name, n.GetType());

      InvalidateVisitorValNOs();
      return true;
    }

    InvalidateVisitorValNOs();
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

    std::string iv_name = SSTab().ScopedName("@" + n.bpv->name);
    vn.AssociateSignatureWithValueNumber(iv_name, cur_vn);
    n.bpv->SetType(MakeBoundedITupleType(s, "pv"));
    DefineASymbol("@" + n.bpv->name, MakeMDSpanType(s));
    DefineASymbol(n.bpv->name, n.bpv->GetType());

    std::map<size_t, std::string> idx2dim;
    idx2dim[0] = "x";
    idx2dim[1] = "y";
    idx2dim[2] = "z";
    for (size_t i = 0; i < n.SubCount(); ++i) {
      const auto& [sym, b] = n.GetIV(i);
      std::string bound;
      if (auto il = dyn_cast<AST::IntLiteral>(b))
        bound = "const_" + std::to_string(il->Val());
      else if (auto id = dyn_cast<AST::Identifier>(b)) {
        if (auto name_in_scope = SSTab().NameInScopeOrNull(id->name)) {
          if (vn.HasValueNumberOfSignature(*name_in_scope))
            bound = *name_in_scope;
          else
            choreo_unreachable("symbol `" + *name_in_scope +
                               "' is not associated with a value number.");
        } else {
          choreo_unreachable("expect symbol `" + *name_in_scope +
                             "' to be defined in symtab!");
        }
      } else
        choreo_unreachable("unexpected type of parallelby bound item");
      int valno = vn.GetOrInsertValueNumberFromSignature(bound);
      std::string iv_name = SSTab().ScopedName("@" + sym->name);
      vn.AssociateSignatureWithValueNumber(iv_name, valno);
      Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(valno));
      sym->SetType(MakeBoundedITupleType(s, "pi:" + idx2dim[i]));
      DefineASymbol("@" + sym->name, MakeMDSpanType(s));
      DefineASymbol(sym->name, sym->GetType());
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
          Error(n.LOC(), "inconsistent with-in values and cmpt_bounds.");
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
            "unable to apply shape inference for function '" + fname + "'.");
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
        DefineASymbol("@" + sym->name, MakeMDSpanType(s));

        // because we use bounded integer var as identifier
        name = SSTab().ScopedName(sym->name);
        DefineASymbol(sym->name, sym->GetType());
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
      DefineASymbol("@" + n.with->name, MakeMDSpanType(s));
      DefineASymbol(n.with->name, n.with->GetType());
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
      DefineASymbol(n.future, MakePlaceHolderFutureType());
      DefineASymbol(n.future + ".span", MakePlaceHolderMDSpanType());
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
      DefineASymbol(n.future, n.GetType());
      DefineASymbol(f_span, MakeMDSpanType(s)); // implicit symbol
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
      auto bpv = dyn_cast<AST::Identifier>(pos);
      if (!bpv) {
        auto expr = cast<AST::Expr>(pos);
        assert(expr->op == "getith");
        bpv = cast<AST::Expr>(expr->GetL())->GetSymbol();
      }
      assert(bpv && "failed to obtain the identifier.");
      int bound_vn = GetInvalidValueNumber();
      if (n.cmpt_bounds) {
        // when explicit bound exists
        auto bnode = n.cmpt_bounds->ValueAt(index);
        if (isa<AST::IntLiteral>(AST::Ref(bnode)))
          bound_vn =
              vn.GetOrInsertValueNumberFromSignature("const_" + STR(*bnode));
        else
          bound_vn = vn.GenerateValueNumberForNode(*bnode);
      } else {
        auto bound_name = SSTab().InScopeName("@" + bpv->name);
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
        // multiple cmpt_bounds
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
    InvalidateVisitorValNOs();
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Trigger& n) {
    TraceEachVisit(n);
    InvalidateVisitorValNOs();
    if (cannot_proceed) return true;

    return true;
  }

  bool Visit(AST::Call& n) {
    TraceEachVisit(n);

    if (cannot_proceed) return true;

    // value the scalars
    for (auto& s : n.arguments->AllValues()) {
      if (!CanBeValueNumbered(s.get())) continue;
      if (isa<IntegerType>(NodeType(*s))) {
        auto expr = cast<AST::Expr>(s);
        expr->s = GenShapeFromSignature(vn.GetSignatureForNode(*s));
        VST_DEBUG(dbgs() << "[ExprShape] Shape for " << PSTR(s) << ": "
                         << STR(expr->s) << "\n");
        assert(expr->s.DimCount() == 1);
        expr->SetOptValExpr(expr->s.ValueAt(0));
        VST_DEBUG(dbgs() << "[ExprVal] Value for " << PSTR(expr) << ": "
                         << STR(expr->s.ValueAt(0)) << "\n");
      }
    }

    InvalidateVisitorValNOs();
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

    InvalidateVisitorValNOs();

    return true;
  };

  bool Visit(AST::Synchronize& n) {
    TraceEachVisit(n);
    return true;
  }

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
    InvalidateVisitorValNOs();
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
    InvalidateVisitorValNOs();

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::InThreadsBlock& n) {
    TraceEachVisit(n);

    // invalidate any current value generated
    InvalidateVisitorValNOs();

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::IfElseBlock& n) {
    TraceEachVisit(n);

    // invalidate any current value generated
    InvalidateVisitorValNOs();

    if (cannot_proceed) return true;

    return true;
  };

  bool Visit(AST::IncrementBlock& n) {
    TraceEachVisit(n);

    gen_values = true; // allow generate values for statements

    // invalidate any current value generated
    InvalidateVisitorValNOs();

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

  ValueItem GenValueItemFromSignature(const std::string& input) {
    if (auto iv = RemovePrefixOrNull("const_", input)) {
      if (input.find(".") !=
          std::string::npos) // do not handle floating numbers
        return nullptr;
      return sbe::nu(std::stoll(*iv));
    } else if (PrefixedWith(input, "::")) {
      // must be a scoped symbol
      return sbe::sym(input);
    }

    // it is an operation
    auto parts = SplitStringByDelimiter(input, ":");
    if (parts.size() != 3) return nullptr;
    if (PrefixedWith(input, "+:") || PrefixedWith(input, "-:") ||
        PrefixedWith(input, "*:") || PrefixedWith(input, "/:") ||
        PrefixedWith(input, "%:")) {
      auto lvi = GenValueItemFromSignature(
          vn.GetSignatureFromValueNumber(std::stoi(parts[1].substr(1))));
      auto rvi = GenValueItemFromSignature(
          vn.GetSignatureFromValueNumber(std::stoi(parts[2].substr(1))));
      if (lvi && rvi)
        return sbe::bop(ToOpCode(input.substr(0, 1)), lvi, rvi)->Normalize();
    } else if (PrefixedWith(input, "cdiv:")) {
      auto lvi = GenValueItemFromSignature(
          vn.GetSignatureFromValueNumber(std::stoi(parts[1].substr(1))));
      auto rvi = GenValueItemFromSignature(
          vn.GetSignatureFromValueNumber(std::stoi(parts[2].substr(1))));
      if (lvi && rvi)
        return sbe::bop(OpCode::DIVIDE, lvi + (rvi - sbe::nu(1)), rvi)
            ->Normalize();
    }
    return nullptr;
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

      if (auto vi = GenValueItemFromSignature(component)) {
        result.push_back(vi);
      } else {
        // TODO: remove the legacy method totally
        auto expr = GenerateExpression(component);
        int int_val;
        auto [ptr, ec] =
            std::from_chars(expr.data(), expr.data() + expr.size(), int_val);
        if (ec == std::errc() && ptr == expr.data() + expr.size()) {
          result.emplace_back(sbe::nu(int_val));
        } else
          result.emplace_back(sbe::sym(expr));
      }
    }

    return {result.size(), result};
  }

  std::string GenerateExpression(const std::string& sig) {
    if (auto digit = RemovePrefixOrNull("const_", sig)) return *digit;

    // a value number reference
    if (auto digit = RemovePrefixOrNull("#", sig);
        digit && !PrefixedWith(sig, "#:"))
      return GenerateExpression(
          vn.GetSignatureFromValueNumber(std::stoi(*digit)));

    // unary expressions
    if (PrefixedWith(sig, "!:")) {
      return "!" + GenerateExpression(sig.substr(2));
    }
    if (PrefixedWith(sig, "sizeof:")) {
      auto all = vn.GetSignatureFromValueNumber(std::stoi(sig.substr(8)));
      auto parts = SplitStringByDelimiter(all, ",");
      std::string res;
      for (auto& p : parts) {
        if (res != "") res += "*";
        res += "(" + GenerateExpression(p) + ")";
      }
      return res;
    }

    // binary expressions
    if (PrefixedWith(sig, "cdiv:")) {
      auto parts = SplitStringByDelimiter(sig.substr(5), ":");
      assert(parts.size() == 2);
      auto lhs = GenerateExpression(parts[0]);
      auto rhs = GenerateExpression(parts[1]);
      return "((" + lhs + ")+(" + rhs + ")-1)/(" + rhs + ")";
    }
    if (PrefixedWith(sig, "#:")) {
      auto parts = SplitStringByDelimiter(sig.substr(2), ":");
      assert(parts.size() == 2);
      std::string res;
      for (auto& p : parts) {
        if (res != "") res += "*";
        res += "(" + GenerateExpression(p) + ")";
      }
      return res;
    }
    static const std::initializer_list<std::string> prefixes = {
        "+:",  "-:", "*:", "/:",  "%:",  ">=:", "||:",
        "&&:", "<:", ">:", "==:", "!=:", "<=:", ">=:"};
    if (std::any_of(prefixes.begin(), prefixes.end(),
                    [&](const std::string& prefix) {
                      return PrefixedWith(sig, prefix);
                    })) {
      std::istringstream stream(sig);
      std::vector<std::string> parts;
      std::string part;

      while (std::getline(stream, part, ':')) parts.push_back(part);
      assert(parts.size() == 3);
      return "(" + GenerateExpression(parts[1]) + ")" + parts[0] + "(" +
             GenerateExpression(parts[2]) + ")";
    }

    // ternary expressions
    if (PrefixedWith(sig, "?:")) {
      auto parts = SplitStringByDelimiter(sig.substr(2), ":");
      assert(parts.size() == 3);
      std::string res = "(" + GenerateExpression(parts[0]) + ")?" + "(" +
                        GenerateExpression(parts[1]) + "):(" +
                        GenerateExpression(parts[2]) + ")";
      return res;
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

  bool CanBeValueNumbered(AST::Node* n) const {
    if (!n) return true;

    if (auto mv = dyn_cast<AST::MultiValues>(n)) {
      for (auto v : mv->AllValues())
        if (!CanBeValueNumbered(v.get())) return false;
      return true;
    }

    assert(!n->IsBlock() && "do not pass in block node.");

    if (isa<AST::ChunkAt>(n)) return false;
    if (isa<AST::StringLiteral>(n)) return false;
    if (isa<AST::DataAccess>(n)) return false;
    if (!NodeType(*n)) {
      // sometimes the symbol is yet to define, simply make it work.
      return true;
    }
    if (IsMutable(*NodeType(*n))) return false;
    if (isa<EventType>(NodeType(*n))) return false;

    if (auto e = dyn_cast<AST::Expr>(n)) {
      if (e->op == "elemof") return false;
      return CanBeValueNumbered(e->GetR().get()) &&
             CanBeValueNumbered(e->GetL().get()) &&
             CanBeValueNumbered(e->GetC().get());
    }
    return true; // could be id/int/...
  }

  void DefineASymbol(const std::string& name, const ptr<Type>& ty) {
    // assert(!SSTab().IsDeclared(name) && "symbol has been declared.");
    SSTab().DefineSymbol(name, ty);
    if (debug_visit)
      dbgs() << "[symtab] add: " << SSTab().InScopeName(name)
             << ", type: " << PSTR(ty) << "\n";
  }
};

} // end namespace Choreo

#endif // __CHOREO_SHAPE_INFERENCE_HPP__
