#ifndef __CHOREO_SHAPE_INFERENCE_HPP__
#define __CHOREO_SHAPE_INFERENCE_HPP__

#include "symvals.hpp"
#include "valno.hpp"

namespace Choreo {

using namespace Choreo::valno;

// An AST node may have multiple value numbers
enum class VNKind {
  VNK_VALUE,  // valno of the current symbol, like 'a'
  VNK_UBOUND, // valno of the implied upper-bound, like '#a'
  VNK_MDSPAN, // valno of the implied mdspan, like 'a.span'
};

inline const std::string STR(VNKind vnt) {
  switch (vnt) {
  case VNKind::VNK_VALUE: return "value"; break;
  case VNKind::VNK_UBOUND: return "ubound"; break;
  case VNKind::VNK_MDSPAN: return "mdspan"; break;
  default: choreo_unreachable("unsupported valno kind."); break;
  }
  return "";
}

// value number for each AST node
class NodeValNo {
private:
  std::vector<std::unordered_map<const AST::Node*, std::map<VNKind, NumTy>>>
      node_valno; // cache to direct map node to value number

  bool debug = false;

public:
  void EnterScope() { node_valno.push_back({}); }
  void LeaveScope() { node_valno.pop_back(); }

public:
  NodeValNo(bool d = false) : debug(d) {}

public:
  bool Hit(const AST::Node* node, VNKind vnt = VNKind::VNK_VALUE) const {
    for (auto nv = node_valno.rbegin(); nv != node_valno.rend(); nv++) {
      if (!nv->count(node)) continue;
      if (!(nv->at(node).count(vnt))) continue;
      return true;
    }
    return false;
  }

  std::optional<NumTy> GetOrNull(const AST::Node* node,
                                 VNKind vnt = VNKind::VNK_VALUE) const {
    for (auto nv = node_valno.rbegin(); nv != node_valno.rend(); nv++) {
      if (!nv->count(node)) continue;
      if (!(nv->at(node).count(vnt))) continue;
      return nv->at(node).at(vnt);
    }
    return std::nullopt;
  }

  NumTy Get(const AST::Node* node, VNKind vnt = VNKind::VNK_VALUE) const {
    auto v = GetOrNull(node, vnt);
    if (v.has_value())
      return v.value();
    else
      choreo_unreachable("can not find valno of node: " + PSTR(node) + ".");
  }

  void Update(const AST::Node* node, NumTy vn, VNKind vnt = VNKind::VNK_VALUE) {
    for (auto nv = node_valno.rbegin(); nv != node_valno.rend(); nv++) {
      if (!nv->count(node)) continue;
      if (!(*nv)[node].count(vnt)) continue;
      (*nv)[node][vnt] = vn;
    }
    assert(!node_valno.empty() && "empty node value number map.");
    node_valno.back()[node][vnt] = vn;

    if (debug)
      dbgs() << " |-<node-valno> update [" << node->TypeNameString() << "] "
             << PSTR(node) << " - " << STR(vnt) << ": " << STR(vn) << "\n";
  }

  void Copy(const AST::Node* fn, const AST::Node* tn) {
    if (auto v = GetOrNull(fn, VNKind::VNK_MDSPAN)) {
      Update(tn, *v, VNKind::VNK_MDSPAN);
      if (debug)
        dbgs() << " |-<node-valno> copy (mdspan): '" << PSTR(fn) << "' -> '"
               << PSTR(tn) << "': " << STR(*v) << "\n";
    }
    if (auto v = GetOrNull(fn, VNKind::VNK_UBOUND)) {
      Update(tn, *v, VNKind::VNK_UBOUND);
      if (debug)
        dbgs() << " |-<node-valno> copy (ubound): '" << PSTR(fn) << "' -> '"
               << PSTR(tn) << "': " << STR(*v) << "\n";
    }
    if (auto v = GetOrNull(fn, VNKind::VNK_VALUE)) {
      Update(tn, *v, VNKind::VNK_VALUE);
      if (debug)
        dbgs() << " |-<node-valno> copy (value): '" << PSTR(fn) << "' -> '"
               << PSTR(tn) << "': " << STR(*v) << "\n";
    }
  }
  void Copy(const AST::Node* fn, const AST::Node* tn, VNKind vnt) {
    if (auto v = GetOrNull(fn, vnt)) {
      Update(tn, *v, vnt);
      if (debug)
        dbgs() << " |-<node-valno> copy (" << STR(vnt) << "): '" << PSTR(fn)
               << "' -> '" << PSTR(tn) << "': " << STR(*v) << "\n";
    } else
      choreo_unreachable("failed to copy " + STR(vnt) + ".");
  }
}; // NodeValNo

class ShapeInference : public VisitorWithScope {
private:
  valno::ValueNumbering vn;

  // valno rendered from current ast node
  NumTy cur_vn = GetInvalidValueNumber();

  // implicit valno of spanned-type with ".span" annotation
  NumTy cur_mdspan_vn = GetInvalidValueNumber();

  // implicit valno of upper-bound
  NumTy cur_ub_vn = GetInvalidValueNumber();

  // when values are consumed instead of generated
  bool gen_values = true;

  bool allow_named_dim = false; // named dimension (mdspan param only)

  TypeConstraints type_equals{this};

  NodeValNo ast_vn;

  OptimizedValues& SymVal(const std::string sym) {
    return FCtx(fname).GetSymbolValues(sym);
  }

private:
  // for debugging purpose only
  bool cannot_proceed = false;

  void TraceEachVisit(AST::Node& n, bool detail = false,
                      const std::string& m = "") const;

  void InvalidateVisitorValNOs();

  // Generate the signature for a node, simplify the signature when optimiz flag
  // is set.
  const SignTy SignNode(const AST::Node& node);

  const SignTy SignSpan(const AST::Node& node);

  std::pair<const SignTy, const SignTy> SignBounded(const AST::Node&);

  // Directly get the value number. Abort when it fails.
  NumTy GetValNo(const AST::Node&, VNKind vnt = VNKind::VNK_VALUE) const;

  // Generate the new value number. Abort when the value number exists.
  NumTy GenValNo(const AST::Node&);

  // Check if the value number exists for the node
  bool HasValNo(const AST::Node&, VNKind vnt = VNKind::VNK_VALUE) const;

  // Symbol names related to the value numbering
  const SignTy VNSymbolName(const AST::Identifier&) const;

  const SignTy GetSign(const AST::Node& n,
                       VNKind vnt = VNKind::VNK_VALUE) const {
    return SignValNo(ast_vn.Get(&n, vnt));
  }

private:
  // short-hands
  const SignTy SignValNo(NumTy valno) const {
    return vn.GetSignatureFromValueNumber(valno);
  }
  NumTy ValNoSign(const SignTy& sign) const {
    return vn.GetValueNumberOfSignature(sign);
  }

  void ValNoAliasSign(const SignTy& sign, NumTy valno) {
    vn.AssociateSignatureWithValueNumber(sign, valno);
  }

public:
  ShapeInference() : VisitorWithScope("valno"), vn(this), ast_vn(debug_visit) {
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
  virtual bool BeforeVisitImpl(AST::Node& n) override;
  virtual bool InMidVisitImpl(AST::Node& n) override;
  virtual bool AfterVisitImpl(AST::Node& n) override;

public:
  // enable NodeType to retrieve a scoped name
  ptr<Type> GetSymbolType(const std::string& n) const override {
    return SSTab().LookupSymbol(n);
  }

  ptr<Type> NodeType(const AST::Node& n) const override;

  void SetNodeType(AST::Node& n, const ptr<Type>& ty) {
    n.SetType(ty);
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      VST_DEBUG(dbgs() << " |-<type> set: [" << pb->TypeNameString() << "]";
                pb->PrintWithoutStmts(dbgs(), "   ");
                dbgs() << "\n   |-> " << PSTR(n.GetType()) << "\n");
    } else {
      VST_DEBUG(dbgs() << " |-<type> set: [" << n.TypeNameString() << "]";
                n.Print(dbgs(), "   ");
                dbgs() << "   |-> " << PSTR(n.GetType()) << "\n");
    }
  }

  void SetMdsShape(AST::MultiDimSpans& n, const Shape& s) {
    n.SetTypeDetail(s);
    VST_DEBUG(dbgs() << " |-<shape> set: [" << n.TypeNameString() << "] "
                     << STR(n) << " |-> " << STR(s) << "\n");
  }

  VNKind NodeValNoKind(const AST::Node& n) {
    // multivalues may not be typed
    if (isa<AST::MultiValues>(&n)) return VNKind::VNK_VALUE;
    auto nty = NodeType(n);
    if (!nty || isa<UnknownType>(nty))
      choreo_unreachable("failed to get the type of " + STR(n) + ": " +
                         n.TypeNameString());
    if (isa<BoundedType>(nty))
      return VNKind::VNK_UBOUND;
    else if (isa<SpannedType>(nty) || GeneralFutureType(nty))
      return VNKind::VNK_MDSPAN;
    else
      return VNKind::VNK_VALUE;
  }

public:
  bool Visit(AST::MultiNodes& n) override;
  bool Visit(AST::MultiValues& n) override;
  bool Visit(AST::IntLiteral& n) override;
  bool Visit(AST::FloatLiteral& n) override;
  bool Visit(AST::StringLiteral& n) override;
  bool Visit(AST::BoolLiteral& n) override;
  bool Visit(AST::Expr& n) override;
  bool Visit(AST::CastExpr& n) override;
  bool Visit(AST::MultiDimSpans& n) override;
  bool Visit(AST::NamedTypeDecl& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::IntTuple& n) override;
  bool Visit(AST::DataAccess& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::IntIndex& n) override;
  bool Visit(AST::DataType& n) override;
  bool Visit(AST::NoValue& n) override;
  bool Visit(AST::Identifier& n) override;
  bool Visit(AST::Parameter& n) override;
  bool Visit(AST::ParamList& n) override;
  bool Visit(AST::ParallelBy& n) override;
  bool Visit(AST::WhereBind& n) override;
  bool Visit(AST::WithIn& n) override;
  bool Visit(AST::WithBlock& n) override;
  bool Visit(AST::Memory& n) override;
  bool Visit(AST::SpanAs& n) override;
  bool Visit(AST::DMA& n) override;
  bool Visit(AST::ChunkAt& n) override;
  bool Visit(AST::Wait& n) override;
  bool Visit(AST::Trigger& n) override;
  bool Visit(AST::Call& n) override;
  bool Visit(AST::Rotate& n) override;
  bool Visit(AST::Synchronize& n) override;
  bool Visit(AST::Select& n) override;
  bool Visit(AST::Return& n) override;
  bool Visit(AST::LoopRange& n) override;
  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::InThreadsBlock& n) override;
  bool Visit(AST::IfElseBlock& n) override;
  bool Visit(AST::IncrementBlock& n) override;
  bool Visit(AST::FunctionDecl& n) override;
  bool Visit(AST::ChoreoFunction& n) override;
  bool Visit(AST::CppSourceCode& n) override;
  bool Visit(AST::Program& n) override;

private:
  void CollapseMultiValues(const AST::MultiValues&);
  const std::optional<std::string> GenerateExpression(const SignTy&) const;
  NumTy GetOnlyValueNumberFromMultiValues(const AST::MultiValues&);
  void UpdateValueNumberForMultiValues(const AST::MultiValues&, const NumTy&);
  bool CanBeValueNumbered(AST::Node* n) const;
  void DefineASymbol(const SignTy& name, const ptr<Type>& ty);
  Shape GenShapeFromSignature(const SignTy&, const AST::Node&);
}; // class ShapeInference

} // end namespace Choreo

#endif // __CHOREO_SHAPE_INFERENCE_HPP__
