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
                      const std::string& m = "") const;

  void InvalidateVisitorValNOs();

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
  virtual bool BeforeVisitImpl(AST::Node& n) override;
  virtual bool InMidVisitImpl(AST::Node& n) override;
  virtual bool AfterVisitImpl(AST::Node& n) override;

public:
  // enable NodeType to retrieve a scoped name
  ptr<Type> GetSymbolType(const std::string& n) const override {
    return SSTab().LookupSymbol(n);
  }

  ptr<Type> NodeType(const AST::Node& n) const override;

  void SetNodeType(AST::Node& n, const ptr<Type>& ty, bool is_mutable = false) {
    if (is_mutable && MutableType(*ty))
      n.SetType(MutateType(*ty));
    else
      n.SetType(ty);
    if (debug_visit)
      dbgs() << "Set type of " << STR(n) << " as " << PSTR(n.GetType()) << "\n";
  }

  void SetMdsShape(AST::MultiDimSpans& n, const Shape& s) {
    n.SetTypeDetail(s);
    if (debug_visit)
      dbgs() << "Set shape of " << STR(n) << " as " << STR(s) << "\n";
  }

public:
  bool Visit(AST::MultiNodes& n) override;
  bool Visit(AST::MultiValues& n) override;
  bool Visit(AST::IntLiteral& n) override;
  bool Visit(AST::FloatLiteral& n) override;
  bool Visit(AST::StringLiteral& n) override;
  bool Visit(AST::Boolean& n) override;
  bool Visit(AST::Expr& n) override;
  bool Visit(AST::MultiDimSpans& n) override;
  bool Visit(AST::NamedTypeDecl& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::IntTuple& n) override;
  bool Visit(AST::DataAccess& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::IntIndex& n) override;
  bool Visit(AST::DataType& n) override;
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
  std::vector<int> Collapse(const ptr<AST::MultiValues>&, bool = false);
  std::string GenerateExpression(const std::string& sig);
  int GetOnlyValueNumberFromMultiValues(const AST::MultiValues& mv);
  void UpdateValueNumberForMultiValues(const AST::MultiValues& mv, int valno);
  bool CanBeValueNumbered(AST::Node* n) const;
  void DefineASymbol(const std::string& name, const ptr<Type>& ty);
  Shape GenShapeFromSignature(const std::string&);
}; // class ShapeInference

} // end namespace Choreo

#endif // __CHOREO_SHAPE_INFERENCE_HPP__
