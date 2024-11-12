#ifndef __CHOREO_TYPE_INFERENCE_HPP__
#define __CHOREO_TYPE_INFERENCE_HPP__

#include <iostream>

#include "typeresolve.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct TypeInference : public Visitor {
private:
  bool Dump = false;
  std::ostream& os;
  size_t error_count = 0;

  TypeConstraints type_equals{this};

private:
  ptr<Type> cur_type = nullptr;
  std::vector<ptr<Type>> cur_param_types;
  Shape cur_mdspan_value;
  std::string cur_func_name; // current function name
  BaseType dma_fmty = BaseType::UNKNOWN;
  Storage dma_mem = Storage::NONE;
  bool allow_named_dim = false; // named dimensions (mdspan param only)

  bool BeforeVisit(AST::Node&) override;
  bool AfterVisit(AST::Node&) override;

  bool AssignSymbolWithType(const location&, const std::string&,
                            const ptr<Type>&);
  ptr<Type> GetSymbolType(const location&, const std::string&);
  bool ModifySymbolType(const location&, const std::string&, const ptr<Type>&);

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) {
      os << n.TypeNameString() << ": ";
      os << "\n";
    }
  }

public:
  TypeInference(bool d, std::ostream& o = std::cout,
                const ptr<SymbolTable> s_tab = std::make_shared<SymbolTable>())
      : Visitor("infer", s_tab), Dump(d), os(o) {
    type_equals.SetTypeReport(true);
  }

  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::Boolean&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::MultiDimSpans&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::IntTuple&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::IntIndex&) override;
  bool Visit(AST::DataType&) override;
  bool Visit(AST::Identifier&) override;
  bool Visit(AST::Parameter&) override;
  bool Visit(AST::ParamList&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::Memory&) override;
  bool Visit(AST::SpanAs&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::Program&) override;
  bool HasError() {
    if (error_count)
      os << "Totally " << error_count << " errors have been detected.\n";
    return error_count != 0;
  }

private:
  bool SetAsCurrentType(AST::Node&, const std::string&);
};

} // end namespace Choreo

#endif // __CHOREO_TYPE_INFERENCE_HPP__
