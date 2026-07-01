#ifndef CHOREO_MOCK_INTERP_HPP_
#define CHOREO_MOCK_INTERP_HPP_

#include "mock_memory.hpp"
#include "visitor.hpp"

#include <functional>
#include <mutex>

namespace Choreo {
namespace Mock {

class Debugger;

struct ControlFlow {
  enum Kind { None, Return, Break, Continue };
  Kind kind = None;
  Value return_value;
};

class MockInterpreter : public VisitorWithSymTab {
public:
  MockInterpreter();
  ~MockInterpreter() override = default;

  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }

  bool RunOnProgramImpl(AST::Node& root) override;

  Value ExprEval(const ptr<AST::Node>& e);

  void SetDebugger(Debugger* dbg) { debugger_ = dbg; }
  Debugger* GetDebugger() const { return debugger_; }

  MockMemory& GetMemory() { return mem; }

private:
  void ExecBlock(const ptr<AST::MultiNodes>& stmts);
  void ExecStatement(const ptr<AST::Node>& stmt);

  void ExecParallelBy(AST::ParallelBy& n);
  void ExecAssignment(AST::Assignment& n);
  void ExecNamedVariableDecl(AST::NamedVariableDecl& n);
  void ExecIfElse(AST::IfElseBlock& n);
  void ExecWhile(AST::WhileBlock& n,
                 const std::vector<AST::Assignment*>& pred_refreshes = {});
  void ExecForeach(AST::ForeachBlock& n);
  void ExecCall(AST::Call& n);
  void ExecReturn(AST::Return& n);
  void ExecDMA(AST::DMA& n);
  void ExecWait(AST::Wait& n);
  void ExecRotate(AST::Rotate& n);
  void ExecInThreads(AST::InThreadsBlock& n);
  void ExecMMA(AST::MMA& n);
  void ExecApply(AST::ApplyBlock& n);
  void ExecFragReduce(AST::FragReduce& n);
  void ExecFragTransfer(AST::FragTransfer& n);

  Value EvalBinaryOp(Opcode op, const Value& lhs, const Value& rhs);
  Value EvalUnaryOp(Opcode op, const Value& operand);

  bool IsKnownBIF(const std::string& name) const;
  Value CallBIF(const std::string& name, const std::vector<Value>& args,
                const AST::Call& node);

  BaseType ResolveBaseType(const ptr<Type>& ty) const;
  std::vector<size_t> ResolveShape(const ptr<Type>& ty) const;
  Storage ResolveStorage(const ptr<Type>& ty) const;

  size_t ComputeLinearIndex(const std::vector<size_t>& indices,
                            const std::vector<size_t>& shape) const;

  Value CastValue(const Value& v, BaseType target_type) const;

  void ExecParallelByBody(AST::ParallelBy& n, int64_t i, int64_t bound,
                          const std::string& pv_name, bool has_sub,
                          const std::vector<std::string>& sub_names,
                          const std::vector<int64_t>& sub_bounds);

  MockMemory mem;
  ControlFlow cf;
  std::map<std::string, AST::ChoreoFunction*> functions;
  Debugger* debugger_ = nullptr;
  bool quit_requested_ = false;
  std::mutex* print_mutex_ = nullptr;
};

} // namespace Mock
} // namespace Choreo

#endif // CHOREO_MOCK_INTERP_HPP_
