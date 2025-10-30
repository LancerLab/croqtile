#ifndef __CHOREO_PIPELINE_HPP__
#define __CHOREO_PIPELINE_HPP__

#include "ast.hpp"
#include "verifier.hpp"
#include "visitor.hpp"
#include <mutex>
#include <unordered_map>

namespace Choreo {

class ASTPipeline;

struct PipelineStage {
  std::unique_ptr<Visitor> v;
  std::function<bool()> pred = {};
  std::function<void(ASTPipeline&)> cond_action = {};
  std::function<void(ASTPipeline&)> action = {};
  PipelineStage(std::unique_ptr<Visitor> visitor) : v(std::move(visitor)) {};
  PipelineStage(std::unique_ptr<Visitor>&& visitor,
                std::function<bool()> p_lambda,
                std::function<void(ASTPipeline&)> post_lambda = {},
                std::function<void(ASTPipeline&)> a_lambda = {})
      : v(std::move(visitor)), pred(p_lambda), cond_action(post_lambda),
        action(a_lambda) {};
};

class ASTPipeline {
private:
  ASTVerify vf;

  std::vector<PipelineStage> pl;

  ptr<SymbolTable> symtab = nullptr;

  bool abend = false;
  bool debug = false;
  int state = 0; // no error

public:
  ASTPipeline() {}

  void Append(PipelineStage&& ps) {
    pl.emplace_back(std::move(ps.v), ps.pred, ps.cond_action, ps.action);
  }

  // This does not require to invoke a visitor
  // user may provide lambda to abort pipeline conditionally
  void AddAction(std::function<void(ASTPipeline&)> action) {
    Append({nullptr, {}, {}, action});
  }

  template <typename VisitorType>
  void AddStage() {
    Append({std::make_unique<VisitorType>()});
  }

  template <typename VTy, typename... VTys>
  void AddStages() {
    AddStage<VTy>();
    if constexpr (sizeof...(VTys) != 0) AddStages<VTys...>();
  }

  template <typename VisitorType>
  void AddStageIf(std::function<bool()> pred) {
    Append({std::make_unique<VisitorType>(), pred});
  }

  template <typename VisitorType>
  void AddStageIfWithPost(std::function<bool()> pred,
                          std::function<void(ASTPipeline&)> post) {
    Append({std::make_unique<VisitorType>(), pred, post});
  }

  template <typename VisitorType>
  void AddStageWithPost(std::function<void(ASTPipeline&)> post) {
    Append({std::make_unique<VisitorType>(), {}, post});
  }

  void SetAbend() { abend = true; }

  int Status() const { return state; }

  const ptr<SymbolTable> LastSymTab() const {
    if (!symtab) choreo_unreachable("unable to find a valid last symtab.");
    return symtab;
  }

  void Dump() const;

  bool RunOnProgram(AST::Node&);

  // contains normal visitors
  ASTPipeline& PlanSemanticRoutine();
  ASTPipeline& PlanCodeGenRoutine();
  ASTPipeline& PlanAllRoutines() {
    return PlanSemanticRoutine().PlanCodeGenRoutine();
  }

private:
  static std::once_flag init_flag;
  static std::unique_ptr<ASTPipeline> instance;

public:
  static ASTPipeline& Get();
}; // ASTPipeline

} // end namespace Choreo

#endif // __CHOREO_PIPELINE_HPP__
