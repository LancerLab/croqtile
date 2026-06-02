#ifndef __CHOREO_FRAGMENT_LAYOUT_PASS_HPP__
#define __CHOREO_FRAGMENT_LAYOUT_PASS_HPP__

/// FragmentLayoutPass -- infer and assign physical register layouts for
/// fragment declarations based on MMA usage and scope context.
///
/// This pass walks each function body and records:
///   - Fragment declarations (shape, element type, declaring scope)
///   - MMA operations (which fragments serve as A, B, C operands)
///
/// At function exit it applies layout inference rules:
///   1. MMA Anchor:   C operand of mma.exec -> WGMMA_ACC or CTMMA_ACC
///   2. RS Operand:   A operand in RS mode  -> WGMMA_RS_A
///   3. Propagation:  element-wise connected to anchored fragment -> inherit
///   4. Reduction:    target of __reduce from anchored accumulator ->
///   REPLICATED_1D
///   5. Fallback:     everything else -> UNIFORM
///
/// Results are stored in FunctionContext::fragment_layouts and consumed by
/// CuteCodeGen for register allocation and index codegen.
///
/// Pipeline position: after GPUAdaptor, before CuteCodeGen.
/// Dependency: ActiveThreadsAnalysis must have run (reads InThreadsBlock data).

#include "ast.hpp"
#include "codegen.hpp"
#include "fragment_layout.hpp"

#include <map>
#include <string>
#include <vector>

namespace Choreo {

struct FragmentLayoutPass : public CodeGenerator {
  FragmentLayoutPass() : CodeGenerator("frag-layout") {}
  ~FragmentLayoutPass() override = default;

  // Visitor interface -- only override nodes we care about.
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::MMA& n) override;

private:
  // Track parallel scope thread count during traversal.
  size_t current_thread_count_ = 0;
  std::string current_thread_count_expr_;
  AST::InThreadsBlock* current_inthreads_ = nullptr;

  size_t EffectiveThreadCount() const;
  std::string EffectiveThreadCountExpr() const;

  // Collected information per function (cleared at function entry).
  struct FragmentUsage {
    std::string scoped_name;
    bool is_mma_acc = false;
    bool is_mma_operand_a = false;
    bool is_mma_operand_b = false;
    std::string mma_acc_source;
    std::vector<size_t> shape;
    BaseType element_type = BaseType::F32;
    size_t scope_thread_count = 0;
    std::string scope_thread_count_expr;
  };
  std::map<std::string, FragmentUsage> usages_;

  void AssignLayouts();

  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node& n) override;
};

} // namespace Choreo

#endif // __CHOREO_FRAGMENT_LAYOUT_PASS_HPP__
