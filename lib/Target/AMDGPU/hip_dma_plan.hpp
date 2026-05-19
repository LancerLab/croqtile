#ifndef __CHOREO_HIP_DMA_PLAN_HPP__
#define __CHOREO_HIP_DMA_PLAN_HPP__

#include "ast.hpp"
#include "codegen.hpp"
#include "dmaconf.hpp"
#include "types.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace Choreo {
namespace HIP {

enum class HIPDMAStrategy {
  UNKNOWN,
  NAIVE_COPY,
  TILED_COPY,
};

enum class HIPDMADirection {
  UNKNOWN,
  G2S,
  S2G,
  G2L,
  L2G,
  S2S,
  OTHER,
};

inline const std::string STR(HIPDMAStrategy s) {
  switch (s) {
  case HIPDMAStrategy::UNKNOWN: return "unknown";
  case HIPDMAStrategy::NAIVE_COPY: return "naive_copy";
  case HIPDMAStrategy::TILED_COPY: return "tiled_copy";
  }
  return "?";
}

inline const std::string STR(HIPDMADirection d) {
  switch (d) {
  case HIPDMADirection::UNKNOWN: return "unknown";
  case HIPDMADirection::G2S: return "g2s";
  case HIPDMADirection::S2G: return "s2g";
  case HIPDMADirection::G2L: return "g2l";
  case HIPDMADirection::L2G: return "l2g";
  case HIPDMADirection::S2S: return "s2s";
  case HIPDMADirection::OTHER: return "other";
  }
  return "?";
}

struct HIPDMALoweringDecision {
  HIPDMAStrategy strategy = HIPDMAStrategy::UNKNOWN;
  HIPDMADirection direction = HIPDMADirection::UNKNOWN;

  std::string operation;
  bool is_async = false;
  bool has_future = false;

  BaseType elem_type = BaseType::UNKNOWN;
  int rank = -1;

  Shape from_ca_shape;
  Shape to_ca_shape;
  Shape from_parent_shape;
  Shape to_parent_shape;

  std::vector<size_t> transpose_perm;

  bool IsResolved() const { return strategy != HIPDMAStrategy::UNKNOWN; }
  bool IsNaive() const { return strategy == HIPDMAStrategy::NAIVE_COPY; }
  bool IsTiled() const { return strategy == HIPDMAStrategy::TILED_COPY; }
  bool IsCopy() const { return operation == ".copy"; }
  bool IsPad() const { return operation == ".pad"; }
  bool IsTranspose() const { return operation == ".transp"; }
};

struct HIPDMAPlan : public CodeGenerator {
  HIPDMAPlan() : CodeGenerator("hip-dma-plan") {}
  ~HIPDMAPlan() override = default;

  bool Visit(AST::DMA& n) override;

  static const HIPDMALoweringDecision* Lookup(const AST::DMA* n) {
    auto it = decisions().find(n);
    return it != decisions().end() ? &it->second : nullptr;
  }

  static void Reset() { decisions().clear(); }

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::Program>(&n)) Reset();
    return true;
  }
  bool AfterVisitImpl(AST::Node&) override { return true; }

  HIPDMADirection ResolveDirection(const AST::DMA& n) const;

  static std::unordered_map<const AST::DMA*, HIPDMALoweringDecision>&
  decisions() {
    static std::unordered_map<const AST::DMA*, HIPDMALoweringDecision> store;
    return store;
  }
};

} // namespace HIP
} // namespace Choreo

#endif // __CHOREO_HIP_DMA_PLAN_HPP__
