#ifndef __CHOREO_TARGET_UTILS_HPP__
#define __CHOREO_TARGET_UTILS_HPP__

#include "types.hpp"
#include <mutex>
#include <thread>

namespace Choreo {

// parallel level <-> parallel depth mapping
class PlDepthMap {
private:
  std::unordered_map<int, ParallelLevel>* to_levels = nullptr;
  std::unordered_map<ParallelLevel, int>* to_depths = nullptr;

  int max_depth = -1;
  ParallelLevel max_level = ParallelLevel::UNKNOWN;

public:
  PlDepthMap();

  ParallelLevel ToLevel(int depth) const {
    if (depth == -1) return ParallelLevel::NONE;
    if (!to_levels->count(depth)) return ParallelLevel::UNKNOWN;
    return (*to_levels)[depth];
  }

  int ToDepth(ParallelLevel pl) const {
    if (pl == ParallelLevel::NONE) return -1;
    if (pl == ParallelLevel::UNKNOWN)
      choreo_unreachable(
          "can not get the parallel depth of an unknown parallel level");
    if (!to_depths->count(pl))
      choreo_unreachable("unsupported parallel level.");
    return (*to_depths)[pl];
  }

  bool HasLevel(ParallelLevel pl) const { return to_depths->count(pl); }

  ParallelLevel MaxLevel() const { return max_level; }
  int MaxDepth() const { return max_depth; }

public:
  static const PlDepthMap& Get();
  static std::once_flag init_flag;
  static std::unique_ptr<PlDepthMap> instance;
};

inline int TargetDepth(ParallelLevel pl) {
  if (CCtx().GetTarget() == CompileTarget::Topscc ||
      CCtx().GetTarget() == CompileTarget::Cute ||
      CCtx().GetTarget() == CompileTarget::MPI)
    return PlDepthMap::Get().ToDepth(pl);
  else
    choreo_unreachable("unsupported target: " + STR(CCtx().GetTarget()) + ".");
  return -1;
}

inline ParallelLevel TargetLevel(int depth) {
  if (CCtx().GetTarget() == CompileTarget::Topscc ||
      CCtx().GetTarget() == CompileTarget::Cute)
    return PlDepthMap::Get().ToLevel(depth);
  else
    choreo_unreachable("unsupported target: " + STR(CCtx().GetTarget()) + ".");
  return ParallelLevel::NONE;
}

inline int TargetMaxDepth() { return PlDepthMap::Get().MaxDepth(); }

inline bool TargetHasLevel(ParallelLevel pl) {
  return PlDepthMap::Get().HasLevel(pl);
}

inline ParallelLevel TargetMaxLevel() { return PlDepthMap::Get().MaxLevel(); }

// return the inner level
inline ParallelLevel operator++(ParallelLevel& pl) {
  auto& pld = PlDepthMap::Get();
  auto l = pld.ToLevel(pld.ToDepth(pl) + 1);
  switch (l) {
  case ParallelLevel::UNKNOWN:
    choreo_unreachable("no higher parallel level exists for an unknown level.");
    break;
  case ParallelLevel::NONE:
    choreo_unreachable("no higher parallel level exists for an none level.");
    break;
  default: pl = l; return l;
  }
  return ParallelLevel::UNKNOWN;
}

inline ParallelLevel operator--(ParallelLevel& pl) {
  auto& pld = PlDepthMap::Get();
  auto l = pld.ToLevel(pld.ToDepth(pl) - 1);
  switch (l) {
  case ParallelLevel::UNKNOWN:
    choreo_unreachable("no higher parallel level exists for an unknown level.");
    break;
  case ParallelLevel::NONE:
    choreo_unreachable("no higher parallel level exists for an none level.");
    break;
  default: pl = l; return l;
  }
  return ParallelLevel::UNKNOWN;
}

inline int operator-(ParallelLevel lhs, ParallelLevel rhs) {
  auto& pld = PlDepthMap::Get();
  return pld.ToDepth(lhs) - pld.ToDepth(rhs);
}

// MMA related static limitation
namespace MMALimit {

enum Sparsity { DENSE, SPARSE };
struct MMAShape {
  int64_t m, n, k;
  bool operator<(const MMAShape& rhs) const {
    return std::tie(m, n, k) < std::tie(rhs.m, rhs.n, rhs.k);
  }
};
struct MMAConfig {
  Sparsity sparsity;
  BaseType mul_ty;   // type of Multiplicands(A and B)
  BaseType scale_ty; // UNKNOWN if NA
  BaseType acc_ty;   // type of Accumulators(C and D)
  MMAShape shape;

  bool operator<(const MMAConfig& rhs) const {
    return std::tie(sparsity, mul_ty, scale_ty, acc_ty, shape) <
           std::tie(rhs.sparsity, rhs.mul_ty, rhs.scale_ty, rhs.acc_ty,
                    rhs.shape);
  }
};

using BT = BaseType;
using PTX_ISA_VER = uint8_t;

static const std::map<MMAConfig, PTX_ISA_VER> mma_configs = {
    {{DENSE, BT::BF16, BT::UNKNOWN, BT::F32, {16, 16, 16}}, 60},
    {{DENSE, BT::F16, BT::UNKNOWN, BT::F16, {16, 16, 16}}, 60},
    {{DENSE, BT::F16, BT::UNKNOWN, BT::F32, {16, 16, 16}}, 60},
    // {{DENSE, BT::F16, BT::UNKNOWN, BT::F16, {8, 32, 16}}, 60},
    // {{DENSE, BT::F16, BT::UNKNOWN, BT::F16, {32, 8, 16}}, 60},
    // // TODO: accumulator could be f32, what's the ISA version?
    // {{DENSE, BT::BF16, BT::UNKNOWN, BT::BF16, {16, 16, 16}}, 70},
    // {{DENSE, BT::BF16, BT::UNKNOWN, BT::BF16, {8, 32, 16}}, 70},
    // {{DENSE, BT::BF16, BT::UNKNOWN, BT::BF16, {32, 8, 16}}, 70},
};

} // namespace MMALimit

} // end namespace Choreo

#endif // __CHOREO_TARGET_UTILS_HPP__
