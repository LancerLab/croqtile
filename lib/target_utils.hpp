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
  BaseType a_ty;     // type of lhs multiplicand(A)
  BaseType b_ty;     // type of rhs multiplicand(B)
  BaseType c_ty;     // type of accumulator(C)
  BaseType d_ty;     // type of accumulator(D)
  BaseType scale_ty; // UNKNOWN if NA
  MMAShape shape;

  MMAConfig(Sparsity s, BaseType a_ty, BaseType b_ty, BaseType c_ty,
            BaseType d_ty, BaseType scale_ty, MMAShape mma_shape)
      : sparsity(s), a_ty(a_ty), b_ty(b_ty), c_ty(c_ty), d_ty(d_ty),
        scale_ty(scale_ty), shape(mma_shape) {}

  MMAConfig(Sparsity s, BaseType a_ty, BaseType b_ty, BaseType c_ty,
            BaseType d_ty, BaseType scale_ty, const ValueList& mma_shape)
      : sparsity(s), a_ty(a_ty), b_ty(b_ty), c_ty(c_ty), d_ty(d_ty),
        scale_ty(scale_ty) {
    if (mma_shape.size() != 3)
      choreo_unreachable("unexpected dims size of MMA shape!");
    assert(IsValueListNumeric(mma_shape));
    this->shape = MMALimit::MMAShape{.m = *VIInt(mma_shape[0]),
                                     .n = *VIInt(mma_shape[1]),
                                     .k = *VIInt(mma_shape[2])};
  }

  bool operator<(const MMAConfig& rhs) const {
    return std::tie(sparsity, a_ty, b_ty, scale_ty, c_ty, d_ty, shape) <
           std::tie(rhs.sparsity, rhs.a_ty, rhs.b_ty, rhs.scale_ty, rhs.c_ty,
                    rhs.d_ty, rhs.shape);
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << "MMAConfig(sparsity=" << (sparsity == DENSE ? "DENSE" : "SPARSE")
        << ", a_ty=" << STR(a_ty) << ", b_ty=" << STR(b_ty)
        << ", scale_ty=" << STR(scale_ty) << ", c_ty=" << STR(c_ty)
        << ", d_ty=" << STR(d_ty) << ", shape=(" << shape.m << ", " << shape.n
        << ", " << shape.k << "))";
    return oss.str();
  }

  std::string ToPTXWrappedHeader(const std::string& sep = "_") {
    // example: mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
    std::vector<std::string> strs;
    strs.push_back("mma");
    strs.push_back("sync");
    strs.push_back("aligned");
    strs.push_back("m" + std::to_string(shape.m) + "n" +
                   std::to_string(shape.n) + "k" + std::to_string(shape.k));
    strs.push_back("row");
    strs.push_back("col");
    // TODO: STR is not worked for F8_E4M3...
    strs.push_back(STR(a_ty));
    strs.push_back(STR(b_ty));
    strs.push_back(STR(c_ty));
    strs.push_back(STR(d_ty));
    return DelimitedString(strs, sep);
  }
};

using BT = BaseType;
using CUDA_CC = uint8_t;

static const std::map<MMAConfig, CUDA_CC> wmma_configs = {
    // 16 x 16 x 16
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {16, 16, 16}},
     70},
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F32, BT::UNKNOWN, {16, 16, 16}},
     70},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {16, 16, 16}},
     70},
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {16, 16, 16}}, 72},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {16, 16, 16}}, 72},
    {{DENSE, BT::BF16, BT::BF16, BT::F32, BT::F32, BT::UNKNOWN, {16, 16, 16}},
     80},
    // 32 x 8 x 16
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F32, BT::UNKNOWN, {32, 8, 16}}, 70},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {32, 8, 16}}, 70},
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {32, 8, 16}}, 72},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {32, 8, 16}}, 72},
    {{DENSE, BT::BF16, BT::BF16, BT::F32, BT::F32, BT::UNKNOWN, {32, 8, 16}},
     80},
    // 8 x 32 x 16
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F32, BT::UNKNOWN, {8, 32, 16}}, 70},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {8, 32, 16}}, 70},
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {8, 32, 16}}, 72},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {8, 32, 16}}, 72},
    {{DENSE, BT::BF16, BT::BF16, BT::F32, BT::F32, BT::UNKNOWN, {8, 32, 16}},
     80},
    // 8 x 8 x 32
    {{DENSE, BT::U4, BT::U4, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 32}}, 73},
    {{DENSE, BT::S4, BT::S4, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 32}}, 73},
    // 8 x 8 x 128
    {{DENSE, BT::BIN1, BT::BIN1, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 128}},
     73},
    // 16 x 16 x 8
    {{DENSE, BT::TF32, BT::TF32, BT::F32, BT::F32, BT::UNKNOWN, {16, 16, 8}},
     80},
    // 8 x 8 x 4
    {{DENSE, BT::F64, BT::F64, BT::F64, BT::F64, BT::UNKNOWN, {8, 8, 4}}, 80},
};

static const std::map<MMAConfig, CUDA_CC> mma_configs = {
    // sm70
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {8, 8, 4}}, 70},
    // sm80
    // 16 x 8 x 8
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {16, 8, 8}}, 80},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {16, 8, 8}}, 80},
    {{DENSE, BT::BF16, BT::BF16, BT::F32, BT::F32, BT::UNKNOWN, {16, 8, 8}},
     80},
    {{DENSE, BT::TF32, BT::TF32, BT::F32, BT::F32, BT::UNKNOWN, {16, 8, 8}},
     80},

    // 16 x 8 x 16
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {16, 8, 16}}, 80},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {16, 8, 16}}, 80},
    {{DENSE, BT::BF16, BT::BF16, BT::F32, BT::F32, BT::UNKNOWN, {16, 8, 16}},
     80},

    // 16 x 8 x 4 (TF32)
    {{DENSE, BT::TF32, BT::TF32, BT::F32, BT::F32, BT::UNKNOWN, {16, 8, 4}},
     80},
    // 8 x 8 x 4 (F64)
    {{DENSE, BT::F64, BT::F64, BT::F64, BT::F64, BT::UNKNOWN, {8, 8, 4}}, 80},

    // Integer S8/U8 -> S32 accumulators
    // 8 x 8 x 16
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 16}}, 80},
    {{DENSE, BT::S8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 16}}, 80},
    {{DENSE, BT::U8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 16}}, 80},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 16}}, 80},
    // 16 x 8 x 16
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 16}}, 80},
    {{DENSE, BT::S8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 16}}, 80},
    {{DENSE, BT::U8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 16}}, 80},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 16}}, 80},
    // 16 x 8 x 32, satfinite and non-saturate
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},
    {{DENSE, BT::S8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},
    {{DENSE, BT::U8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},
    // 8 x 8 x 32, 4-bit S4/U4 variants -> S32
    {{DENSE, BT::S4, BT::S4, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 32}}, 80},
    {{DENSE, BT::S4, BT::U4, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 32}}, 80},
    {{DENSE, BT::U4, BT::S4, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 32}}, 80},
    {{DENSE, BT::U4, BT::U4, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 32}}, 80},

    // 16 x 8 x 32
    {{DENSE, BT::S4, BT::S4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},
    {{DENSE, BT::S4, BT::U4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},
    {{DENSE, BT::U4, BT::S4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},
    {{DENSE, BT::U4, BT::U4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 32}}, 80},

    // 16 x 8 x 64
    {{DENSE, BT::S4, BT::S4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 64}}, 80},
    {{DENSE, BT::S4, BT::U4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 64}}, 80},
    {{DENSE, BT::U4, BT::S4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 64}}, 80},
    {{DENSE, BT::U4, BT::U4, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 64}}, 80},

    // Binary (b1) popc -> S32
    // 8 x 8 x 128
    {{DENSE, BT::BIN1, BT::BIN1, BT::S32, BT::S32, BT::UNKNOWN, {8, 8, 128}},
     80},
    // 16 x 8 x 128
    {{DENSE, BT::BIN1, BT::BIN1, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 128}},
     80},
    // 16 x 8 x 256
    {{DENSE, BT::BIN1, BT::BIN1, BT::S32, BT::S32, BT::UNKNOWN, {16, 8, 256}},
     80},

    // sm89 (all 16 x 8 x 32)
    {{DENSE,
      BT::F8_E4M3,
      BT::F8_E4M3,
      BT::F32,
      BT::F32,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},
    {{DENSE,
      BT::F8_E4M3,
      BT::F8_E5M2,
      BT::F32,
      BT::F32,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},
    {{DENSE,
      BT::F8_E5M2,
      BT::F8_E5M2,
      BT::F32,
      BT::F32,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},
    {{DENSE,
      BT::F8_E5M2,
      BT::F8_E4M3,
      BT::F32,
      BT::F32,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},

    {{DENSE,
      BT::F8_E4M3,
      BT::F8_E4M3,
      BT::F16,
      BT::F16,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},
    {{DENSE,
      BT::F8_E4M3,
      BT::F8_E5M2,
      BT::F16,
      BT::F16,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},
    {{DENSE,
      BT::F8_E5M2,
      BT::F8_E4M3,
      BT::F16,
      BT::F16,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},
    {{DENSE,
      BT::F8_E5M2,
      BT::F8_E5M2,
      BT::F16,
      BT::F16,
      BT::UNKNOWN,
      {16, 8, 32}},
     89},

    // sm90 (all for FP64) Complex double is supported now
    // 16 x 8 x 4
    {{DENSE, BT::F64, BT::F64, BT::F64, BT::F64, BT::UNKNOWN, {16, 8, 4}}, 90},
    // 16 x 8 x 8
    {{DENSE, BT::F64, BT::F64, BT::F64, BT::F64, BT::UNKNOWN, {16, 8, 8}}, 90},
    // 16 x 8 x 16
    {{DENSE, BT::F64, BT::F64, BT::F64, BT::F64, BT::UNKNOWN, {16, 8, 16}}, 90},
    // sm100
    // 2 x 1 x 1
    {{DENSE, BT::F32, BT::F32, BT::F32, BT::F32, BT::UNKNOWN, {2, 1, 1}}, 100},
    // 1 x 2 x 1
    {{DENSE, BT::F32, BT::F32, BT::F32, BT::F32, BT::UNKNOWN, {1, 2, 1}}, 100},
    // sm120 todo
};

inline bool ConfigIsWMMA(const MMAConfig& config) {
  return wmma_configs.count(config);
}

} // namespace MMALimit

} // end namespace Choreo

#endif // __CHOREO_TARGET_UTILS_HPP__
