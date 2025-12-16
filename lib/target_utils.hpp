#ifndef __CHOREO_TARGET_UTILS_HPP__
#define __CHOREO_TARGET_UTILS_HPP__

#include "types.hpp"
#include <mutex>

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
};

using BT = BaseType;
using CUDA_CC = uint8_t;

static const std::map<MMAConfig, CUDA_CC> wmma_configs = {
    // 16 x 16 x 16
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {16, 16, 16}},
     70},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {16, 16, 16}},
     70},
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {16, 16, 16}}, 72},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {16, 16, 16}}, 72},
    {{DENSE, BT::BF16, BT::BF16, BT::F32, BT::F32, BT::UNKNOWN, {16, 16, 16}},
     80},
    // 32 x 8 x 16
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {32, 8, 16}}, 70},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {32, 8, 16}}, 70},
    {{DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN, {32, 8, 16}}, 72},
    {{DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN, {32, 8, 16}}, 72},
    {{DENSE, BT::BF16, BT::BF16, BT::F32, BT::F32, BT::UNKNOWN, {32, 8, 16}},
     80},
    // 8 x 32 x 16
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {8, 32, 16}}, 70},
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
    // WGMMA configs (SM90+, 128-thread warp groups)
    // 64 x 64 x 16 (K=16 only for F16→F16, per GMMA spec)
    {{DENSE, BT::F16, BT::F16, BT::F16, BT::F16, BT::UNKNOWN, {64, 64, 16}},
     90},
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {64, 64, 16}},
     90},
};

static const std::map<MMAConfig, CUDA_CC> cute_mma_configs = {
    // sm70
    {{DENSE, BT::F16, BT::F16, BT::F32, BT::F32, BT::UNKNOWN, {8, 8, 4}}, 70},
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

// WGMMA configs
// m is fixed to 64, k * sizeof(ty) == 32 bytes
static std::map<MMAConfig, CUDA_CC> GenerateWGMMAConfigs() {
  std::map<MMAConfig, CUDA_CC> out;
  // floating point types
  int m = 64, k = 16;
  for (int n = 8; n <= 256; n += 8) {
    for (auto cd_ty : {BT::F16, BT::F32}) {
      MMAConfig config(DENSE, BT::F16, BT::F16, cd_ty, cd_ty, BT::UNKNOWN,
                       MMAShape{m, n, k});
      out[config] = 90;
    }
    for (auto cd_ty : {BT::F32}) {
      MMAConfig config(DENSE, BT::BF16, BT::BF16, cd_ty, cd_ty, BT::UNKNOWN,
                       MMAShape{m, n, k});
      out[config] = 90;
    }
  }

  m = 64, k = 8;
  for (int n = 16; n <= 256; n += 8) {
    MMAConfig config(DENSE, BT::TF32, BT::TF32, BT::F32, BT::F32, BT::UNKNOWN,
                     MMAShape{m, n, k});
    out[config] = 90;
  }

  m = 64, k = 32;
  for (int n = 8; n <= 128; n += 8)
    for (auto a_ty : {BT::F8_E4M3, BT::F8_E5M2})
      for (auto b_ty : {BT::F8_E4M3, BT::F8_E5M2})
        for (auto cd_ty : {BT::F16, BT::F32}) {
          MMAConfig config(DENSE, a_ty, b_ty, cd_ty, cd_ty, BT::UNKNOWN,
                           MMAShape{m, n, k});
          out[config] = 90;
        }
  // integer types
  m = 64, k = 32;
  for (int n : {8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192,
                208, 224}) {
    MMAConfig config(DENSE, BT::S8, BT::S8, BT::S32, BT::S32, BT::UNKNOWN,
                     MMAShape{m, n, k});
    out[config] = 90;
    MMAConfig config_u(DENSE, BT::U8, BT::U8, BT::S32, BT::S32, BT::UNKNOWN,
                       MMAShape{m, n, k});
    out[config_u] = 90;
  }
  // binary types
  m = 64, k = 256;
  for (int n : {8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192,
                208, 224, 240, 256}) {
    MMAConfig config(DENSE, BT::BIN1, BT::BIN1, BT::S32, BT::S32, BT::UNKNOWN,
                     MMAShape{m, n, k});
    out[config] = 90;
  }
  return out;
}

class MMAConfigRegistry {
public:
  static MMAConfigRegistry& Get() {
    static MMAConfigRegistry instance;
    return instance;
  }

  const std::map<MMAConfig, CUDA_CC>& GetCuteMMAConfigs() const {
    return cute_mma_configs;
  }

  const std::map<MMAConfig, CUDA_CC>& GetWMMAConfigs() const {
    return wmma_configs;
  }

  const std::map<MMAConfig, CUDA_CC>& GetWGMMMAConfigs() const {
    return wgmma_configs_;
  }

  bool IsWMMAConfig(const MMAConfig& config) const {
    return wmma_configs.count(config);
  }

  bool IsCuteMMAConfig(const MMAConfig& config) const {
    return cute_mma_configs.count(config);
  }

  bool IsWGMMMAConfig(const MMAConfig& config) const {
    return wgmma_configs_.count(config);
  }

  bool IsValidMMAConfig(const MMAConfig& config, MMALimit::CUDA_CC cc) const {
    if (IsWMMAConfig(config))
      return wmma_configs.at(config) <= cc;
    else if (IsWGMMMAConfig(config))
      return wgmma_configs_.at(config) <= cc;
    else if (IsCuteMMAConfig(config))
      return cute_mma_configs.at(config) <= cc;
    else
      return false;
  }

  MMAType GetMMAType(const MMAConfig& config) const {
    if (IsWMMAConfig(config))
      return MMAType::WMMA;
    else if (IsWGMMMAConfig(config))
      return MMAType::WGMMA;
    else if (IsCuteMMAConfig(config))
      return MMAType::CTMMA;
    else
      choreo_unreachable("unsupported MMA config: " + config.ToString());
    return MMAType::WMMA;
  }

private:
  std::map<MMAConfig, CUDA_CC> wgmma_configs_;
  MMAConfigRegistry() { wgmma_configs_ = GenerateWGMMAConfigs(); }

  MMAConfigRegistry(const MMAConfigRegistry&) = delete;
  MMAConfigRegistry& operator=(const MMAConfigRegistry&) = delete;
  MMAConfigRegistry(MMAConfigRegistry&&) = delete;
  MMAConfigRegistry& operator=(MMAConfigRegistry&&) = delete;
};
// Helper functions for WGMMA detection and thread group size
inline bool IsWGMMAShape(const MMAShape& shape) {
  // WGMMA uses 64x64 tile sizes (M and N both >= 64)
  return shape.m == 64 && shape.n == 64;
}

inline bool IsWGMMAShape(const ValueList& shape) {
  // WGMMA uses 64x64 tile sizes (M and N both >= 64)
  return sbe::ceq(shape[0], sbe::nu(64)) && sbe::ceq(shape[1], sbe::nu(64));
}

inline size_t GetThreadGroupSize(const MMAShape& shape) {
  // WGMMA: 64x64 shapes use 128 threads (warp group)
  // WMMA/MMA: all others use 32 threads (warp)
  return IsWGMMAShape(shape) ? 128 : 32;
}

inline size_t GetThreadGroupSize(const ValueList& shape) {
  // WGMMA: 64x64 shapes use 128 threads (warp group)
  // WMMA/MMA: all others use 32 threads (warp)
  return IsWGMMAShape(shape) ? 128 : 32;
}

inline bool ConfigIsWMMA(const MMAConfig& config) {
  return wmma_configs.count(config);
}

inline bool ConfigIsWGMMA(const MMAConfig& config) {
  // WGMMA configs are in wmma_configs with 64x64 shapes
  if (!wmma_configs.count(config)) return false;
  return IsWGMMAShape(config.shape);
}

inline std::string MMAConfig2CuteMMAName(const MMAConfig& mma_config,
                                         const std::string& sep = "_") {
  // example: SM80_16x8x8_F16F16F16F16_TN
  assert(cute_mma_configs.count(mma_config));
  std::vector<std::string> strs;
  strs.push_back("SM" + std::to_string(cute_mma_configs.at(mma_config)));
  strs.push_back(std::to_string(mma_config.shape.m) + "x" +
                 std::to_string(mma_config.shape.n) + "x" +
                 std::to_string(mma_config.shape.k));
  auto ty_str = [](BaseType bt) -> std::string {
    switch (bt) {
    case BaseType::F8_E4M3: return "E4M3";
    case BaseType::F8_E5M2: return "E5M2";
    case BaseType::F8_UE4M3: return "UE4M3";
    case BaseType::F8_UE8M0: return "UE8M0";
    default: return STR(bt);
    }
  };
  strs.push_back(ToUpper(ty_str(mma_config.d_ty) + ty_str(mma_config.a_ty) +
                         ty_str(mma_config.b_ty) + ty_str(mma_config.c_ty)));
  // row, col
  strs.push_back("TN");
  // TODO: STR is not worked for F8_E4M3...
  return DelimitedString(strs, sep);
}

} // namespace MMALimit

} // end namespace Choreo

#endif // __CHOREO_TARGET_UTILS_HPP__
