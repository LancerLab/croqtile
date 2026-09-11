#ifndef __CHOREO_GPU_TARGET_HPP__
#define __CHOREO_GPU_TARGET_HPP__

#include <algorithm>

#include "target.hpp"

namespace Choreo {

class GPUTarget : public Target {};

// Common base for NVIDIA GPU targets (sm_70+). Provides the shared
// architecture table, memory capacity/alignment tables, feature set,
// scalar type support, atomic capabilities, and parallel levels that
// both CuteTarget and NVPTXTarget use.
class NVGPUTarget : public GPUTarget {
public:
  const std::string DeviceName() const override { return "gpu"; }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {
        {"sm_70", "CUDA Compute Capability Architecture 7.0"},
        {"sm_75", "CUDA Compute Capability Architecture 7.5"},
        {"sm_80", "CUDA Compute Capability Architecture 8.0"},
        {"sm_86", "CUDA Compute Capability Architecture 8.6"},
        {"sm_89", "CUDA Compute Capability Architecture 8.9"},
        {"sm_90", "CUDA Compute Capability Architecture 9.0"},
        {"sm_90a", "CUDA Compute Capability Architecture 9.0a"},
        {"sm_100", "CUDA Compute Capability Architecture 10.0"},
        {"sm_120", "CUDA Compute Capability Architecture 12.0"},
    };
  }

  const ArchId DefaultArch() const override { return "sm_86"; }

  // Max resident threads per streaming multiprocessor. Drives the per-thread
  // `local` budget derived from the per-SM register file below.
  size_t GetMaxThreadsPerSM(const ArchId& arch) const override {
    int arch_num = ArchNum(arch);
    static std::map<int, size_t> threads_per_sm = {
        {70, 2048}, {75, 1024}, {80, 2048},  {86, 1536},
        {89, 1536}, {90, 2048}, {100, 2048}, {120, 1536},
    };
    if (!threads_per_sm.count(arch_num))
      choreo_unreachable("unsupported architecture: " + arch + ".");
    return threads_per_sm[arch_num];
  }

  // Coarse best-practice per-thread `local` budget. `local` is backed by
  // device global memory, so it is not a fixed hardware resource; instead of
  // modeling exact register promotion we reserve an even share of the per-SM
  // register file per resident thread at full occupancy and clamp it by an
  // absolute per-thread spill budget. Exceeding this budget warns; exceeding
  // GetMemCapacity(LOCAL) errors. `--max-local-mem-capacity` overrides the
  // hard cap.
  size_t GetLocalMemBudget(const ArchId& arch) const override {
    constexpr size_t regfile_bytes_per_sm = 64ull * 1024 * 4; // 64K x 32-bit
    constexpr size_t local_spill_budget = 256; // absolute per-thread ceiling
    size_t regfile_share = regfile_bytes_per_sm / GetMaxThreadsPerSM(arch);
    return std::max<size_t>(1, std::min(regfile_share, local_spill_budget));
  }

  size_t GetMemCapacity(const Storage& sto, const ArchId& arch) const override {
    int arch_num = ArchNum(arch);
    // Per-SM shared-memory capacity, keyed by arch.
    static std::map<int, size_t> shared_caps = {
        {70, 48ull * 1024},   {75, 64ull * 1024},   {80, 164ull * 1024},
        {86, 100ull * 1024},  {89, 100ull * 1024},  {90, 228ull * 1024},
        {100, 228ull * 1024}, {120, 300ull * 1024},
    };

    if (!shared_caps.count(arch_num))
      choreo_unreachable("unsupported architecture: " + arch + ".");

    if (sto == Storage::LOCAL)
      // Hard cap: 2x the best-practice budget, leaving a warn-only band.
      return 2 * GetLocalMemBudget(arch);
    else if (sto == Storage::SHARED)
      return shared_caps[arch_num];
    else if (sto == Storage::GLOBAL)
      return 8ull * 1024 * 1024 * 1024; // 8GB

    // TODO: Memory capacity of GPU is not only determined by the
    // arch, but also by the specific model?
    choreo_unreachable("unsupported memory level.");
    return 0;
  }

  size_t GetMinGroupDim(const ArchId&) const override { return 32; }
  // CUDA constraint: all SM70+ architectures allow at most 1024 threads per
  // CTA.  The GPU adaptor checks the product of thread x group x group-4
  // dimensions against this value.
  size_t GetMaxThreadsPerBlock(const ArchId&) const override { return 1024; }

  size_t GetMemAlignmentByte(const Storage& sto,
                             const ArchId& arch) const override {
    // global buffer from cudaMalloc is aligned with 256 bytes by default.
    int arch_num = ArchNum(arch);
    if (arch_num < 90) {
      switch (sto) {
      case Storage::LOCAL: return 16;
      case Storage::SHARED: return 16;
      case Storage::GLOBAL: return 256;
      default: choreo_unreachable("Unsupported mem level.");
      }
    } else {
      switch (sto) {
      case Storage::LOCAL: return 16;
      case Storage::SHARED: return 128; // req of wgmma and tma
      case Storage::GLOBAL: return 256;
      default: choreo_unreachable("Unsupported mem level.");
      }
    }
    return 0;
  }

  const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId& arch) const override {
    int arch_num = ArchNum(arch);
    if (arch_num >= 90) {
      std::vector<FeatureToggle> feats = {
          {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
          {STR(ChoreoFeature::MMA), Description(ChoreoFeature::MMA)},
          {STR(ChoreoFeature::TMA), Description(ChoreoFeature::TMA)},
          {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
          {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
          {STR(ChoreoFeature::ASYNC_DMA),
           Description(ChoreoFeature::ASYNC_DMA)},
          {STR(ChoreoFeature::SLML), Description(ChoreoFeature::SLML)},
          {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
          {STR(ChoreoFeature::BARRIER), Description(ChoreoFeature::BARRIER)},
          {STR(ChoreoFeature::FENCE), Description(ChoreoFeature::FENCE)},
          {STR(ChoreoFeature::COOPERATIVE_LAUNCH),
           Description(ChoreoFeature::COOPERATIVE_LAUNCH)},
      };
      if (arch == "sm_90a")
        feats.push_back(
            {STR(ChoreoFeature::WGMMA), Description(ChoreoFeature::WGMMA)});
      return feats;
    }
    return {
        {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
        {STR(ChoreoFeature::MMA), Description(ChoreoFeature::MMA)},
        {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
        {STR(ChoreoFeature::ASYNC_DMA), Description(ChoreoFeature::ASYNC_DMA)},
        {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
        {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
        {STR(ChoreoFeature::SLML), Description(ChoreoFeature::SLML)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
        {STR(ChoreoFeature::BARRIER), Description(ChoreoFeature::BARRIER)},
        {STR(ChoreoFeature::FENCE), Description(ChoreoFeature::FENCE)},
        {STR(ChoreoFeature::COOPERATIVE_LAUNCH),
         Description(ChoreoFeature::COOPERATIVE_LAUNCH)},
    };
  }

  const std::set<BaseType>
  SupportedScalarTypes(const ArchId& arch) const override {
    std::set<BaseType> types = {
        BaseType::F64,  BaseType::F32,     BaseType::TF32,    BaseType::F16,
        BaseType::BF16, BaseType::F8_E4M3, BaseType::F8_E5M2, BaseType::S64,
        BaseType::U64,  BaseType::S32,     BaseType::U32,     BaseType::S16,
        BaseType::U16,  BaseType::S8,      BaseType::U8,
    };
    int arch_num = ArchNum(arch);
    if (arch_num >= 90) {
      types.insert(BaseType::F8_UE8M0);
      types.insert(BaseType::F8_UE4M3);
      types.insert(BaseType::F6_E2M3);
      types.insert(BaseType::F6_E3M2);
      types.insert(BaseType::F4_E2M1);
    }
    return types;
  }

  std::vector<AtomicCapability>
  SupportedAtomicOps(const ArchId&) const override {
    // Choreo only targets SM70+; all types below are available on SM70+.
    std::vector<AtomicCapability> caps;
    caps.push_back({AtomicOp::ADD,
                    {BaseType::S32, BaseType::U32, BaseType::U64, BaseType::F32,
                     BaseType::F64, BaseType::F16}});
    caps.push_back({AtomicOp::SUB, {BaseType::S32, BaseType::U32}});
    caps.push_back(
        {AtomicOp::EXCH,
         {BaseType::S32, BaseType::U32, BaseType::U64, BaseType::F32}});
    caps.push_back(
        {AtomicOp::MIN,
         {BaseType::S32, BaseType::U32, BaseType::S64, BaseType::U64}});
    caps.push_back(
        {AtomicOp::MAX,
         {BaseType::S32, BaseType::U32, BaseType::S64, BaseType::U64}});
    caps.push_back(
        {AtomicOp::AND,
         {BaseType::S32, BaseType::U32, BaseType::S64, BaseType::U64}});
    caps.push_back(
        {AtomicOp::OR,
         {BaseType::S32, BaseType::U32, BaseType::S64, BaseType::U64}});
    caps.push_back(
        {AtomicOp::XOR,
         {BaseType::S32, BaseType::U32, BaseType::S64, BaseType::U64}});
    caps.push_back(
        {AtomicOp::CAS,
         {BaseType::S32, BaseType::U32, BaseType::U64, BaseType::U16}});
    return caps;
  }

  std::set<Storage> SupportedAtomicStorages(const ArchId&) const override {
    return {Storage::SHARED, Storage::GLOBAL};
  }

  const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId& arch) const override {
    if (IsFeatureSupported(arch, STR(ChoreoFeature::WGMMA)))
      return {ParallelLevel::SEQ,   ParallelLevel::CLUSTER,
              ParallelLevel::BLOCK, ParallelLevel::GROUPx4,
              ParallelLevel::GROUP, ParallelLevel::THREAD};
    else
      return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::GROUP,
              ParallelLevel::THREAD};
    return {};
  }

  // CUDA's standalone `fence` instruction has no release-only / acquire-only
  // form (only .acq_rel and .sc), so a directional fence order collapses to
  // the full __threadfence*() barrier.
  bool SupportsDirectionalFence(const ArchId&) const override { return false; }

  // CUDA exposes a sequentially-consistent fence via `fence.sc`, so the
  // default fence order is seq_cst and `sync.fence.seq_cst` is accepted.
  bool SupportsSeqCstFence(const ArchId&) const override { return true; }

  ArchId ResolveNativeArch() const override;
};

} // end namespace Choreo

#endif // __CHOREO_GPU_TARGET_HPP__
