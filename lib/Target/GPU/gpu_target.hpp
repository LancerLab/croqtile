#ifndef __CHOREO_GPU_TARGET_HPP__
#define __CHOREO_GPU_TARGET_HPP__

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

  size_t GetMemCapacity(const Storage& sto, const ArchId& arch) const override {
    int arch_num = ArchNum(arch);
    // arch -> {local, shared}
    static std::map<int, std::pair<size_t, size_t>> caps = {
        {70, {1024 /*1KB*/, 48ull * 1024 /* 48KB */}},
        {75, {1024 /*1KB*/, 64ull * 1024 /* 64KB */}},
        {80, {2048 /*1KB*/, 164ull * 1024 /* 164KB */}},
        {86, {2048 /*1KB*/, 100ull * 1024 /* 100KB */}},
        {89, {2048 /*1KB*/, 100ull * 1024 /* 100KB */}},
        {90, {2048 /*1KB*/, 228ull * 1024 /* 228KB */}},
        {100, {2048 /*1KB*/, 228ull * 1024 /* 228KB */}},
        {120, {2048 /*1KB*/, 300ull * 1024 /* 228KB */}},
    };

    if (!caps.count(arch_num))
      choreo_unreachable("unsupported architecture: " + arch + ".");

    if (sto == Storage::LOCAL)
      return caps[arch_num].first;
    else if (sto == Storage::SHARED)
      return caps[arch_num].second;
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
          {STR(ChoreoFeature::ASYNC_DMA),
           Description(ChoreoFeature::ASYNC_DMA)},
          {STR(ChoreoFeature::SLML), Description(ChoreoFeature::SLML)},
          {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
      };
      if (arch == "sm_90a")
        feats.push_back(
            {STR(ChoreoFeature::WGMMA), Description(ChoreoFeature::WGMMA)});
      return feats;
    }
    return {
        {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
        {STR(ChoreoFeature::MMA), Description(ChoreoFeature::MMA)},
        {STR(ChoreoFeature::ASYNC_DMA), Description(ChoreoFeature::ASYNC_DMA)},
        {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
        {STR(ChoreoFeature::SLML), Description(ChoreoFeature::SLML)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
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

  ArchId ResolveNativeArch() const override;
};

} // end namespace Choreo

#endif // __CHOREO_GPU_TARGET_HPP__
