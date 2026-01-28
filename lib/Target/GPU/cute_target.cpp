#include "cute_codegen.hpp"
#include "dmaconf.hpp"
#include "gpu_adapt.hpp"
#include "gpu_target.hpp"
#include "memcheck.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"
#include "types.hpp"

using namespace Choreo;

namespace {

class CuteTarget : public GPUTarget {
public:
  ~CuteTarget() {}
  const std::string Name() const override { return "cute"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  int DefaultOptLevel(const ArchId&) const override { return 3; }

  const std::set<SwizMode>
  SupportedSwizzleModes(const ArchId& arch) const override {
    if (IsFeatureSupported(arch, STR(ChoreoFeature::TMA)))
      return {SwizMode::NONE, SwizMode::B32, SwizMode::B64, SwizMode::B128};
    else
      return {};
  }

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
        {90, {2048 /*1KB*/, 164ull * 1024 /* 164KB */}},
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
  size_t GetMemAlignment(const Storage& sto,
                         const ArchId& arch) const override {
    int arch_num = ArchNum(arch);
    if (arch_num < 90) {
      switch (sto) {
      case Storage::LOCAL: return 16;
      case Storage::SHARED: return 16;
      default: choreo_unreachable("Unsupported mem level.");
      }
    } else {
      switch (sto) {
      case Storage::LOCAL: return 16;
      case Storage::SHARED: return 128; // req of wgmma and tma
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
        {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
        {STR(ChoreoFeature::SLML), Description(ChoreoFeature::SLML)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
    };
  }

  const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId& arch) const override {
    if (IsFeatureSupported(arch, STR(ChoreoFeature::WGMMA)))
      return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::GROUPx4,
              ParallelLevel::GROUP, ParallelLevel::THREAD};
    else
      return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::GROUP,
              ParallelLevel::THREAD};
    return {};
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    p.AddStage<GPUAdaptor>();
    p.AddStage<MemUsageCheck>();
    p.AddStage<Cute::CuteCodeGen>();
    return true;
  }

private:
  static int id;
};

int CuteTarget::id;
std::unique_ptr<Target> CreateCute() { return std::make_unique<CuteTarget>(); }

} // end anonymous namespace

static bool registered = [] {
  TargetRegistry::Register(CuteTarget::Id(), "cute",
                           "CUDA Cute target for NVidia GPUs (sm_70+).",
                           &CreateCute);
  return true;
}();
