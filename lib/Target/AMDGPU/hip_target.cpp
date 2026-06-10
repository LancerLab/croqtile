#include "assert_site.hpp"
#include "gpu_adapt.hpp"
#include "gpu_target.hpp"
#include "hip_codegen.hpp"
#include "hip_dma_plan.hpp"
#include "io.hpp"
#include "memcheck.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"
#include "types.hpp"

using namespace Choreo;

namespace {

class AMDGPUTarget : public GPUTarget {
public:
  ~AMDGPUTarget() {}
  const std::string Name() const override { return "hip"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  int DefaultOptLevel(const ArchId&) const override { return 3; }

  const std::set<SwizMode> SupportedSwizzleModes(const ArchId&) const override {
    return {};
  }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {
        {"gfx1030", "AMD RDNA 2 (Navi 21)"},
        {"gfx1100", "AMD RDNA 3 (Navi 31)"},
    };
  }

  const ArchId DefaultArch() const override { return "gfx1030"; }

  ArchId ResolveNativeArch() const override {
    std::string cfg_dir;
#ifdef __CHOREO_ROCM_DIR__
    cfg_dir = STRINGIZE(__CHOREO_ROCM_DIR__);
#endif
    auto hipcc = FindToolchain(cfg_dir, "hipcc");
    if (hipcc.empty()) return "";
    auto arch =
        CompileAndRun(hipcc,
                      "#include <cstdio>\n"
                      "#include <hip/hip_runtime.h>\n"
                      "int main(){hipDeviceProp_t p;"
                      "if(hipGetDeviceProperties(&p,0)!=hipSuccess)return 1;"
                      "printf(\"%s\",p.gcnArchName);return 0;}\n",
                      ".cpp");
    if (!arch.empty() && IsArchSupported(arch)) return arch;
    return "";
  }

  size_t GetMemCapacity(const Storage& sto, const ArchId&) const override {
    if (sto == Storage::LOCAL)
      return 1024;
    else if (sto == Storage::SHARED)
      return 64ull * 1024; // 64KB LDS
    else if (sto == Storage::GLOBAL)
      return 16ull * 1024 * 1024 * 1024; // 16GB
    choreo_unreachable("unsupported memory level.");
    return 0;
  }

  size_t GetMinGroupDim(const ArchId&) const override { return 32; }
  size_t GetMaxThreadsPerBlock(const ArchId&) const override { return 1024; }

  size_t GetMemAlignmentByte(const Storage& sto, const ArchId&) const override {
    switch (sto) {
    case Storage::LOCAL: return 16;
    case Storage::SHARED: return 16;
    case Storage::GLOBAL: return 256;
    default: choreo_unreachable("Unsupported mem level.");
    }
    return 0;
  }

  const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId&) const override {
    return {
        {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
        {STR(ChoreoFeature::SLML), Description(ChoreoFeature::SLML)},
        {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
    };
  }

  const std::set<BaseType> SupportedScalarTypes(const ArchId&) const override {
    return {
        BaseType::F64, BaseType::F32, BaseType::F16, BaseType::BF16,
        BaseType::S64, BaseType::U64, BaseType::S32, BaseType::U32,
        BaseType::S16, BaseType::U16, BaseType::S8,  BaseType::U8,
    };
  }

  std::vector<AtomicCapability>
  SupportedAtomicOps(const ArchId&) const override {
    std::vector<AtomicCapability> caps;
    caps.push_back(
        {AtomicOp::ADD,
         {BaseType::S32, BaseType::U32, BaseType::U64, BaseType::F32}});
    caps.push_back(
        {AtomicOp::EXCH, {BaseType::S32, BaseType::U32, BaseType::U64}});
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
        {AtomicOp::CAS, {BaseType::S32, BaseType::U32, BaseType::U64}});
    return caps;
  }

  std::set<Storage> SupportedAtomicStorages(const ArchId&) const override {
    return {Storage::SHARED, Storage::GLOBAL};
  }

  size_t GetVectorLength(const ArchId&) const override { return 4; }

  size_t VectorizeLimit(const ArchId&) const override { return 128; }

  const std::set<BaseType> VectorizableTypes(const ArchId&) const override {
    return {BaseType::F32, BaseType::F16, BaseType::BF16, BaseType::S32,
            BaseType::U32};
  }

  bool EnforceVectorAlignment(const ArchId&) const override { return false; }

  const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId&) const override {
    return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::GROUP,
            ParallelLevel::THREAD};
  }

  const std::unordered_map<std::string, std::string>
  ChoreoMacros(const ArchId& arch) const override {
    return {
        {"__CHOREO_TARGET_AMDGPU__", "1"},
        {"__CHOREO_AMDGPU_ARCH__", arch},
    };
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    errs() << "warning: AMDGPU target is experimental and not yet fully "
              "supported.\n";
    p.AddStage<GPUAdaptor>();
    p.AddStage<HIP::HIPDMAPlan>();
    p.AddStage<MemUsageCheck>();
    p.AddStage<AssertSite>();
    p.AddStage<HIP::HIPCodeGen>();
    return true;
  }

private:
  static int id;
};

int AMDGPUTarget::id;
std::unique_ptr<Target> CreateAMDGPU() {
  return std::make_unique<AMDGPUTarget>();
}

} // end anonymous namespace

static bool registered = [] {
  TargetRegistry::Register(AMDGPUTarget::Id(), "hip",
                           "HIP target for AMD GPUs (RDNA 2+).", &CreateAMDGPU);
  return true;
}();
