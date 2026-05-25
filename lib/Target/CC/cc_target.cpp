#include "assert_site.hpp"
#include "cc_codegen.hpp"
#include "memcheck.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"
#include "types.hpp"

using namespace Choreo;

namespace {

class CCTarget : public Target {
public:
  ~CCTarget() {}
  const std::string Name() const override { return "cc"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  int DefaultOptLevel(const ArchId&) const override { return 2; }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {
        {"x86_64", "Generic x86-64 CPU"},
    };
  }

  const ArchId DefaultArch() const override { return "x86_64"; }

  size_t GetMemCapacity(const Storage& sto, const ArchId&) const override {
    switch (sto) {
    case Storage::LOCAL: return 1ull * 1024 * 1024;
    case Storage::SHARED: return 1ull * 1024 * 1024;
    case Storage::GLOBAL: return 1ull * 1024 * 1024 * 1024;
    default: choreo_unreachable("unsupported memory level.");
    }
    return 0;
  }

  size_t GetMemAlignmentByte(const Storage&, const ArchId&) const override {
    return 64;
  }

  const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId&) const override {
    return {
        {STR(ChoreoFeature::ASYNC_DMA), Description(ChoreoFeature::ASYNC_DMA)},
    };
  }

  const std::set<BaseType> SupportedScalarTypes(const ArchId&) const override {
    return {
        BaseType::F64,  BaseType::F32, BaseType::F16, BaseType::BF16,
        BaseType::S64,  BaseType::U64, BaseType::S32, BaseType::U32,
        BaseType::S16,  BaseType::U16, BaseType::S8,  BaseType::U8,
        BaseType::BOOL,
    };
  }

  const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId&) const override {
    return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::THREAD};
  }

  size_t GetMaxParallelByCount(ParallelLevel, const ArchId&) const override {
    return 0;
  }

  std::vector<AtomicCapability>
  SupportedAtomicOps(const ArchId&) const override {
    std::set<BaseType> int_types = {BaseType::S32, BaseType::U32, BaseType::S64,
                                    BaseType::U64};
    std::set<BaseType> add_types = {BaseType::S32, BaseType::U32,
                                    BaseType::S64, BaseType::U64,
                                    BaseType::F32, BaseType::F64};
    return {
        {AtomicOp::ADD, add_types},  {AtomicOp::SUB, add_types},
        {AtomicOp::EXCH, int_types}, {AtomicOp::MIN, int_types},
        {AtomicOp::MAX, int_types},  {AtomicOp::AND, int_types},
        {AtomicOp::OR, int_types},   {AtomicOp::XOR, int_types},
        {AtomicOp::CAS, int_types},
    };
  }

  std::set<Storage> SupportedAtomicStorages(const ArchId&) const override {
    return {Storage::LOCAL, Storage::SHARED, Storage::GLOBAL};
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    p.AddStage<MemUsageCheck>();
    p.AddStage<AssertSite>();
    p.AddStage<CC::CCCodeGen>();
    return true;
  }

private:
  static int id;
};

int CCTarget::id;
std::unique_ptr<Target> CreateCC() { return std::make_unique<CCTarget>(); }

} // end anonymous namespace

static bool registered = [] {
  TargetRegistry::Register(CCTarget::Id(), "cc",
                           "C++ target for portable CPU code.", &CreateCC);
  return true;
}();
