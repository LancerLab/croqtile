#include "assert_site.hpp"
#include "hetero_codegen.hpp"
#include "memcheck.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"
#include "types.hpp"

using namespace Choreo;

namespace {

class HeteroTarget : public Target {
public:
  ~HeteroTarget() {}
  const std::string Name() const override { return "hetero"; }
  const std::string DeviceName() const override { return "hetero"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  int DefaultOptLevel(const ArchId&) const override { return 2; }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {{"x86_64", "Generic x86-64 host CPU"}};
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
    return {ParallelLevel::SEQ, ParallelLevel::DEVICE, ParallelLevel::BLOCK,
            ParallelLevel::THREAD};
  }

  size_t GetMaxParallelByCount(ParallelLevel, const ArchId&) const override {
    return 0;
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    p.AddStage<MemUsageCheck>();
    p.AddStage<AssertSite>();
    p.AddStage<Hetero::HeteroCodeGen>();
    return true;
  }

private:
  static int id;
};

int HeteroTarget::id;
std::unique_ptr<Target> CreateHetero() {
  return std::make_unique<HeteroTarget>();
}

} // end anonymous namespace

static bool registered = [] {
  TargetRegistry::Register(HeteroTarget::Id(), "hetero",
                           "Heterogeneous target for multi-device code.",
                           &CreateHetero);
  return true;
}();
