#include "assert_site.hpp"
#include "gcu_check.hpp"
#include "gcu_target.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"
#include "topscc_codegen.hpp"
#include "topscc_preprocess.hpp"
#include "topscc_transform.hpp"
#include "types.hpp"

using namespace Choreo;

namespace {

class TopsccTarget : public GCUTarget {
public:
  ~TopsccTarget() {}
  const std::string Name() const override { return "topscc"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {
        {"gcu200", "GCU Architecture 2.0"}, {"gcu210", "GCU Architecture 2.1"},
        {"gcu300", "GCU Architecture 3.0"}, {"gcu400", "GCU Architecture 4.0"},
        {"gcu450", "GCU Architecture 4.5"}, {"gcu500", "GCU Architecture 5.0"},
    };
  }

  const std::unordered_map<std::string, std::string>
  ChoreoMacros(const ArchId& arch) const override {
    std::unordered_map<std::string, std::string> macros;
    macros.emplace("__TOPSCC__", "");
    macros.emplace("__GCU_ARCH__", std::to_string(ArchNum(arch)));
    return macros;
  }
  const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId& arch) const override {
    if (ArchNum(arch) >= 400)
      return {
          {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
          {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
          {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
          {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
          {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
          {STR(ChoreoFeature::HDRPARSE), Description(ChoreoFeature::HDRPARSE)},
          {STR(ChoreoFeature::VECTORIZE),
           Description(ChoreoFeature::VECTORIZE)},
      };
    else if (ArchNum(arch) >= 300)
      return {
          {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
          {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
          {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
          {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
          {STR(ChoreoFeature::HDRPARSE), Description(ChoreoFeature::HDRPARSE)},
          {STR(ChoreoFeature::VECTORIZE),
           Description(ChoreoFeature::VECTORIZE)},
      };
    return {
        {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
        {STR(ChoreoFeature::HDRPARSE), Description(ChoreoFeature::HDRPARSE)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
    };
  }
  size_t GetVectorLength(const ArchId& arch) const override {
    int arch_num = ArchNum(arch);
    switch (arch_num) {
    case 300: return 128;
    case 400: return 512;
    default:
      choreo_unreachable("unsupported target architecture: " + arch + ".");
    }
  }
  int DefaultOptLevel(const ArchId&) const override { return 3; }

  size_t VectorizeLimit(const ArchId& arch) const override {
    if (IsFeatureSupported(STR(ChoreoFeature::VECTORIZE)))
      choreo_unreachable("target architecture: " + arch +
                         " does not support vectorization.");
    if (arch == "gcu300")
      return 128;
    else if (arch == "gcu400")
      return 512;
    else
      choreo_unreachable("target architecture: " + arch +
                         " vectorization is yet to support.");
    return 0;
  }

  const std::set<BaseType>
  VectorizableTypes(const ArchId& arch) const override {
    if (arch == "gcu300")
      return {BaseType::S64, BaseType::U64,  BaseType::S32, BaseType::U32,
              BaseType::S16, BaseType::U16,  BaseType::S8,  BaseType::U8,
              BaseType::F32, BaseType::BF16, BaseType::F16};
    else if (arch == "gcu400")
      return {BaseType::S64,    BaseType::U64, BaseType::S32,
              BaseType::U32,    BaseType::S16, BaseType::U16,
              BaseType::S8,     BaseType::U8,  BaseType::F32,
              BaseType::BF16,   BaseType::F16, BaseType::F8_E4M3,
              BaseType::F8_E5M2};
    else
      return {};
  }

  bool EnforceVectorAlignment(const ArchId& arch) const override {
    return ArchNum(arch) < 400;
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    // apply GCU specific checks
    p.AddStage<GCUCheck>();
    p.AddStage<MemUsageCheck>();
    p.AddStage<AssertSite>();
    p.AddStage<Topscc::TopsccCodeGen>();
    return true;
  }

  const std::unique_ptr<Preprocess> MakePP(std::ostream& os) const override {
    return std::make_unique<TopsccPreprocess>(os);
  }

private:
  static int id;
};

int TopsccTarget::id;

std::unique_ptr<Target> CreateTopscc() {
  return std::make_unique<TopsccTarget>();
}

} // end anonymous namespace

// register at the initialization time
static bool registered = [] {
  TargetRegistry::Register(TopsccTarget::Id(), "topscc",
                           "Topscc target for GCU.", &CreateTopscc);
  return true;
}();
