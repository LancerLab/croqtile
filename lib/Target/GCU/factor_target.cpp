#include "factor_codegen.hpp"
#include "factor_transform.hpp"
#include "gcu_check.hpp"
#include "gcu_target.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"

using namespace Choreo;

namespace {

class FactorTarget : public GCUTarget {
public:
  ~FactorTarget() {}
  const std::string Name() const override { return "factor"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {
        {"gcu200", "GCU Architecture 2.0"},
        {"gcu210", "GCU Architecture 2.1"},
        {"gcu300", "GCU Architecture 3.0"},
    };
  }
  int DefaultOptLevel(const ArchId&) const override { return 3; }

  const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId&) const override {
    return {
        {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
        {STR(ChoreoFeature::RSTM0), Description(ChoreoFeature::RSTM0)},
        {STR(ChoreoFeature::NSVR), Description(ChoreoFeature::NSVR)},
    };
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    p.AddStage<GCUCheck>();
    p.AddStage<FactorTrans>();
    p.AddStage<MemUsageCheck>();
    p.AddStage<Factor::FactorCodeGen>();
    return true;
  }

private:
  static int id;
};

int FactorTarget::id;

std::unique_ptr<Target> CreateFactor() {
  return std::make_unique<FactorTarget>();
}

} // end anonymous namespace

static bool registered = [] {
  TargetRegistry::Register(FactorTarget::Id(), "factor",
                           "Factor target for GCU.", &CreateFactor);
  return true;
}();
