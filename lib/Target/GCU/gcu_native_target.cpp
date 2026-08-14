#include "codegen_prepare.hpp"
#include "gcu_adapt.hpp"
#include "gcu_target.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"

using namespace Choreo;

namespace {

// GCU native target: lowers through CoIR -> GPU MLIR -> gcu-compiler-opt.
// GCU300+ only. Does not use the classic AST codegen pipeline.
class GCUNativeTarget : public GCUTarget {
public:
  const std::string Name() const override { return "gcu"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {
        {"gcu300", "GCU Architecture 3.0"},
        {"gcu400", "GCU Architecture 4.0"},
        {"gcu450", "GCU Architecture 4.5"},
        {"gcu500", "GCU Architecture 5.0"},
    };
  }

  const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId&) const override {
    return {
        {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
        {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
        {STR(ChoreoFeature::ASYNC_DMA), Description(ChoreoFeature::ASYNC_DMA)},
        {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
        {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
        {STR(ChoreoFeature::HDRPARSE), Description(ChoreoFeature::HDRPARSE)},
        {STR(ChoreoFeature::VECTORIZE), Description(ChoreoFeature::VECTORIZE)},
        {STR(ChoreoFeature::LIBCALL), Description(ChoreoFeature::LIBCALL)},
        {STR(ChoreoFeature::MMA), Description(ChoreoFeature::MMA)},
        {STR(ChoreoFeature::MMA_UKERNEL),
         Description(ChoreoFeature::MMA_UKERNEL)},
        {STR(ChoreoFeature::COOPERATIVE_LAUNCH),
         Description(ChoreoFeature::COOPERATIVE_LAUNCH)},
    };
  }

  bool IsAsmSupported(const ArchId& arch) const override {
    return IsArchSupported(arch);
  }

  bool PlanPreCodegenStages(ASTPipeline& p) const override {
    p.AddStage<CodegenPrepare>();
    p.AddStage<GCUAdaptor>();
    return true;
  }

  bool IsBinaryOnlyCodeGen() const override { return false; }

  bool PlanCodeGenStages(ASTPipeline&) const override {
    choreo_unreachable("GCU native target uses CoIR pipeline, not AST codegen");
  }

private:
  static int id;
};

int GCUNativeTarget::id;

std::unique_ptr<Target> CreateGCUNative() {
  return std::make_unique<GCUNativeTarget>();
}

} // end anonymous namespace

static bool registered = [] {
  TargetRegistry::Register(GCUNativeTarget::Id(), "gcu",
                           "GCU native target (CoIR -> GPU MLIR).",
                           &CreateGCUNative);
  return true;
}();
