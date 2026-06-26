#include "codegen_prepare.hpp"
#include "gcu_adapt.hpp"
#include "gcu_target.hpp"
#include "pipeline.hpp"
#include "sys_utils.hpp"
#include "target_registry.hpp"
#include <filesystem>

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

  ArchId ResolveNativeArch() const override {
    std::string cfg_dir;
#ifdef __CHOREO_TOPSCC_DIR__
    cfg_dir = __CHOREO_TOPSCC_DIR__;
#endif
    auto topscc = FindToolchain(cfg_dir, "topscc");
    if (topscc.empty()) return "";
    namespace fs = std::filesystem;
    auto lib_dir =
        (fs::path(topscc).parent_path().parent_path() / "lib").string();
    std::string flags = "-arch gcu300 -ltops -Wl,-rpath," + lib_dir;
    auto output = CompileAndRun(
        topscc,
        R"(
#include <stdio.h>
#include "tops_runtime.h"
int main() {
  topsDeviceAttr_t attr;
  topsDeviceGetAttribute(&attr, 0);
  printf("gcu%d", attr.dieArch);
  return 0;
}
)",
        flags);
    if (output.empty()) return "";
    return output;
  }

  const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId& arch) const override {
    if (ArchNum(arch) >= 400)
      return {
          {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
          {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
          {STR(ChoreoFeature::ASYNC_DMA),
           Description(ChoreoFeature::ASYNC_DMA)},
          {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
          {STR(ChoreoFeature::DGMA), Description(ChoreoFeature::DGMA)},
          {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
          {STR(ChoreoFeature::HDRPARSE), Description(ChoreoFeature::HDRPARSE)},
          {STR(ChoreoFeature::VECTORIZE),
           Description(ChoreoFeature::VECTORIZE)},
          {STR(ChoreoFeature::LIBCALL), Description(ChoreoFeature::LIBCALL)},
          {STR(ChoreoFeature::MMA), Description(ChoreoFeature::MMA)},
          {STR(ChoreoFeature::MMA_UKERNEL),
           Description(ChoreoFeature::MMA_UKERNEL)},
      };
    return {
        {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
        {STR(ChoreoFeature::DSDMA), Description(ChoreoFeature::DSDMA)},
        {STR(ChoreoFeature::ASYNC_DMA),
         Description(ChoreoFeature::ASYNC_DMA)},
        {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
        {STR(ChoreoFeature::HDRPARSE), Description(ChoreoFeature::HDRPARSE)},
        {STR(ChoreoFeature::VECTORIZE),
         Description(ChoreoFeature::VECTORIZE)},
        {STR(ChoreoFeature::LIBCALL), Description(ChoreoFeature::LIBCALL)},
        {STR(ChoreoFeature::MMA), Description(ChoreoFeature::MMA)},
        {STR(ChoreoFeature::MMA_UKERNEL),
         Description(ChoreoFeature::MMA_UKERNEL)},
    };
  }

  bool PlanPreCodegenStages(ASTPipeline& p) const override {
    p.AddStage<CodegenPrepare>();
    p.AddStage<GCUAdaptor>();
    return true;
  }

  bool IsBinaryOnlyCodeGen() const override { return true; }

  bool PlanCodeGenStages(ASTPipeline&) const override { return false; }

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
