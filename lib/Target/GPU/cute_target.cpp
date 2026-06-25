#include "assert_site.hpp"
#include "codegen_prepare.hpp"
#include "cute_codegen.hpp"
#include "cute_device_codegen.hpp"
#include "dma_plan.hpp"
#include "dmaconf.hpp"
#include "fragment_layout_pass.hpp"
#include "gpu_adapt.hpp"
#include "gpu_target.hpp"
#include "memcheck.hpp"
#include "pipeline.hpp"
#include "sys_utils.hpp"
#include "target_registry.hpp"
#include "types.hpp"

using namespace Choreo;

// Shared native-arch detection for all NVIDIA GPU targets.
ArchId NVGPUTarget::ResolveNativeArch() const {
  std::string cfg_dir;
#ifdef __CHOREO_CUDA_DIR__
  cfg_dir = STRINGIZE(__CHOREO_CUDA_DIR__);
#endif
  auto nvcc = FindToolchain(cfg_dir, "nvcc");
  if (nvcc.empty()) return "";
  auto cc =
      CompileAndRun(nvcc,
                    "#include <cstdio>\n"
                    "#include <cstring>\n"
                    "#include <cuda_runtime.h>\n"
                    "int main(){cudaDeviceProp p;"
                    "if(cudaGetDeviceProperties(&p,0)!=cudaSuccess)return 1;"
                    "if(p.major==9&&p.minor==0){"
                    "  const char*n=p.name;"
                    "  if(strstr(n,\"H100\")||strstr(n,\"H200\")"
                    "    ||strstr(n,\"H800\")||strstr(n,\"H20\")"
                    "    ||strstr(n,\"GH200\")||strstr(n,\"GH100\")"
                    "    ||strstr(n,\"B100\")||strstr(n,\"B200\")"
                    "    ||strstr(n,\"GB200\")||strstr(n,\"GB100\"))"
                    "    {printf(\"90a\");return 0;}}"
                    "printf(\"%d%d\",p.major,p.minor);return 0;}\n",
                    ".cu", "-lcudart");
  if (cc.empty()) return "";
  ArchId arch = "sm_" + cc;
  if (IsArchSupported(arch)) return arch;
  return "";
}

namespace {

class CuteTarget : public NVGPUTarget {
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

  bool PlanPreCodegenStages(ASTPipeline& p) const override {
    p.AddStage<CodegenPrepare>();
    p.AddStage<GPUAdaptor>();
    return true;
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    p.AddStage<GPUAdaptor>();
    p.AddStage<FragmentLayoutPass>();
    p.AddStage<DMAPlan>();
    p.AddStage<MemUsageCheck>();
    p.AddStage<AssertSite>();
    p.AddStage<Cute::CuteCodeGen>();
    return true;
  }

  std::unique_ptr<DeviceCodeGen> MakeDeviceCodeGen() const override {
    return std::make_unique<CuteDeviceCodeGen>();
  }

private:
  static int id;
};

// NVPTX target: lowers through MLIR GPU/NVVM/LLVM dialects to PTX.
// Does not use the classic AST pipeline.
class NVPTXTarget : public NVGPUTarget {
public:
  const std::string Name() const override { return "gpu"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  bool PlanCodeGenStages(ASTPipeline&) const override { return false; }

private:
  static int id;
};

int CuteTarget::id;
int NVPTXTarget::id;

std::unique_ptr<Target> CreateCute() { return std::make_unique<CuteTarget>(); }
std::unique_ptr<Target> CreateNVPTX() {
  return std::make_unique<NVPTXTarget>();
}

} // end anonymous namespace

static bool registered = [] {
  TargetRegistry::Register(CuteTarget::Id(), "cute",
                           "CUDA Cute target for NVidia GPUs (sm_70+).",
                           &CreateCute);
  TargetRegistry::Register(NVPTXTarget::Id(), "gpu",
                           "GPU native target (CoIR -> PTX).", &CreateNVPTX);
  return true;
}();
