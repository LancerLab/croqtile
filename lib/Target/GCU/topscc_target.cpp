#include "assert_site.hpp"
#include "gcu_adapt.hpp"
#include "gcu_target.hpp"
#include "pipeline.hpp"
#include "sys_utils.hpp"
#include "target_registry.hpp"
#include "topscc_codegen.hpp"
#include "topscc_device_codegen.hpp"
#include "topscc_preprocess.hpp"
#include "topscc_transform.hpp"
#include "types.hpp"
#include <filesystem>

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

  ArchId ResolveNativeArch() const override {
    std::string cfg_dir;
#ifdef __CHOREO_TOPSCC_DIR__
    cfg_dir = __CHOREO_TOPSCC_DIR__;
#endif
    auto topscc = FindToolchain(cfg_dir, "topscc");
    if (topscc.empty()) return "";
    // topscc's tops_runtime.h includes device intrinsics that require an
    // arch target for __bf16 support. Use a baseline arch (gcu300) to enable
    // the compilation environment; the detection program only queries host-
    // side runtime APIs so any valid arch suffices.
    // Embed rpath so the detection binary finds libtopsrt.so at runtime.
    namespace fs = std::filesystem;
    auto lib_dir =
        (fs::path(topscc).parent_path().parent_path() / "lib").string();
    std::string flags = "-arch gcu300 -ltops -Wl,-rpath," + lib_dir;
    auto output = CompileAndRun(
        topscc,
        "#include <cstdio>\n"
        "#include \"tops/tops_runtime.h\"\n"
        "int main(){topsDeviceProp_t p;"
        "if(topsGetDeviceProperties(&p,0)!=topsSuccess)return 1;"
        "printf(\"CHOREO_ARCH:gcu%d:\",p.major*100+p.minor*10);return 0;}\n",
        ".cpp", flags);
    // Extract arch between markers (runtime may emit logs to stdout).
    auto pos = output.find("CHOREO_ARCH:");
    if (pos == std::string::npos) return "";
    pos += 12;
    auto end = output.find(':', pos);
    if (end == std::string::npos) return "";
    auto arch = output.substr(pos, end - pos);
    if (IsArchSupported(arch)) return arch;
    return "";
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
    else if (ArchNum(arch) >= 300)
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
    return {
        {STR(ChoreoFeature::MGM), Description(ChoreoFeature::MGM)},
        {STR(ChoreoFeature::ASYNC_DMA), Description(ChoreoFeature::ASYNC_DMA)},
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

  const std::set<BaseType>
  SupportedScalarTypes(const ArchId& arch) const override {
    std::set<BaseType> types = {
        BaseType::F32, BaseType::F16, BaseType::BF16,
        BaseType::S32, BaseType::U32, BaseType::S16,
        BaseType::U16, BaseType::S8,  BaseType::U8,
    };
    int arch_num = ArchNum(arch);
    if (arch_num >= 400) {
      types.insert(BaseType::S64);
      types.insert(BaseType::U64);
      types.insert(BaseType::F8_E4M3);
      types.insert(BaseType::F8_E5M2);
    }
    return types;
  }

  bool EnforceVectorAlignment(const ArchId& arch) const override {
    return ArchNum(arch) < 400;
  }

  bool PlanCodeGenStages(ASTPipeline& p) const override {
    p.AddStage<GCUAdaptor>();
    p.AddStage<MemUsageCheck>();
    p.AddStage<AssertSite>();
    p.AddStage<Topscc::TopsccCodeGen>();
    return true;
  }

  std::unique_ptr<DeviceCodeGen> MakeDeviceCodeGen() const override {
    return std::make_unique<TopsccDeviceCodeGen>();
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
