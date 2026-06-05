#include "assert_site.hpp"
#include "cc_codegen.hpp"
#include "cc_device_codegen.hpp"
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
  const std::string DeviceName() const override { return "cpu"; }
  static TargetID Id() { return reinterpret_cast<TargetID>(&id); }

  int DefaultOptLevel(const ArchId&) const override { return 2; }

  const std::vector<ArchInfo> SupportedArchs() const override {
    return {
        {"x86_64", "Generic x86-64 CPU"},
    };
  }

  const ArchId DefaultArch() const override { return "x86_64"; }

  ArchId ResolveNativeArch() const override { return "x86_64"; }

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
        {STR(ChoreoFeature::EVENT), Description(ChoreoFeature::EVENT)},
        {STR(ChoreoFeature::MMA), Description(ChoreoFeature::MMA)},
        {STR(ChoreoFeature::MEMALLOC), Description(ChoreoFeature::MEMALLOC)},
        {STR(ChoreoFeature::LIBCALL), Description(ChoreoFeature::LIBCALL)},
        {STR(ChoreoFeature::VECTORIZE), Description(ChoreoFeature::VECTORIZE)},
    };
  }

  size_t GetVectorLength(const ArchId&) const override { return 8; }
  size_t VectorizeLimit(const ArchId&) const override { return 8; }

  const std::set<BaseType> VectorizableTypes(const ArchId&) const override {
    return {BaseType::F32, BaseType::F64, BaseType::S32, BaseType::S64};
  }

  bool IsLibCallSupported(const std::string& name) const override {
    static const std::set<std::string> supported = {
        "__lib_gemm",        "__lib_add",        "__lib_sub",
        "__lib_mul",         "__lib_div",        "__lib_max",
        "__lib_min",         "__lib_pow",        "__lib_atan2",
        "__lib_fmod",        "__lib_gt",         "__lib_ge",
        "__lib_lt",          "__lib_le",         "__lib_eq",
        "__lib_ne",          "__lib_abs",        "__lib_neg",
        "__lib_sqrt",        "__lib_rsqrt",      "__lib_exp",
        "__lib_log",         "__lib_ceil",       "__lib_floor",
        "__lib_round",       "__lib_sin",        "__lib_cos",
        "__lib_tan",         "__lib_asin",       "__lib_acos",
        "__lib_atan",        "__lib_sinh",       "__lib_cosh",
        "__lib_tanh",        "__lib_erf",        "__lib_erfc",
        "__lib_cbrt",        "__lib_reciprocal", "__lib_sign",
        "__lib_relu",        "__lib_gelu",       "__lib_sigmoid",
        "__lib_silu",        "__lib_convert",    "__lib_where",
        "__lib_reduce_sum",  "__lib_reduce_max", "__lib_reduce_min",
        "__lib_reduce_mean",
    };
    return supported.count(name) > 0;
  }

  std::pair<int, int> LibCallArgRange(const std::string& name) const override {
    if (name == "__lib_gemm") return {5, 6};
    if (name == "__lib_where") return {5, 5};
    if (name == "__lib_convert") return {3, 3};
    if (name == "__lib_reduce_sum" || name == "__lib_reduce_max" ||
        name == "__lib_reduce_min" || name == "__lib_reduce_mean")
      return {5, 5};
    // Binary: 4 args; Unary: 3 args
    if (name.find("__lib_add") != std::string::npos ||
        name.find("__lib_sub") != std::string::npos ||
        name.find("__lib_mul") != std::string::npos ||
        name.find("__lib_div") != std::string::npos ||
        name.find("__lib_max") != std::string::npos ||
        name.find("__lib_min") != std::string::npos ||
        name.find("__lib_pow") != std::string::npos ||
        name.find("__lib_fmod") != std::string::npos ||
        name.find("__lib_atan2") != std::string::npos ||
        name.find("__lib_gt") != std::string::npos ||
        name.find("__lib_ge") != std::string::npos ||
        name.find("__lib_lt") != std::string::npos ||
        name.find("__lib_le") != std::string::npos ||
        name.find("__lib_eq") != std::string::npos ||
        name.find("__lib_ne") != std::string::npos)
      return {4, 4};
    return {3, 3};
  }

  const std::set<BaseType> SupportedScalarTypes(const ArchId&) const override {
    return {
        BaseType::F64,  BaseType::F32,     BaseType::TF32,    BaseType::F16,
        BaseType::BF16, BaseType::F8_E4M3, BaseType::F8_E5M2, BaseType::S64,
        BaseType::U64,  BaseType::S32,     BaseType::U32,     BaseType::S16,
        BaseType::U16,  BaseType::S8,      BaseType::U8,      BaseType::BOOL,
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

  std::unique_ptr<DeviceCodeGen> MakeDeviceCodeGen() const override {
    return std::make_unique<CCDeviceCodeGen>();
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
