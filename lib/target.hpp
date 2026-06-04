#ifndef __CHOREO_TARGET_HPP__
#define __CHOREO_TARGET_HPP__

#include "aux.hpp"
#include "preprocess.hpp"
#include <memory>
#include <set>

namespace Choreo {

enum class Storage;
enum class BaseType;
enum class ParallelLevel;
enum class SwizMode;
class ASTPipeline;
class Preprocess;
struct DeviceCodeGen;
struct CodeGenerator;

// see the Description
enum class ChoreoFeature {
  EVENT,
  MMA,
  WGMMA,
  TMA,
  MEMALLOC,
  DGMA,
  DSDMA,
  ASYNC_DMA,
  NSVR,
  SLML,
  MGM,
  VECTORIZE,
  HDRPARSE,
  LIBCALL,
};

inline static const std::string STR(ChoreoFeature cf) {
  switch (cf) {
  case ChoreoFeature::EVENT: return "event";
  case ChoreoFeature::MMA: return "mma";
  case ChoreoFeature::WGMMA: return "wgmma";
  case ChoreoFeature::TMA: return "tma";
  case ChoreoFeature::MEMALLOC: return "memalloc";
  case ChoreoFeature::DGMA: return "dgma";
  case ChoreoFeature::DSDMA: return "dsdma";
  case ChoreoFeature::ASYNC_DMA: return "async_dma";
  case ChoreoFeature::NSVR: return "nsvr";
  case ChoreoFeature::SLML: return "slml";
  case ChoreoFeature::MGM: return "mgm";
  case ChoreoFeature::VECTORIZE: return "vectorize";
  case ChoreoFeature::HDRPARSE: return "hdrparse";
  case ChoreoFeature::LIBCALL: return "libcall";
  default: choreo_unreachable("unsupported feature kind.");
  }
}

inline static const std::string Description(ChoreoFeature cf) {
  switch (cf) {
  case ChoreoFeature::EVENT:
    return "Choreo Event for Asynchronized Computation.";
  case ChoreoFeature::MMA: return "Choreo MMA for Matrix Computation.";
  case ChoreoFeature::WGMMA: return "Choreo MMA for Wide Matrix Computation.";
  case ChoreoFeature::TMA:
    return "Hardware TMA for Tensor Asynchronous Data Movement across Memory "
           "Hierarchy.";
  case ChoreoFeature::MEMALLOC: return "Chroeo Memory Analysis and Allocation.";
  case ChoreoFeature::DGMA: return "Hardware with Direct Global Memory Access.";
  case ChoreoFeature::DSDMA:
    return "Single DMA/TMA with Both Slice and DeSlice.";
  case ChoreoFeature::ASYNC_DMA:
    return "Hardware-accelerated async DMA (e.g. cp.async).";
  case ChoreoFeature::NSVR: return "No Scalar Value Return Support.";
  case ChoreoFeature::SLML: return "Allow User to Set Local Memory Limit.";
  case ChoreoFeature::MGM: return "Choreo to Manage Global Memory.";
  case ChoreoFeature::VECTORIZE: return "Choreo Scalar Code Vectorization.";
  case ChoreoFeature::HDRPARSE:
    return "Parse C++ Header Files Included by Choreo Source.";
  case ChoreoFeature::LIBCALL:
    return "Target Library (__lib_*) Builtin Support.";
  default: choreo_unreachable("unsupported feature kind.");
  }
}

using TargetID = uintptr_t;
using ArchId = std::string;
using Feature = std::string;

struct FeatureToggle {
  Feature name;
  std::string descption;
  bool operator==(const FeatureToggle& other) const {
    return name == other.name; // only compare name
  }
};

struct ArchInfo {
  ArchId id;
  std::string description;
  bool operator==(const ArchInfo& other) const {
    return id == other.id; // Only compare id
  }
};

struct TargetInfo {
  TargetID id;
  std::string name;
  std::string description;
};

class Target {
public:
  virtual ~Target() {};

  // Target hooks -- keep these abstract
  virtual const std::string Name() const = 0;
  virtual const std::string DeviceName() const { return Name(); }
  virtual const std::vector<ArchInfo> SupportedArchs() const { return {}; }
  virtual const std::unordered_map<std::string, std::string>
  ChoreoMacros(const ArchId&) const {
    return {};
  }
  virtual const ArchId DefaultArch() const {
    auto& archs = SupportedArchs();
    if (archs.empty())
      choreo_unreachable("the target '" + Name() +
                         "' supports no architecture.");
    return archs.begin()->id;
  }
  virtual const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId&) const {
    return {};
  }

public:
  virtual size_t GetMemCapacity(const Storage&, const ArchId&) const = 0;
  virtual size_t GetMemAlignmentByte(const Storage&, const ArchId&) const = 0;
  virtual size_t GetMinGroupDim(const ArchId& arch) const {
    choreo_unreachable("unsupported target '" + Name() + "(" + arch + ")'.");
  }
  // Max total count for a parallel-by at the given level.  Returns 0 if the
  // level is unconstrained for this target/arch. Implemented by targets to
  // enforce per-arch parallel-by count limits.
  virtual size_t GetMaxParallelByCount(ParallelLevel /*pl*/,
                                       const ArchId& /*arch*/) const {
    return 0;
  }
  // Max physical threads per block for GPU targets.  Returns 0 if
  // unconstrained.  For GPU, the real thread count is thread x group x
  // group-4 dims; this value bounds that product.
  virtual size_t GetMaxThreadsPerBlock(const ArchId& /*arch*/) const {
    return 0;
  }
  virtual size_t GetVectorLength(const ArchId& arch) const {
    choreo_unreachable("unsupported target '" + Name() + "(" + arch + ")'.");
    return 0;
  }
  virtual bool IsEventSupported() const { return false; }
  virtual bool IsMMASupported() const { return false; }
  virtual bool IsArchSupported(const ArchId& arch) const {
    for (auto& ai : SupportedArchs())
      if (ai.id == arch) return true;
    return false;
  }
  virtual bool IsFeatureSupported(const ArchId& arch,
                                  const Feature& feat) const {
    for (auto& fi : SupportedFeatures(arch))
      if (fi.name == feat) return true;
    return false;
  }
  virtual bool IsFeatureSupported(const Feature& feat) const {
    if (SupportedArchs().empty()) return false;
    for (auto arch : SupportedArchs())
      if (!IsFeatureSupported(arch.id, feat)) return false;
    return true;
  }
  virtual int DefaultOptLevel(const ArchId&) const { return 0; }

  virtual size_t VectorizeLimit(const ArchId& arch) const {
    choreo_unreachable("unexpected architecture: " + arch + ".");
  }

  virtual const std::set<BaseType> VectorizableTypes(const ArchId&) const {
    return {};
  }

  // Scalar types supported by the target for explicit type conversions.
  // An empty set means no restriction (all types allowed).
  virtual const std::set<BaseType> SupportedScalarTypes(const ArchId&) const {
    return {};
  }

  // Whether an explicit cast from `from` to `to` is supported.
  // Default: allowed if both types are in SupportedScalarTypes.
  virtual bool IsCastSupported(const ArchId& arch, BaseType from,
                               BaseType to) const {
    auto types = SupportedScalarTypes(arch);
    return types.count(from) && types.count(to);
  }

  virtual const std::set<SwizMode> SupportedSwizzleModes(const ArchId&) const {
    return {};
  }

  virtual bool EnforceVectorAlignment(const ArchId&) const { return false; }

  // Atomic operation support -- targets override to declare which atomic ops
  // they support for each data type and storage level on each architecture.
  enum class AtomicOp { ADD, SUB, EXCH, MIN, MAX, AND, OR, XOR, CAS };

  struct AtomicCapability {
    AtomicOp op;
    std::set<BaseType> supported_types;
  };

  virtual std::vector<AtomicCapability>
  SupportedAtomicOps(const ArchId&) const {
    return {};
  }

  // Storage levels where atomics are available (e.g. shared, global).
  // Targets with cluster-scoped or other storage may restrict this.
  virtual std::set<Storage> SupportedAtomicStorages(const ArchId&) const {
    return {};
  }

  virtual bool IsAtomicSupported(const ArchId& arch, AtomicOp op,
                                 BaseType ty) const {
    for (auto& cap : SupportedAtomicOps(arch))
      if (cap.op == op && cap.supported_types.count(ty)) return true;
    return false;
  }

  virtual bool IsAtomicStorageSupported(const ArchId& arch, Storage sto) const {
    return SupportedAtomicStorages(arch).count(sto) > 0;
  }

  virtual bool IsAtomicSupported(const ArchId& arch, AtomicOp op, BaseType ty,
                                 Storage sto) const {
    return IsAtomicSupported(arch, op, ty) &&
           IsAtomicStorageSupported(arch, sto);
  }

  static AtomicOp ParseAtomicOp(const std::string& name) {
    if (name == "__atomic_add") return AtomicOp::ADD;
    if (name == "__atomic_sub") return AtomicOp::SUB;
    if (name == "__atomic_exch") return AtomicOp::EXCH;
    if (name == "__atomic_min") return AtomicOp::MIN;
    if (name == "__atomic_max") return AtomicOp::MAX;
    if (name == "__atomic_and") return AtomicOp::AND;
    if (name == "__atomic_or") return AtomicOp::OR;
    if (name == "__atomic_xor") return AtomicOp::XOR;
    if (name == "__atomic_cas") return AtomicOp::CAS;
    choreo_unreachable("unknown atomic op: " + name);
  }

  // Library call support -- targets override to declare which __lib_* builtins
  // they support and the expected argument counts for early sema validation.
  // name is the full builtin name, e.g. "__lib_gemm".
  virtual bool IsLibCallSupported(const std::string& /*name*/) const {
    return false;
  }
  // Returns {min_args, max_args} for the given lib call.  {-1,-1} = unknown.
  virtual std::pair<int, int>
  LibCallArgRange(const std::string& /*name*/) const {
    return {-1, -1};
  }
  // Whether this target enables target-library lowering by default.
  virtual bool DefaultUseTargetLib() const { return false; }

  virtual bool PlanCodeGenStages(ASTPipeline&) const = 0;

  // Factory for DeviceCodeGen used by HeteroCodeGen. Returns nullptr if
  // the target does not support heterogeneous device code generation.
  virtual std::unique_ptr<DeviceCodeGen> MakeDeviceCodeGen() const;

  virtual const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId&) const = 0;

  virtual const std::unique_ptr<Preprocess> MakePP(std::ostream& os) const;

public:
  virtual int ArchNum(std::string arch) const {
    if (arch.empty())
      choreo_unreachable("unexpected architecture: " + arch + ".");

    // Find the last non-digit character from the end
    auto it = arch.rbegin();
    // skip any tailing characters
    auto rb = it;
    while (!std::isdigit(static_cast<unsigned char>(*it))) {
      ++it;
      rb = it;
    }
    while (it != arch.rend() && std::isdigit(static_cast<unsigned char>(*it)))
      ++it;

    if (it == rb) choreo_unreachable("incorrect arch: " + arch + ".");

    // Keep the trailing digits
    arch.erase(arch.begin(), it.base());
    return std::stoi(arch);
  }
}; // class Target

} // end namespace Choreo

#endif //__CHOREO_TARGET_HPP__
