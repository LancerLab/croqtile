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
struct FenceKind; // (space, entity, order) fence requirement; see types.hpp
struct FenceSelection;
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
  MMA_UKERNEL,
  BARRIER,
  FENCE,
  COOPERATIVE_LAUNCH,
  BUFFER_MAP,
  ASM,
  INTRINSIC_PASSTHROUGH,
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
  case ChoreoFeature::MMA_UKERNEL: return "mma_ukernel";
  case ChoreoFeature::BARRIER: return "barrier";
  case ChoreoFeature::FENCE: return "fence";
  case ChoreoFeature::COOPERATIVE_LAUNCH: return "cooperative_launch";
  case ChoreoFeature::BUFFER_MAP: return "buffer_map";
  case ChoreoFeature::ASM: return "asm";
  case ChoreoFeature::INTRINSIC_PASSTHROUGH: return "intrinsic_passthrough";
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
  case ChoreoFeature::MMA_UKERNEL:
    return "Micro-kernel based MMA via target library calls.";
  case ChoreoFeature::BARRIER:
    return "Execution barrier synchronization at parallel levels.";
  case ChoreoFeature::FENCE:
    return "Memory fence for ordering memory operations.";
  case ChoreoFeature::COOPERATIVE_LAUNCH:
    return "Cooperative kernel launch for grid-level synchronization.";
  case ChoreoFeature::BUFFER_MAP:
    return "Explicit memory mapping (map/remap/unmap) for zero-copy "
           "source-to-destination memory aliasing.";
  case ChoreoFeature::ASM:
    return "Inline assembly support via GCC extended asm syntax.";
  case ChoreoFeature::INTRINSIC_PASSTHROUGH:
    return "Intrinsic passthrough via #pragma croq intrinsic prefix.";
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

// FenceKind / FenceSelection are defined in types.hpp (next to Storage, which
// is only forward-declared here); see the DMA fence-insertion plan section 4.

class Target {
public:
  virtual ~Target(){};

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

  // Resolve "native" to a concrete arch by probing local hardware.
  // Returns empty string if detection is not supported or fails.
  virtual ArchId ResolveNativeArch() const { return ""; }

  // Return path to the host C++ compiler for this target.
  // Used by --lib multi-file driver to compile .cpp wrapper files.
  virtual std::string HostCXXCompiler() const { return "c++"; }

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
  virtual bool IsAsmSupported(const ArchId& /*arch*/) const { return false; }
  virtual std::string LowerAsmOperand(const std::string& varName) const {
    return varName;
  }
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

  // Barrier, fence, and cooperative launch support queries.
  // Targets override these to declare which barrier levels, fence scopes,
  // and cooperative launch are supported for each architecture.

  // Whether execution barrier at the given parallel level is supported.
  virtual bool IsBarrierSupported(const ArchId& arch, ParallelLevel) const {
    return IsFeatureSupported(arch, STR(ChoreoFeature::BARRIER));
  }
  // Whether memory fence at the given visibility level is supported.
  virtual bool IsFenceSupported(const ArchId& arch, ParallelLevel) const {
    return IsFeatureSupported(arch, STR(ChoreoFeature::FENCE));
  }
  // Whether the target honors the fence order axis with a directional
  // (release-only / acquire-only) standalone fence. Targets whose fence
  // instruction only offers a full acq_rel barrier (e.g. CUDA's standalone
  // `fence`) return false, so the semantic checker can note when a
  // `sync.fence.rel` / `sync.fence.acq` collapses to the full fence.
  virtual bool SupportsDirectionalFence(const ArchId&) const { return true; }
  // Whether the target has a sequentially-consistent standalone fence
  // (`fence.sc` on CUDA, `memory_order_seq_cst` on CPU, `__ATOMIC_SEQ_CST`
  // on AMDGPU). Targets without one return false, so the default fence order
  // degrades to acq_rel and an explicit `sync.fence.seq_cst` is rejected by the
  // semantic checker.
  virtual bool SupportsSeqCstFence(const ArchId&) const { return false; }
  // Default memory scope for a fence at the given visibility level.
  // Returns Storage::NONE if the default is target-defined (not specifiable).
  virtual Storage GetDefaultFenceMemory(const ArchId&,
                                        ParallelLevel visibility) const;

  // Fence requirements to insert at the producer/consumer sites of a DMA edge
  // moving data from `src` storage to `dst` storage on `arch`. Targets without
  // a fence requirement return the default no-op selection. (The entity-class
  // and LCA-level refinements from the fence-insertion plan are resolved in
  // later stages; this query keys the table on storage direction.)
  virtual FenceSelection SelectDMAFences(const ArchId& arch, Storage src,
                                         Storage dst) const;

  // Whether the target supports cooperative kernel launches.
  virtual bool SupportsCooperativeLaunch(const ArchId& arch) const {
    return IsFeatureSupported(arch, STR(ChoreoFeature::COOPERATIVE_LAUNCH));
  }

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

  // Whether misaligned vector loads/stores should produce a non-fatal warning
  // (rather than silently proceeding) on architectures where alignment is not
  // enforced. Default: no warning.
  virtual bool WarnVectorAlignment(const ArchId&) const { return false; }

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

  // Intrinsic passthrough: allow bare identifier calls matching a declared
  // prefix to be auto-rewritten into Call nodes and emitted verbatim.
  virtual bool IsIntrinsicPassthroughSupported(const ArchId& /*arch*/) const {
    return false;
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

  // MMA target strategy name for CoIR lowering.
  // Returns the string used to annotate coir.mma.exec operations:
  //   "wgmma"    -- warp-group MMA (SM90+)
  //   "mma_sync" -- cooperative MMA (SM80)
  //   "ukernel"  -- library-based micro-kernel MMA
  //   ""         -- target does not support MMA
  virtual std::string MMATargetName(const ArchId& arch) const {
    if (IsFeatureSupported(arch, STR(ChoreoFeature::MMA_UKERNEL)))
      return "ukernel";
    if (IsFeatureSupported(arch, STR(ChoreoFeature::WGMMA))) return "wgmma";
    if (IsFeatureSupported(arch, STR(ChoreoFeature::MMA))) return "mma_sync";
    return "";
  }

  // Whether TMA is supported for the given architecture (for CoIR lowering).
  virtual bool HasTMA(const ArchId& arch) const {
    return IsFeatureSupported(arch, STR(ChoreoFeature::TMA));
  }

  // Whether the target requires a hardware DMA engine for global<->local
  // memory transfers (for CoIR lowering).  When true, ClassifyCopies
  // produces coir.dma.copy instead of coir.thread.copy for global<->local.
  virtual bool HasDMA(const ArchId& arch) const {
    return IsFeatureSupported(arch, STR(ChoreoFeature::ASYNC_DMA));
  }

  // Whether the target supports hardware MMU buffer mapping (map_mem_m /
  // remap_mem_m / unmap_mem_m) for global-to-local memory aliasing.
  virtual bool HasBufferMap(const ArchId& arch) const {
    return IsFeatureSupported(arch, STR(ChoreoFeature::BUFFER_MAP));
  }

  // Whether the given src->dst storage pair is valid for buffer mapping
  // on the given architecture.  Default: only GLOBAL->LOCAL.
  virtual bool IsBufferMappingValid(const ArchId& arch, Storage src,
                                    Storage dst) const;

  // Maximum number of bytes a single buffer.map/remap may cover on the given
  // architecture. Returns 0 when the mapped size is unbounded (no check).
  virtual size_t GetMaxBufferMapBytes(const ArchId& arch) const {
    (void)arch;
    return 0;
  }

  // Maximum GROUP dimension for targets with a hardware GROUP tier: the
  // number of THREADs (SIMT lanes) per GROUP. A THREAD-level `local` buffer is
  // replicated once per THREAD, so its physical footprint is the declared
  // per-thread size times max_group_dim times replicas_per_pool. Default 1: no
  // lane dimension (THREAD == GROUP), e.g. targets without a GROUP tier.
  virtual size_t GetMaxGroupDim(const ArchId& /*arch*/) const { return 1; }

  // Some targets have no dedicated shared-memory level: SHARED memory aliases
  // the same physical on-chip scratchpad that backs the per-GROUP
  // `shared<group>` tier (Storage::GROUP_SHARED). For such targets
  // shared<group> and SHARED cannot be budgeted independently; the memcheck
  // pass enforces them jointly as
  //     group_shared * replicas_per_pool + shared <= pool_bytes
  // where replicas_per_pool is the number of GROUPs sharing one pool.
  struct SharedScratchpadPool {
    bool aliased = false; // whether GROUP_SHARED and SHARED alias one pool
    size_t replicas_per_pool = 0; // GROUPs sharing one pool
    size_t pool_bytes =
        0; // total combined on-chip pool (shared<group> + SHARED)
  };
  virtual SharedScratchpadPool
  GetSharedScratchpadPool(const ArchId& arch) const {
    (void)arch;
    return {};
  }

  // Whether `shared<group>` (Storage::GROUP_SHARED) is a valid storage tier
  // for the given architecture. Only targets with a hardware GROUP tier
  // between BLOCK and THREAD support it; the default is false so targets
  // without a group tier reject it in early semantic analysis.
  virtual bool IsGroupSharedStorageSupported(const ArchId& /*arch*/) const {
    return false;
  }

  // Whether `local` (Storage::LOCAL) is a valid storage tier for the given
  // architecture. Most targets keep `local` as a per-thread private
  // (register/stack) tier. Targets whose DMA engines cannot source/sink
  // per-thread stack (e.g. targets whose on-chip scratchpad is instead
  // exposed as `shared<group>`) override this to false so `local`
  // declarations are rejected by default in early semantic analysis (the
  // `--allow-local` opt-in relaxes this for declarations only).
  virtual bool IsLocalStorageSupported(const ArchId& /*arch*/) const {
    return true;
  }

  // Whether static local-memory reuse should expose physically disjoint
  // ranges as independent target allocation roots. This lets source-emitting
  // targets preserve alias-analysis boundaries without changing the reuse
  // layout or dynamic-allocation path.
  virtual bool UseDistinctLocalAllocationRoots(const ArchId& /*arch*/) const {
    return false;
  }

  // Whether the target's code generation only produces binaries (no text
  // source or script emission).  Targets that return true default to
  // compile_binary mode and reject -es/-gs.
  virtual bool IsBinaryOnlyCodeGen() const { return false; }

  virtual bool PlanCodeGenStages(ASTPipeline&) const = 0;

  /// Run the pre-codegen AST stages that populate CodeGenInfo and register
  /// target-specific assessments (HW constraints).  Called by CoCC before
  /// ASTCoIRGen so that the assessor contains both sema-level and
  /// target-level assertions for the COIR path.  Targets override to add
  /// their adaptor (e.g. GPUAdaptor).  The base implementation adds only
  /// CodegenPrepare.
  virtual bool PlanPreCodegenStages(ASTPipeline& p) const;

  // Factory for DeviceCodeGen used by HeteroCodeGen. Returns nullptr if
  // the target does not support heterogeneous device code generation.
  virtual std::unique_ptr<DeviceCodeGen> MakeDeviceCodeGen() const;
  virtual bool HasDeviceCodeGen() const;

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
