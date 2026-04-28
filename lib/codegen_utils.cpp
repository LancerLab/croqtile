#include "codegen_utils.hpp"
#include "visitor.hpp"

namespace Choreo {

Option<bool>
    dma_verbose(OptionKind::Hidden, "--dma-verbose", "", false,
                " print DMA related informtion at runtime (debug only).");

Option<bool> no_decay_spanview(OptionKind::Hidden, "--no-decay-spanview",
                               "-ndecay-spv", false,
                               " decay spanview to be pointers.");
Option<bool> dma_opt(OptionKind::Hidden, "-fopt-dma", "", true,
                     "optimize dma to linear copy.");
Option<bool> tma_cluster_aware(
    OptionKind::User, "--tma-cluster-aware", "", false,
    "Enable cluster-aware PTX mbarrier TMA codegen for global->shared copy.");
Option<bool> ptx_barrier(
    OptionKind::User, "--ptx-barrier", "", false,
    "Enable PTX mbarrier-style synchronization for TMA cluster-aware path.");
Option<bool> use_stmatrix(OptionKind::User, "--stmatrix", "", false,
                          "Use stmatrix PTX instruction for WGMMA accumulator "
                          "store to shared memory.");
Option<bool> hoist_offset(
    OptionKind::User, "--hoist-offset", "", false,
    "Hoist loop-invariant offset/address calculations in GPU codegen.");
Option<bool>
    hoist_scale(OptionKind::User, "--hoist-scale", "", false,
                "Hoist loop-invariant scale calculations in GPU codegen.");
Option<bool> assume_aligned_global(
    OptionKind::User, "--assume-aligned-global", "", false,
    "Assume global memory pointers are 128-bit aligned for DMA.");

void PrintSubscriptions(std::ostream& os, const std::string& prefix,
                        const std::string& suffix, const ValueList& dims,
                        std::vector<size_t>& indices, size_t depth) {
  if (depth == dims.size()) {
    os << prefix;
    for (size_t i : indices) os << "[" << i << "]";
    os << suffix;
    return;
  }

  for (int i = 0; i < *VIInt(dims[depth]); ++i) {
    indices[depth] = i;
    PrintSubscriptions(os, prefix, suffix, dims, indices, depth + 1);
  }
}

} // end namespace Choreo
