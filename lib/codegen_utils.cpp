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
// WARNING: --tma-cluster-aware emits cluster-scoped PTX mbarrier TMA loads
// that require a matching cluster launch configuration.  Incorrect cluster
// dimensions silently produce wrong results or hangs.  Only enable when the
// kernel launch guarantees the expected cluster geometry.
Option<bool> tma_cluster_aware(
    OptionKind::User, "--tma-cluster-aware", "", false,
    "Enable cluster-aware PTX mbarrier TMA codegen for global->shared copy. "
    "DANGEROUS: requires matching cluster launch geometry.");
Option<bool> use_stmatrix(OptionKind::User, "--stmatrix", "", true,
                          "Use stmatrix PTX instruction for WGMMA accumulator "
                          "store to shared memory (default: on).");
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
