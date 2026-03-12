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
} // end namespace Choreo
