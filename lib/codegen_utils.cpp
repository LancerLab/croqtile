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
} // end namespace Choreo
