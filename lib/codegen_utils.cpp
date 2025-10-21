#include "codegen_utils.hpp"

namespace Choreo {

Option<bool>
    dma_verbose(OptionKind::Hidden, "--dma-verbose", "", false,
                " print DMA related informtion at runtime (debug only).");

} // end namespace Choreo
