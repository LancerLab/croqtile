//===- TileOffset.h - Shared tensor-tile offset helpers ----------*- C++
//-*-===//
//
// Centralizes the coir.element_offset concept so every lowering pass and
// codegen backend interprets tensor-tile indices consistently.
//
// A TensorTileOp marked coir.element_offset carries pre-multiplied element
// offsets in its leading indices (the trailing indices, if any, are dynamic
// shape values). Consumers must pass those offsets through unchanged instead
// of scaling them by a chunk size. A plain tile instead carries raw chunk
// indices, which must be scaled by the tile's per-dim chunk size -- falling
// back to the DMA counterpart's per-dim extent for anchor dims.
//
//===----------------------------------------------------------------------===//

#ifndef COIR_DIALECT_COIR_TILEOFFSET_H
#define COIR_DIALECT_COIR_TILEOFFSET_H

#include "Dialect/CoIR/CoIROps.h"

namespace coir {

/// True when a tile's leading indices are already element offsets (the
/// coir.element_offset attribute is set). Consumers must pass those values
/// through unchanged instead of scaling them by a chunk size.
inline bool isElementOffsetTile(TensorTileOp tile) {
  return tile->hasAttr("coir.element_offset");
}

/// Element-offset multiplier for a raw chunk index in one dimension. Prefers
/// the tile's own chunk size; anchor tiles (chunk size 1) fall back to the DMA
/// counterpart's per-dim extent. Callers must resolve a dynamic counterpart
/// extent themselves (never pass ShapedType::kDynamic here). Returns 1 when no
/// scaling is required, otherwise the value the index must be multiplied by.
inline int64_t effectiveChunkSize(int64_t tileDim, int64_t counterpartExtent) {
  if (tileDim > 1) return tileDim;
  return counterpartExtent;
}

} // namespace coir

#endif // COIR_DIALECT_COIR_TILEOFFSET_H
