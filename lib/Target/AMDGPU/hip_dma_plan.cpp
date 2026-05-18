#include "hip_dma_plan.hpp"
#include "types.hpp"

using namespace Choreo;
using namespace Choreo::HIP;

bool HIPDMAPlan::Visit(AST::DMA& n) {
  if (isa<PlaceHolderType>(NodeType(n))) return true;

  HIPDMALoweringDecision dec;
  dec.direction = ResolveDirection(n);

  auto from_sty = GetSpannedType(NodeType(*n.GetFrom()));
  auto to_sty = GetSpannedType(NodeType(*n.GetTo()));

  if (from_sty) {
    dec.elem_type = from_sty->ElementType();
    dec.rank = from_sty->Dims();
  } else if (to_sty) {
    dec.elem_type = to_sty->ElementType();
    dec.rank = to_sty->Dims();
  }

  if (auto ca = dyn_cast<AST::ChunkAt>(n.GetFrom())) {
    auto ca_sty = GetSpannedType(NodeType(*ca));
    if (ca_sty) dec.from_ca_shape = ca_sty->GetShape();
    auto sym_name = ca->RefSymbol();
    auto sty_p = GetSpannedType(GetSymbolType(sym_name));
    if (sty_p) dec.from_parent_shape = sty_p->GetShape();
  }
  if (auto ca = dyn_cast<AST::ChunkAt>(n.GetTo())) {
    auto ca_sty = GetSpannedType(NodeType(*ca));
    if (ca_sty) dec.to_ca_shape = ca_sty->GetShape();
    auto sym_name = ca->RefSymbol();
    auto sty_p = GetSpannedType(GetSymbolType(sym_name));
    if (sty_p) dec.to_parent_shape = sty_p->GetShape();
  }

  bool has_shared = (dec.direction == HIPDMADirection::G2S ||
                     dec.direction == HIPDMADirection::S2G ||
                     dec.direction == HIPDMADirection::S2S);

  if (has_shared && dec.rank >= 1 && dec.rank <= 2) {
    dec.strategy = HIPDMAStrategy::TILED_COPY;
    size_t elem_bytes = SizeOf(dec.elem_type);
    if (elem_bytes >= 4)
      dec.vec_width = 4;
    else if (elem_bytes == 2)
      dec.vec_width = 8;
    else
      dec.vec_width = 16;
  } else {
    dec.strategy = HIPDMAStrategy::NAIVE_COPY;
    dec.vec_width = 1;
  }

  decisions()[&n] = std::move(dec);
  return true;
}

HIPDMADirection HIPDMAPlan::ResolveDirection(const AST::DMA& n) const {
  auto from_sty = GetSpannedType(NodeType(*n.GetFrom()));
  auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
  if (!from_sty || !to_sty) return HIPDMADirection::UNKNOWN;

  const Storage from_sto = ProjectStorage(from_sty->GetStorage());
  const Storage to_sto = ProjectStorage(to_sty->GetStorage());

  if (from_sto == Storage::GLOBAL && to_sto == Storage::SHARED)
    return HIPDMADirection::G2S;
  if (from_sto == Storage::SHARED && to_sto == Storage::GLOBAL)
    return HIPDMADirection::S2G;
  if (from_sto == Storage::GLOBAL && to_sto == Storage::LOCAL)
    return HIPDMADirection::G2L;
  if (from_sto == Storage::LOCAL && to_sto == Storage::GLOBAL)
    return HIPDMADirection::L2G;
  if (from_sto == Storage::SHARED && to_sto == Storage::SHARED)
    return HIPDMADirection::S2S;
  return HIPDMADirection::OTHER;
}
