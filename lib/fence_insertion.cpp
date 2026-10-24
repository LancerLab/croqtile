#include "fence_insertion.hpp"

#include "ast.hpp"
#include "codegen_utils.hpp"
#include "context.hpp"
#include "io.hpp"
#include "target.hpp"

using namespace Choreo;

FenceInsertion::FenceInsertion()
    : VisitorWithSymTab("fenceinsertion"),
      enabled_(insert_dma_fences.GetValue()),
      dump_(dump_fence_insertion.GetValue()) {
  if (trace_visit) debug_visit = true;
}

bool FenceInsertion::BeforeVisitImpl(AST::Node& n) {
  if (!indexed_ && isa<AST::Program>(&n)) BuildIndex();
  return true;
}

void FenceInsertion::BuildIndex() {
  indexed_ = true;
  for (const auto& ev : CCtx().GetBufferAccessLog().events) {
    by_buffer_[ev.buffer].push_back(&ev);
    if (ev.entity == AccessEntity::DMA || ev.entity == AccessEntity::TMA) {
      if (ev.kind == AccessKind::READ && !dma_src_.count(ev.stmt))
        dma_src_[ev.stmt] = &ev;
      else if (ev.kind == AccessKind::WRITE && !dma_dst_.count(ev.stmt))
        dma_dst_[ev.stmt] = &ev;
    }
  }
}

const BufferAccessEvent* FenceInsertion::PrevWrite(const std::string& buffer,
                                                   size_t order) const {
  auto it = by_buffer_.find(buffer);
  if (it == by_buffer_.end()) return nullptr;
  const BufferAccessEvent* result = nullptr;
  for (const auto* ev : it->second) {
    if (ev->order >= order) break;
    if (ev->kind == AccessKind::WRITE) result = ev;
  }
  return result;
}

const BufferAccessEvent* FenceInsertion::NextRead(const std::string& buffer,
                                                  size_t order) const {
  auto it = by_buffer_.find(buffer);
  if (it == by_buffer_.end()) return nullptr;
  for (const auto* ev : it->second) {
    if (ev->order > order && ev->kind == AccessKind::READ) return ev;
  }
  return nullptr;
}

std::string FenceInsertion::JoinKinds(const std::vector<FenceKind>& kinds) {
  std::string out;
  for (size_t i = 0; i < kinds.size(); ++i) {
    if (i) out += ",";
    out += kinds[i].Name();
  }
  return out;
}

void FenceInsertion::Count(const std::vector<FenceKind>& producer,
                           const std::vector<FenceKind>& consumer) {
  auto& fs = CCtx().GetAssessmentStats().fence_stats;
  for (const auto& k : producer) {
    fs.inserted++;
    fs.producer_fences++;
    fs.by_kind[k]++;
  }
  for (const auto& k : consumer) {
    fs.inserted++;
    fs.consumer_fences++;
    fs.by_kind[k]++;
  }
  if (producer.empty() && consumer.empty()) fs.elided++;
}

bool FenceInsertion::Visit(AST::DMA& n) {
  if (!enabled_ || n.IsDummy()) return true;

  const auto* src_ev = [&]() -> const BufferAccessEvent* {
    auto it = dma_src_.find(&n);
    return it == dma_src_.end() ? nullptr : it->second;
  }();
  if (!src_ev) return true;

  const BufferAccessEvent* dst_ev = nullptr;
  if (auto it = dma_dst_.find(&n); it != dma_dst_.end()) dst_ev = it->second;

  // Unannotated buffers default to global storage; normalize so the target's
  // storage-directional table sees canonical GLOBAL (mirrors dma_plan.cpp).
  const Storage src_storage = ProjectStorage(src_ev->storage);
  const Storage dst_storage =
      dst_ev ? ProjectStorage(dst_ev->storage) : Storage::NONE;

  FenceSelection sel = CCtx().GetTarget().SelectDMAFences(
      CCtx().GetArch(), src_storage, dst_storage);
  if (sel.IsNoop()) return true;

  // Elide fences already covered by same-engine in-order execution (e.g. a
  // chain of DMAs on one engine needs no fence between its own segments).
  const AccessEntity entity = n.IsTMA() ? AccessEntity::TMA : AccessEntity::DMA;
  if (!sel.producer.empty()) {
    if (const auto* prev = PrevWrite(src_ev->buffer, src_ev->order))
      if (prev->entity == entity) sel.producer.clear();
  }
  if (!sel.consumer.empty() && dst_ev) {
    if (const auto* next = NextRead(dst_ev->buffer, dst_ev->order))
      if (next->entity == entity) sel.consumer.clear();
  }

  // Producer fences attach to the DMA node itself.
  if (!sel.producer.empty()) {
    n.AddNote("dma_fence_producer", JoinKinds(sel.producer));
    if (dump_)
      dbgs() << "[dma-fence] producer " << STR(src_storage) << "->"
             << STR(dst_storage) << " : " << JoinKinds(sel.producer) << "\n";
  }

  // Consumer fences attach to the wait (async) or the DMA (sync).
  if (!sel.consumer.empty()) {
    if (n.IsAsync() && !n.future.empty()) {
      consumer_by_future_[InScopeNameForRef(n.future)] = sel.consumer;
    } else {
      n.AddNote("dma_fence_consumer", JoinKinds(sel.consumer));
      if (dump_)
        dbgs() << "[dma-fence] consumer " << STR(src_storage) << "->"
               << STR(dst_storage) << " : " << JoinKinds(sel.consumer) << "\n";
    }
  }

  Count(sel.producer, sel.consumer);
  return true;
}

bool FenceInsertion::Visit(AST::Wait& n) {
  if (!enabled_) return true;

  std::vector<FenceKind> consumer;
  for (const auto& f : n.GetTargets()) {
    auto id = AST::GetIdentifier(f);
    if (!id) continue;
    auto it = consumer_by_future_.find(InScopeNameForRef(id->name));
    if (it != consumer_by_future_.end()) {
      for (const auto& k : it->second) consumer.push_back(k);
    }
  }
  if (consumer.empty()) return true;

  n.AddNote("dma_fence_consumer", JoinKinds(consumer));
  if (dump_)
    dbgs() << "[dma-fence] consumer wait : " << JoinKinds(consumer) << "\n";
  return true;
}
