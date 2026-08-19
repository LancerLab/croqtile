#include <gtest/gtest.h>

#include "Target/GCU/gcu_fence.hpp"

using namespace Choreo;

class GCUFenceTableTest : public ::testing::Test {};

namespace {
inline FenceKind Release(Storage s, FenceEntity e) {
  return {s, e, FenceOrder::RELEASE};
}
} // namespace

TEST_F(GCUFenceTableTest, GlobalToShared) {
  auto sel = SelectGCUDMAFences(Storage::GLOBAL, Storage::SHARED);
  ASSERT_TRUE(sel.producer.empty());
  ASSERT_EQ(sel.consumer.size(), 1u);
  ASSERT_EQ(sel.consumer[0], Release(Storage::SHARED, FenceEntity::DMA));
}

TEST_F(GCUFenceTableTest, SharedToLocal) {
  auto sel = SelectGCUDMAFences(Storage::SHARED, Storage::LOCAL);
  ASSERT_EQ(sel.producer.size(), 1u);
  ASSERT_EQ(sel.producer[0], Release(Storage::LOCAL, FenceEntity::DMA));
  ASSERT_EQ(sel.consumer.size(), 1u);
  ASSERT_EQ(sel.consumer[0], Release(Storage::LOCAL, FenceEntity::DMA));
}

TEST_F(GCUFenceTableTest, LocalToGlobal) {
  auto sel = SelectGCUDMAFences(Storage::LOCAL, Storage::GLOBAL);
  ASSERT_EQ(sel.producer.size(), 1u);
  ASSERT_EQ(sel.producer[0], Release(Storage::LOCAL, FenceEntity::THREADS));
  ASSERT_TRUE(sel.consumer.empty());
}

TEST_F(GCUFenceTableTest, UntabulatedDirectionsNeedNoFence) {
  // Directions absent from the table (handled by multi-hop decomposition in
  // later stages, or simply not requiring a fence) return a no-op selection.
  ASSERT_TRUE(SelectGCUDMAFences(Storage::LOCAL, Storage::SHARED).IsNoop());
  ASSERT_TRUE(SelectGCUDMAFences(Storage::SHARED, Storage::GLOBAL).IsNoop());
  ASSERT_TRUE(SelectGCUDMAFences(Storage::GLOBAL, Storage::GLOBAL).IsNoop());
}
