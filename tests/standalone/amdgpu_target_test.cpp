#include <gtest/gtest.h>

#include "target_registry.hpp"

using namespace Choreo;

class AMDGPUTargetTest : public ::testing::Test {};

TEST_F(AMDGPUTargetTest, RegistryIsStable) {
  auto targets = TargetRegistry::List();
  auto targets2 = TargetRegistry::List();
  ASSERT_EQ(targets.size(), targets2.size());
}

TEST_F(AMDGPUTargetTest, CreateNonexistentTargetReturnsNull) {
  auto target = TargetRegistry::Create("nonexistent_target_xyz");
  ASSERT_EQ(target, nullptr);
}

TEST_F(AMDGPUTargetTest, FeatureStringRoundtrip) {
  ASSERT_EQ(STR(ChoreoFeature::ASYNC_DMA), "async_dma");
  ASSERT_EQ(Description(ChoreoFeature::ASYNC_DMA),
            "Hardware-accelerated async DMA (e.g. cp.async).");
}
