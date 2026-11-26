#include "runtime/choreo_vec.h"

#include <gtest/gtest.h>

TEST(VectorSimd, SplatAndLaneAccess) {
  auto value = choreo::splat<int, 4>(7);
  EXPECT_EQ(value[0], 7);
  EXPECT_EQ(value[3], 7);
  value[2] = 11;
  EXPECT_EQ(value[2], 11);
}

TEST(VectorSimd, IotaAndArithmetic) {
  auto lane = choreo::iota<int, 4>(1, 2);
  EXPECT_EQ(lane[0], 1);
  EXPECT_EQ(lane[3], 7);

  auto sum = lane + choreo::splat<int, 4>(3);
  EXPECT_EQ(sum[0], 4);
  EXPECT_EQ(sum[3], 10);

  auto scaled = 2 * lane;
  EXPECT_EQ(scaled[2], 10);
}

TEST(VectorSimd, CompareMaskAndSelect) {
  auto lane = choreo::iota<int, 4>(0, 1);
  auto predicate = lane < 2;
  EXPECT_TRUE(predicate[0]);
  EXPECT_FALSE(predicate[2]);

  auto selected = choreo::select(predicate, lane, choreo::splat<int, 4>(-1));
  EXPECT_EQ(selected[0], 0);
  EXPECT_EQ(selected[1], 1);
  EXPECT_EQ(selected[2], -1);

  auto both = predicate & (lane >= 1);
  EXPECT_FALSE(both[0]);
  EXPECT_TRUE(both[1]);
  EXPECT_FALSE(both[2]);
}

TEST(VectorSimd, CastPreservesLanes) {
  auto values = choreo::simd<int, 4>{{1, 2, 3, 4}};
  auto converted = choreo::cast<float>(values);
  EXPECT_FLOAT_EQ(converted[0], 1.0f);
  EXPECT_FLOAT_EQ(converted[3], 4.0f);
}
