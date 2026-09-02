#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "runtime/choreo.h"

namespace {

template <typename To, typename From>
To bitCopy(const From& value) {
  static_assert(sizeof(To) == sizeof(From));
  To result;
  std::memcpy(&result, &value, sizeof(To));
  return result;
}

float floatFromBits(uint32_t bits) { return bitCopy<float>(bits); }

uint32_t floatBits(float value) { return bitCopy<uint32_t>(value); }

uint16_t f16Bits(choreo::f16 value) { return bitCopy<uint16_t>(value); }

uint16_t bf16Bits(choreo::bf16 value) { return bitCopy<uint16_t>(value); }

TEST(ChoreoTypesConversion, F16UnderflowAndSubnormalRounding) {
  using choreo::f16_to_f32;
  using choreo::f32_to_f16;

  EXPECT_EQ(floatBits(f16_to_f32(f32_to_f16(std::ldexp(1.0f, -33)))),
            0x00000000u);
  EXPECT_EQ(floatBits(f16_to_f32(f32_to_f16(-std::ldexp(1.0f, -33)))),
            0x80000000u);

  const float subnormal_tie = std::ldexp(1.0f, -25);
  EXPECT_EQ(floatBits(f16_to_f32(f32_to_f16(subnormal_tie))), 0x00000000u);
  EXPECT_EQ(
      floatBits(f16_to_f32(f32_to_f16(std::nextafter(subnormal_tie, 1.0f)))),
      floatBits(std::ldexp(1.0f, -24)));

  EXPECT_EQ(f16Bits(f32_to_f16(floatFromBits(0x3F801000u))), 0x3C00u);
  EXPECT_EQ(f16Bits(f32_to_f16(floatFromBits(0x3F803000u))), 0x3C02u);
}

TEST(ChoreoTypesConversion, F16SpecialValues) {
  using choreo::f16_to_f32;
  using choreo::f32_to_f16;

  EXPECT_EQ(floatBits(f16_to_f32(f32_to_f16(0.0f))), 0x00000000u);
  EXPECT_EQ(floatBits(f16_to_f32(f32_to_f16(-0.0f))), 0x80000000u);
  EXPECT_TRUE(std::isinf(
      f16_to_f32(f32_to_f16(std::numeric_limits<float>::infinity()))));
  EXPECT_TRUE(std::isnan(
      f16_to_f32(f32_to_f16(std::numeric_limits<float>::quiet_NaN()))));

  EXPECT_EQ(f16Bits(f32_to_f16(65504.0f)), 0x7BFFu);
  EXPECT_EQ(f16Bits(f32_to_f16(65520.0f)), 0x7C00u);
  const auto min_subnormal = f32_to_f16(std::ldexp(1.0f, -24));
  EXPECT_EQ(f16Bits(min_subnormal), 0x0001u);
  EXPECT_EQ(floatBits(f16_to_f32(min_subnormal)), 0x33800000u);
  const auto max_subnormal = f32_to_f16(std::ldexp(1023.0f, -24));
  EXPECT_EQ(f16Bits(max_subnormal), 0x03FFu);
  EXPECT_EQ(floatBits(f16_to_f32(max_subnormal)), 0x387FC000u);
}

TEST(ChoreoTypesConversion, BF16SubnormalAndTieToEven) {
  using choreo::bf16_to_f32;
  using choreo::f32_to_bf16;

  const float min_subnormal = floatFromBits(0x00010000u);
  EXPECT_EQ(floatBits(bf16_to_f32(f32_to_bf16(min_subnormal))), 0x00010000u);

  EXPECT_EQ(floatBits(bf16_to_f32(f32_to_bf16(floatFromBits(0x3F808000u)))),
            0x3F800000u);
  EXPECT_EQ(floatBits(bf16_to_f32(f32_to_bf16(floatFromBits(0x3F818000u)))),
            0x3F820000u);

  EXPECT_EQ(bf16Bits(f32_to_bf16(floatFromBits(0x00008000u))), 0x0000u);
  EXPECT_EQ(bf16Bits(f32_to_bf16(floatFromBits(0x00008001u))), 0x0001u);
  EXPECT_EQ(bf16Bits(f32_to_bf16(floatFromBits(0x007F0000u))), 0x007Fu);
}

TEST(ChoreoTypesConversion, BF16SpecialValues) {
  using choreo::bf16_to_f32;
  using choreo::f32_to_bf16;

  EXPECT_EQ(floatBits(bf16_to_f32(f32_to_bf16(-0.0f))), 0x80000000u);
  EXPECT_TRUE(std::isinf(
      bf16_to_f32(f32_to_bf16(std::numeric_limits<float>::infinity()))));
  EXPECT_TRUE(std::isnan(
      bf16_to_f32(f32_to_bf16(std::numeric_limits<float>::quiet_NaN()))));

  EXPECT_EQ(bf16Bits(f32_to_bf16(floatFromBits(0x7F7F0000u))), 0x7F7Fu);
  EXPECT_EQ(bf16Bits(f32_to_bf16(floatFromBits(0x7F7F8000u))), 0x7F80u);
}

} // namespace
