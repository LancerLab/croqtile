#include "types.hpp"
#include <gtest/gtest.h>

using namespace Choreo;

using BT = Choreo::BaseType;

struct PromoteCase {
  BT lty;
  BT rty;
  PromoteResult expected;
};

void PrintTo(const PromoteCase& param, std::ostream* os) {
  *os << "PromoteCase{left = " << STR(param.lty)
      << ", right = " << STR(param.rty) << ", expected = ("
      << STR(param.expected.lty) << ", " << STR(param.expected.rty) << ")}";
}

class PromoteTestSuite : public ::testing::TestWithParam<PromoteCase> {};

TEST_P(PromoteTestSuite, PromoteBehavior) {
  const auto& param = GetParam();
  auto actual = PromoteType(param.lty, param.rty);
  if (actual.lty != param.expected.lty || actual.rty != param.expected.rty) {
    std::string message = "Promote {" + STR(param.lty) + ", " + STR(param.rty) +
                          "}\n\tresult in {" + STR(actual.lty) + ", " +
                          STR(actual.rty) + "}\n\texpected  {" +
                          STR(param.expected.lty) + ", " +
                          STR(param.expected.rty) + "}";
    FAIL() << message;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllScalarPromote, PromoteTestSuite,
    ::testing::Values(
        // F64 with other
        PromoteCase{BT::F64, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::F32, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U64, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S64, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U32, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S32, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U16, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S16, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U8, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S8, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::UNKSCALAR, {BT::F64, BT::UNKSCALAR}},
        // F32 with other
        PromoteCase{BT::F32, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::F32, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U64, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S64, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U32, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S32, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U16, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S16, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U8, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S8, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::UNKSCALAR, {BT::F32, BT::UNKSCALAR}},
        // F16 with other
        // PromoteCase{BT::F16, BT::F64, {BT::F64, BT::F64}},
        // PromoteCase{BT::F16, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::F16, {BT::F16, BT::F16}},
        // PromoteCase{BT::F16, BT::BF16, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::F8_E4M3, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::U64, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::S64, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::U32, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::S32, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::U16, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::S16, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::U8, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::S8, {BT::F32, BT::F32}},
        // PromoteCase{BT::F16, BT::UNKSCALAR, {BT::F16, BT::UNKSCALAR}},
        // BF16 with other
        // PromoteCase{BT::BF16, BT::F64, {BT::F64, BT::F64}},
        // PromoteCase{BT::BF16, BT::F32, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::BF16, {BT::BF16, BT::BF16}},
        // PromoteCase{BT::BF16, BT::F8_E4M3, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::U64, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::S64, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::U32, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::S32, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::U16, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::S16, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::U8, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::S8, {BT::F32, BT::F32}},
        // PromoteCase{BT::BF16, BT::UNKSCALAR, {BT::BF16, BT::UNKSCALAR}},
        // F8 with other
        // PromoteCase{BT::F8_E4M3, BT::F64, {BT::F64, BT::F64}},
        // PromoteCase{BT::F8_E4M3, BT::F32, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::F16, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::F8_E4M3, BT::F8_E4M3, {BT::F8_E4M3, BT::F8_E4M3}},
        // PromoteCase{BT::F8_E4M3, BT::U64, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::S64, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::U32, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::S32, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::U16, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::S16, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::U8, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::S8, {BT::F32, BT::F32}},
        // PromoteCase{BT::F8_E4M3, BT::UNKSCALAR, {BT::F8_E4M3,
        // BT::UNKSCALAR}}, U64 with other
        PromoteCase{BT::U64, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U64, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U64, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S64, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::U32, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S32, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::U16, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S16, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::U8, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S8, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::UNKSCALAR, {BT::U64, BT::UNKSCALAR}},
        // S64 with other
        PromoteCase{BT::S64, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S64, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S64, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S64, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::U32, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::S32, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::U16, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::S16, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::U8, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::S8, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::UNKSCALAR, {BT::S64, BT::UNKSCALAR}},
        // U32 with other
        PromoteCase{BT::U32, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U32, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U32, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U32, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::U32, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::S32, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::U16, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::S16, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::U8, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::S8, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::UNKSCALAR, {BT::U32, BT::UNKSCALAR}},
        // S32 with other
        PromoteCase{BT::S32, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S32, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S32, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S32, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S32, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::S32, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::U16, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::S16, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::U8, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::S8, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::UNKSCALAR, {BT::S32, BT::UNKSCALAR}},
        // U16 with other
        PromoteCase{BT::U16, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U16, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U16, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U16, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::U16, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::U16, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::U16, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::S16, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::U8, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::S8, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::UNKSCALAR, {BT::U16, BT::UNKSCALAR}},
        // S16 with other
        PromoteCase{BT::S16, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S16, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S16, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S16, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S16, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::S16, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::S16, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::S16, BT::S16, {BT::S16, BT::S16}},
        PromoteCase{BT::S16, BT::U8, {BT::S16, BT::S16}},
        PromoteCase{BT::S16, BT::S8, {BT::S16, BT::S16}},
        PromoteCase{BT::S16, BT::UNKSCALAR, {BT::S16, BT::UNKSCALAR}},
        // U8 with other
        PromoteCase{BT::U8, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U8, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U8, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U8, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::U8, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::U8, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::U8, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::U8, BT::S16, {BT::S16, BT::S16}},
        PromoteCase{BT::U8, BT::U8, {BT::U8, BT::U8}},
        PromoteCase{BT::U8, BT::S8, {BT::U8, BT::U8}},
        PromoteCase{BT::U8, BT::UNKSCALAR, {BT::U8, BT::UNKSCALAR}},
        // S8 with other
        PromoteCase{BT::S8, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S8, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S8, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S8, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S8, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::S8, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::S8, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::S8, BT::S16, {BT::S16, BT::S16}},
        PromoteCase{BT::S8, BT::U8, {BT::U8, BT::U8}},
        PromoteCase{BT::S8, BT::S8, {BT::S8, BT::S8}},
        PromoteCase{BT::S8, BT::UNKSCALAR, {BT::S8, BT::UNKSCALAR}},
        // UNKSCALAR with other
        PromoteCase{BT::UNKSCALAR, BT::F64, {BT::UNKSCALAR, BT::F64}},
        PromoteCase{BT::UNKSCALAR, BT::F32, {BT::UNKSCALAR, BT::F32}},
        PromoteCase{BT::UNKSCALAR, BT::U64, {BT::UNKSCALAR, BT::U64}},
        PromoteCase{BT::UNKSCALAR, BT::S64, {BT::UNKSCALAR, BT::S64}},
        PromoteCase{BT::UNKSCALAR, BT::U32, {BT::UNKSCALAR, BT::U32}},
        PromoteCase{BT::UNKSCALAR, BT::S32, {BT::UNKSCALAR, BT::S32}},
        PromoteCase{BT::UNKSCALAR, BT::U16, {BT::UNKSCALAR, BT::U16}},
        PromoteCase{BT::UNKSCALAR, BT::S16, {BT::UNKSCALAR, BT::S16}},
        PromoteCase{BT::UNKSCALAR, BT::U8, {BT::UNKSCALAR, BT::U8}},
        PromoteCase{BT::UNKSCALAR, BT::S8, {BT::UNKSCALAR, BT::S8}},
        PromoteCase{
            BT::UNKSCALAR, BT::UNKSCALAR, {BT::UNKSCALAR, BT::UNKSCALAR}}));