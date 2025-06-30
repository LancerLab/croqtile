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
        PromoteCase{BT::F64, BT::F16, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::BF16, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::F8, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U64, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S64, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U32, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S32, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U16, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S16, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::U8, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::S8, {BT::F64, BT::F64}},
        PromoteCase{BT::F64, BT::UNKNOWN, {BT::F64, BT::UNKNOWN}},
        // F32 with other
        PromoteCase{BT::F32, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::F32, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U64, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S64, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U32, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S32, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U16, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S16, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::U8, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::S8, {BT::F32, BT::F32}},
        PromoteCase{BT::F32, BT::UNKNOWN, {BT::F32, BT::UNKNOWN}},
        // F16 with other
        PromoteCase{BT::F16, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::F16, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::U64, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::S64, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::U32, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::S32, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::U16, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::S16, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::U8, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::S8, {BT::F32, BT::F32}},
        PromoteCase{BT::F16, BT::UNKNOWN, {BT::F16, BT::UNKNOWN}},
        // BF16 with other
        PromoteCase{BT::BF16, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::BF16, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::U64, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::S64, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::U32, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::S32, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::U16, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::S16, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::U8, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::S8, {BT::F32, BT::F32}},
        PromoteCase{BT::BF16, BT::UNKNOWN, {BT::BF16, BT::UNKNOWN}},
        // F8 with other
        PromoteCase{BT::F8, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::F8, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::U64, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::S64, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::U32, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::S32, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::U16, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::S16, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::U8, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::S8, {BT::F32, BT::F32}},
        PromoteCase{BT::F8, BT::UNKNOWN, {BT::F8, BT::UNKNOWN}},
        // U64 with other
        PromoteCase{BT::U64, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U64, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U64, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::U64, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::U64, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::U64, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S64, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::U32, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S32, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::U16, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S16, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::U8, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::S8, {BT::U64, BT::U64}},
        PromoteCase{BT::U64, BT::UNKNOWN, {BT::U64, BT::UNKNOWN}},
        // S64 with other
        PromoteCase{BT::S64, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S64, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S64, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::S64, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::S64, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::S64, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S64, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::U32, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::S32, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::U16, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::S16, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::U8, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::S8, {BT::S64, BT::S64}},
        PromoteCase{BT::S64, BT::UNKNOWN, {BT::S64, BT::UNKNOWN}},
        // U32 with other
        PromoteCase{BT::U32, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U32, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U32, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::U32, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::U32, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::U32, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U32, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::U32, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::S32, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::U16, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::S16, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::U8, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::S8, {BT::U32, BT::U32}},
        PromoteCase{BT::U32, BT::UNKNOWN, {BT::U32, BT::UNKNOWN}},
        // S32 with other
        PromoteCase{BT::S32, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S32, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S32, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::S32, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::S32, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::S32, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S32, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S32, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::S32, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::U16, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::S16, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::U8, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::S8, {BT::S32, BT::S32}},
        PromoteCase{BT::S32, BT::UNKNOWN, {BT::S32, BT::UNKNOWN}},
        // U16 with other
        PromoteCase{BT::U16, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U16, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U16, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::U16, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::U16, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::U16, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U16, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::U16, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::U16, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::U16, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::S16, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::U8, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::S8, {BT::U16, BT::U16}},
        PromoteCase{BT::U16, BT::UNKNOWN, {BT::U16, BT::UNKNOWN}},
        // S16 with other
        PromoteCase{BT::S16, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S16, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S16, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::S16, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::S16, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::S16, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S16, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S16, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::S16, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::S16, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::S16, BT::S16, {BT::S16, BT::S16}},
        PromoteCase{BT::S16, BT::U8, {BT::S16, BT::S16}},
        PromoteCase{BT::S16, BT::S8, {BT::S16, BT::S16}},
        PromoteCase{BT::S16, BT::UNKNOWN, {BT::S16, BT::UNKNOWN}},
        // U8 with other
        PromoteCase{BT::U8, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::U8, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::U8, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::U8, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::U8, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::U8, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::U8, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::U8, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::U8, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::U8, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::U8, BT::S16, {BT::S16, BT::S16}},
        PromoteCase{BT::U8, BT::U8, {BT::U8, BT::U8}},
        PromoteCase{BT::U8, BT::S8, {BT::U8, BT::U8}},
        PromoteCase{BT::U8, BT::UNKNOWN, {BT::U8, BT::UNKNOWN}},
        // S8 with other
        PromoteCase{BT::S8, BT::F64, {BT::F64, BT::F64}},
        PromoteCase{BT::S8, BT::F32, {BT::F32, BT::F32}},
        PromoteCase{BT::S8, BT::F16, {BT::F32, BT::F32}},
        PromoteCase{BT::S8, BT::BF16, {BT::F32, BT::F32}},
        PromoteCase{BT::S8, BT::F8, {BT::F32, BT::F32}},
        PromoteCase{BT::S8, BT::U64, {BT::U64, BT::U64}},
        PromoteCase{BT::S8, BT::S64, {BT::S64, BT::S64}},
        PromoteCase{BT::S8, BT::U32, {BT::U32, BT::U32}},
        PromoteCase{BT::S8, BT::S32, {BT::S32, BT::S32}},
        PromoteCase{BT::S8, BT::U16, {BT::U16, BT::U16}},
        PromoteCase{BT::S8, BT::S16, {BT::S16, BT::S16}},
        PromoteCase{BT::S8, BT::U8, {BT::U8, BT::U8}},
        PromoteCase{BT::S8, BT::S8, {BT::S8, BT::S8}},
        PromoteCase{BT::S8, BT::UNKNOWN, {BT::S8, BT::UNKNOWN}},
        // UNKNOWN with other
        PromoteCase{BT::UNKNOWN, BT::F64, {BT::UNKNOWN, BT::F64}},
        PromoteCase{BT::UNKNOWN, BT::F32, {BT::UNKNOWN, BT::F32}},
        PromoteCase{BT::UNKNOWN, BT::F16, {BT::UNKNOWN, BT::F16}},
        PromoteCase{BT::UNKNOWN, BT::BF16, {BT::UNKNOWN, BT::BF16}},
        PromoteCase{BT::UNKNOWN, BT::F8, {BT::UNKNOWN, BT::F8}},
        PromoteCase{BT::UNKNOWN, BT::U64, {BT::UNKNOWN, BT::U64}},
        PromoteCase{BT::UNKNOWN, BT::S64, {BT::UNKNOWN, BT::S64}},
        PromoteCase{BT::UNKNOWN, BT::U32, {BT::UNKNOWN, BT::U32}},
        PromoteCase{BT::UNKNOWN, BT::S32, {BT::UNKNOWN, BT::S32}},
        PromoteCase{BT::UNKNOWN, BT::U16, {BT::UNKNOWN, BT::U16}},
        PromoteCase{BT::UNKNOWN, BT::S16, {BT::UNKNOWN, BT::S16}},
        PromoteCase{BT::UNKNOWN, BT::U8, {BT::UNKNOWN, BT::U8}},
        PromoteCase{BT::UNKNOWN, BT::S8, {BT::UNKNOWN, BT::S8}},
        PromoteCase{BT::UNKNOWN, BT::UNKNOWN, {BT::UNKNOWN, BT::UNKNOWN}}));