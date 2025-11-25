#include "types.hpp"
#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace Choreo;

struct CastCheckParam {
  BaseType from;
  BaseType to;
};

void PrintTo(const CastCheckParam& param, std::ostream* os) {
  *os << "CastCheckParam{from = " << STR(param.from)
      << ", to = " << STR(param.to) << "}";
}

class ScalarCastCoverageTest : public ::testing::TestWithParam<CastCheckParam> {
};

TEST_P(ScalarCastCoverageTest, AllCastsHandled) {
  auto param = GetParam();
  BaseType from = param.from;
  BaseType to = param.to;

  bool is_value_preserving = IsValuePreservingCast(from, to);
  bool is_lossy = IsLossyCast(from, to);
  bool is_reinterpreted = IsReinterpretiveCast(from, to);

  int match_count =
      int(is_value_preserving) + int(is_lossy) + int(is_reinterpreted);

  if (match_count != 1) {
    std::vector<std::string> kinds;
    if (is_value_preserving) kinds.push_back("ValuePreserving");
    if (is_lossy) kinds.push_back("Lossy");
    if (is_reinterpreted) kinds.push_back("Reinterpreted");

    std::string message = "Conversion from " + STR(from) + " to " + STR(to) +
                          " matches " + std::to_string(match_count) +
                          " kinds: [" +
                          [&] {
                            std::string out;
                            for (size_t i = 0; i < kinds.size(); ++i) {
                              if (i > 0) out += ", ";
                              out += kinds[i];
                            }
                            return out;
                          }() +
                          "]";

    FAIL() << message;
  }
}

std::vector<CastCheckParam> GenerateAllCastPairs() {
  std::vector<CastCheckParam> result;
  constexpr int N = 29;
  assert(static_cast<BaseType>(N - 1) == BaseType::BOOL);
  auto SkipNow = [](BaseType bt) {
    // skip sub-byte types and BOOL
    if (Choreo::IsSubByteType(bt)) return true;
    if (bt == BaseType::BOOL || bt == BaseType::F8_UE4M3 ||
        bt == BaseType::F8_UE8M0 || bt == BaseType::TF32)
      return true;

    return false;
  };
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      // ignore BOOL
      if (SkipNow(static_cast<BaseType>(i)) ||
          SkipNow(static_cast<BaseType>(j)))
        continue;
      result.push_back({static_cast<BaseType>(i), static_cast<BaseType>(j)});
    }
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(AllCasts, ScalarCastCoverageTest,
                         ::testing::ValuesIn(GenerateAllCastPairs()));

/*
// To classify the conversion

#include "utils/choreo.h"
#include <iostream>
#include <numeric>

using namespace choreo;

template <class A, class B>
bool CheckOK() {
  // std::cout << __PRETTY_FUNCTION__ << "\n";
  for (A i = std::numeric_limits<A>::max(); i >= std::numeric_limits<A>::min();
       i--) {
    auto res = static_cast<A>(static_cast<B>(i));
    if (res != i) {
      std::cerr << "\terror: " << i << " vs " << res << "\n";
      // std::cerr << "\t\tstatic_cast<B>(i): " << static_cast<B>(i) << "\n";
      // std::cerr << "\t\tstatic_cast<A>(static_cast<B>(i)): "
      //           << static_cast<A>(static_cast<B>(i)) << "\n";
      return false;
    }
    if (i == std::numeric_limits<A>::min()) break;
  }
  std::cerr << "\tdone\n";
  return true;
}

template <class A>
bool CheckF16OK() {
  for (A i = std::numeric_limits<A>::max(); i >= std::numeric_limits<A>::min();
       i--) {
    auto res = static_cast<A>(f16_to_f32(f32_to_f16(static_cast<float>(i))));
    if (res != i) {
      std::cerr << "\terror: " << i << " vs " << res << "\n";
      return false;
    }
    if (i == std::numeric_limits<A>::min()) break;
  }
  std::cerr << "\tdone\n";
  return true;
}

template <class A>
bool CheckBF16OK() {
  for (A i = std::numeric_limits<A>::max(); i >= std::numeric_limits<A>::min();
       i--) {
    auto _bf16 = bf16(static_cast<float>(i));
    auto res = static_cast<A>((float)_bf16);
    if (res != i) {
      std::cerr << "\terror: " << i << " vs " << res << "\n";
      return false;
    }
    if (i == std::numeric_limits<A>::min()) break;
  }
  std::cerr << "\tdone\n";
  return true;
}

int main() {

  // Note: The rules for floating-point conversions may differ across platforms
  // and compilers, which can lead to inconsistent results. However, non-safe
  // casts are definitively unsafe.

  std::cout << "int64_t, bf16\n";
  CheckBF16OK<int64_t>();
  std::cout << "int64_t, f16\n";
  CheckF16OK<int64_t>();
  std::cout << "int64_t, float\n";
  CheckOK<int64_t, float>();
  std::cout << "int64_t, double\n";
  CheckOK<int64_t, double>();
  std::cout << std::endl;

  std::cout << "uint64_t, bf16\n";
  CheckBF16OK<uint64_t>();
  std::cout << "uint64_t, f16\n";
  CheckF16OK<uint64_t>();
  std::cout << "uint64_t, float\n";
  CheckOK<uint64_t, float>();
  std::cout << "uint64_t, double\n";
  CheckOK<uint64_t, double>();
  std::cout << std::endl;

  std::cout << "uint32_t, bf16\n";
  CheckBF16OK<uint32_t>();
  std::cout << "uint32_t, f16\n";
  CheckF16OK<uint32_t>();
  std::cout << "uint32_t, float\n";
  CheckOK<uint32_t, float>();
  std::cout << "uint32_t, double\n";
  CheckOK<uint32_t, double>();
  std::cout << std::endl;

  std::cout << "int32_t, bf16\n";
  CheckBF16OK<int32_t>();
  std::cout << "int32_t, f16\n";
  CheckF16OK<int32_t>();
  std::cout << "int32_t, float\n";
  CheckOK<int32_t, float>();
  std::cout << "int32_t, double\n";
  CheckOK<int32_t, double>();
  std::cout << std::endl;

  std::cout << "int16_t, bf16\n";
  CheckBF16OK<int16_t>();
  std::cout << "int16_t, f16\n";
  CheckF16OK<int16_t>();
  std::cout << "int16_t, float\n";
  CheckOK<int16_t, float>();
  std::cout << "int16_t, double\n";
  CheckOK<int16_t, double>();
  std::cout << std::endl;

  std::cout << "uint16_t, bf16\n";
  CheckBF16OK<uint16_t>();
  std::cout << "uint16_t, f16\n";
  CheckF16OK<uint16_t>();
  std::cout << "uint16_t, float\n";
  CheckOK<uint16_t, float>();
  std::cout << "uint16_t, double\n";
  CheckOK<uint16_t, double>();
  std::cout << std::endl;

  std::cout << "int8_t, bf16\n";
  CheckBF16OK<int8_t>();
  std::cout << "int8_t, f16\n";
  CheckF16OK<int8_t>();
  std::cout << "int8_t, float\n";
  CheckOK<int8_t, float>();
  std::cout << "int8_t, double\n";
  CheckOK<int8_t, double>();
  std::cout << std::endl;

  std::cout << "uint8_t, bf16\n";
  CheckBF16OK<uint8_t>();
  std::cout << "uint8_t, f16\n";
  CheckF16OK<uint8_t>();
  std::cout << "uint8_t, float\n";
  CheckOK<uint8_t, float>();
  std::cout << "uint8_t, double\n";
  CheckOK<uint8_t, double>();
  std::cout << std::endl;

  std::cout << "int8_t, uint16_t\n";
  CheckOK<int8_t, uint16_t>();
  CheckOK<int8_t, uint32_t>();
  CheckOK<int8_t, uint64_t>();
  std::cout << "int8_t, uint8_t\n";
  CheckOK<int8_t, uint8_t>();
  std::cout << "uint8_t, int8_t\n";
  CheckOK<uint8_t, int8_t>();

  std::cout << std::endl;

  // uint8_t i = 255;
  // auto _bf16 = bf16(static_cast<float>(i));
  // auto res = static_cast<uint8_t>((float)_bf16);
  // auto ff = (float)_bf16;
  // std::cout << "result: " << (int)i << ' ' << (int)res << "\n";
  // printf("0X%x\n%f\n", _bf16.Bits(), ff);
}
// clang++ .tr.cc -std=c++2a -Wall -Wcast-qual -Wsign-compare -Wconversion
// -frounding-math
*/
