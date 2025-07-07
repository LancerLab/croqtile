#include "operator_info.hpp"
#include "symbexpr.hpp"
#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace Choreo;

bool AlreadyHandled(enum OpCode opc) {
  if (!Operator::op_table.count(STR(opc))) return false;
  return true;
}

TEST(EnumTest, AllOpsAreHandled) {
  using UnderlyingType = std::underlying_type_t<OpCode>;

  constexpr UnderlyingType MAX_OP_CODE =
      static_cast<UnderlyingType>(OpCode::NUM_CODES) - 1;

  std::vector<OpCode> op_codes;
  for (UnderlyingType i = 0; i <= MAX_OP_CODE; ++i) {
    OpCode op = static_cast<OpCode>(i);
    if (op == OpCode::NONE) continue;
    op_codes.push_back(op);
  }

  for (OpCode e : op_codes) {
    EXPECT_EQ(AlreadyHandled(e), true)
        << "The OpCode is declared in sbe but not handled in operator_info: "
        << STR(e);
  }
}
