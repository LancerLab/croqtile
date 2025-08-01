#include "symbexpr.hpp"
#include <gtest/gtest.h>

using namespace Choreo;
using namespace Choreo::sbe;

class ExpressionTest : public ::testing::Test {
protected:
  void SetUp() override {
    a = make_symbolic("a");
    b = make_symbolic("b");
    c = make_symbolic("c");
    x = make_symbolic("x");
    y = make_symbolic("y");
    two = make_numeric(2);
    three = make_numeric(3);
    four = make_numeric(4);
    five = make_numeric(5);
    tru = make_boolean(true);
    fls = make_boolean(false);
  }

  std::shared_ptr<SymbolicExpression> a, b, c, x, y, two, three, four, five,
      tru, fls;
};

TEST_F(ExpressionTest, BasicNormalization) {
  // a * 3 should stay the same
  auto expr1 = make_operation(OpCode::MULTIPLY, a, three);
  auto norm1 = SimplifyExpression(expr1);
  EXPECT_EQ(norm1->ToString(), "(a * 3)");

  // 3 * a should become a * 3
  auto expr2 = make_operation(OpCode::MULTIPLY, three, a);
  auto norm2 = SimplifyExpression(expr2);
  EXPECT_EQ(norm2->ToString(), "(a * 3)");
}

TEST_F(ExpressionTest, MultiplicativeNormalization) {
  // 2 * a * 3 * b should become a * b * 6
  auto part1 = make_operation(OpCode::MULTIPLY, two, a);
  auto part2 = make_operation(OpCode::MULTIPLY, part1, three);
  auto expr = make_operation(OpCode::MULTIPLY, part2, b);

  auto simplified = SimplifyAndPrint(expr);
  EXPECT_EQ(simplified, "(a * (b * 6))");
}

TEST_F(ExpressionTest, AdditiveNormalization) {
  // 1 + a + 4 + b should become a + b + 5
  auto part1 = make_operation(OpCode::ADD, make_numeric(1), a);
  auto part2 = make_operation(OpCode::ADD, part1, four);
  auto expr = make_operation(OpCode::ADD, part2, b);

  auto simplified = SimplifyAndPrint(expr);
  EXPECT_EQ(simplified, "(a + (b + 5))");
}

TEST_F(ExpressionTest, MixedOperationsNormalization) {
  // a + 2 * b + 3 * a should become a*3 + a + b*2
  // (but exact order may depend on implementation details)
  auto term1 = make_operation(OpCode::MULTIPLY, two, b);
  auto term2 = make_operation(OpCode::MULTIPLY, three, a);
  auto part1 = make_operation(OpCode::ADD, a, term1);
  auto expr = make_operation(OpCode::ADD, part1, term2);

  auto simplified = SimplifyAndPrint(expr);
  // The exact string might vary based on your normalization rules
  EXPECT_EQ(simplified, "((a * 3) + (a + (b * 2)))");
}

TEST_F(ExpressionTest, PrecedenceRespected) {
  // a + b * c should stay the same (multiplication has higher precedence)
  auto mul = make_operation(OpCode::MULTIPLY, b, c);
  auto expr = make_operation(OpCode::ADD, a, mul);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "(a + (b * c))");
}

TEST_F(ExpressionTest, SymbolicOrdering) {
  // c * b * a should become a * b * c
  auto part1 = make_operation(OpCode::MULTIPLY, c, b);
  auto expr = make_operation(OpCode::MULTIPLY, part1, a);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "(a * (b * c))");
}

TEST_F(ExpressionTest, ReassociateToSimplify0) {
  // (a * 2) * (3 * b) should become a * (b * 6)
  auto part1 = make_operation(OpCode::MULTIPLY, a, two);
  auto part2 = make_operation(OpCode::MULTIPLY, three, b);
  auto expr = make_operation(OpCode::MULTIPLY, part1, part2);

  auto simplified = SimplifyAndPrint(expr);
  EXPECT_EQ(simplified, "(a * (b * 6))");
}

TEST_F(ExpressionTest, ReassociateToSimplify1) {
  // ((a * 2) * x) * ((3 * b) * c) should become a * b * c * x * 6
  auto part1 = make_operation(OpCode::MULTIPLY, a, two);
  auto part2 = make_operation(OpCode::MULTIPLY, three, b);
  auto part3 = make_operation(OpCode::MULTIPLY, part2, c);
  auto part4 = make_operation(OpCode::MULTIPLY, part1, x);
  auto expr = make_operation(OpCode::MULTIPLY, part4, part3);

  auto simplified = SimplifyAndPrint(expr);
  EXPECT_EQ(simplified, "((a * b) * (c * (x * 6)))");
}

TEST_F(ExpressionTest, ComplexExpression) {
  // (2 + a) * (3 + b) should normalize to (a + 2) * (b + 3)
  auto left = make_operation(OpCode::ADD, two, a);
  auto right = make_operation(OpCode::ADD, three, b);
  auto expr = make_operation(OpCode::MULTIPLY, left, right);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "((a + 2) * (b + 3))");
}

TEST_F(ExpressionTest, MuliplyDivide) {
  // (a * 4) / 2 should normalize to a * 2
  auto term = make_operation(OpCode::MULTIPLY, a, four);
  auto expr = make_operation(OpCode::DIVIDE, term, two);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "(a * 2)");
}

TEST_F(ExpressionTest, Muliply3) {
  // 0 * ((H * W) * 3) should normalize to 0
  auto H = make_symbolic("H");
  auto W = make_symbolic("W");
  auto zero = make_numeric(0);
  auto three = make_numeric(3);

  auto x = make_operation(OpCode::MULTIPLY, H, W);
  auto y = make_operation(OpCode::MULTIPLY, x, three);
  auto z = make_operation(OpCode::MULTIPLY, zero, y);

  auto norm = SimplifyExpression(z);
  EXPECT_EQ(norm->ToString(), "0");
}

TEST_F(ExpressionTest, DivideDivide) {
  // a / (a / 2) should normalize to 2
  auto term = make_operation(OpCode::DIVIDE, a, two);
  auto expr = make_operation(OpCode::DIVIDE, a, term);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "2");
}

TEST_F(ExpressionTest, DivideDivide2) {
  // (a + 3) / ((a + 3) / 2) should normalize to 2
  auto expr = (a + three) / ((a + three) / two);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "2");
}

TEST_F(ExpressionTest, SimpleSelect) {
  // (false) ? a : b should normalize to b
  auto expr = sel(fls, a, b);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "b");
}

TEST_F(ExpressionTest, Select1) {
  // (3 < 2) ? (a + 3) : b - 4 / 2) should normalize to b - 2
  auto expr = sel(oc_lt(three, two), a + three, b - four / two);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "(b - 2)");
}

TEST_F(ExpressionTest, Select2) {
  // (a <= 2) ? (2 + 3) : 4 + b) should normalize to (a <= 2) ? 5 : b + 4
  auto expr = sel(oc_le(a, two), two + three, four + b);

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "((a <= 2) ? 5 : (b + 4))");
}

TEST_F(ExpressionTest, Divdiv) {
  // (2 + a) / (2 + 3) / 4 should normalize to (a + 2) / 20
  auto expr = (two + a) / (two + three) / four;

  auto norm = SimplifyExpression(expr);
  EXPECT_EQ(norm->ToString(), "((a + 2) / 20)");
}

TEST_F(ExpressionTest, SubSubAdd) {
  // ((H - 1) - 0) + 1 should normalize to H
  auto H = make_symbolic("H");
  auto one = make_numeric(1);
  auto zero = make_numeric(0);

  auto r = H - one - zero + one;

  auto norm = SimplifyExpression(r);
  EXPECT_EQ(norm->ToString(), H->ToString());
}
