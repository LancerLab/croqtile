#include "interval.hpp"
#include <gtest/gtest.h>

using namespace Choreo;
using namespace Choreo::sbe;

// =========================================================================
// Interval basic tests
// =========================================================================

class IntervalTest : public ::testing::Test {};

TEST_F(IntervalTest, Empty) {
  auto iv = Interval::Empty();
  EXPECT_TRUE(iv.IsEmpty());
  EXPECT_FALSE(iv.IsUniverse());
  EXPECT_FALSE(iv.IsSingleton());
  EXPECT_FALSE(iv.Contains(0));
}

TEST_F(IntervalTest, Universe) {
  auto iv = Interval::Universe();
  EXPECT_FALSE(iv.IsEmpty());
  EXPECT_TRUE(iv.IsUniverse());
  EXPECT_TRUE(iv.Contains(0));
  EXPECT_TRUE(iv.Contains(-999));
  EXPECT_TRUE(iv.Contains(999));
}

TEST_F(IntervalTest, Point) {
  auto iv = Interval::Point(5);
  EXPECT_FALSE(iv.IsEmpty());
  EXPECT_TRUE(iv.IsSingleton());
  EXPECT_TRUE(iv.Contains(5));
  EXPECT_FALSE(iv.Contains(4));
  EXPECT_FALSE(iv.Contains(6));
}

TEST_F(IntervalTest, HalfOpen) {
  auto iv = Interval::HalfOpen(0, 5);
  EXPECT_TRUE(iv.Contains(0));
  EXPECT_TRUE(iv.Contains(4));
  EXPECT_FALSE(iv.Contains(5));
  EXPECT_FALSE(iv.Contains(-1));
}

TEST_F(IntervalTest, HalfOpenBelow) {
  auto iv = Interval::HalfOpenBelow(4);
  EXPECT_TRUE(iv.Contains(-100));
  EXPECT_TRUE(iv.Contains(3));
  EXPECT_FALSE(iv.Contains(4));
}

TEST_F(IntervalTest, HalfOpenAbove) {
  auto iv = Interval::HalfOpenAbove(3);
  EXPECT_FALSE(iv.Contains(2));
  EXPECT_TRUE(iv.Contains(3));
  EXPECT_TRUE(iv.Contains(100));
}

TEST_F(IntervalTest, IntersectBasic) {
  auto a = Interval::HalfOpen(0, 10);
  auto b = Interval::HalfOpen(5, 15);
  auto c = Intersect(a, b);
  EXPECT_EQ(c, Interval::HalfOpen(5, 10));
}

TEST_F(IntervalTest, IntersectDisjoint) {
  auto a = Interval::HalfOpen(0, 3);
  auto b = Interval::HalfOpen(5, 10);
  auto c = Intersect(a, b);
  EXPECT_TRUE(c.IsEmpty());
}

TEST_F(IntervalTest, IntersectUnbounded) {
  auto a = Interval::HalfOpenBelow(5);
  auto b = Interval::HalfOpenAbove(3);
  auto c = Intersect(a, b);
  EXPECT_EQ(c, Interval::HalfOpen(3, 5));
}

TEST_F(IntervalTest, Equality) {
  EXPECT_EQ(Interval::Empty(), Interval::HalfOpen(3, 3));
  EXPECT_EQ(Interval::HalfOpen(0, 5), Interval::HalfOpen(0, 5));
  EXPECT_NE(Interval::HalfOpen(0, 5), Interval::HalfOpen(0, 6));
}

TEST_F(IntervalTest, ToString) {
  EXPECT_EQ(Interval::Empty().ToString(), "{}");
  EXPECT_EQ(Interval::HalfOpen(0, 5).ToString(), "[0, 5)");
  EXPECT_EQ(Interval::HalfOpenBelow(3).ToString(), "(-inf, 3)");
  EXPECT_EQ(Interval::HalfOpenAbove(3).ToString(), "[3, +inf)");
  EXPECT_EQ(Interval::Universe().ToString(), "(-inf, +inf)");
}

// =========================================================================
// IntervalSet tests
// =========================================================================

class IntervalSetTest : public ::testing::Test {};

TEST_F(IntervalSetTest, Empty) {
  auto s = IntervalSet::MakeEmpty();
  EXPECT_TRUE(s.IsEmpty());
  EXPECT_FALSE(s.IsUniverse());
}

TEST_F(IntervalSetTest, Universe) {
  auto s = IntervalSet::MakeUniverse();
  EXPECT_TRUE(s.IsUniverse());
  EXPECT_FALSE(s.IsEmpty());
}

TEST_F(IntervalSetTest, Point) {
  auto s = IntervalSet::MakePoint(7);
  EXPECT_TRUE(s.IsSingleton());
  EXPECT_EQ(s.SingletonValue(), 7);
  EXPECT_TRUE(s.Contains(7));
  EXPECT_FALSE(s.Contains(8));
}

TEST_F(IntervalSetTest, UnionDisjoint) {
  auto a = IntervalSet(Interval::HalfOpen(0, 3));
  auto b = IntervalSet(Interval::HalfOpen(5, 8));
  auto c = Unite(a, b);
  EXPECT_TRUE(c.Contains(0));
  EXPECT_TRUE(c.Contains(2));
  EXPECT_FALSE(c.Contains(3));
  EXPECT_FALSE(c.Contains(4));
  EXPECT_TRUE(c.Contains(5));
  EXPECT_TRUE(c.Contains(7));
  EXPECT_EQ(c.Intervals().size(), 2u);
}

TEST_F(IntervalSetTest, UnionOverlapping) {
  auto a = IntervalSet(Interval::HalfOpen(0, 5));
  auto b = IntervalSet(Interval::HalfOpen(3, 8));
  auto c = Unite(a, b);
  EXPECT_EQ(c.Intervals().size(), 1u);
  EXPECT_TRUE(c.Contains(0));
  EXPECT_TRUE(c.Contains(7));
  EXPECT_FALSE(c.Contains(8));
}

TEST_F(IntervalSetTest, UnionAdjacent) {
  auto a = IntervalSet(Interval::HalfOpen(0, 5));
  auto b = IntervalSet(Interval::HalfOpen(5, 10));
  auto c = Unite(a, b);
  EXPECT_EQ(c.Intervals().size(), 1u);
  EXPECT_TRUE(c.Contains(0));
  EXPECT_TRUE(c.Contains(9));
  EXPECT_FALSE(c.Contains(10));
}

TEST_F(IntervalSetTest, IntersectSets) {
  auto a = IntervalSet(Interval::HalfOpen(0, 10));
  auto b = IntervalSet(Interval::HalfOpen(5, 15));
  auto c = Intersect(a, b);
  EXPECT_TRUE(c.Contains(5));
  EXPECT_TRUE(c.Contains(9));
  EXPECT_FALSE(c.Contains(4));
  EXPECT_FALSE(c.Contains(10));
}

TEST_F(IntervalSetTest, IntersectDisjointSets) {
  auto a = IntervalSet(Interval::HalfOpen(0, 3));
  auto b = IntervalSet(Interval::HalfOpen(5, 8));
  auto c = Intersect(a, b);
  EXPECT_TRUE(c.IsEmpty());
}

TEST_F(IntervalSetTest, IntersectMultipleIntervals) {
  auto a = Unite(IntervalSet(Interval::HalfOpen(0, 3)),
                 IntervalSet(Interval::HalfOpen(5, 8)));
  auto b = IntervalSet(Interval::HalfOpen(2, 6));
  auto c = Intersect(a, b);
  EXPECT_TRUE(c.Contains(2));
  EXPECT_FALSE(c.Contains(3));
  EXPECT_FALSE(c.Contains(4));
  EXPECT_TRUE(c.Contains(5));
  EXPECT_FALSE(c.Contains(6));
}

TEST_F(IntervalSetTest, ComplementFinite) {
  auto s = IntervalSet(Interval::HalfOpen(3, 7));
  auto c = Complement(s);
  EXPECT_TRUE(c.Contains(2));
  EXPECT_FALSE(c.Contains(3));
  EXPECT_FALSE(c.Contains(6));
  EXPECT_TRUE(c.Contains(7));
  EXPECT_TRUE(c.Contains(100));
  EXPECT_TRUE(c.Contains(-100));
}

TEST_F(IntervalSetTest, ComplementUniverse) {
  auto c = Complement(IntervalSet::MakeUniverse());
  EXPECT_TRUE(c.IsEmpty());
}

TEST_F(IntervalSetTest, ComplementEmpty) {
  auto c = Complement(IntervalSet::MakeEmpty());
  EXPECT_TRUE(c.IsUniverse());
}

TEST_F(IntervalSetTest, ComplementMultiple) {
  auto s = Unite(IntervalSet(Interval::HalfOpen(0, 3)),
                 IntervalSet(Interval::HalfOpen(5, 8)));
  auto c = Complement(s);
  EXPECT_TRUE(c.Contains(-1));
  EXPECT_FALSE(c.Contains(0));
  EXPECT_FALSE(c.Contains(2));
  EXPECT_TRUE(c.Contains(3));
  EXPECT_TRUE(c.Contains(4));
  EXPECT_FALSE(c.Contains(5));
  EXPECT_FALSE(c.Contains(7));
  EXPECT_TRUE(c.Contains(8));
  EXPECT_TRUE(c.Contains(100));
}

TEST_F(IntervalSetTest, ProvablyLT) {
  auto s = IntervalSet(Interval::HalfOpen(0, 5));
  EXPECT_TRUE(s.ProvablyLT(5));
  EXPECT_TRUE(s.ProvablyLT(6));
  EXPECT_FALSE(s.ProvablyLT(4));
}

TEST_F(IntervalSetTest, ProvablyGE) {
  auto s = IntervalSet(Interval::HalfOpen(3, 10));
  EXPECT_TRUE(s.ProvablyGE(3));
  EXPECT_TRUE(s.ProvablyGE(2));
  EXPECT_FALSE(s.ProvablyGE(4));
}

TEST_F(IntervalSetTest, ProvablyLE) {
  auto s = IntervalSet(Interval::HalfOpen(0, 5));
  EXPECT_TRUE(s.ProvablyLE(4));
  EXPECT_TRUE(s.ProvablyLE(5));
  EXPECT_FALSE(s.ProvablyLE(3));
}

TEST_F(IntervalSetTest, ProvablyGT) {
  auto s = IntervalSet(Interval::HalfOpen(3, 10));
  EXPECT_TRUE(s.ProvablyGT(2));
  EXPECT_FALSE(s.ProvablyGT(3));
}

TEST_F(IntervalSetTest, IsSubsetOf) {
  auto a = IntervalSet(Interval::HalfOpen(2, 5));
  auto b = IntervalSet(Interval::HalfOpen(0, 10));
  EXPECT_TRUE(a.IsSubsetOf(b));
  EXPECT_FALSE(b.IsSubsetOf(a));
}

// =========================================================================
// Interval arithmetic tests
// =========================================================================

class IntervalArithTest : public ::testing::Test {};

TEST_F(IntervalArithTest, Add) {
  auto a = IntervalSet(Interval::HalfOpen(0, 5));
  auto b = IntervalSet(Interval::HalfOpen(10, 20));
  auto c = IntervalAdd(a, b);
  EXPECT_TRUE(c.ProvablyGE(10));
  EXPECT_TRUE(c.ProvablyLT(24));
}

TEST_F(IntervalArithTest, Sub) {
  auto a = IntervalSet(Interval::HalfOpen(5, 10));
  auto b = IntervalSet(Interval::HalfOpen(0, 3));
  auto c = IntervalSub(a, b);
  EXPECT_TRUE(c.ProvablyGE(3));
  EXPECT_TRUE(c.ProvablyLT(10));
}

TEST_F(IntervalArithTest, Neg) {
  auto a = IntervalSet(Interval::HalfOpen(3, 7));
  auto c = IntervalNeg(a);
  EXPECT_TRUE(c.Contains(-6));
  EXPECT_TRUE(c.Contains(-3));
  EXPECT_FALSE(c.Contains(-7));
  EXPECT_FALSE(c.Contains(-2));
}

TEST_F(IntervalArithTest, MulNonNeg) {
  auto a = IntervalSet(Interval::HalfOpen(2, 5));
  auto b = IntervalSet(Interval::HalfOpen(3, 7));
  auto c = IntervalMul(a, b);
  EXPECT_TRUE(c.ProvablyGE(6));
  EXPECT_TRUE(c.ProvablyLT(25));
}

TEST_F(IntervalArithTest, MulReturnsUniverseForNegative) {
  auto a = IntervalSet(Interval::HalfOpen(-5, 5));
  auto b = IntervalSet(Interval::HalfOpen(3, 7));
  auto c = IntervalMul(a, b);
  EXPECT_TRUE(c.IsUniverse());
}

TEST_F(IntervalArithTest, AddEmpty) {
  auto a = IntervalSet::MakeEmpty();
  auto b = IntervalSet(Interval::HalfOpen(0, 5));
  EXPECT_TRUE(IntervalAdd(a, b).IsEmpty());
}

// =========================================================================
// Constraint projection tests
// =========================================================================

class ProjectTest : public ::testing::Test {
protected:
  Operand p, a, q;
  void SetUp() override {
    p = sym("p");
    a = sym("a");
    q = sym("q");
  }
};

TEST_F(ProjectTest, SimpleLT) {
  auto pred = oc_lt(p, nu(4))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.ProvablyLT(4));
  EXPECT_TRUE(iv.Contains(3));
  EXPECT_FALSE(iv.Contains(4));
}

TEST_F(ProjectTest, SimpleGE) {
  auto pred = oc_ge(p, nu(0))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.ProvablyGE(0));
  EXPECT_TRUE(iv.Contains(0));
  EXPECT_FALSE(iv.Contains(-1));
}

TEST_F(ProjectTest, SimpleLE) {
  auto pred = oc_le(p, nu(5))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.Contains(5));
  EXPECT_FALSE(iv.Contains(6));
}

TEST_F(ProjectTest, SimpleGT) {
  auto pred = oc_gt(p, nu(2))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_FALSE(iv.Contains(2));
  EXPECT_TRUE(iv.Contains(3));
}

TEST_F(ProjectTest, SimpleEQ) {
  auto pred = bop(OpCode::EQ, p, nu(5))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.IsSingleton());
  EXPECT_EQ(iv.SingletonValue(), 5);
}

TEST_F(ProjectTest, AndConjunction) {
  auto pred = bl_and(oc_ge(p, nu(0)), oc_lt(p, nu(4)))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.ProvablyGE(0));
  EXPECT_TRUE(iv.ProvablyLT(4));
  EXPECT_TRUE(iv.Contains(0));
  EXPECT_TRUE(iv.Contains(3));
  EXPECT_FALSE(iv.Contains(4));
  EXPECT_FALSE(iv.Contains(-1));
}

TEST_F(ProjectTest, OrDisjunction) {
  auto pred = bl_or(oc_lt(p, nu(3)), oc_ge(p, nu(5)))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.Contains(2));
  EXPECT_FALSE(iv.Contains(3));
  EXPECT_FALSE(iv.Contains(4));
  EXPECT_TRUE(iv.Contains(5));
  EXPECT_TRUE(iv.Contains(100));
}

TEST_F(ProjectTest, NotPredicate) {
  auto pred = uop(OpCode::NOT, oc_ge(p, nu(4)))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.Contains(3));
  EXPECT_FALSE(iv.Contains(4));
}

TEST_F(ProjectTest, LinearIsolation) {
  auto pred = oc_lt(p + nu(1), nu(5))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.Contains(3));
  EXPECT_FALSE(iv.Contains(4));
}

TEST_F(ProjectTest, LinearIsolationSubtract) {
  auto pred = oc_ge(p - nu(2), nu(0))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.Contains(2));
  EXPECT_FALSE(iv.Contains(1));
}

TEST_F(ProjectTest, UnrelatedVariable) {
  auto pred = oc_lt(a, nu(4))->Normalize();
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.IsUniverse());
}

TEST_F(ProjectTest, MixedAndIsolation) {
  auto pred = bl_and(oc_lt(p, nu(4)), oc_lt(a, nu(1)))->Normalize();
  auto iv_p = ProjectConstraint(pred, "p");
  auto iv_a = ProjectConstraint(pred, "a");
  EXPECT_TRUE(iv_p.ProvablyLT(4));
  EXPECT_TRUE(iv_a.ProvablyLT(1));
}

TEST_F(ProjectTest, TransitiveWithEnv) {
  ConstraintEnv env;
  env["a"] = IntervalSet(Interval::HalfOpen(0, 3));
  auto pred = oc_lt(p, a)->Normalize();
  auto iv = ProjectConstraint(pred, "p", env);
  EXPECT_TRUE(iv.ProvablyLT(3));
}

TEST_F(ProjectTest, BooleanTrue) {
  auto pred = bl(true);
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.IsUniverse());
}

TEST_F(ProjectTest, BooleanFalse) {
  auto pred = bl(false);
  auto iv = ProjectConstraint(pred, "p");
  EXPECT_TRUE(iv.IsEmpty());
}

// =========================================================================
// EvalPredInterval tests
// =========================================================================

class EvalPredTest : public ::testing::Test {
protected:
  Operand p, a;
  void SetUp() override {
    p = sym("p");
    a = sym("a");
  }
};

TEST_F(EvalPredTest, ConstantTrue) {
  auto result = EvalPredInterval(bl(true), {});
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, ConstantFalse) {
  auto result = EvalPredInterval(bl(false), {});
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(*result);
}

TEST_F(EvalPredTest, ProvablyTrueLT) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 5));
  auto pred = oc_lt(p, nu(5))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, ProvablyFalseLT) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(5, 10));
  auto pred = oc_lt(p, nu(5))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(*result);
}

TEST_F(EvalPredTest, ProvablyTrueGE) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 5));
  auto pred = oc_ge(p, nu(0))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, Unknown) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 10));
  auto pred = oc_lt(p, nu(5))->Normalize();
  auto result = EvalPredInterval(pred, env);
  EXPECT_FALSE(result.has_value());
}

TEST_F(EvalPredTest, ArithmeticExpr) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 4));
  auto pred = oc_lt(p + nu(1), nu(5))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, AndPredicate) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 5));
  auto pred = bl_and(oc_ge(p, nu(0)), oc_lt(p, nu(5)))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, OrPredicate) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 3));
  auto pred = bl_or(oc_lt(p, nu(5)), oc_gt(p, nu(10)))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, NotPredicate) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 3));
  auto pred = uop(OpCode::NOT, oc_ge(p, nu(5)))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, TwoVariables) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 4));
  env["a"] = IntervalSet(Interval::HalfOpen(0, 8));
  auto pred = oc_lt(p, a)->Normalize();
  auto result = EvalPredInterval(pred, env);
  EXPECT_FALSE(result.has_value());
}

TEST_F(EvalPredTest, TwoVariablesProvable) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 4));
  env["a"] = IntervalSet(Interval::HalfOpen(5, 10));
  auto pred = oc_lt(p, a)->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, EqualitySingleton) {
  ConstraintEnv env;
  env["p"] = IntervalSet::MakePoint(5);
  auto pred = bop(OpCode::EQ, p, nu(5))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EvalPredTest, NEDisjoint) {
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 3));
  auto pred = bop(OpCode::NE, p, nu(5))->Normalize();
  auto result = EvalPredInterval(pred, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

// =========================================================================
// End-to-end scenario tests
// =========================================================================

class EndToEndTest : public ::testing::Test {};

TEST_F(EndToEndTest, ParallelIfArrayAccess) {
  auto p = sym("p");
  IntervalSet type_bound(Interval::HalfOpen(0, 5));
  auto scope_pred = oc_lt(p, nu(4))->Normalize();
  auto scope_iv = ProjectConstraint(scope_pred, "p");
  auto narrowed = Intersect(type_bound, scope_iv);
  EXPECT_TRUE(narrowed.ProvablyGE(0));
  EXPECT_TRUE(narrowed.ProvablyLT(4));

  ConstraintEnv env;
  env["p"] = narrowed;

  auto lb_check = oc_ge(p, nu(0))->Normalize();
  auto lb_result = EvalPredInterval(lb_check, env);
  ASSERT_TRUE(lb_result.has_value());
  EXPECT_TRUE(*lb_result);

  auto ub_check = oc_lt(p, nu(4))->Normalize();
  auto ub_result = EvalPredInterval(ub_check, env);
  ASSERT_TRUE(ub_result.has_value());
  EXPECT_TRUE(*ub_result);
}

TEST_F(EndToEndTest, ParallelIfArithmeticIndex) {
  auto p = sym("p");
  IntervalSet type_bound(Interval::HalfOpen(0, 5));
  auto scope_iv = ProjectConstraint(oc_lt(p, nu(3))->Normalize(), "p");
  auto narrowed = Intersect(type_bound, scope_iv);

  ConstraintEnv env;
  env["p"] = narrowed;

  auto check = oc_lt(p + nu(1), nu(5))->Normalize();
  auto result = EvalPredInterval(check, env);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(EndToEndTest, DisjunctiveRange) {
  auto p = sym("p");
  auto range = Unite(IntervalSet(Interval::HalfOpen(0, 3)),
                     IntervalSet(Interval::HalfOpen(5, 8)));

  ConstraintEnv env;
  env["p"] = range;

  auto check_lb = oc_ge(p, nu(0))->Normalize();
  auto result_lb = EvalPredInterval(check_lb, env);
  ASSERT_TRUE(result_lb.has_value());
  EXPECT_TRUE(*result_lb);

  auto check_ub = oc_lt(p, nu(8))->Normalize();
  auto result_ub = EvalPredInterval(check_ub, env);
  ASSERT_TRUE(result_ub.has_value());
  EXPECT_TRUE(*result_ub);

  auto check_mid = oc_lt(p, nu(5))->Normalize();
  auto result_mid = EvalPredInterval(check_mid, env);
  EXPECT_FALSE(result_mid.has_value());
}

// Nested if narrowing: parallel p by 5, if (p < 6) if (p > 2)
// p in [0,5) narrowed by (p<6) = [0,5), then by (p>2) = [3,5)
TEST_F(EndToEndTest, NestedIfNarrowing) {
  auto p = sym("p");
  IntervalSet type_bound(Interval::HalfOpen(0, 5));

  auto pred1 = oc_lt(p, nu(6))->Normalize();
  auto pred2 = oc_gt(p, nu(2))->Normalize();
  auto combined = bl_and(pred1, pred2)->Normalize();

  auto scope_iv = ProjectConstraint(combined, "p");
  auto narrowed = Intersect(type_bound, scope_iv);
  // [0,5) intersect [3,6) = [3,5)
  EXPECT_TRUE(narrowed.ProvablyGE(3));
  EXPECT_TRUE(narrowed.ProvablyLT(5));

  ConstraintEnv env;
  env["p"] = narrowed;

  // buf.at(p - 2): p-2 in [1,3), buf size 3 => p-2 >= 0 and p-2 < 3
  auto check_lb = oc_ge(p - nu(2), nu(0))->Normalize();
  auto result_lb = EvalPredInterval(check_lb, env);
  ASSERT_TRUE(result_lb.has_value());
  EXPECT_TRUE(*result_lb);

  auto check_ub = oc_lt(p - nu(2), nu(3))->Normalize();
  auto result_ub = EvalPredInterval(check_ub, env);
  ASSERT_TRUE(result_ub.has_value());
  EXPECT_TRUE(*result_ub);
}

// Transitive: parallel p by 5, if (p < a && a < b) if (b <= 5) => p < 5
TEST_F(EndToEndTest, SymbolicTransitive) {
  auto p = sym("p");
  auto a = sym("a");
  auto b = sym("b");

  IntervalSet type_bound_p(Interval::HalfOpen(0, 5));

  // First scope: (p < a && a < b)
  // Second scope: (b <= 5)
  // Combined: p < a && a < b && b <= 5
  auto pred =
      bl_and(bl_and(oc_lt(p, a), oc_lt(a, b)), oc_le(b, nu(5)))->Normalize();

  // Build environment iteratively: first b, then a, then p
  ConstraintEnv env;

  // b <= 5 => b in (-inf, 6)
  auto b_iv = ProjectConstraint(pred, "b", env);
  env["b"] = b_iv;
  EXPECT_TRUE(b_iv.ProvablyLT(6));

  // a < b, with b's UB = 6 => a in (-inf, 6)
  auto a_iv = ProjectConstraint(pred, "a", env);
  env["a"] = a_iv;
  EXPECT_TRUE(a_iv.ProvablyLT(6));

  // p < a, with a's UB = 6 => p in (-inf, 6)
  // Intersect with type_bound [0, 5) => [0, 5)
  auto p_scope_iv = ProjectConstraint(pred, "p", env);
  auto p_iv = Intersect(type_bound_p, p_scope_iv);
  env["p"] = p_iv;
  EXPECT_TRUE(p_iv.ProvablyGE(0));
  EXPECT_TRUE(p_iv.ProvablyLT(5));

  // buf of size 5: p >= 0 and p < 5
  auto check_lb = oc_ge(p, nu(0))->Normalize();
  auto result_lb = EvalPredInterval(check_lb, env);
  ASSERT_TRUE(result_lb.has_value());
  EXPECT_TRUE(*result_lb);

  auto check_ub = oc_lt(p, nu(5))->Normalize();
  auto result_ub = EvalPredInterval(check_ub, env);
  ASSERT_TRUE(result_ub.has_value());
  EXPECT_TRUE(*result_ub);
}

// Stats tracking test
TEST_F(EndToEndTest, StatsTracking) {
  auto& profiler = IntervalProfiler::Get();
  profiler.Reset();

  auto p = sym("p");
  ConstraintEnv env;
  env["p"] = IntervalSet(Interval::HalfOpen(0, 5));

  auto pred = oc_lt(p, nu(5))->Normalize();
  ProjectConstraint(pred, "p", env);
  EvalPredInterval(pred, env);

  auto snap = profiler.Snapshot();
  EXPECT_GT(snap.project_calls, 0u);
  EXPECT_GT(snap.eval_pred_calls, 0u);
  EXPECT_GT(snap.proven_true, 0u);
}
