#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "sigutils/neumaier.hpp"

// Edge cases for the compensated sum: non finite input from a dead sensor.

template <typename T>
class NeumaierEdges : public ::testing::Test {};
using Floats = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NeumaierEdges, Floats);

TYPED_TEST(NeumaierEdges, InfinityKeepsTheSumInfiniteNotNaN) {
    constexpr TypeParam inf = std::numeric_limits<TypeParam>::infinity();
    StableFloatSum<TypeParam> s;
    s += 1;
    s += inf;
    EXPECT_EQ(static_cast<TypeParam>(s), inf);
    s += 5;
    EXPECT_EQ(static_cast<TypeParam>(s), inf);
}

TYPED_TEST(NeumaierEdges, NegativeInfinityKeepsTheSumNegativeInfinite) {
    constexpr TypeParam inf = std::numeric_limits<TypeParam>::infinity();
    StableFloatSum<TypeParam> s;
    s -= inf;
    EXPECT_EQ(static_cast<TypeParam>(s), -inf);
}

TYPED_TEST(NeumaierEdges, OppositeInfinitiesGiveNaN) {
    constexpr TypeParam inf = std::numeric_limits<TypeParam>::infinity();
    StableFloatSum<TypeParam> s;
    s += inf;
    s -= inf;
    EXPECT_TRUE(std::isnan(static_cast<TypeParam>(s)));
}

TYPED_TEST(NeumaierEdges, NaNPropagatesAndResetRecovers) {
    StableFloatSum<TypeParam> s;
    s += 2;
    s += std::numeric_limits<TypeParam>::quiet_NaN();
    EXPECT_TRUE(std::isnan(static_cast<TypeParam>(s)));
    s += 3;
    EXPECT_TRUE(std::isnan(static_cast<TypeParam>(s)));

    s.reset();
    s += 5;
    EXPECT_FLOAT_EQ(static_cast<TypeParam>(s), 5);
}

TYPED_TEST(NeumaierEdges, AssignmentRecoversFromNaNAndInfinity) {
    StableFloatSum<TypeParam> s;
    s += std::numeric_limits<TypeParam>::quiet_NaN();
    s = 7;
    EXPECT_FLOAT_EQ(static_cast<TypeParam>(s), 7);

    s += std::numeric_limits<TypeParam>::infinity();
    s = 1;
    s += 1;
    EXPECT_FLOAT_EQ(static_cast<TypeParam>(s), 2);
}

// Float precision: the compensation term is itself a float, so the error grows
// with the number of adds. Still far better than the naive sum.
TEST(NeumaierEdgesLongRun, FloatHundredThousandTenthsIsWithinAHundredth) {
    StableFloatSum<float> s;
    float naive = 0;
    for (int i = 0; i < 100000; i++) {
        s += 0.1f;
        naive += 0.1f;
    }
    const double exact = 1e5 * static_cast<double>(0.1f);
    EXPECT_NEAR(static_cast<float>(s), exact, 0.01);
    EXPECT_GT(std::abs(naive - exact), 1.0);
}

TEST(NeumaierEdgesLongRun, FloatTenMillionTenthsDriftIsBoundedAndBeatsNaive) {
    // Measured: about 2000 off of 1e6, naive is about 88000 off.
    StableFloatSum<float> s;
    float naive = 0;
    for (int i = 0; i < 10000000; i++) {
        s += 0.1f;
        naive += 0.1f;
    }
    const double exact = 1e7 * static_cast<double>(0.1f);
    EXPECT_NEAR(static_cast<float>(s), exact, 3000.0);
    EXPECT_GT(std::abs(naive - exact), 50000.0);
}
