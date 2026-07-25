#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include "sigutils/neumaier.hpp"

// ============================================================================
// 1. Type-Parameterized Tests (Runs all tests for both float and double)
// ============================================================================

template <typename T>
class StableFloatSumTypedTest : public ::testing::Test {};

using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StableFloatSumTypedTest, FloatingTypes);

TYPED_TEST(StableFloatSumTypedTest, InitialStateIsZero) {
    StableFloatSum<TypeParam> acc;
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(0));
}

TYPED_TEST(StableFloatSumTypedTest, SingleAdditionAndSubtraction) {
    StableFloatSum<TypeParam> acc;
    acc += TypeParam(42.5);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(42.5));

    acc -= TypeParam(12.5);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(30.0));
}

TYPED_TEST(StableFloatSumTypedTest, AddingAndSubtractingZero) {
    StableFloatSum<TypeParam> acc;
    acc += TypeParam(10.0);
    acc += TypeParam(0.0);
    acc += TypeParam(-0.0);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(10.0));

    acc -= TypeParam(0.0);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(10.0));
}

TYPED_TEST(StableFloatSumTypedTest, AbsoluteEqualityBranch) {
    // Tests std::abs(sum) == std::abs(val) specifically
    StableFloatSum<TypeParam> acc;
    acc += TypeParam(10.0);
    acc += TypeParam(10.0); // |10.0| >= |10.0| (True branch)
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(20.0));

    acc += TypeParam(-20.0); // |20.0| >= |-20.0| (True branch)
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(0.0));
}

TYPED_TEST(StableFloatSumTypedTest, NegativeAccumulation) {
    StableFloatSum<TypeParam> acc;
    acc -= TypeParam(15.0);
    acc -= TypeParam(25.0);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(-40.0));

    acc += TypeParam(10.0);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(-30.0));
}

TYPED_TEST(StableFloatSumTypedTest, ResetClearsAccumulatedCompensation) {
    StableFloatSum<TypeParam> acc;
    
    // Accumulate value and force error compensation c to be non-zero
    acc += TypeParam(1e6);
    acc += TypeParam(1e-6);
    acc -= TypeParam(1e6);

    // Reset should clear both sum and c
    acc.reset();
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(0));

    // Verify state after reset behaves cleanly like a new object
    acc += TypeParam(5.0);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(5.0));
}

TYPED_TEST(StableFloatSumTypedTest, OperatorAssignOverwritesEverything) {
    StableFloatSum<TypeParam> acc;
    
    // Fill accumulator with drift/compensation
    acc += TypeParam(1e5);
    acc += TypeParam(1e-5);
    
    // Overwrite via operator=
    acc = TypeParam(123.45);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(123.45));

    // Ensure compensation was reset by operator=
    acc -= TypeParam(23.45);
    EXPECT_EQ(static_cast<TypeParam>(acc), TypeParam(100.0));
}

// ============================================================================
// 2. Syntax & Chaining Tests
// ============================================================================

TEST(StableFloatSumSyntaxTest, OperatorChaining) {
    StableFloatSum<double> acc;
    
    // Test operator+= chaining
    (acc += 10.0) += 20.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(acc), 30.0);

    // Test operator-= chaining
    (acc -= 5.0) -= 5.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(acc), 20.0);

    // Test operator= chaining
    StableFloatSum<double> acc2;
    acc2 = acc = 99.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(acc), 99.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(acc2), 99.0);
}

TEST(StableFloatSumSyntaxTest, ConstCorrectness) {
    StableFloatSum<double> acc;
    acc += 50.0;

    // Const reference check on conversion operator
    const auto& const_ref = acc;
    double val = const_ref;
    EXPECT_DOUBLE_EQ(val, 50.0);
}

// ============================================================================
// 3. Numerical Precision & Stress Tests
// ============================================================================

TEST(StableFloatSumStressTest, FloatCatastrophicCancellation) {
    // In naive float arithmetic: (1e8f + 1.0f) - 1e8f == 0.0f
    // Neumaier recovers the 1.0f precisely.
    StableFloatSum<float> acc;
    
    acc += 1e8f;
    acc += 1.0f;
    acc -= 1e8f;

    EXPECT_FLOAT_EQ(static_cast<float>(acc), 1.0f);
}

TEST(StableFloatSumStressTest, DoubleCatastrophicCancellation) {
    // In naive double arithmetic: (1e16 + 1.0) - 1e16 == 0.0
    StableFloatSum<double> acc;

    acc += 1e16;
    acc += 1.0;
    acc -= 1e16;

    EXPECT_DOUBLE_EQ(static_cast<double>(acc), 1.0);
}

TEST(StableFloatSumStressTest, AlternatingLargeAndSmallValues) {
    StableFloatSum<double> acc;

    const double large_val = 1e12;
    const double small_val = 1e-4;
    const int iterations = 10000;

    acc += large_val;
    for (int i = 0; i < iterations; ++i) {
        acc += small_val;
    }
    acc -= large_val;

    // Compare with expected total: 10000 * 1e-4 = 1.0
    EXPECT_NEAR(static_cast<double>(acc), 1.0, 1e-9);
}

TEST(StableFloatSumStressTest, CompareWithNaiveSummation) {
    std::vector<float> data(100000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (i % 2 == 0) ? 10000.0f : 1e-4f;
    }

    // 1. Naive loop
    float naive_sum = 0.0f;
    for (float v : data) {
        naive_sum += v;
    }

    // 2. Neumaier sum
    StableFloatSum<float> neumaier_acc;
    for (float v : data) {
        neumaier_acc += v;
    }

    // Exact analytical sum: 50000 * 10000.0f + 50000 * 1e-4f = 500000000.0 + 5.0
    double exact_sum = 500000005.0;

    double naive_error = std::abs(static_cast<double>(naive_sum) - exact_sum);
    double neumaier_error = std::abs(static_cast<double>(neumaier_acc) - exact_sum);

    // Neumaier must be significantly more accurate than naive addition
    EXPECT_LT(neumaier_error, naive_error);
}

TEST(StableFloatSumStressTest, LargeArraySequenceRandomizedScale) {
    StableFloatSum<double> acc;
    
    // Sequence of numbers that sum to exactly 0.0
    std::vector<double> numbers = { 1e15, 3.1415926535, -1e15, 2.7182818284, -5.8598744819 };
    
    for (double num : numbers) {
        acc += num;
    }

    EXPECT_NEAR(static_cast<double>(acc), 0.0, 1e-12);
}