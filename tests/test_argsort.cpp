#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "sigutils/argsort.hpp"

// =======================================================================
// Helpers
// =======================================================================

// Verifies idx is a valid permutation of [0, length) and that
// values[idx[0]] <= values[idx[1]] <= ... <= values[idx[length-1]].
template <typename FloatType>
void expect_valid_ascending_permutation(const std::vector<FloatType>& values,
                                         const std::vector<uint16_t>& idx) {
    size_t length = values.size();
    ASSERT_EQ(idx.size(), length);

    // Permutation check: every index in [0,length) appears exactly once.
    std::vector<bool> seen(length, false);
    for (size_t i = 0; i < length; i++) {
        ASSERT_LT(idx[i], length) << "idx[" << i << "] out of range";
        ASSERT_FALSE(seen[idx[i]]) << "index " << idx[i] << " repeated";
        seen[idx[i]] = true;
    }

    // Ascending order check.
    for (size_t i = 0; i + 1 < length; i++) {
        EXPECT_LE(values[idx[i]], values[idx[i + 1]])
            << "not ascending at position " << i;
    }
}

template <typename FloatType>
std::vector<uint16_t> run_argsort(std::vector<FloatType> values) {
    std::vector<uint16_t> idx(values.size());
    argsort(values.data(), idx.data(), values.size());
    return idx;
}

// =======================================================================
// Basic correctness
// =======================================================================

TEST(Argsort, EmptyInputProducesEmptyOutput) {
    std::vector<double> values;
    std::vector<uint16_t> idx(0);
    argsort(values.data(), idx.data(), 0);  // must not read/write anything
    EXPECT_TRUE(idx.empty());
}

TEST(Argsort, SingleElement) {
    std::vector<double> values = {42.0};
    auto idx = run_argsort(values);
    ASSERT_EQ(idx.size(), 1u);
    EXPECT_EQ(idx[0], 0);
}

TEST(Argsort, AlreadyAscending) {
    std::vector<double> values = {1, 2, 3, 4, 5};
    auto idx = run_argsort(values);
    std::vector<uint16_t> expected = {0, 1, 2, 3, 4};
    EXPECT_EQ(idx, expected);
    expect_valid_ascending_permutation(values, idx);
}

TEST(Argsort, FullyDescendingGetsReversed) {
    std::vector<double> values = {5, 4, 3, 2, 1};
    auto idx = run_argsort(values);
    std::vector<uint16_t> expected = {4, 3, 2, 1, 0};
    EXPECT_EQ(idx, expected);
    expect_valid_ascending_permutation(values, idx);
}

TEST(Argsort, ArbitraryOrder) {
    std::vector<double> values = {30, 10, 40, 10, 50, 90, 20};
    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);
    // Sanity check on a couple of fixed points.
    EXPECT_EQ(values[idx.front()], 10);
    EXPECT_EQ(values[idx.back()], 90);
}

TEST(Argsort, NegativeAndPositiveValuesMixed) {
    std::vector<double> values = {-5, 3, -100, 0, 42, -1};
    auto idx = run_argsort(values);
    std::vector<uint16_t> expected = {2, 0, 5, 3, 1, 4};  // -100,-5,-1,0,3,42
    EXPECT_EQ(idx, expected);
    expect_valid_ascending_permutation(values, idx);
}

TEST(Argsort, AllEqualValues) {
    std::vector<double> values = {7, 7, 7, 7, 7};
    auto idx = run_argsort(values);
    // No ordering guarantee is made among equal keys (see the
    // "NOT stable" tests below) -- only that the result is a valid
    // permutation and trivially non-decreasing, since every value is equal.
    expect_valid_ascending_permutation(values, idx);
}

// =======================================================================
// Ties: NOT stable (documented, expected behavior)
// =======================================================================
//
// This argsort is a Shellsort (gap-sequence insertion sort). Shellsort is
// NOT a stable sort: during the large-gap passes, elements are compared and
// swapped against others `gap` positions away using a strict '>' test, which
// can permute equal-valued elements relative to each other. The final
// gap=1 pass (plain insertion sort) guarantees the output is correctly
// non-decreasing, but it does *not* restore original relative order for
// ties, since by that point the equal-valued elements may already sit in a
// different relative order than they started in.
//
// (One exception worth knowing about: if *every* value in the array is
// equal, as in AllEqualValues above, the strict '>' comparison never fires
// at all, so the buffer never moves from its identity initialization and
// original order happens to be preserved. That's an artifact of comparing
// identical values, not a general stability guarantee -- as soon as any
// other value is present to move around, ties can be reordered, as shown
// below.)

TEST(Argsort, TiesAreNotGuaranteedStableWhenOtherValuesArePresent) {
    // Indices of value==1 are originally {1,3,6}; indices of value==5 are
    // originally {0,2,4}. With a stable sort these would stay in that
    // relative order. This Shellsort does not make that guarantee, and in
    // practice reorders them for this input -- this test pins down and
    // documents that actual (non-stable) behavior as a regression check.
    std::vector<double> values = {5, 1, 5, 1, 5, 2, 1};
    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);  // correctness still holds

    std::vector<uint16_t> ones;
    for (uint16_t i : idx) if (values[i] == 1) ones.push_back(i);
    std::vector<uint16_t> fives;
    for (uint16_t i : idx) if (values[i] == 5) fives.push_back(i);

    // Both groups must still be exactly {1,3,6} and {0,2,4} as *sets* --
    // argsort must not lose, duplicate, or invent indices.
    std::vector<uint16_t> expected_ones_set = {1, 3, 6};
    std::vector<uint16_t> expected_fives_set = {0, 2, 4};
    std::vector<uint16_t> sorted_ones = ones, sorted_fives = fives;
    std::sort(sorted_ones.begin(), sorted_ones.end());
    std::sort(sorted_fives.begin(), sorted_fives.end());
    EXPECT_EQ(sorted_ones, expected_ones_set);
    EXPECT_EQ(sorted_fives, expected_fives_set);

    // But relative order within each group is NOT preserved for this input --
    // this is the documented non-stability, not a bug.
    EXPECT_NE(ones, expected_ones_set)
        << "If this starts passing, either the algorithm changed to be "
           "stable, or this particular input stopped exercising the "
           "reordering -- worth double-checking either way.";
}

TEST(Argsort, TiesRemainAValidPermutationEvenWhenReordered) {
    // Three interleaved tied groups. We only assert the two properties this
    // argsort actually guarantees: a valid permutation, and correct
    // non-decreasing order -- explicitly NOT relative order within a tied
    // group (see TiesAreNotGuaranteedStableWhenOtherValuesArePresent).
    std::vector<double> values = {1, 2, 1, 3, 2, 1, 3, 2, 1};
    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);

    for (double v : {1.0, 2.0, 3.0}) {
        std::vector<uint16_t> original, sorted_out;
        for (size_t i = 0; i < values.size(); i++)
            if (values[i] == v) original.push_back(static_cast<uint16_t>(i));
        for (uint16_t i : idx)
            if (values[i] == v) sorted_out.push_back(i);

        std::vector<uint16_t> sorted_copy = sorted_out;
        std::sort(sorted_copy.begin(), sorted_copy.end());
        EXPECT_EQ(sorted_copy, original)
            << "argsort must not lose/duplicate/invent indices for value " << v;
    }
}

// =======================================================================
// Boundary / extreme values
// =======================================================================

TEST(Argsort, HandlesZeroAndNegativeZero) {
    // -0.0 == 0.0 under operator<, so these are "tied"; no relative-order
    // guarantee is made between them (see the ties/stability tests above).
    std::vector<double> values = {0.0, -0.0, 1.0, -1.0};
    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);
    EXPECT_EQ(idx.front(), 3);  // -1.0 must sort first
    EXPECT_EQ(idx.back(), 2);   // 1.0 must sort last
    // The middle two positions must be exactly {0,1} (the two zeros), in
    // either order.
    std::vector<uint16_t> middle = {idx[1], idx[2]};
    std::sort(middle.begin(), middle.end());
    std::vector<uint16_t> expected_middle = {0, 1};
    EXPECT_EQ(middle, expected_middle);
}

TEST(Argsort, HandlesExtremeMagnitudes) {
    std::vector<double> values = {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::min(),
        0.0,
        -1.0,
    };
    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);
    EXPECT_EQ(values[idx.front()], std::numeric_limits<double>::lowest());
    EXPECT_EQ(values[idx.back()], std::numeric_limits<double>::max());
}

// =======================================================================
// Does not overrun the requested length
// =======================================================================

TEST(Argsort, LargeArrayExercisesEveryGapInTheSequence) {
    // The implementation is a Shellsort with gaps {701,301,132,57,23,10,4,1}.
    // Arrays smaller than 701 never let the largest gaps' inner loops
    // actually shift anything (the loop body is skipped, not exercised).
    // Use an array well past the largest gap so every pass does real work.
    const int length = 2000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> value_dist(-1e6, 1e6);

    std::vector<double> values(length);
    for (int i = 0; i < length; i++) values[i] = value_dist(rng);

    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);
}

TEST(Argsort, LargeArrayWithReverseSortedInput) {
    // Worst case for insertion-sort-style passes: fully reverse order,
    // sized past the largest gap so every pass has real shifting to do.
    const int length = 1500;
    std::vector<double> values(length);
    for (int i = 0; i < length; i++) values[i] = static_cast<double>(length - i);

    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);
    EXPECT_EQ(values[idx.front()], 1.0);
    EXPECT_EQ(values[idx.back()], static_cast<double>(length));
}

TEST(Argsort, LargeArrayAtExactGapBoundary) {
    // Length exactly equal to the largest gap (701) is an edge case worth
    // pinning down explicitly: the first gap's inner loop condition
    // (idx < length) is never true for idx starting at src+701 when
    // length==701, so that pass is a no-op; later gaps must still finish
    // the sort correctly.
    const int length = 701;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> value_dist(-100.0, 100.0);

    std::vector<double> values(length);
    for (int i = 0; i < length; i++) values[i] = value_dist(rng);

    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);
}

TEST(Argsort, OnlyTouchesFirstLengthElements) {
    // Buffer is larger than the requested length; argsort must not read or
    // write past `length`, and must not be influenced by the extra data.
    std::vector<double> values = {50, 10, 30, /* untouched tail: */ -999, -999};
    std::vector<uint16_t> idx(5, 0xFFFF);  // sentinel for untouched slots

    argsort(values.data(), idx.data(), 3);

    expect_valid_ascending_permutation(
        std::vector<double>(values.begin(), values.begin() + 3),
        std::vector<uint16_t>(idx.begin(), idx.begin() + 3));

    // The tail of idx must be untouched.
    EXPECT_EQ(idx[3], 0xFFFF);
    EXPECT_EQ(idx[4], 0xFFFF);
}

// =======================================================================
// Type genericity (float vs double)
// =======================================================================

template <typename FloatType>
class ArgsortTypedTest : public ::testing::Test {};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ArgsortTypedTest, FloatTypes);

TYPED_TEST(ArgsortTypedTest, SortsCorrectlyForType) {
    using FloatType = TypeParam;
    std::vector<FloatType> values = {
        static_cast<FloatType>(3.5), static_cast<FloatType>(-2.25),
        static_cast<FloatType>(0.0), static_cast<FloatType>(1.75),
        static_cast<FloatType>(-10.0)};
    auto idx = run_argsort(values);
    expect_valid_ascending_permutation(values, idx);
    EXPECT_EQ(values[idx.front()], static_cast<FloatType>(-10.0));
    EXPECT_EQ(values[idx.back()], static_cast<FloatType>(3.5));
}

// =======================================================================
// Randomized property-based test
// =======================================================================

TYPED_TEST(ArgsortTypedTest, RandomizedPermutationAndOrderingProperty) {
    using FloatType = TypeParam;
    std::mt19937 rng(987654321);
    std::uniform_int_distribution<int> length_dist(0, 60);
    std::uniform_real_distribution<double> value_dist(-1000.0, 1000.0);

    // Bias some trials toward heavy duplication to stress tie-handling.
    std::uniform_int_distribution<int> duplicate_bias(0, 3);

    for (int trial = 0; trial < 300; trial++) {
        int length = length_dist(rng);
        std::vector<FloatType> values(length);
        for (int i = 0; i < length; i++) {
            double raw = value_dist(rng);
            if (duplicate_bias(rng) == 0) {
                // Round to force more ties.
                raw = std::round(raw / 50.0) * 50.0;
            }
            values[i] = static_cast<FloatType>(raw);
        }

        SCOPED_TRACE(::testing::Message() << "trial=" << trial << " length=" << length);
        auto idx = run_argsort(values);
        expect_valid_ascending_permutation(values, idx);
        // Note: no pairwise-stability check here -- this argsort (Shellsort)
        // does not guarantee stable ordering among tied values; see the
        // dedicated tie tests above for that documented behavior.
    }
}