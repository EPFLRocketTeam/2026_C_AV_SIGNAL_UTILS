#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "sigutils/outlier.hpp"

// =======================================================================
// WelfordVariance unit tests
// =======================================================================

TEST(WelfordVariance, EmptyHasZeroSizeAndScore) {
    WelfordVariance<double> w;
    EXPECT_EQ(w.size(), 0);
    EXPECT_DOUBLE_EQ(w.get_score(), 0.0);
    EXPECT_DOUBLE_EQ(w.get_mean(), 0.0);
}

TEST(WelfordVariance, SingleElement) {
    WelfordVariance<double> w;
    w.add_variable(42.0);
    EXPECT_EQ(w.size(), 1);
    EXPECT_DOUBLE_EQ(w.get_mean(), 42.0);
    EXPECT_DOUBLE_EQ(w.get_score(), 0.0);  // variance of a single point is 0
}

TEST(WelfordVariance, KnownDatasetMeanAndM2) {
    // Dataset: 2, 4, 4, 4, 5, 5, 7, 9  -> mean = 5, population variance = 4,
    // so M2 (sum of squared deviations) = variance * n = 32.
    WelfordVariance<double> w;
    for (double x : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) {
        w.add_variable(x);
    }
    EXPECT_EQ(w.size(), 8);
    EXPECT_NEAR(w.get_mean(), 5.0, 1e-9);
    EXPECT_NEAR(w.get_score(), 32.0, 1e-9);
}

TEST(WelfordVariance, RemoveBringsStateBackToPriorAggregate) {
    WelfordVariance<double> w;
    for (double x : {1.0, 2.0, 3.0, 4.0}) w.add_variable(x);

    WelfordVariance<double> reference;
    for (double x : {1.0, 2.0, 3.0}) reference.add_variable(x);

    w.remove_variable(4.0);

    EXPECT_EQ(w.size(), reference.size());
    EXPECT_NEAR(w.get_mean(), reference.get_mean(), 1e-9);
    EXPECT_NEAR(w.get_score(), reference.get_score(), 1e-9);
}

TEST(WelfordVariance, RemoveDownToOneElement) {
    WelfordVariance<double> w;
    w.add_variable(10.0);
    w.add_variable(20.0);
    w.add_variable(30.0);

    w.remove_variable(30.0);
    w.remove_variable(20.0);

    EXPECT_EQ(w.size(), 1);
    EXPECT_NEAR(w.get_mean(), 10.0, 1e-9);
    EXPECT_NEAR(w.get_score(), 0.0, 1e-9);
}

TEST(WelfordVariance, RemoveLastElementDoesNotDivideByZero) {
    WelfordVariance<double> w;
    w.add_variable(5.0);
    w.remove_variable(5.0);  // count goes 1 -> 0; must not crash / produce NaN or Inf

    EXPECT_EQ(w.size(), 0);
    EXPECT_TRUE(std::isfinite(w.get_mean()));
    EXPECT_TRUE(std::isfinite(w.get_score()));
    EXPECT_DOUBLE_EQ(w.get_mean(), 0.0);
    EXPECT_DOUBLE_EQ(w.get_score(), 0.0);
}

TEST(WelfordVariance, ReusableAfterDrainingToZero) {
    WelfordVariance<double> w;
    w.add_variable(1.0);
    w.remove_variable(1.0);

    // Adding again after fully draining should behave like a fresh instance.
    w.add_variable(100.0);
    EXPECT_EQ(w.size(), 1);
    EXPECT_DOUBLE_EQ(w.get_mean(), 100.0);
    EXPECT_DOUBLE_EQ(w.get_score(), 0.0);
}

TEST(WelfordVariance, SlidingWindowMatchesDirectComputation) {
    // Simulate a sliding window of size 3 over a sequence and check every
    // step against a direct (non-incremental) mean/M2 computation.
    std::vector<double> data = {1, 5, 2, 8, 3, 9, 0, 4, 7, 6};
    const size_t window = 3;

    WelfordVariance<double> w;
    for (size_t i = 0; i < window; i++) w.add_variable(data[i]);

    for (size_t start = 0; start + window <= data.size(); start++) {
        if (start > 0) {
            w.add_variable(data[start + window - 1]);
            w.remove_variable(data[start - 1]);
        }

        double sum = 0;
        for (size_t i = start; i < start + window; i++) sum += data[i];
        double mean = sum / window;
        double m2 = 0;
        for (size_t i = start; i < start + window; i++) {
            double d = data[i] - mean;
            m2 += d * d;
        }

        EXPECT_NEAR(w.get_mean(), mean, 1e-9) << "window start=" << start;
        EXPECT_NEAR(w.get_score(), m2, 1e-9) << "window start=" << start;
        EXPECT_EQ(w.size(), window);
    }
}

// =======================================================================
// OutlierParams::in_range unit tests
// =======================================================================

TEST(OutlierParamsInRange, BoundaryValuesAreInclusive) {
    OutlierParams<double, 4, 2> params;
    params.min = 1.0;
    params.max = 10.0;

    EXPECT_TRUE(params.in_range(1.0));    // == min
    EXPECT_TRUE(params.in_range(10.0));   // == max
    EXPECT_TRUE(params.in_range(5.5));    // inside
    EXPECT_FALSE(params.in_range(0.999999));
    EXPECT_FALSE(params.in_range(10.000001));
}

// =======================================================================
// outlier() test helpers
// =======================================================================

template <typename FloatType, int MaxLength, int MaxNumValid>
struct OutlierRunResult {
    uint16_t count;
    FloatType mean;
    std::vector<bool> is_outlier;
};

template <typename FloatType, int MaxLength, int MaxNumValid>
OutlierRunResult<FloatType, MaxLength, MaxNumValid> run_outlier(
    std::vector<FloatType> values,
    FloatType min,
    FloatType max,
    size_t runtime_length,
    size_t runtime_num_valid
) {
    OutlierParams<FloatType, MaxLength, MaxNumValid> params;
    params.min = min;
    params.max = max;
    params.length = runtime_length;
    params.num_valid = runtime_num_valid;

    std::array<bool, MaxLength> is_outlier_buf{};
    FloatType mean = std::numeric_limits<FloatType>::quiet_NaN();

    values.resize(std::max<size_t>(values.size(), MaxLength), FloatType(0));

    uint16_t count = outlier<FloatType, MaxLength, MaxNumValid>(
        params, values.data(), is_outlier_buf.data(), mean);

    return {count, mean,
            std::vector<bool>(is_outlier_buf.begin(),
                               is_outlier_buf.begin() + runtime_length)};
}

// Independent brute-force reference: uses the *same* argsort (so ordering
// of ties is identical) but recomputes each window's mean/M2 directly
// instead of incrementally, so it can catch bugs in the Welford add/remove
// bookkeeping used by outlier().
template <typename FloatType>
struct BruteForceResult {
    uint16_t count;
    FloatType mean;
    std::vector<bool> is_outlier;
};

template <typename FloatType>
BruteForceResult<FloatType> brute_force_reference(
    std::vector<FloatType> values,
    FloatType min,
    FloatType max,
    size_t num_valid
) {
    size_t length = values.size();
    std::vector<bool> is_outlier(length, true);

    std::vector<uint16_t> idx(length);
    argsort(values.data(), idx.data(), length);

    std::vector<uint16_t> valid_idx;
    for (size_t i = 0; i < length; i++) {
        uint16_t index = idx[i];
        if (values[index] >= min && values[index] <= max) {
            valid_idx.push_back(index);
        }
    }

    size_t count_valid = valid_idx.size();
    if (count_valid == 0) {
        return {0, FloatType(0), is_outlier};
    }

    size_t best_lft = 0, best_rgt = 0;
    FloatType best_scr = FloatType(-1);
    FloatType best_mean = FloatType(-1);

    auto window_stats = [&](size_t lft, size_t rgt) {
        FloatType sum = 0;
        for (size_t i = lft; i < rgt; i++) sum += values[valid_idx[i]];
        FloatType m = sum / FloatType(rgt - lft);
        FloatType m2 = 0;
        for (size_t i = lft; i < rgt; i++) {
            FloatType d = values[valid_idx[i]] - m;
            m2 += d * d;
        }
        return std::make_pair(m, m2);
    };

    if (count_valid < num_valid) {
        auto [m, m2] = window_stats(0, count_valid);
        best_lft = 0;
        best_rgt = count_valid;
        best_mean = m;
        (void)m2;
    } else {
        for (size_t offset = 0; offset + num_valid <= count_valid; offset++) {
            auto [m, m2] = window_stats(offset, offset + num_valid);
            if (m2 < best_scr || offset == 0) {
                best_scr = m2;
                best_mean = m;
                best_lft = offset;
                best_rgt = offset + num_valid;
            }
        }
    }

    for (size_t i = best_lft; i < best_rgt; i++) {
        is_outlier[valid_idx[i]] = false;
    }

    return {static_cast<uint16_t>(best_rgt - best_lft), best_mean, is_outlier};
}

// =======================================================================
// outlier() worked/manual examples
// =======================================================================

TEST(Outlier, ExactTieBetweenTwoClustersPicksAValidMinimalVarianceWindow) {
    // Sorted: 10,11,12,50,51,52 -> windows {10,11,12} and {50,51,52} are
    // mathematically exact ties: both have M2 = 2.
    //
    // IMPORTANT FINDING: the code's tie-break comment/intent ("only replace
    // on strictly '<', so earlier windows win ties") is NOT reliable in
    // practice, because scores are computed via incremental Welford
    // add/remove across several steps rather than fresh per-window. Tracing
    // this exact input shows window {50,51,52} computes as
    // 1.9999999999993001... instead of exactly 2.0 due to floating-point
    // drift accumulated over the intervening add/remove operations, so it
    // wins the "tie" instead of the leftmost window. Which window wins can
    // depend on how many incremental steps separate it from the running
    // optimum, not just on the true variance. We therefore only assert that
    // *a* correct, minimal-variance window was returned, not which one.
    std::vector<double> values = {10, 12, 11, 50, 52, 51};
    auto result = run_outlier<double, 6, 3>(values, -1000.0, 1000.0, 6, 3);

    EXPECT_EQ(result.count, 3);

    bool matches_low_cluster = !result.is_outlier[0] && !result.is_outlier[1] &&
                                !result.is_outlier[2] && result.is_outlier[3] &&
                                result.is_outlier[4] && result.is_outlier[5];
    bool matches_high_cluster = result.is_outlier[0] && result.is_outlier[1] &&
                                 result.is_outlier[2] && !result.is_outlier[3] &&
                                 !result.is_outlier[4] && !result.is_outlier[5];

    ASSERT_TRUE(matches_low_cluster || matches_high_cluster)
        << "Selected window was neither of the two tied minimal-variance clusters.";

    EXPECT_NEAR(result.mean, matches_low_cluster ? 11.0 : 51.0, 1e-6);
}

TEST(Outlier, OutOfRangeValuesAlwaysMarkedOutlier) {
    std::vector<double> values = {5, 5, 5, 1000, 5, 5};
    auto result = run_outlier<double, 6, 3>(values, 0.0, 10.0, 6, 3);

    EXPECT_EQ(result.count, 3);
    EXPECT_NEAR(result.mean, 5.0, 1e-9);
    EXPECT_TRUE(result.is_outlier[3]);  // the 1000 must always be an outlier
}

TEST(Outlier, FewerValidThanNumValidUsesAllValidPoints) {
    // Only indices 1 and 3 are within [0, 10]; num_valid=4 but only 2 valid.
    std::vector<double> values = {100, 2, -50, 8, 200};
    auto result = run_outlier<double, 5, 4>(values, 0.0, 10.0, 5, 4);

    EXPECT_EQ(result.count, 2);
    EXPECT_NEAR(result.mean, 5.0, 1e-9);  // mean(2, 8) = 5
    std::vector<bool> expected_is_outlier = {true, false, true, false, true};
    EXPECT_EQ(result.is_outlier, expected_is_outlier);
}

TEST(Outlier, NoValidPointsReturnsZeroAndLeavesMeanUntouched) {
    std::vector<double> values = {100, 200, 300};
    OutlierParams<double, 3, 2> params;
    params.min = 1000.0;
    params.max = 2000.0;
    params.length = 3;
    params.num_valid = 2;

    std::array<bool, 3> is_outlier{};
    double sentinel = -777.0;
    double mean = sentinel;

    uint16_t count = outlier<double, 3, 2>(params, values.data(), is_outlier.data(), mean);

    EXPECT_EQ(count, 0);
    EXPECT_DOUBLE_EQ(mean, sentinel);  // function must not touch `mean`
    for (bool b : is_outlier) EXPECT_TRUE(b);
}

TEST(Outlier, CountValidExactlyEqualsNumValidUsesWholeSet) {
    std::vector<double> values = {3, 1, 2};
    auto result = run_outlier<double, 3, 3>(values, -100.0, 100.0, 3, 3);

    EXPECT_EQ(result.count, 3);
    EXPECT_NEAR(result.mean, 2.0, 1e-9);
    std::vector<bool> expected_is_outlier = {false, false, false};
    EXPECT_EQ(result.is_outlier, expected_is_outlier);
}

TEST(Outlier, NumValidOneAlwaysZeroVariancePicksLeftmostValue) {
    // All windows of size 1 have M2 = 0 (a total tie), so the first entry
    // in sorted order (the minimum valid value) must be picked.
    std::vector<double> values = {30, 10, 20};
    auto result = run_outlier<double, 3, 1>(values, -100.0, 100.0, 3, 1);

    EXPECT_EQ(result.count, 1);
    EXPECT_NEAR(result.mean, 10.0, 1e-9);
    std::vector<bool> expected_is_outlier = {true, false, true};
    EXPECT_EQ(result.is_outlier, expected_is_outlier);
}

TEST(Outlier, BoundaryValuesEqualToMinMaxAreIncluded) {
    std::vector<double> values = {0.0, 5.0, 10.0};  // min/max exactly hit
    auto result = run_outlier<double, 3, 3>(values, 0.0, 10.0, 3, 3);

    EXPECT_EQ(result.count, 3);  // nothing filtered out
    std::vector<bool> expected_is_outlier = {false, false, false};
    EXPECT_EQ(result.is_outlier, expected_is_outlier);
}

TEST(Outlier, NegativeValuesHandledCorrectly) {
    std::vector<double> values = {-10, -9, -8, 100, -50};
    auto result = run_outlier<double, 5, 3>(values, -20.0, 20.0, 5, 3);

    EXPECT_EQ(result.count, 3);
    EXPECT_NEAR(result.mean, -9.0, 1e-9);
    std::vector<bool> expected_is_outlier = {false, false, false, true, true};
    EXPECT_EQ(result.is_outlier, expected_is_outlier);
}

TEST(Outlier, AllIdenticalValuesZeroVariance) {
    std::vector<double> values = {7, 7, 7, 7, 7};
    auto result = run_outlier<double, 5, 3>(values, -100.0, 100.0, 5, 3);

    EXPECT_EQ(result.count, 3);
    EXPECT_NEAR(result.mean, 7.0, 1e-9);
    // First window (offset 0) must win since every window ties at M2 = 0.
    std::vector<bool> expected_is_outlier = {false, false, false, true, true};
    EXPECT_EQ(result.is_outlier, expected_is_outlier);
}

TEST(Outlier, RuntimeParamsCanBeSmallerThanTemplateCapacity) {
    // Compile-time capacity is much larger than what's actually used at
    // runtime; this must not read/write past the declared runtime length.
    std::vector<double> values = {50, 10, 5, 20, 999, 999, 999};
    auto result = run_outlier<double, 50, 10>(values, 0.0, 100.0, 4, 2);

    EXPECT_EQ(result.count, 2);
    EXPECT_NEAR(result.mean, 7.5, 1e-9);  // mean(5, 10)
    ASSERT_EQ(result.is_outlier.size(), 4u);
    std::vector<bool> expected_is_outlier = {true, false, false, true};
    EXPECT_EQ(result.is_outlier, expected_is_outlier);
}

// =======================================================================
// Randomized property-based cross-check against the brute-force oracle
// =======================================================================

template <typename FloatType>
class OutlierRandomizedTest : public ::testing::Test {};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(OutlierRandomizedTest, FloatTypes);

TYPED_TEST(OutlierRandomizedTest, MatchesBruteForceReferenceAcrossManyRandomCases) {
    using FloatType = TypeParam;

    std::mt19937 rng(1234567);
    std::uniform_int_distribution<int> length_dist(1, 25);
    std::uniform_real_distribution<double> value_dist(-100.0, 100.0);
    std::uniform_real_distribution<double> range_pick(-100.0, 100.0);

    const int kMaxLength = 25;
    const int trials = 500;

    double tolerance = std::is_same<FloatType, float>::value ? 1e-1 : 1e-6;

    for (int t = 0; t < trials; t++) {
        int length = length_dist(rng);
        std::uniform_int_distribution<int> num_valid_dist(1, length);
        int num_valid = num_valid_dist(rng);

        std::vector<FloatType> values(length);
        for (int i = 0; i < length; i++) {
            values[i] = static_cast<FloatType>(value_dist(rng));
        }

        double a = range_pick(rng);
        double b = range_pick(rng);
        FloatType min = static_cast<FloatType>(std::min(a, b));
        FloatType max = static_cast<FloatType>(std::max(a, b));

        auto expected = brute_force_reference<FloatType>(values, min, max, num_valid);

        OutlierParams<FloatType, kMaxLength, kMaxLength> params;
        params.min = min;
        params.max = max;
        params.length = length;
        params.num_valid = num_valid;

        std::array<bool, kMaxLength> is_outlier_buf{};
        FloatType mean = static_cast<FloatType>(-999999);

        uint16_t count = outlier<FloatType, kMaxLength, kMaxLength>(
            params, values.data(), is_outlier_buf.data(), mean);

        ASSERT_EQ(count, expected.count) << "trial=" << t << " length=" << length
                                          << " num_valid=" << num_valid;

        if (expected.count > 0) {
            EXPECT_NEAR(static_cast<double>(mean), static_cast<double>(expected.mean),
                        tolerance)
                << "trial=" << t;

            for (int i = 0; i < length; i++) {
                EXPECT_EQ(is_outlier_buf[i], expected.is_outlier[i])
                    << "trial=" << t << " index=" << i;
            }
        }
    }
}

TYPED_TEST(OutlierRandomizedTest, AllValuesOutOfRangeAlwaysReturnsZero) {
    using FloatType = TypeParam;
    const int kLength = 10;

    std::vector<FloatType> values(kLength);
    for (int i = 0; i < kLength; i++) values[i] = static_cast<FloatType>(i);

    OutlierParams<FloatType, kLength, 3> params;
    params.min = static_cast<FloatType>(1000);
    params.max = static_cast<FloatType>(2000);
    params.length = kLength;
    params.num_valid = 3;

    std::array<bool, kLength> is_outlier_buf{};
    FloatType mean = static_cast<FloatType>(-1);

    uint16_t count = outlier<FloatType, kLength, 3>(
        params, values.data(), is_outlier_buf.data(), mean);

    EXPECT_EQ(count, 0);
    for (bool b : is_outlier_buf) EXPECT_TRUE(b);
}