#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <deque>
#include <vector>
#include <numeric>
#include "sigutils/average.hpp"

// ============================================================================
// 1. Type-Parameterized Tests across Floating Types
// ============================================================================

template <typename T>
class RunningAverageTypedTest : public ::testing::Test {};

using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RunningAverageTypedTest, FloatingTypes);

TYPED_TEST(RunningAverageTypedTest, InitialStateIsNotReady) {
    RunningAverage<TypeParam, 5> avg{};
    EXPECT_FALSE(avg.ready());
    EXPECT_EQ(avg.num_samples, 5u);
}

TYPED_TEST(RunningAverageTypedTest, BecomesReadyAfterNSamples) {
    RunningAverage<TypeParam, 4> avg{};

    for (size_t i = 0; i < 3; ++i) {
        avg.push(static_cast<TypeParam>(10.0));
        EXPECT_FALSE(avg.ready());
    }

    // 4th sample reaches num_samples, setting has_mean to true
    avg.push(static_cast<TypeParam>(10.0));
    EXPECT_TRUE(avg.ready());
}

TYPED_TEST(RunningAverageTypedTest, ConstantSignalAverage) {
    constexpr size_t N = 10;
    RunningAverage<TypeParam, N> avg{};

    const TypeParam constant_val = static_cast<TypeParam>(7.5);
    for (size_t i = 0; i < N; ++i) {
        avg.push(constant_val);
    }

    EXPECT_TRUE(avg.ready());
    EXPECT_NEAR(avg.mean(), constant_val, static_cast<TypeParam>(1e-5));
}

// ============================================================================
// 2. Specific Functional & Boundary Tests
// ============================================================================

TEST(RunningAverageTest, ExactMeanCalculation) {
    constexpr size_t N = 5;
    RunningAverage<double, N> avg{};

    // Push 5 samples: sum = 15, mean = 3.0
    avg.push(1.0);
    avg.push(2.0);
    avg.push(3.0);
    avg.push(4.0);
    avg.push(5.0);

    EXPECT_TRUE(avg.ready());
    EXPECT_DOUBLE_EQ(avg.mean(), 3.0);
}

TEST(RunningAverageTest, WindowSizeOne) {
    RunningAverage<double, 1> avg{};

    EXPECT_FALSE(avg.ready());

    avg.push(42.0);
    EXPECT_TRUE(avg.ready());
    EXPECT_DOUBLE_EQ(avg.mean(), 42.0);

    avg.push(100.0);
    EXPECT_TRUE(avg.ready());
    EXPECT_DOUBLE_EQ(avg.mean(), 100.0);
}

TEST(RunningAverageTest, SlidingWindowBehaviorAcrossCycles) {
    constexpr size_t N = 3;
    RunningAverage<double, N> avg{};

    // First cycle: [10, 20, 30] -> mean = 20.0
    avg.push(10.0);
    avg.push(20.0);
    avg.push(30.0);
    EXPECT_DOUBLE_EQ(avg.mean(), 20.0); // Offset resets to 0, sum1 resynced from sum2

    // Second cycle: replace 10 with 40 -> buffer: [40, 20, 30] -> mean = 30.0
    avg.push(40.0);
    EXPECT_DOUBLE_EQ(avg.mean(), 30.0);

    // Replace 20 with 50 -> buffer: [40, 50, 30] -> mean = 40.0
    avg.push(50.0);
    EXPECT_DOUBLE_EQ(avg.mean(), 40.0);

    // Replace 30 with 60 -> buffer: [40, 50, 60] -> mean = 50.0
    avg.push(60.0);
    EXPECT_DOUBLE_EQ(avg.mean(), 50.0); // Resync triggered again
}

TEST(RunningAverageTest, PeriodicResyncResetVerification) {
    constexpr size_t N = 4;
    RunningAverage<double, N> avg{};

    // Fill first full window with 10.0
    for (size_t i = 0; i < N; ++i) {
        avg.push(10.0);
    }
    EXPECT_DOUBLE_EQ(avg.mean(), 10.0);

    // Push 4 new samples (0.0) to complete second cycle
    for (size_t i = 0; i < N; ++i) {
        avg.push(0.0);
    }

    // sum1 should now equal sum2 (which was accumulating 0.0), resulting in mean = 0.0
    EXPECT_DOUBLE_EQ(avg.mean(), 0.0);
}

TEST(RunningAverageTest, LargeDatasetMovingAverageVerification) {
    constexpr size_t N = 5;
    RunningAverage<double, N> avg{};

    std::vector<double> stream = { 12.0, -4.0, 8.0, 16.0, 3.0, 9.0, -1.0, 5.0, 10.0, 2.0 };

    for (size_t i = 0; i < stream.size(); ++i) {
        avg.push(stream[i]);

        if (i >= N - 1) {
            EXPECT_TRUE(avg.ready());

            // Compute expected mean directly from vector window
            double expected_sum = 0.0;
            for (size_t k = i - N + 1; k <= i; ++k) {
                expected_sum += stream[k];
            }
            double expected_mean = expected_sum / static_cast<double>(N);

            EXPECT_NEAR(avg.mean(), expected_mean, 1e-9);
        }
    }
}





// ============================================================================
// Helper Class: Exact High-Precision Reference Window
// ============================================================================
// Maintained alongside RunningAverage during stress testing to calculate the
// true ground-truth mean using high-precision long double arithmetic.
template <typename FloatType>
class ReferenceSlidingWindow {
private:
    std::deque<long double> window;
    size_t capacity;

public:
    explicit ReferenceSlidingWindow(size_t N) : capacity(N) {}

    void push(FloatType point) {
        window.push_back(static_cast<long double>(point));
        if (window.size() > capacity) {
            window.pop_front();
        }
    }

    bool ready() const {
        return window.size() == capacity;
    }

    FloatType exact_mean() const {
        long double sum = 0.0L;
        for (long double val : window) {
            sum += val;
        }
        return static_cast<FloatType>(sum / static_cast<long double>(capacity));
    }
};

// ============================================================================
// Long-Running Randomized Stress Tests
// ============================================================================

TEST(RunningAverageStressTest, LongStreamRandomUniformDistribution) {
    constexpr size_t N = 10;
    constexpr size_t TOTAL_SAMPLES = 200'000;

    RunningAverage<double, N> filter{};
    ReferenceSlidingWindow<double> ref(N);

    std::mt19937_64 rng(1337); // Fixed seed for deterministic test runs
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    double max_delta = 0.0;

    for (size_t step = 0; step < TOTAL_SAMPLES; ++step) {
        const double sample = dist(rng);

        filter.push(sample);
        ref.push(sample);

        if (filter.ready()) {
            const double actual_mean = filter.mean();
            const double expected_mean = ref.exact_mean();

            const double delta = std::abs(actual_mean - expected_mean);
            if (delta > max_delta) {
                max_delta = delta;
            }

            // Delta tolerance: double precision across 200,000 steps
            ASSERT_NEAR(actual_mean, expected_mean, 1e-12)
                << "Failed at step " << step << " with sample " << sample;
        }
    }

    EXPECT_LT(max_delta, 1e-12);
}

TEST(RunningAverageStressTest, FloatPrecisionLongStream) {
    constexpr size_t N = 16;
    constexpr size_t TOTAL_SAMPLES = 100'000;

    RunningAverage<float, N> filter{};
    ReferenceSlidingWindow<float> ref(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);

    float max_delta = 0.0f;

    for (size_t step = 0; step < TOTAL_SAMPLES; ++step) {
        const float sample = dist(rng);

        filter.push(sample);
        ref.push(sample);

        if (filter.ready()) {
            const float actual_mean = filter.mean();
            const float expected_mean = ref.exact_mean();

            const float delta = std::abs(actual_mean - expected_mean);
            if (delta > max_delta) {
                max_delta = delta;
            }

            // Delta tolerance for single-precision float over 100k steps
            ASSERT_NEAR(actual_mean, expected_mean, 1e-5f)
                << "Failed at step " << step;
        }
    }

    EXPECT_LT(max_delta, 1e-5f);
}

TEST(RunningAverageStressTest, RandomLargeDCBiasWithSmallAC) {
    // Tests signal with massive DC offset (1e8) and tiny noise (+/- 0.01)
    // Stress-tests subtraction cancellation when small values are added to large sums.
    constexpr size_t N = 32;
    constexpr size_t TOTAL_SAMPLES = 50'000;

    RunningAverage<double, N> filter{};
    ReferenceSlidingWindow<double> ref(N);

    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> noise(-0.01, 0.01);
    constexpr double DC_OFFSET = 1.0e8;

    for (size_t step = 0; step < TOTAL_SAMPLES; ++step) {
        const double sample = DC_OFFSET + noise(rng);

        filter.push(sample);
        ref.push(sample);

        if (filter.ready()) {
            const double actual_mean = filter.mean();
            const double expected_mean = ref.exact_mean();

            ASSERT_NEAR(actual_mean, expected_mean, 1e-6)
                << "Failed at step " << step;
        }
    }
}

TEST(RunningAverageStressTest, LargeWindowRandomWalk) {
    constexpr size_t N = 1024; // Large window size
    constexpr size_t TOTAL_SAMPLES = 50'000;

    RunningAverage<double, N> filter{};
    ReferenceSlidingWindow<double> ref(N);

    std::mt19937_64 rng(777);
    std::normal_distribution<double> step_dist(0.0, 1.0); // Gaussian random walk

    double current_value = 0.0;

    for (size_t step = 0; step < TOTAL_SAMPLES; ++step) {
        current_value += step_dist(rng);

        filter.push(current_value);
        ref.push(current_value);

        if (filter.ready()) {
            const double actual_mean = filter.mean();
            const double expected_mean = ref.exact_mean();

            ASSERT_NEAR(actual_mean, expected_mean, 1e-9)
                << "Random walk failed at step " << step;
        }
    }
}

TEST(RunningAverageStressTest, MultiCyclePeriodicResetValidation) {
    // Specifically verifies that the algorithm remains exact over multiple full reset boundaries
    constexpr size_t N = 5;
    constexpr size_t CYCLES = 10'000;
    constexpr size_t TOTAL_SAMPLES = N * CYCLES; // Exactly 50,000 steps (10,000 resync boundaries)

    RunningAverage<double, N> filter{};
    ReferenceSlidingWindow<double> ref(N);

    std::mt19937_64 rng(2026);
    std::exponential_distribution<double> dist(0.5);

    for (size_t step = 0; step < TOTAL_SAMPLES; ++step) {
        const double sample = dist(rng);

        filter.push(sample);
        ref.push(sample);

        if (filter.ready()) {
            const double actual_mean = filter.mean();
            const double expected_mean = ref.exact_mean();

            ASSERT_DOUBLE_EQ(actual_mean, expected_mean);
        }
    }
}