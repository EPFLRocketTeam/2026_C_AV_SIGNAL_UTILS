#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <new>

#include "sigutils/average.hpp"

// Edge cases for RunningAverage: first samples, NaN, infinity, long runs.

TEST(RunningAverageEdges, PartialMeanBeforeTheWindowIsFull) {
    // Poison the storage so a read of uninitialised memory cannot pass by luck.
    alignas(RunningAverage<double, 4>) unsigned char raw[sizeof(RunningAverage<double, 4>)];
    for (auto &b : raw) b = 0xFF;
    auto *avg = new (raw) RunningAverage<double, 4>();

    EXPECT_FALSE(avg->ready());
    avg->push(4.0);
    EXPECT_FALSE(avg->ready());
    EXPECT_DOUBLE_EQ(avg->mean(), 4.0);
    avg->push(8.0);
    EXPECT_DOUBLE_EQ(avg->mean(), 6.0);
    avg->push(0.0);
    EXPECT_DOUBLE_EQ(avg->mean(), 4.0);
    EXPECT_FALSE(avg->ready());
    avg->push(12.0);
    EXPECT_TRUE(avg->ready());
    EXPECT_DOUBLE_EQ(avg->mean(), 6.0);
}

TEST(RunningAverageEdges, WindowOneTracksTheLastSample) {
    RunningAverage<double, 1> avg;
    avg.push(3.0);
    EXPECT_TRUE(avg.ready());
    EXPECT_DOUBLE_EQ(avg.mean(), 3.0);
    avg.push(-9.5);
    EXPECT_DOUBLE_EQ(avg.mean(), -9.5);
}

TEST(RunningAverageEdges, NaNPoisonsThenSelfHealsWithinTwoWindows) {
    // Documents current behaviour: a NaN sample makes the mean NaN, and the
    // double sum scheme flushes it once two full windows of finite data have
    // gone through. There is no reset(); this is the only recovery path.
    constexpr int N = 5;
    RunningAverage<double, N> avg;
    for (int i = 0; i < N; i++) avg.push(1.0);
    EXPECT_DOUBLE_EQ(avg.mean(), 1.0);

    avg.push(std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(std::isnan(avg.mean()));

    for (int i = 0; i < 2 * N; i++) avg.push(1.0);
    EXPECT_FALSE(std::isnan(avg.mean()));
    EXPECT_DOUBLE_EQ(avg.mean(), 1.0);
}

TEST(RunningAverageEdges, InfinitySelfHealsLikeNaN) {
    constexpr int N = 4;
    RunningAverage<double, N> avg;
    for (int i = 0; i < N; i++) avg.push(2.0);
    avg.push(std::numeric_limits<double>::infinity());
    EXPECT_FALSE(std::isfinite(avg.mean()));
    for (int i = 0; i < 2 * N; i++) avg.push(2.0);
    EXPECT_DOUBLE_EQ(avg.mean(), 2.0);
}

TEST(RunningAverageEdges, FloatDriftOverAMillionSamples) {
    constexpr int N = 50;
    RunningAverage<float, N> avg;
    // A slow ramp with a large offset: the mean of the last N samples is known exactly.
    long long last = 0;
    for (long long i = 0; i < 1000000; i++) {
        avg.push(static_cast<float>(100000 + (i % 1000)));
        last = i;
    }
    double expected = 0;
    for (long long k = last - N + 1; k <= last; k++) expected += 100000 + (k % 1000);
    expected /= N;
    EXPECT_NEAR(avg.mean(), expected, 0.01);
}

TEST(RunningAverageEdges, LargeWindowIndexIsNotTruncated) {
    constexpr int N = 70000;  // above uint16_t range
    static RunningAverage<double, N> avg;
    for (int i = 0; i < N; i++) avg.push(static_cast<double>(i));
    EXPECT_TRUE(avg.ready());
    EXPECT_DOUBLE_EQ(avg.mean(), (N - 1) / 2.0);
    avg.push(0.0);
    // the oldest sample was 0.0, so replacing it by 0.0 leaves the mean alone
    EXPECT_NEAR(avg.mean(), (N - 1) / 2.0, 1e-9);
}
