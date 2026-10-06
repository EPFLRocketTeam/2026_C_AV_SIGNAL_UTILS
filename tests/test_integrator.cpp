#include <gtest/gtest.h>
#include <random>
#include <cstdint>
#include <cmath>
#include <limits>

#include "sigutils/integrator.hpp"

// -----------------------------------------------------------------------------
// Time & Integrator Conversion Templates
// -----------------------------------------------------------------------------

template <typename T>
constexpr T identity_conversion(T x) {
    return x;
}

template <typename T>
constexpr T ms_to_sec(T dt_ms) {
    return dt_ms / static_cast<T>(1000);
}

template <typename T>
constexpr T us_to_sec(T dt_us) {
    return dt_us / static_cast<T>(1000000);
}

// Custom wrapper to convert uint32_t delta directly to double
constexpr double ms_to_sec_double(uint32_t dt_ms) {
    return static_cast<double>(dt_ms) / 1000.0;
}

constexpr float ms_to_sec_float(uint32_t dt_ms) {
    return static_cast<float>(dt_ms) / 1000.0f;
}

constexpr double us_to_sec_double(uint64_t dt_us) {
    return static_cast<double>(dt_us) / 1000000.0;
}

constexpr double sec_conversion(uint32_t dt_sec) {
    return static_cast<double>(dt_sec);
}

// =============================================================================
// Basic Functionality & Trapezoidal Logic Tests
// =============================================================================

TEST(StableIntegratorTest, ExactTrapezoidManualSteps) {
    StableIntegrator<double, uint32_t, sec_conversion> integrator;

    uint32_t t = 0;
    integrator.start(1.0, t);

    // Step 1: dt = 3, trapezoid = ((1 + 1) / 2) * 3 = 3.0
    t += 3;
    integrator.ingest(1.0, t);
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 3.0);

    // Step 2: dt = 5, trapezoid = ((1 + 2) / 2) * 5 = 7.5 -> total = 10.5
    t += 5;
    integrator.ingest(2.0, t);
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 10.5);

    // Step 3: dt = 1, trapezoid = ((2 + 1) / 2) * 1 = 1.5 -> total = 12.0
    t += 1;
    integrator.ingest(1.0, t);
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 12.0);
}

TEST(StableIntegratorTest, ConstantSignalAt1KHz) {
    StableIntegrator<double, uint32_t, ms_to_sec_double> integrator;

    uint32_t t_ms = 0;
    constexpr double val = 2.0;

    integrator.start(val, t_ms);

    constexpr int steps = 1000;
    for (int i = 0; i < steps; ++i) {
        t_ms += 1;
        integrator.ingest(val, t_ms);
    }

    // Integral of 2.0 over 1.0 sec = 2.0
    EXPECT_NEAR(static_cast<double>(integrator), 2.0, 1e-12);
}

TEST(StableIntegratorTest, AliasTypeIntegrationTest) {
    // Tests StableIntegratorMillis alias or type-conversion functor usage
    StableIntegratorMillis<double> integrator;

    uint32_t t_ms = 0;
    integrator.start(5.0, t_ms);

    t_ms += 2000; // 2 seconds
    integrator.ingest(5.0, t_ms);

    // 5.0 * 2.0s = 10.0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 10.0);
}

TEST(StableIntegratorTest, VariableTimeDeltaWithRandomSamples) {
    StableIntegrator<float, uint32_t, ms_to_sec_float> integrator;

    uint32_t t_ms = 0;
    float expected = 0.0f;
    float prev_val = 4.0f;

    integrator.start(prev_val, t_ms);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist_dt(10, 50);

    for (int i = 0; i < 100; ++i) {
        uint32_t dt_ms = dist_dt(gen);
        t_ms += dt_ms;

        float curr_val = prev_val + 0.5f;
        float dt_sec = static_cast<float>(dt_ms) / 1000.0f;

        expected += ((curr_val + prev_val) / 2.0f) * dt_sec;

        integrator.ingest(curr_val, t_ms);
        prev_val = curr_val;
    }

    EXPECT_NEAR(static_cast<float>(integrator), expected, 1e-4f);
}

// =============================================================================
// Analytical Functions Integrations
// =============================================================================

TEST(StableIntegratorTest, LinearRampFunction) {
    // Integrate f(t) = 3 * t from t = 0 to t = 10 sec
    // Exact analytical integral = [1.5 * t^2] from 0 to 10 = 150.0
    StableIntegrator<double, uint32_t, ms_to_sec_double> integrator;

    uint32_t t_ms = 0;
    integrator.start(0.0, t_ms);

    constexpr uint32_t dt_ms = 10; // 100 Hz sampling rate
    constexpr uint32_t total_time_ms = 10000;

    for (uint32_t elapsed = dt_ms; elapsed <= total_time_ms; elapsed += dt_ms) {
        double t_sec = static_cast<double>(elapsed) / 1000.0;
        double val = 3.0 * t_sec;
        integrator.ingest(val, elapsed);
    }

    // Trapezoidal rule is EXACT for linear functions
    EXPECT_NEAR(static_cast<double>(integrator), 150.0, 1e-11);
}

TEST(StableIntegratorTest, FullSineWaveCycleIntegratesToZero) {
    // Integrate f(t) = sin(2 * pi * t) over T = 1.0 sec
    // Exact analytical integral over full period = 0.0
    StableIntegrator<double, uint32_t, ms_to_sec_double> integrator;

    constexpr double pi = 3.14159265358979323846;
    uint32_t t_ms = 0;

    integrator.start(0.0, t_ms);

    constexpr int steps = 1000; // 1 kHz sampling resolution
    for (int i = 1; i <= steps; ++i) {
        t_ms += 1;
        double t_sec = static_cast<double>(t_ms) / 1000.0;
        double val = std::sin(2.0 * pi * t_sec);
        integrator.ingest(val, t_ms);
    }

    EXPECT_NEAR(static_cast<double>(integrator), 0.0, 1e-7);
}

// =============================================================================
// Edge Cases & Special Conditions
// =============================================================================

TEST(StableIntegratorTest, ZeroTimeDeltaIngest) {
    // Repeated calls with same timestamp (dt = 0) should add 0 to integral
    StableIntegrator<double, uint32_t, sec_conversion> integrator;

    uint32_t t = 5;
    integrator.start(10.0, t);

    integrator.ingest(20.0, t); // dt = 0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 0.0);

    integrator.ingest(30.0, t); // dt = 0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 0.0);

    t += 2;
    integrator.ingest(30.0, t); // dt = 2 -> area = ((30 + 30)/2)*2 = 60.0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 60.0);
}

TEST(StableIntegratorTest, ReinitializationViaStart) {
    // Verify that start() fully resets accumulated sum and sets new reference state
    StableIntegrator<double, uint32_t, sec_conversion> integrator;

    uint32_t t = 0;
    integrator.start(5.0, t);

    t += 10;
    integrator.ingest(5.0, t); // integral = 50.0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 50.0);

    // Reset integrator mid-stream
    t = 100;
    integrator.start(2.0, t);
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 0.0);

    t += 4;
    integrator.ingest(2.0, t); // integral = 8.0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 8.0);
}

TEST(StableIntegratorTest, NegativeSignalValues) {
    // Integrate f(t) = -5.0 for 4 seconds -> -20.0
    StableIntegrator<double, uint32_t, sec_conversion> integrator;

    uint32_t t = 0;
    integrator.start(-5.0, t);

    t += 4;
    integrator.ingest(-5.0, t);

    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), -20.0);
}

// =============================================================================
// Overflow & Numerical Stability Tests
// =============================================================================

TEST(StableIntegratorTest, TimerOverflowRolloverHandledByUnsignedSub) {
    // Verifies standard C++ unsigned integer wrapping behaviour across 32-bit boundary
    StableIntegrator<double, uint32_t, sec_conversion> integrator;

    uint32_t t = std::numeric_limits<uint32_t>::max() - 3; // 4,294,967,292
    integrator.start(2.0, t);

    // Advance across 32-bit boundary (wraps around 0)
    t += 9; // t is now 5
    integrator.ingest(2.0, t);

    // dt should be computed as 9s -> Area = 2.0 * 9 = 18.0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 18.0);
}

TEST(StableIntegratorTest, MicrosecondTicks64BitTime) {
    StableIntegrator<double, uint64_t, us_to_sec_double> integrator;

    uint64_t t_us = 1'000'000'000ULL; // Start at 1000 sec mark
    integrator.start(10.0, t_us);

    t_us += 500'000ULL; // +0.5 sec
    integrator.ingest(10.0, t_us);

    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 5.0);
}

TEST(StableIntegratorTest, LargeAccumulationWithNeumaierStability) {
    // Adds a large DC offset (1e6) and a tiny oscillating signal (1e-6)
    // Neumaier compensated summation prevents dropping precision in tiny deltas.
    StableIntegrator<double, uint32_t, ms_to_sec_double> integrator;

    uint32_t t_ms = 0;
    constexpr double dc_offset = 1e6;
    constexpr double tiny_val  = 1e-6;

    integrator.start(dc_offset + tiny_val, t_ms);

    constexpr int steps = 100'000; // 100 seconds
    for (int i = 0; i < steps; ++i) {
        t_ms += 1;
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        integrator.ingest(dc_offset + sign * tiny_val, t_ms);
    }

    // Integrated DC component = 1e6 * 100 sec = 1e8
    EXPECT_NEAR(static_cast<double>(integrator), 1e8, 1e-4);
}

// =============================================================================
// StableIntegratorMillis Type Alias Specific Tests
// =============================================================================

TEST(StableIntegratorMillisTest, BasicMillisecondIntegration) {
    // Default float/double type tests with millisecond time resolution
    StableIntegratorMillis<double> integrator;

    uint32_t t_ms = 0;
    integrator.start(10.0, t_ms); // f(t) = 10.0

    // Advance 500 ms (0.5 seconds) -> Area = 10.0 * 0.5 = 5.0
    t_ms += 500;
    integrator.ingest(10.0, t_ms);
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 5.0);

    // Advance 1500 ms (1.5 seconds) -> Area = 10.0 * 1.5 = 15.0 -> Total = 20.0
    t_ms += 1500;
    integrator.ingest(10.0, t_ms);
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 20.0);
}

TEST(StableIntegratorMillisTest, FloatPrecision32Bit) {
    // Tests that float template instantiation converts correctly without double demotion warnings
    StableIntegratorMillis<float> integrator;

    uint32_t t_ms = 1000;
    integrator.start(2.5f, t_ms);

    t_ms += 4000; // 4 seconds
    integrator.ingest(2.5f, t_ms);

    // 2.5 * 4.0s = 10.0f
    EXPECT_FLOAT_EQ(static_cast<float>(integrator), 10.0f);
}

TEST(StableIntegratorMillisTest, NonUniformSubMillisecondSampling) {
    // Verifies irregular dt steps (e.g., jitter in loop calls: 1ms, 3ms, 17ms)
    StableIntegratorMillis<double> integrator;

    uint32_t t_ms = 0;
    double expected = 0.0;
    double last_val = 1.0;

    integrator.start(last_val, t_ms);

    const uint32_t intervals[] = {1, 3, 17, 2, 100, 250, 15, 8};
    for (uint32_t dt : intervals) {
        t_ms += dt;
        double curr_val = last_val + 0.5;

        // trapezoidal sum: (f(a) + f(b)) / 2 * (dt / 1000.0)
        expected += ((curr_val + last_val) / 2.0) * (static_cast<double>(dt) / 1000.0);

        integrator.ingest(curr_val, t_ms);
        last_val = curr_val;
    }

    EXPECT_NEAR(static_cast<double>(integrator), expected, 1e-12);
}

TEST(StableIntegratorMillisTest, MillisTimerOverflowRollover) {
    // Tests 32-bit millisecond overflow (~49.7 days of uptime)
    StableIntegratorMillis<double> integrator;

    // Set time to 10 ms before UINT32_MAX
    uint32_t t_ms = std::numeric_limits<uint32_t>::max() - 10;
    integrator.start(5.0, t_ms);

    // Roll over past 0 by 990 ms (Total dt = 1000 ms = 1.0 second)
    t_ms += 1000; // Unsigned arithmetic causes wrap-around to 989
    integrator.ingest(5.0, t_ms);

    // Area = 5.0 * 1.0s = 5.0
    EXPECT_DOUBLE_EQ(static_cast<double>(integrator), 5.0);
}