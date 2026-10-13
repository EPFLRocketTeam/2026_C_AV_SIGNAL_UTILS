#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "sigutils/integrator.hpp"

// Edge cases for the trapezoid integrator as the PRC Impulse module uses it:
// StableIntegratorMillis<float>, start() then ingest() on every pressure sample.

using Millis = StableIntegratorMillis<float>;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

TEST(IntegratorEdges, TrapezoidNotRectangleNotDoubled) {
    // Ramp 0 to 10 over one second: trapezoid area is exactly 5.
    // A rectangle on the new sample would give 10, on the old sample 0.
    Millis i;
    i.start(0.0f, 0);
    i.ingest(10.0f, 1000);
    EXPECT_FLOAT_EQ(static_cast<float>(i), 5.0f);
}

TEST(IntegratorEdges, ConstantSignalIsValueTimesSeconds) {
    // Unit check behind PressureIntegralToImpulse: 4 bar held for 2.5 s is 10 bar s.
    Millis i;
    i.start(4.0f, 100);
    i.ingest(4.0f, 1100);
    i.ingest(4.0f, 2600);
    EXPECT_FLOAT_EQ(static_cast<float>(i), 10.0f);
}

TEST(IntegratorEdges, NothingIngestedIsZeroAndStartDiscardsTheOldIntegral) {
    Millis i;
    i.start(7.0f, 50);
    EXPECT_FLOAT_EQ(static_cast<float>(i), 0.0f);
    i.ingest(7.0f, 1050);
    EXPECT_FLOAT_EQ(static_cast<float>(i), 7.0f);
    i.start(1.0f, 5000);
    EXPECT_FLOAT_EQ(static_cast<float>(i), 0.0f);
}

TEST(IntegratorEdges, ZeroDtSampleAddsNothingButUpdatesLastValue) {
    Millis i;
    i.start(0.0f, 0);
    i.ingest(100.0f, 0);   // dt = 0, contributes 0
    i.ingest(100.0f, 1000);
    EXPECT_FLOAT_EQ(static_cast<float>(i), 100.0f);
}

TEST(IntegratorEdges, TimeGoingBackwardsIsReadAsAHugeForwardStep) {
    // Documents current behaviour: the unsigned difference wraps, so a tick
    // that goes back by 1 ms integrates about 2^32 ms. Not a design the team
    // asked for, listed in the PR for a decision.
    Millis i;
    i.start(0.0f, 1000);
    i.ingest(2.0f, 999);
    EXPECT_NEAR(static_cast<float>(i), 4294967.296 / 1.0, 4294967.296 * 1e-3);
}

TEST(IntegratorEdges, TickCounterWrapIsContinuous) {
    Millis i;
    i.start(2.0f, 0xFFFFFF00u);
    i.ingest(2.0f, 0x000000FFu + 0u);  // 0x1FF ms later, across the wrap
    EXPECT_NEAR(static_cast<float>(i), 2.0f * 0.511f, 1e-5);
}

TEST(IntegratorEdges, NaNSampleIsSkippedNotPoisonous) {
    Millis i;
    i.start(2.0f, 0);
    i.ingest(2.0f, 1000);
    i.ingest(kNaN, 2000);
    i.ingest(2.0f, 3000);
    EXPECT_TRUE(std::isfinite(static_cast<float>(i)));
    // the dropped sample is bridged by the trapezoid between its neighbours
    EXPECT_FLOAT_EQ(static_cast<float>(i), 6.0f);
}

TEST(IntegratorEdges, InfinitySampleIsSkipped) {
    Millis i;
    i.start(1.0f, 0);
    i.ingest(kInf, 1000);
    i.ingest(-kInf, 2000);
    i.ingest(1.0f, 3000);
    EXPECT_FLOAT_EQ(static_cast<float>(i), 3.0f);
}

TEST(IntegratorEdges, NaNAtStartDoesNotPoisonTheFirstStep) {
    Millis i;
    i.start(kNaN, 0);
    i.ingest(4.0f, 1000);
    EXPECT_TRUE(std::isfinite(static_cast<float>(i)));
}

TEST(IntegratorEdges, MillionSamplesAtOneKilohertzStayExact) {
    Millis i;
    i.start(1.0f, 0);
    for (uint32_t t = 1; t <= 1000000; t++) i.ingest(1.0f, t);
    EXPECT_NEAR(static_cast<float>(i), 1000.0f, 1e-3);
}

TEST(IntegratorEdges, ConvergesOnASmallDtLongBurn) {
    // 60 s burn at 100 Hz of a 35 bar plateau with 0.1f style rounding.
    Millis i;
    i.start(0.1f, 0);
    for (uint32_t t = 10; t <= 60000; t += 10) i.ingest(0.1f, t);
    EXPECT_NEAR(static_cast<float>(i), static_cast<double>(0.1f) * 60.0, 1e-4);
}
