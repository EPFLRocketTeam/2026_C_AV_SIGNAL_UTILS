#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "sigutils/outlier.hpp"

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

namespace {

template <int N, int K, bool Reset = true>
struct Run {
    uint16_t count;
    double mean;
    std::array<bool, N> flag;
};

template <int N, int K, bool Reset = true>
Run<N, K, Reset> run(std::array<double, N> v, double lo, double hi, std::array<bool, N> pre = {}) {
    OutlierParams<double, N, K> p;
    p.min = lo;
    p.max = hi;
    Run<N, K, Reset> r;
    r.flag = pre;
    r.mean = -12345.0;
    r.count = outlier<double, N, K, Reset>(p, v.data(), r.flag.data(), r.mean);
    return r;
}

// Direct reference: finite in range values, sorted, best contiguous window of K by two pass variance.
double reference_mean(std::vector<double> v, double lo, double hi, size_t K, size_t &used) {
    std::vector<double> ok;
    for (double x : v) { if (std::isfinite(x) && lo <= x && x <= hi) ok.push_back(x); }
    std::sort(ok.begin(), ok.end());
    used = std::min(K, ok.size());
    if (ok.empty()) return 0;
    if (ok.size() <= K) {
        double s = 0; for (double x : ok) s += x; return s / ok.size();
    }
    double best = 1e300, best_mean = 0;
    for (size_t a = 0; a + K <= ok.size(); a++) {
        double m = 0; for (size_t j = 0; j < K; j++) m += ok[a + j]; m /= K;
        double q = 0; for (size_t j = 0; j < K; j++) q += (ok[a + j] - m) * (ok[a + j] - m);
        if (q < best - 1e-12) { best = q; best_mean = m; }
    }
    return best_mean;
}

}  // namespace

TEST(OutlierEdges, NaNInputIsFlaggedAndNeverReachesTheMean) {
    auto r = run<5, 3>({5, kNaN, 1, 2, 3}, -1000, 1000);
    EXPECT_TRUE(std::isfinite(r.mean));
    EXPECT_EQ(r.count, 3);
    EXPECT_NEAR(r.mean, 2.0, 1e-12);
    EXPECT_TRUE(r.flag[1]);
    EXPECT_TRUE(r.flag[0]);
}

TEST(OutlierEdges, InfinityIsExcludedEvenWithAnUnboundedRange) {
    const double lim = std::numeric_limits<double>::max();
    auto r = run<4, 3>({1, 1.1, kInf, 0.9}, -lim, kInf);
    EXPECT_TRUE(std::isfinite(r.mean));
    EXPECT_NEAR(r.mean, 1.0, 1e-12);
    EXPECT_TRUE(r.flag[2]);
}

TEST(OutlierEdges, EveryInputNaNLeavesMeanAndFlagsAll) {
    auto r = run<3, 2>({kNaN, kNaN, kNaN}, -1000, 1000);
    EXPECT_EQ(r.count, 0);
    EXPECT_DOUBLE_EQ(r.mean, -12345.0);
    EXPECT_TRUE(r.flag[0] && r.flag[1] && r.flag[2]);
}

TEST(OutlierEdges, RandomMixWithNaNMatchesReference) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> d(0, 100);
    for (int round = 0; round < 3000; round++) {
        std::array<double, 7> v;
        for (auto &x : v) { x = (rng() % 5 == 0) ? kNaN : d(rng); }
        auto r = run<7, 4>(v, 10, 90);
        size_t used;
        double want = reference_mean(std::vector<double>(v.begin(), v.end()), 10, 90, 4, used);
        ASSERT_EQ(r.count, used) << "round " << round;
        if (used > 0) { ASSERT_NEAR(r.mean, want, 1e-9) << "round " << round; }
    }
}

TEST(OutlierEdges, AllEqualValuesDoNotDivideByZero) {
    auto r = run<6, 4>({2.5, 2.5, 2.5, 2.5, 2.5, 2.5}, 0, 10);
    EXPECT_EQ(r.count, 4);
    EXPECT_DOUBLE_EQ(r.mean, 2.5);
    EXPECT_TRUE(std::isfinite(r.mean));
}

TEST(OutlierEdges, AllEqualAcrossManyCallsStaysExact) {
    OutlierParams<double, 5, 3> p;
    p.min = 0; p.max = 200000;
    std::array<double, 5> v;
    for (int k = 0; k < 1000; k++) {
        v.fill(101325.123 + (k % 3));
        bool f[5] = {};
        double m = 0;
        outlier<double, 5, 3, true>(p, v.data(), f, m);
        ASSERT_DOUBLE_EQ(m, 101325.123 + (k % 3));
    }
}

TEST(OutlierEdges, SingleSpikeDoesNotMoveTheBaselineAndIsForgotten) {
    OutlierParams<double, 5, 3> p;
    p.min = 0; p.max = 5000;
    double base[5] = {10.0, 10.1, 9.9, 10.05, 9.95};
    double spiked[5] = {10.0, 10.1, 9.9, 4000.0, 9.95};
    bool f[5] = {};
    double m = 0;

    outlier<double, 5, 3, true>(p, spiked, f, m);
    EXPECT_NEAR(m, 10.0, 0.1);
    EXPECT_TRUE(f[3]);

    outlier<double, 5, 3, true>(p, base, f, m);  // state is fully recomputed
    EXPECT_FALSE(f[3]);
    EXPECT_NEAR(m, 10.0, 0.1);
}

TEST(OutlierEdges, ResetTrueIgnoresIncomingFlags) {
    // flag[0] arrives set, but reset=true clears it so the in range value counts
    auto r = run<3, 3, true>({1, 2, 3}, 0, 10, {true, false, false});
    EXPECT_EQ(r.count, 3);
    EXPECT_DOUBLE_EQ(r.mean, 2.0);
    EXPECT_FALSE(r.flag[0]);
}

TEST(OutlierEdges, ResetFalseKeepsIncomingFlagsExcluded) {
    // this is how the PRC pipeline marks invalid sensors: flag = !valid
    auto r = run<3, 3, false>({1, 2, 30}, 0, 100, {false, false, true});
    EXPECT_EQ(r.count, 2);
    EXPECT_DOUBLE_EQ(r.mean, 1.5);
    EXPECT_TRUE(r.flag[2]);
    EXPECT_FALSE(r.flag[0]);
}

TEST(OutlierEdges, ZeroNumValidAtRuntimeIsSafeAndReturnsNothing) {
    OutlierParams<double, 3, 3> p;
    p.min = 0; p.max = 10;
    p.num_valid = 0;
    double v[3] = {1, 2, 3};
    bool f[3] = {};
    double m = -1;
    const uint16_t used = outlier<double, 3, 3, true>(p, v, f, m);
    EXPECT_EQ(used, 0);
    EXPECT_TRUE(f[0] && f[1] && f[2]);
}

TEST(OutlierEdges, ZeroLengthAtRuntimeIsSafe) {
    OutlierParams<double, 3, 3> p;
    p.min = 0; p.max = 10;
    p.length = 0;
    double v[3] = {1, 2, 3};
    bool f[3] = {};
    double m = -1;
    const uint16_t used = outlier<double, 3, 3, true>(p, v, f, m);
    EXPECT_EQ(used, 0);
    EXPECT_DOUBLE_EQ(m, -1);
}

TEST(OutlierEdges, SingleSensorPassesThrough) {
    auto r = run<1, 1>({42.0}, 0, 100);
    EXPECT_EQ(r.count, 1);
    EXPECT_DOUBLE_EQ(r.mean, 42.0);
    EXPECT_FALSE(r.flag[0]);
}

TEST(WelfordEdges, ScoreNeverGoesNegativeOnASlidingWindowOfFloats) {
    WelfordVariance<float> w;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(101000.f, 101010.f);
    std::vector<float> win;
    for (int k = 0; k < 4; k++) { win.push_back(d(rng)); w.add_variable(win.back()); }
    for (int k = 0; k < 20000; k++) {
        float x = d(rng);
        w.add_variable(x);
        w.remove_variable(win.front());
        win.erase(win.begin());
        win.push_back(x);
        ASSERT_GE(w.get_score(), 0.0f) << "step " << k;
    }
}

TEST(WelfordEdges, NaNInWindowRecoversOnceItIsRemovedAndWindowDrains) {
    WelfordVariance<double> w;
    w.add_variable(1.0);
    w.add_variable(kNaN);
    w.remove_variable(1.0);
    w.remove_variable(kNaN);  // count back to 0 resets mean and M2
    EXPECT_EQ(w.size(), 0);
    w.add_variable(3.0);
    EXPECT_DOUBLE_EQ(w.get_mean(), 3.0);
    EXPECT_DOUBLE_EQ(w.get_score(), 0.0);
}
