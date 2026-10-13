#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "sigutils/argsort.hpp"

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

TEST(ArgsortEdges, NaNSortsLastAndTheRestStaysOrdered) {
    std::vector<double> v = {5, kNaN, 1, 2, 3, kNaN, 0};
    std::vector<uint16_t> idx(v.size());
    argsort(v.data(), idx.data(), v.size());
    // finite values ascending, then both NaNs
    for (size_t k = 0; k + 1 < 5; k++) EXPECT_LE(v[idx[k]], v[idx[k + 1]]);
    EXPECT_TRUE(std::isnan(v[idx[5]]));
    EXPECT_TRUE(std::isnan(v[idx[6]]));
}

TEST(ArgsortEdges, RandomNaNMixKeepsFiniteValuesSorted) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> d(-50, 50);
    for (int round = 0; round < 500; round++) {
        size_t n = 1 + rng() % 40;
        std::vector<double> v(n);
        size_t nan_count = 0;
        for (auto &x : v) {
            if (rng() % 4 == 0) { x = kNaN; nan_count++; } else { x = d(rng); }
        }
        std::vector<uint16_t> idx(n);
        argsort(v.data(), idx.data(), n);
        for (size_t k = 0; k + 1 < n - nan_count; k++) ASSERT_LE(v[idx[k]], v[idx[k + 1]]) << "round " << round;
        for (size_t k = n - nan_count; k < n; k++) ASSERT_TRUE(std::isnan(v[idx[k]]));
    }
}

TEST(ArgsortEdges, InfinitiesSortAtTheEnds) {
    std::vector<double> v = {kInf, 0, -kInf, 3};
    std::vector<uint16_t> idx(4);
    argsort(v.data(), idx.data(), 4);
    EXPECT_EQ(idx[0], 2);
    EXPECT_EQ(idx[3], 0);
}

TEST(ArgsortEdges, LengthAtTheIndexTypeLimit) {
    // The index buffer is uint16_t, so 65535 is the largest length that can be
    // addressed. Loop counters must not wrap before reaching it.
    const size_t n = 65535;
    std::mt19937 rng(3);
    std::vector<float> v(n);
    for (auto &x : v) x = static_cast<float>(rng() % 100000);
    std::vector<uint16_t> idx(n);
    argsort(v.data(), idx.data(), n);
    std::vector<bool> seen(n, false);
    for (size_t k = 0; k < n; k++) {
        ASSERT_FALSE(seen[idx[k]]);
        seen[idx[k]] = true;
        if (k + 1 < n) { ASSERT_LE(v[idx[k]], v[idx[k + 1]]) << "at " << k; }
    }
}
