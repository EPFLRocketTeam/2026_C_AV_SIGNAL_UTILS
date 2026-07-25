#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>

#include "sigutils/neumaier.hpp"
#include "sigutils/average.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <N_sample> <N_window>\n";
        return 1;
    }

    const size_t N_sample = static_cast<size_t>(std::atol(argv[1]));
    const size_t N_window = static_cast<size_t>(std::atol(argv[2]));

    if (N_window == 0 || N_sample < N_window) {
        std::cerr << "Error: N_window must be > 0 and <= N_sample.\n";
        return 1;
    }

    // 1. Generate Raw Signal S: sin(x / 1000) + Gaussian Noise
    std::vector<float> S(N_sample);
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::normal_distribution<float> noise(0.0f, 0.2f);

    for (size_t i = 0; i < N_sample; ++i) {
        S[i] = std::sin(static_cast<float>(i) / 1000.0f) + noise(rng);
    }

    // Allocate memory for computed signals
    std::vector<float> S0(N_sample, 0.0f);
    std::vector<float> S1(N_sample, 0.0f);
    std::vector<float> S2(N_sample, 0.0f);
    std::vector<float> S3(N_sample, 0.0f);
    std::vector<float> S4(N_sample, 0.0f);

    // --- Signal S0: O(N * W) True Moving Average using long double ---
    for (size_t i = N_window - 1; i < N_sample; ++i) {
        long double win_sum = 0.0L;
        for (size_t j = i + 1 - N_window; j <= i; ++j) {
            win_sum += static_cast<long double>(S[j]);
        }
        S0[i] = static_cast<float>(win_sum / static_cast<long double>(N_window));
    }

    // --- Signal S1: O(N * W) True Moving Average using Neumaier ---
    for (size_t i = N_window - 1; i < N_sample; ++i) {
        StableFloatSum<float> win_sum;
        for (size_t j = i + 1 - N_window; j <= i; ++j) {
            win_sum += S[j];
        }
        S1[i] = static_cast<float>(win_sum) / static_cast<float>(N_window);
    }

    // --- Signal S2: O(N * W) Naive True Moving Average ---
    for (size_t i = N_window - 1; i < N_sample; ++i) {
        float win_sum = 0.0f;
        for (size_t j = i + 1 - N_window; j <= i; ++j) {
            win_sum += S[j];
        }
        S2[i] = win_sum / static_cast<float>(N_window);
    }

    // --- Signal S3: O(N) Sliding Moving Average (Naive Float without Resets) ---
    {
        float running_sum = 0.0f;
        for (size_t i = 0; i < N_sample; ++i) {
            running_sum += S[i];
            if (i >= N_window) {
                running_sum -= S[i - N_window];
            }
            if (i >= N_window - 1) {
                S3[i] = running_sum / static_cast<float>(N_window);
            }
        }
    }

    // --- Signal S4: O(N) Sliding Moving Average via Dynamic Buffer Resync ---
    {
        std::vector<float> buffer(N_window, 0.0f);
        size_t offset = 0;
        StableFloatSum<float> sum1;
        StableFloatSum<float> sum2;

        for (size_t i = 0; i < N_sample; ++i) {
            sum1 -= buffer[offset];
            sum1 += S[i];
            sum2 += S[i];

            buffer[offset] = S[i];
            offset++;

            if (offset == N_window) {
                offset = 0;
                sum1 = sum2;
                sum2.reset();
            }

            if (i >= N_window - 1) {
                S4[i] = static_cast<float>(sum1) / static_cast<float>(N_window);
            }
        }
    }

    // 2. Export to CSV
    // Moving averages S0..S4 are valid from index (N_window - 1) to (N_sample - 1).
    // To align S visually with the center of the moving average window,
    // sample S at index (i - N_window / 2).
    const size_t half_window = N_window / 2;
    const size_t start_idx = N_window - 1;
    const size_t end_idx = N_sample;

    std::ofstream csv("signals.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to open output CSV file.\n";
        return 1;
    }

    csv << "sample_index,S,S0,S1,S2,S3,S4\n";
    for (size_t i = start_idx; i < end_idx; ++i) {
        const size_t s_aligned_idx = i - half_window;
        csv << i << ","
            << S[s_aligned_idx] << ","
            << S0[i] << ","
            << S1[i] << ","
            << S2[i] << ","
            << S3[i] << ","
            << S4[i] << "\n";
    }

    csv.close();
    std::cout << "Successfully exported signals.csv with " 
              << (end_idx - start_idx) << " aligned samples.\n";

    return 0;
}