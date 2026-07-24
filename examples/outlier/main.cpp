#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <limits>
#include <numeric>

#include "sigutils/outlier.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <N_devices> <N_keep_devices> <N_time>\n";
        return 1;
    }

    const int N_devices      = std::stoi(argv[1]);
    const int N_keep_devices = std::stoi(argv[2]);
    const int N_time         = std::stoi(argv[3]);

    if (N_keep_devices > N_devices) {
        std::cerr << "Error: N_keep_devices cannot be greater than N_devices.\n";
        return 1;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(1.0f, 1.0f);     // Clean signal ~ N(1, 1)
    std::uniform_real_distribution<float> unif_dist(- 9.f, 11.0f); // Glitch signal ~ U(0.5, 2.0)
    std::bernoulli_distribution glitch_prob(0.25);              // 25% chance of glitch per device

    constexpr int MAX_DEVICES = 128; 

    if (N_devices > MAX_DEVICES) {
        std::cerr << "Error: N_devices exceeds template bounds (" << MAX_DEVICES << ")\n";
        return 1;
    }

    std::ofstream csv("output.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to create output.csv\n";
        return 1;
    }

    // Write CSV Header
    csv << "time";
    for (int d = 0; d < N_devices; ++d) csv << ",dev_" << d;
    for (int k = 0; k < N_keep_devices; ++k) csv << ",keep_" << k;
    csv << ",algo_mean,raw_mean,true_mean\n";

    std::vector<float> values(N_devices);
    bool  is_outlier[N_devices];

    OutlierParams<float, MAX_DEVICES, MAX_DEVICES> params;
    params.length = N_devices;
    params.num_valid = N_keep_devices;
    params.min = -std::numeric_limits<float>::infinity();
    params.max =  std::numeric_limits<float>::infinity();

    const float true_mean = 1.0f; // Target signal ground truth

    for (int t = 0; t < N_time; ++t) {
        // 1. Generate signal with 25% chance of Uniform[0.5, 2.0] glitch per device
        double raw_sum = 0.0;
        for (int d = 0; d < N_devices; ++d) {
            if (glitch_prob(rng)) {
                values[d] = unif_dist(rng); // Injected Uniform[0.5, 2.0] glitch
            } else {
                values[d] = norm_dist(rng); // Clean N(1, 1) measurement
            }
            raw_sum += values[d];
        }

        // Raw mean across ALL N_devices (includes glitched devices)
        float raw_mean = static_cast<float>(raw_sum / N_devices);

        // 2. Run minimum-variance outlier selection algorithm
        float algo_mean = 0.0f;
        outlier<float, MAX_DEVICES, MAX_DEVICES>(params, values.data(), is_outlier, algo_mean);

        // 3. Extract kept values
        std::vector<float> kept_values;
        for (int d = 0; d < N_devices; ++d) {
            if (!is_outlier[d]) {
                kept_values.push_back(values[d]);
            }
        }

        while (kept_values.size() < static_cast<size_t>(N_keep_devices)) {
            kept_values.push_back(0.0f);
        }

        // 4. Output row to CSV
        csv << t;
        for (int d = 0; d < N_devices; ++d) {
            csv << "," << values[d];
        }
        for (int k = 0; k < N_keep_devices; ++k) {
            csv << "," << kept_values[k];
        }
        csv << "," << algo_mean << "," << raw_mean << "," << true_mean << "\n";
    }

    csv.close();
    std::cout << "Successfully generated output.csv across " << N_time << " timesteps.\n";

    return 0;
}