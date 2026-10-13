
#pragma once

#include <cstddef>
#include <cstdint>
#include "sigutils/neumaier.hpp"

template<typename FloatType, const int _num_samples>
struct RunningAverage {
    static_assert(_num_samples > 0, "RunningAverage needs a window of at least one sample");
private:
    FloatType buffer[_num_samples] = {};
    size_t offset = 0;
    // samples seen while the first window fills, so mean() is the partial mean
    size_t filled = 0;

    StableFloatSum<FloatType> sum1;
    StableFloatSum<FloatType> sum2;

    bool has_mean = false;
public:
    size_t num_samples = _num_samples;

    bool ready () {
        return has_mean;
    }

    void push (FloatType point) {
        // update sums
        sum1 -= buffer[offset];
        sum1 += point;
        sum2 += point;
        
        // update buffer
        buffer[offset] = point;

        offset ++;
        if (!has_mean) {
            filled ++;
        }
        
        // if reached the end, put sum2 into sum1
        // and mark the system as ready, reset sum1
        if (offset == num_samples) {
            offset = 0;

            sum1 = sum2;
            sum2 = 0.;
            has_mean = true;
        }
    }
    FloatType mean () {
        if (has_mean) {
            return sum1 / num_samples;
        }
        // window not full yet: mean of what has arrived, 0 before the first push
        return filled == 0 ? FloatType(0) : sum1 / filled;
    }
};
