
#pragma once

#include <cstdint>
#include "sigutils/neumaier.hpp"

template<typename FloatType, const int _num_samples>
struct RunningAverage {
private:
    FloatType buffer[_num_samples];
    size_t offset = 0;

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
        return sum1 / num_samples;
    }
};