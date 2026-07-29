
#pragma once

#include <cstdint>

#include "sigutils/argsort.hpp"

template<typename FloatType>
struct WelfordVariance {
private:
    FloatType mean = 0;
    FloatType M2 = 0;

    uint16_t count = 0;
public:
    void add_variable(FloatType x) noexcept {
        count += 1;
        
        FloatType old_mean = mean;
        mean += (x - mean) / count;
        M2 += (x - old_mean) * (x - mean);
    }

    void remove_variable(FloatType x) noexcept {
        count -= 1;
        if (count == 0) {
            mean = 0;
            M2 = 0;
            return ;
        }

        FloatType new_mean = mean;
        mean -= (x - mean) / count;
        M2 -= (x - new_mean) * (x - mean);
    }

    uint16_t size () noexcept {
        return count;
    }
        
    FloatType get_mean() noexcept {
        return mean;
    }

    // For a fixed count, equals the variance scaled proportionally.
    FloatType get_score () noexcept {
        return M2;
    }
};

template<typename FloatType, const int _length, const int _num_valid>
struct OutlierParams {
    FloatType min, max;

    size_t   num_valid = _num_valid;
    size_t   length    = _length;
    uint16_t idx_buffer[_length];
    uint16_t idx_valid_buffer[_length];

    inline bool in_range (FloatType val) noexcept {
        return min <= val && val <= max;
    }
};

template<typename FloatType, const int _length, const int _num_valid, const bool reset = true>
uint16_t outlier (
    OutlierParams<FloatType, _length, _num_valid> &params,

    FloatType* values,
    bool*      is_outlier,

    FloatType& mean
) noexcept {
    argsort(values, params.idx_buffer, params.length);

    uint16_t count_valid = 0;
    for (size_t offset = 0; offset < params.length; offset ++) {
        uint16_t index = params.idx_buffer[offset];

        if (reset) {
            is_outlier[index] = false;
        }

        if (params.in_range(values[index]) && !is_outlier[index]) {
            params.idx_valid_buffer[count_valid] = index;
            count_valid ++;
        }

        is_outlier[index] = true;
    }

    if (count_valid == 0) {
        return 0;
    }

    uint16_t  opt_lft  = 0;
    uint16_t  opt_rgt  = 0;
    FloatType opt_scr  = 0;
    FloatType opt_mean = 0;

    WelfordVariance<FloatType> scoreComputer;

    for (size_t offset = 0; offset < count_valid && offset + 1 < params.num_valid; offset ++) {
        uint16_t index = params.idx_valid_buffer[offset];
        scoreComputer.add_variable(values[index]);
    }

    if (count_valid < params.num_valid) {
        opt_lft  = 0;
        opt_rgt  = count_valid;
        opt_scr  = scoreComputer.get_score();
        opt_mean = scoreComputer.get_mean();
    } else {
        for (size_t offset = 0; offset + params.num_valid - 1 < count_valid; offset ++) {
            uint16_t new_index = params.idx_valid_buffer[offset + params.num_valid - 1];
            scoreComputer.add_variable(values[new_index]);

            if (scoreComputer.get_score() < opt_scr || offset == 0) {
                opt_scr  = scoreComputer.get_score();
                opt_mean = scoreComputer.get_mean();
                opt_lft  = offset;
                opt_rgt  = offset + params.num_valid;
            }
            
            uint16_t old_index = params.idx_valid_buffer[offset];
            scoreComputer.remove_variable(values[old_index]);
        }
    }

    for (size_t opt_offset = opt_lft; opt_offset < opt_rgt; opt_offset ++) {
        uint16_t opt_index = params.idx_valid_buffer[opt_offset];

        is_outlier[opt_index] = false;
    }

    mean = opt_mean;
    return opt_rgt - opt_lft;
}
