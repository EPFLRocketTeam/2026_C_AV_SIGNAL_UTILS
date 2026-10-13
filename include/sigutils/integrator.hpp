
#pragma once

#include <cmath>
#include <cstdint>

#include "sigutils/neumaier.hpp"

template<typename Src, typename Dst>
constexpr Dst identity_cast_conversion(Src x) { return static_cast<Dst>(x); }

template<typename T>
constexpr T millis_to_seconds(T x) { return x / static_cast<T>(1000); }

template<
    typename FloatType,
    typename TimeType,
    auto TimeConversion,
    auto IntegratorConversion = identity_cast_conversion<FloatType, FloatType>
>
struct StableIntegrator {
private:
    StableFloatSum<FloatType> integrator_;

    TimeType  last_time_;
    FloatType last_value_;
public:
    void start (FloatType value, TimeType current_time) {
        integrator_ = 0;
        last_time_  = current_time;
        // a dead sensor at start counts as zero so the first step stays finite
        last_value_ = std::isfinite(value) ? value : static_cast<FloatType>(0);
    }
    void ingest (FloatType value, TimeType current_time) {
        // a NaN or infinite sample is skipped; the next good sample is
        // bridged by the trapezoid from the last good one
        if (!std::isfinite(value)) {
            return ;
        }

        integrator_ += (value + last_value_) * TimeConversion(current_time - last_time_);

        last_time_  = current_time;
        last_value_ = value;
    }

    operator FloatType() const {
        return IntegratorConversion( static_cast<FloatType>(integrator_) / ((FloatType) 2) );
    }
};

template<
    typename FloatType
>
using StableIntegratorMillis = StableIntegrator<
    FloatType,
    uint32_t,
    identity_cast_conversion<uint32_t, FloatType>,
    millis_to_seconds<FloatType>
>;
