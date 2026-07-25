
#pragma once

#include <cmath>

template<typename FloatType>
class StableFloatSum {
private:
    FloatType sum = 0;
    FloatType c = 0; // Accumulated error compensation

public:
    StableFloatSum() = default;

    void add(FloatType val) {
        FloatType t = sum + val;
        if (std::abs(sum) >= std::abs(val)) {
            c += (sum - t) + val;
        } else {
            c += (val - t) + sum;
        }
        sum = t;
    }

    StableFloatSum& operator= (FloatType val) {
        sum = val;
        c = 0;
        return *this;
    }

    StableFloatSum& operator+=(FloatType val) {
        add(val);
        return *this;
    }

    StableFloatSum& operator-=(FloatType val) {
        add(-val);
        return *this;
    }

    operator FloatType() const {
        return sum + c;
    }

    void reset() {
        sum = 0;
        c = 0;
    }
};
