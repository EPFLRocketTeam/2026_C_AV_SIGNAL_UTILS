
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace _ {
    constexpr uint16_t gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};
    constexpr uint16_t numberGaps = sizeof(gaps) / sizeof(uint16_t);

    // NaN sorts after every number, so it can not block the ones behind it.
    template<typename T>
    inline bool greater (T a, T b) noexcept {
        return a > b || (std::isnan(a) && !std::isnan(b));
    }

    template<typename T>
    void argsort_shell (T* data, uint16_t* buffer, size_t length) noexcept {
        for (size_t offset = 0; offset < length; offset ++) buffer[offset] = offset;

        for (uint16_t iGap = 0; iGap < numberGaps; iGap ++) {
            size_t gap = gaps[iGap];

            for (size_t src = 0; src < gap; src ++) {
                for (size_t idx = src + gap; idx < length; idx += gap) {
                    size_t jdx = idx;
                    
                    uint16_t val = buffer[idx];
                    while (jdx > src && greater(data[buffer[jdx - gap]], data[val])) {
                        buffer[jdx] = buffer[jdx - gap];
                        jdx -= gap;
                    }

                    buffer[jdx] = val;
                }
            }
        }
    }
}

template<typename T>
void argsort (T* data, uint16_t* buffer, size_t length) noexcept {
    _::argsort_shell(data, buffer, length);
}
