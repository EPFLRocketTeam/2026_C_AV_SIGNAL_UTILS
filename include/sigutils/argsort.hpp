
#pragma once

#include <cstdint>

namespace _ {
    constexpr uint16_t gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};
    constexpr uint16_t numberGaps = sizeof(gaps) / sizeof(uint16_t);

    template<typename T>
    void argsort_shell (T* data, uint16_t* buffer, size_t length) noexcept {
        for (size_t offset = 0; offset < length; offset ++) buffer[offset] = offset;

        for (uint16_t iGap = 0; iGap < numberGaps; iGap ++) {
            uint16_t gap = gaps[iGap];

            for (uint16_t src = 0; src < gap; src ++) {
                for (uint16_t idx = src + gap; idx < length; idx += gap) {
                    uint16_t jdx = idx;
                    
                    uint16_t val = buffer[idx];
                    while (jdx > src && data[buffer[jdx - gap]] > data[val]) {
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
