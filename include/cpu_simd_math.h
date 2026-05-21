#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace network {
namespace simd_math {

inline float dotProductScalar(const float* left, const float* right, std::size_t count) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        sum += left[i] * right[i];
    }
    return sum;
}

inline float dotProductSimd(const float* left, const float* right, std::size_t count) {
#if defined(__wasm_simd128__)
    v128_t acc = wasm_f32x4_splat(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const v128_t a = wasm_v128_load(left + i);
        const v128_t b = wasm_v128_load(right + i);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(a, b));
    }

    alignas(16) float lanes[4];
    wasm_v128_store(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (; i < count; ++i) {
        sum += left[i] * right[i];
    }
    return sum;
#else
    return dotProductScalar(left, right, count);
#endif
}

inline float sumScalar(const float* data, std::size_t count) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

inline float sumSimd(const float* data, std::size_t count) {
#if defined(__wasm_simd128__)
    v128_t acc = wasm_f32x4_splat(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        acc = wasm_f32x4_add(acc, wasm_v128_load(data + i));
    }

    alignas(16) float lanes[4];
    wasm_v128_store(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (; i < count; ++i) {
        sum += data[i];
    }
    return sum;
#else
    return sumScalar(data, count);
#endif
}

inline void reluScalar(const float* input, float* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = std::max(input[i], 0.0f);
    }
}

inline void reluSimd(const float* input, float* output, std::size_t count) {
#if defined(__wasm_simd128__)
    const v128_t zero = wasm_f32x4_splat(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        wasm_v128_store(output + i, wasm_f32x4_max(wasm_v128_load(input + i), zero));
    }
    for (; i < count; ++i) {
        output[i] = std::max(input[i], 0.0f);
    }
#else
    reluScalar(input, output, count);
#endif
}

inline float maxPoolScalar(
    const float* input,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t kernel_y,
    uint32_t kernel_x,
    uint32_t stride_y,
    uint32_t stride_x,
    uint32_t padding_y,
    uint32_t padding_x,
    uint32_t oy,
    uint32_t ox
) {
    float best = -std::numeric_limits<float>::infinity();
    for (uint32_t ky = 0; ky < kernel_y; ++ky) {
        for (uint32_t kx = 0; kx < kernel_x; ++kx) {
            const int iy = static_cast<int>(oy * stride_y + ky) - static_cast<int>(padding_y);
            const int ix = static_cast<int>(ox * stride_x + kx) - static_cast<int>(padding_x);
            if (iy < 0 || ix < 0 || iy >= static_cast<int>(in_h) || ix >= static_cast<int>(in_w)) {
                continue;
            }
            best = std::max(best, input[static_cast<std::size_t>(iy) * in_w + ix]);
        }
    }
    return best;
}

inline float maxPoolSimd(
    const float* input,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t kernel_y,
    uint32_t kernel_x,
    uint32_t stride_y,
    uint32_t stride_x,
    uint32_t padding_y,
    uint32_t padding_x,
    uint32_t oy,
    uint32_t ox
) {
#if defined(__wasm_simd128__)
    float best = -std::numeric_limits<float>::infinity();
    for (uint32_t ky = 0; ky < kernel_y; ++ky) {
        const int iy = static_cast<int>(oy * stride_y + ky) - static_cast<int>(padding_y);
        if (iy < 0 || iy >= static_cast<int>(in_h)) {
            continue;
        }

        uint32_t kx = 0;
        for (; kx + 4 <= kernel_x; kx += 4) {
            const int ix = static_cast<int>(ox * stride_x + kx) - static_cast<int>(padding_x);
            if (ix < 0 || ix + 3 >= static_cast<int>(in_w)) {
                break;
            }
            const v128_t values = wasm_v128_load(input + static_cast<std::size_t>(iy) * in_w + ix);
            alignas(16) float lanes[4];
            wasm_v128_store(lanes, values);
            best = std::max(best, std::max(std::max(lanes[0], lanes[1]), std::max(lanes[2], lanes[3])));
        }
        for (; kx < kernel_x; ++kx) {
            const int ix = static_cast<int>(ox * stride_x + kx) - static_cast<int>(padding_x);
            if (ix < 0 || ix >= static_cast<int>(in_w)) {
                continue;
            }
            best = std::max(best, input[static_cast<std::size_t>(iy) * in_w + ix]);
        }
    }
    return best;
#else
    return maxPoolScalar(input, in_h, in_w, kernel_y, kernel_x, stride_y, stride_x, padding_y, padding_x, oy, ox);
#endif
}

} // namespace simd_math
} // namespace network
