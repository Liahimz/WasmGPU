#include "cpu_conv_kernels.h"

#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <cstddef>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace network {
namespace cpu_conv {
namespace {

template <typename Body>
void runRange(std::size_t begin, std::size_t end, std::size_t grain_size, bool use_threads, const Body& body) {
    if (use_threads) {
        parallel::parallelFor(begin, end, grain_size, [&](std::size_t index) {
            body(index);
        });
        return;
    }

    for (std::size_t index = begin; index < end; ++index) {
        body(index);
    }
}

bool isSupportedShape(const LayerDesc& layer) {
    if (layer.kernel_y == 1 && layer.kernel_x == 1 && layer.stride_y == 1 && layer.stride_x == 1 &&
        layer.padding_y == 0 && layer.padding_x == 0) {
        return true;
    }
    if (layer.kernel_y == 3 && layer.kernel_x == 3 && layer.stride_y == 1 && layer.stride_x == 1 &&
        layer.padding_y == 1 && layer.padding_x == 1) {
        return true;
    }
    if (layer.kernel_y == 7 && layer.kernel_x == 7 && layer.stride_y == 2 && layer.stride_x == 2 &&
        layer.padding_y == 3 && layer.padding_x == 3) {
        return true;
    }
    return false;
}

float convScalarAt(
    const float* input,
    const float* weights,
    const float* bias,
    uint32_t in_c,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t kernel_y,
    uint32_t kernel_x,
    uint32_t stride_y,
    uint32_t stride_x,
    uint32_t padding_y,
    uint32_t padding_x,
    uint32_t oc,
    uint32_t oy,
    uint32_t ox
) {
    float sum = bias[oc];
    for (uint32_t ic = 0; ic < in_c; ++ic) {
        for (uint32_t ky = 0; ky < kernel_y; ++ky) {
            const int iy = static_cast<int>(oy * stride_y + ky) - static_cast<int>(padding_y);
            if (iy < 0 || iy >= static_cast<int>(in_h)) {
                continue;
            }
            for (uint32_t kx = 0; kx < kernel_x; ++kx) {
                const int ix = static_cast<int>(ox * stride_x + kx) - static_cast<int>(padding_x);
                if (ix < 0 || ix >= static_cast<int>(in_w)) {
                    continue;
                }

                const std::size_t input_index =
                    static_cast<std::size_t>(ic) * in_h * in_w +
                    static_cast<std::size_t>(iy) * in_w +
                    static_cast<std::size_t>(ix);
                const std::size_t weight_index =
                    static_cast<std::size_t>(oc) * in_c * kernel_y * kernel_x +
                    static_cast<std::size_t>(ic) * kernel_y * kernel_x +
                    static_cast<std::size_t>(ky) * kernel_x +
                    kx;
                sum += input[input_index] * weights[weight_index];
            }
        }
    }
    return sum;
}

#if defined(__wasm_simd128__)
v128_t loadOiHwOc4(
    const float* weights,
    uint32_t in_c,
    uint32_t kernel_y,
    uint32_t kernel_x,
    uint32_t oc,
    uint32_t ic,
    uint32_t ky,
    uint32_t kx
) {
    const std::size_t oc_stride = static_cast<std::size_t>(in_c) * kernel_y * kernel_x;
    const std::size_t inner = static_cast<std::size_t>(ic) * kernel_y * kernel_x +
                              static_cast<std::size_t>(ky) * kernel_x +
                              kx;
    return wasm_f32x4_make(
        weights[static_cast<std::size_t>(oc) * oc_stride + inner],
        weights[static_cast<std::size_t>(oc + 1) * oc_stride + inner],
        weights[static_cast<std::size_t>(oc + 2) * oc_stride + inner],
        weights[static_cast<std::size_t>(oc + 3) * oc_stride + inner]
    );
}

void storeOc4(
    float* output,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t oc,
    uint32_t oy,
    uint32_t ox,
    v128_t value
) {
    alignas(16) float lanes[4];
    wasm_v128_store(lanes, value);
    const std::size_t plane = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t spatial = static_cast<std::size_t>(oy) * out_w + ox;
    output[static_cast<std::size_t>(oc) * plane + spatial] = lanes[0];
    output[static_cast<std::size_t>(oc + 1) * plane + spatial] = lanes[1];
    output[static_cast<std::size_t>(oc + 2) * plane + spatial] = lanes[2];
    output[static_cast<std::size_t>(oc + 3) * plane + spatial] = lanes[3];
}

void runConvOc4Simd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    bool use_threads
) {
    const uint32_t in_c = layer.input_shape.dims[0];
    const uint32_t in_h = layer.input_shape.dims[1];
    const uint32_t in_w = layer.input_shape.dims[2];
    const uint32_t out_c = layer.output_shape.dims[0];
    const uint32_t out_h = layer.output_shape.dims[1];
    const uint32_t out_w = layer.output_shape.dims[2];
    const uint32_t oc4 = out_c / 4;
    const std::size_t spatial = static_cast<std::size_t>(out_h) * out_w;

    runRange(0, static_cast<std::size_t>(oc4) * spatial, 16, use_threads, [&](std::size_t index) {
        const uint32_t oc_block = static_cast<uint32_t>(index / spatial);
        const uint32_t rem = static_cast<uint32_t>(index % spatial);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;
        const uint32_t oc = oc_block * 4;

        v128_t sum = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
        for (uint32_t ic = 0; ic < in_c; ++ic) {
            for (uint32_t ky = 0; ky < layer.kernel_y; ++ky) {
                const int iy = static_cast<int>(oy * layer.stride_y + ky) - static_cast<int>(layer.padding_y);
                if (iy < 0 || iy >= static_cast<int>(in_h)) {
                    continue;
                }
                for (uint32_t kx = 0; kx < layer.kernel_x; ++kx) {
                    const int ix = static_cast<int>(ox * layer.stride_x + kx) - static_cast<int>(layer.padding_x);
                    if (ix < 0 || ix >= static_cast<int>(in_w)) {
                        continue;
                    }

                    const std::size_t input_index =
                        static_cast<std::size_t>(ic) * in_h * in_w +
                        static_cast<std::size_t>(iy) * in_w +
                        static_cast<std::size_t>(ix);
                    const v128_t input_value = wasm_f32x4_splat(input[input_index]);
                    const v128_t weight_value = loadOiHwOc4(weights, in_c, layer.kernel_y, layer.kernel_x, oc, ic, ky, kx);
                    sum = wasm_f32x4_add(sum, wasm_f32x4_mul(input_value, weight_value));
                }
            }
        }

        storeOc4(output, out_h, out_w, oc, oy, ox, sum);
    });

    if (oc4 * 4 == out_c) {
        return;
    }

    const uint32_t tail_begin = oc4 * 4;
    const std::size_t tail_total = static_cast<std::size_t>(out_c - tail_begin) * spatial;
    runRange(0, tail_total, 64, use_threads, [&](std::size_t index) {
        const uint32_t oc = tail_begin + static_cast<uint32_t>(index / spatial);
        const uint32_t rem = static_cast<uint32_t>(index % spatial);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;
        output[static_cast<std::size_t>(oc) * spatial + rem] = convScalarAt(
            input,
            weights,
            bias,
            in_c,
            in_h,
            in_w,
            layer.kernel_y,
            layer.kernel_x,
            layer.stride_y,
            layer.stride_x,
            layer.padding_y,
            layer.padding_x,
            oc,
            oy,
            ox
        );
    });
}
#endif

} // namespace

bool runSpecializedConv2d(
    const LayerDesc& layer,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    bool use_simd,
    bool use_threads,
    std::vector<float>& output
) {
    if (!use_simd || !isSupportedShape(layer)) {
        return false;
    }
    if (layer.input_shape.dims.size() != 3 || layer.output_shape.dims.size() != 3) {
        return false;
    }
    if (layer.output_shape.dims[0] < 4) {
        return false;
    }

#if defined(__wasm_simd128__)
    output.assign(layer.output_shape.elementCount(), 0.0f);
    runConvOc4Simd(layer, input.data(), weights.data(), bias.data(), output.data(), use_threads);
    return true;
#else
    (void)input;
    (void)weights;
    (void)bias;
    (void)use_threads;
    (void)output;
    return false;
#endif
}

} // namespace cpu_conv
} // namespace network
