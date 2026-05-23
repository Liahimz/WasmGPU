#include "cpu_conv_kernels.h"

#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

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
    if (layer.kernel_y == 1 && layer.kernel_x == 1 && layer.stride_y == 2 && layer.stride_x == 2 &&
        layer.padding_y == 0 && layer.padding_x == 0) {
        return true;
    }
    if (layer.kernel_y == 3 && layer.kernel_x == 3 && layer.stride_y == 1 && layer.stride_x == 1 &&
        layer.padding_y == 1 && layer.padding_x == 1) {
        return true;
    }
    if (layer.kernel_y == 3 && layer.kernel_x == 3 && layer.stride_y == 2 && layer.stride_x == 2 &&
        layer.padding_y == 1 && layer.padding_x == 1) {
        return true;
    }
    if (layer.kernel_y == 7 && layer.kernel_x == 7 && layer.stride_y == 2 && layer.stride_x == 2 &&
        layer.padding_y == 3 && layer.padding_x == 3) {
        return true;
    }
    return false;
}

std::size_t packedOc4Index(
    uint32_t in_c,
    uint32_t kernel_y,
    uint32_t kernel_x,
    uint32_t oc_block,
    uint32_t ic,
    uint32_t ky,
    uint32_t kx
) {
    return (((static_cast<std::size_t>(oc_block) * in_c + ic) * kernel_y + ky) * kernel_x + kx) * 4;
}

std::size_t c4Index(uint32_t channels, uint32_t height, uint32_t width, uint32_t c, uint32_t y, uint32_t x) {
    const uint32_t c4 = (channels + 3) / 4;
    return ((static_cast<std::size_t>(y) * width + x) * c4 + c / 4) * 4 + c % 4;
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

v128_t loadPackedOc4(
    const float* packed_weights,
    uint32_t in_c,
    uint32_t kernel_y,
    uint32_t kernel_x,
    uint32_t oc_block,
    uint32_t ic,
    uint32_t ky,
    uint32_t kx
) {
    return wasm_v128_load(packed_weights + packedOc4Index(in_c, kernel_y, kernel_x, oc_block, ic, ky, kx));
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

template <int TilePixels>
void storeOc4Tile(
    float* output,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t oc,
    uint32_t oy,
    uint32_t ox,
    const v128_t* values
) {
    for (int tile = 0; tile < TilePixels; ++tile) {
        storeOc4(output, out_h, out_w, oc, oy, ox + static_cast<uint32_t>(tile), values[tile]);
    }
}

template <int TilePixels>
void runConv1x1Oc4TiledSimd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
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
    const uint32_t tile_count_x = out_w / TilePixels;
    const std::size_t row_tiles = static_cast<std::size_t>(out_h) * tile_count_x;
    const bool use_packed = packed_weights != nullptr;

    const std::size_t grain_size = use_threads ? 64 : 4;
    runRange(0, static_cast<std::size_t>(oc4) * row_tiles, grain_size, use_threads, [&](std::size_t index) {
        const uint32_t oc_block = static_cast<uint32_t>(index / row_tiles);
        const uint32_t tile_index = static_cast<uint32_t>(index % row_tiles);
        const uint32_t oy = tile_index / tile_count_x;
        const uint32_t ox = (tile_index % tile_count_x) * TilePixels;
        const uint32_t oc = oc_block * 4;
        const uint32_t iy = oy * layer.stride_y;

        v128_t sum[TilePixels];
        const v128_t bias_value = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
        for (int tile = 0; tile < TilePixels; ++tile) {
            sum[tile] = bias_value;
        }
        for (uint32_t ic = 0; ic < in_c; ++ic) {
            const std::size_t input_base =
                static_cast<std::size_t>(ic) * in_h * in_w +
                static_cast<std::size_t>(iy) * in_w +
                static_cast<std::size_t>(ox) * layer.stride_x;
            const v128_t weight_value = use_packed
                ? loadPackedOc4(packed_weights, in_c, 1, 1, oc_block, ic, 0, 0)
                : loadOiHwOc4(weights, in_c, 1, 1, oc, ic, 0, 0);
            for (int tile = 0; tile < TilePixels; ++tile) {
                const v128_t input_value = wasm_f32x4_splat(input[input_base + static_cast<std::size_t>(tile) * layer.stride_x]);
                sum[tile] = wasm_f32x4_add(sum[tile], wasm_f32x4_mul(input_value, weight_value));
            }
        }

        storeOc4Tile<TilePixels>(output, out_h, out_w, oc, oy, ox, sum);
    });

    const uint32_t tail_x = tile_count_x * TilePixels;
    if (tail_x < out_w) {
        runRange(0, static_cast<std::size_t>(oc4) * out_h * (out_w - tail_x), 64, use_threads, [&](std::size_t index) {
            const uint32_t tail_width = out_w - tail_x;
            const uint32_t oc_block = static_cast<uint32_t>(index / (static_cast<std::size_t>(out_h) * tail_width));
            const uint32_t rem = static_cast<uint32_t>(index % (static_cast<std::size_t>(out_h) * tail_width));
            const uint32_t oy = rem / tail_width;
            const uint32_t ox = tail_x + rem % tail_width;
            const uint32_t oc = oc_block * 4;

            v128_t sum = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
            for (uint32_t ic = 0; ic < in_c; ++ic) {
                const std::size_t input_index =
                    static_cast<std::size_t>(ic) * in_h * in_w +
                    static_cast<std::size_t>(oy * layer.stride_y) * in_w +
                    static_cast<std::size_t>(ox * layer.stride_x);
                const v128_t input_value = wasm_f32x4_splat(input[input_index]);
                const v128_t weight_value = use_packed
                    ? loadPackedOc4(packed_weights, in_c, 1, 1, oc_block, ic, 0, 0)
                    : loadOiHwOc4(weights, in_c, 1, 1, oc, ic, 0, 0);
                sum = wasm_f32x4_add(sum, wasm_f32x4_mul(input_value, weight_value));
            }
            storeOc4(output, out_h, out_w, oc, oy, ox, sum);
        });
    }

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

template <int TilePixels>
void runConv3x3Stride1Pad1Oc4InteriorTiledSimd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
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
    const bool use_packed = packed_weights != nullptr;

    const uint32_t interior_w = out_w > 2 ? out_w - 2 : 0;
    const uint32_t tile_count_x = interior_w / TilePixels;
    const std::size_t row_tiles = out_h > 2 ? static_cast<std::size_t>(out_h - 2) * tile_count_x : 0;
    const std::size_t grain_size = use_threads ? 16 : 4;

    runRange(0, static_cast<std::size_t>(oc4) * row_tiles, grain_size, use_threads, [&](std::size_t index) {
        const uint32_t oc_block = static_cast<uint32_t>(index / row_tiles);
        const uint32_t tile_index = static_cast<uint32_t>(index % row_tiles);
        const uint32_t oy = 1 + tile_index / tile_count_x;
        const uint32_t ox = 1 + (tile_index % tile_count_x) * TilePixels;
        const uint32_t oc = oc_block * 4;

        v128_t sum[TilePixels];
        const v128_t bias_value = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
        for (int tile = 0; tile < TilePixels; ++tile) {
            sum[tile] = bias_value;
        }

        for (uint32_t ic = 0; ic < in_c; ++ic) {
            const std::size_t input_channel_base = static_cast<std::size_t>(ic) * in_h * in_w;
            for (uint32_t ky = 0; ky < 3; ++ky) {
                const uint32_t iy = oy + ky - 1;
                const std::size_t input_row_base = input_channel_base + static_cast<std::size_t>(iy) * in_w + ox - 1;
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    const v128_t weight_value = use_packed
                        ? loadPackedOc4(packed_weights, in_c, 3, 3, oc_block, ic, ky, kx)
                        : loadOiHwOc4(weights, in_c, 3, 3, oc, ic, ky, kx);
                    for (int tile = 0; tile < TilePixels; ++tile) {
                        const v128_t input_value = wasm_f32x4_splat(input[input_row_base + kx + tile]);
                        sum[tile] = wasm_f32x4_add(sum[tile], wasm_f32x4_mul(input_value, weight_value));
                    }
                }
            }
        }

        storeOc4Tile<TilePixels>(output, out_h, out_w, oc, oy, ox, sum);
    });

    const std::size_t spatial = static_cast<std::size_t>(out_h) * out_w;
    runRange(0, static_cast<std::size_t>(oc4) * spatial, 128, use_threads, [&](std::size_t index) {
        const uint32_t oc_block = static_cast<uint32_t>(index / spatial);
        const uint32_t rem = static_cast<uint32_t>(index % spatial);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;
        const uint32_t oc = oc_block * 4;
        const bool is_interior_y = oy > 0 && oy + 1 < out_h;
        const bool is_interior_x = ox > 0 && ox + 1 < out_w;
        const bool is_tiled_x = is_interior_x && ox < 1 + tile_count_x * TilePixels;
        if (is_interior_y && is_tiled_x) {
            return;
        }

        v128_t sum = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
        for (uint32_t ic = 0; ic < in_c; ++ic) {
            for (uint32_t ky = 0; ky < 3; ++ky) {
                const int iy = static_cast<int>(oy + ky) - 1;
                if (iy < 0 || iy >= static_cast<int>(in_h)) {
                    continue;
                }
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    const int ix = static_cast<int>(ox + kx) - 1;
                    if (ix < 0 || ix >= static_cast<int>(in_w)) {
                        continue;
                    }
                    const std::size_t input_index =
                        static_cast<std::size_t>(ic) * in_h * in_w +
                        static_cast<std::size_t>(iy) * in_w +
                        static_cast<std::size_t>(ix);
                    const v128_t input_value = wasm_f32x4_splat(input[input_index]);
                    const v128_t weight_value = use_packed
                        ? loadPackedOc4(packed_weights, in_c, 3, 3, oc_block, ic, ky, kx)
                        : loadOiHwOc4(weights, in_c, 3, 3, oc, ic, ky, kx);
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

void runConvOc4GenericSimd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
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

    const std::size_t grain_size = use_threads ? 256 : 16;
    runRange(0, static_cast<std::size_t>(oc4) * spatial, grain_size, use_threads, [&](std::size_t index) {
        const uint32_t oc_block = static_cast<uint32_t>(index / spatial);
        const uint32_t rem = static_cast<uint32_t>(index % spatial);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;
        const uint32_t oc = oc_block * 4;
        const bool use_packed = packed_weights != nullptr;

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
                    const v128_t weight_value = use_packed
                        ? loadPackedOc4(packed_weights, in_c, layer.kernel_y, layer.kernel_x, oc_block, ic, ky, kx)
                        : loadOiHwOc4(weights, in_c, layer.kernel_y, layer.kernel_x, oc, ic, ky, kx);
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

void runConvOc4TiledSimd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
    const float* bias,
    float* output,
    bool use_threads,
    ConvTileMode tile_mode
) {
    if (layer.kernel_y == 1 && layer.kernel_x == 1 && layer.padding_y == 0 && layer.padding_x == 0) {
        if (tile_mode == ConvTileMode::Oc4x8) {
            runConv1x1Oc4TiledSimd<8>(layer, input, weights, packed_weights, bias, output, use_threads);
            return;
        }
        if (tile_mode == ConvTileMode::Oc4x4) {
            runConv1x1Oc4TiledSimd<4>(layer, input, weights, packed_weights, bias, output, use_threads);
            return;
        }
    }

    if (layer.kernel_y == 3 && layer.kernel_x == 3 && layer.stride_y == 1 && layer.stride_x == 1 &&
        layer.padding_y == 1 && layer.padding_x == 1) {
        if (tile_mode == ConvTileMode::Oc4x8) {
            runConv3x3Stride1Pad1Oc4InteriorTiledSimd<8>(
                layer, input, weights, packed_weights, bias, output, use_threads
            );
            return;
        }
        if (tile_mode == ConvTileMode::Oc4x4) {
            runConv3x3Stride1Pad1Oc4InteriorTiledSimd<4>(
                layer, input, weights, packed_weights, bias, output, use_threads
            );
            return;
        }
    }

    runConvOc4GenericSimd(layer, input, weights, packed_weights, bias, output, use_threads);
}

void storeC4(float* output, uint32_t out_c, uint32_t out_h, uint32_t out_w, uint32_t oc_block, uint32_t oy, uint32_t ox, v128_t value) {
    const uint32_t out_c4 = (out_c + 3) / 4;
    wasm_v128_store(output + ((static_cast<std::size_t>(oy) * out_w + ox) * out_c4 + oc_block) * 4, value);
}

float loadC4Scalar(const float* input, uint32_t in_c, uint32_t in_h, uint32_t in_w, uint32_t ic, uint32_t iy, uint32_t ix) {
    return input[c4Index(in_c, in_h, in_w, ic, iy, ix)];
}

template <int TilePixels>
void runConv1x1C4Oc8xTileSimd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
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
    const uint32_t oc8 = oc4 / 2;
    const uint32_t tile_count_x = out_w / TilePixels;
    const std::size_t row_tiles = static_cast<std::size_t>(out_h) * tile_count_x;
    const bool use_packed = packed_weights != nullptr;

    runRange(0, static_cast<std::size_t>(oc8) * row_tiles, use_threads ? 32 : 4, use_threads, [&](std::size_t index) {
        const uint32_t oc_pair = static_cast<uint32_t>(index / row_tiles);
        const uint32_t tile_index = static_cast<uint32_t>(index % row_tiles);
        const uint32_t oc_block0 = oc_pair * 2;
        const uint32_t oc_block1 = oc_block0 + 1;
        const uint32_t oc0 = oc_block0 * 4;
        const uint32_t oc1 = oc_block1 * 4;
        const uint32_t oy = tile_index / tile_count_x;
        const uint32_t ox = (tile_index % tile_count_x) * TilePixels;
        const uint32_t iy = oy * layer.stride_y;

        v128_t sum0[TilePixels];
        v128_t sum1[TilePixels];
        const v128_t bias0 = wasm_f32x4_make(bias[oc0], bias[oc0 + 1], bias[oc0 + 2], bias[oc0 + 3]);
        const v128_t bias1 = wasm_f32x4_make(bias[oc1], bias[oc1 + 1], bias[oc1 + 2], bias[oc1 + 3]);
        for (int tile = 0; tile < TilePixels; ++tile) {
            sum0[tile] = bias0;
            sum1[tile] = bias1;
        }

        for (uint32_t ic = 0; ic < in_c; ++ic) {
            const v128_t weight0 = use_packed
                ? loadPackedOc4(packed_weights, in_c, 1, 1, oc_block0, ic, 0, 0)
                : loadOiHwOc4(weights, in_c, 1, 1, oc0, ic, 0, 0);
            const v128_t weight1 = use_packed
                ? loadPackedOc4(packed_weights, in_c, 1, 1, oc_block1, ic, 0, 0)
                : loadOiHwOc4(weights, in_c, 1, 1, oc1, ic, 0, 0);
            for (int tile = 0; tile < TilePixels; ++tile) {
                const float scalar = loadC4Scalar(input, in_c, in_h, in_w, ic, iy, ox * layer.stride_x + static_cast<uint32_t>(tile) * layer.stride_x);
                const v128_t input_value = wasm_f32x4_splat(scalar);
                sum0[tile] = wasm_f32x4_add(sum0[tile], wasm_f32x4_mul(input_value, weight0));
                sum1[tile] = wasm_f32x4_add(sum1[tile], wasm_f32x4_mul(input_value, weight1));
            }
        }

        for (int tile = 0; tile < TilePixels; ++tile) {
            storeC4(output, out_c, out_h, out_w, oc_block0, oy, ox + static_cast<uint32_t>(tile), sum0[tile]);
            storeC4(output, out_c, out_h, out_w, oc_block1, oy, ox + static_cast<uint32_t>(tile), sum1[tile]);
        }
    });

    const uint32_t tail_x = tile_count_x * TilePixels;
    const std::size_t spatial = static_cast<std::size_t>(out_h) * out_w;
    runRange(0, static_cast<std::size_t>(oc4) * spatial, 128, use_threads, [&](std::size_t index) {
        const uint32_t oc_block = static_cast<uint32_t>(index / spatial);
        const uint32_t rem_skip = static_cast<uint32_t>(index % spatial);
        const uint32_t ox_skip = rem_skip % out_w;
        if (oc_block < oc8 * 2 && ox_skip < tail_x) {
            return;
        }
        const uint32_t rem = static_cast<uint32_t>(index % spatial);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;
        const uint32_t oc = oc_block * 4;
        const uint32_t iy = oy * layer.stride_y;

        v128_t sum = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
        for (uint32_t ic = 0; ic < in_c; ++ic) {
            const float scalar = loadC4Scalar(input, in_c, in_h, in_w, ic, iy, ox * layer.stride_x);
            const v128_t input_value = wasm_f32x4_splat(scalar);
            const v128_t weight = use_packed
                ? loadPackedOc4(packed_weights, in_c, 1, 1, oc_block, ic, 0, 0)
                : loadOiHwOc4(weights, in_c, 1, 1, oc, ic, 0, 0);
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(input_value, weight));
        }
        storeC4(output, out_c, out_h, out_w, oc_block, oy, ox, sum);
    });
}

void runConvC4GenericSimd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
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
    const bool use_packed = packed_weights != nullptr;

    runRange(0, static_cast<std::size_t>(oc4) * spatial, use_threads ? 256 : 16, use_threads, [&](std::size_t index) {
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
                    const float scalar = loadC4Scalar(input, in_c, in_h, in_w, ic, static_cast<uint32_t>(iy), static_cast<uint32_t>(ix));
                    const v128_t input_value = wasm_f32x4_splat(scalar);
                    const v128_t weight = use_packed
                        ? loadPackedOc4(packed_weights, in_c, layer.kernel_y, layer.kernel_x, oc_block, ic, ky, kx)
                        : loadOiHwOc4(weights, in_c, layer.kernel_y, layer.kernel_x, oc, ic, ky, kx);
                    sum = wasm_f32x4_add(sum, wasm_f32x4_mul(input_value, weight));
                }
            }
        }
        storeC4(output, out_c, out_h, out_w, oc_block, oy, ox, sum);
    });
}

template <int TilePixels>
void runConv3x3Stride1Pad1C4InteriorTiledSimd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
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
    const bool use_packed = packed_weights != nullptr;
    const uint32_t interior_w = out_w > 2 ? out_w - 2 : 0;
    const uint32_t tile_count_x = interior_w / TilePixels;
    const std::size_t row_tiles = out_h > 2 ? static_cast<std::size_t>(out_h - 2) * tile_count_x : 0;

    runRange(0, static_cast<std::size_t>(oc4) * row_tiles, use_threads ? 32 : 4, use_threads, [&](std::size_t index) {
        const uint32_t oc_block = static_cast<uint32_t>(index / row_tiles);
        const uint32_t tile_index = static_cast<uint32_t>(index % row_tiles);
        const uint32_t oy = 1 + tile_index / tile_count_x;
        const uint32_t ox = 1 + (tile_index % tile_count_x) * TilePixels;
        const uint32_t oc = oc_block * 4;

        v128_t sum[TilePixels];
        const v128_t bias_value = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
        for (int tile = 0; tile < TilePixels; ++tile) {
            sum[tile] = bias_value;
        }

        for (uint32_t ic = 0; ic < in_c; ++ic) {
            for (uint32_t ky = 0; ky < 3; ++ky) {
                const uint32_t iy = oy + ky - 1;
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    const v128_t weight_value = use_packed
                        ? loadPackedOc4(packed_weights, in_c, 3, 3, oc_block, ic, ky, kx)
                        : loadOiHwOc4(weights, in_c, 3, 3, oc, ic, ky, kx);
                    for (int tile = 0; tile < TilePixels; ++tile) {
                        const float scalar = loadC4Scalar(input, in_c, in_h, in_w, ic, iy, ox + kx - 1 + static_cast<uint32_t>(tile));
                        const v128_t input_value = wasm_f32x4_splat(scalar);
                        sum[tile] = wasm_f32x4_add(sum[tile], wasm_f32x4_mul(input_value, weight_value));
                    }
                }
            }
        }

        for (int tile = 0; tile < TilePixels; ++tile) {
            storeC4(output, out_c, out_h, out_w, oc_block, oy, ox + static_cast<uint32_t>(tile), sum[tile]);
        }
    });

    const std::size_t spatial = static_cast<std::size_t>(out_h) * out_w;
    runRange(0, static_cast<std::size_t>(oc4) * spatial, 128, use_threads, [&](std::size_t index) {
        const uint32_t rem = static_cast<uint32_t>(index % spatial);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;
        const bool is_interior_y = oy > 0 && oy + 1 < out_h;
        const bool is_tiled_x = ox > 0 && ox < 1 + tile_count_x * TilePixels;
        if (is_interior_y && is_tiled_x) {
            return;
        }

        const uint32_t oc_block = static_cast<uint32_t>(index / spatial);
        const uint32_t oc = oc_block * 4;
        v128_t sum = wasm_f32x4_make(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
        for (uint32_t ic = 0; ic < in_c; ++ic) {
            for (uint32_t ky = 0; ky < 3; ++ky) {
                const int iy = static_cast<int>(oy + ky) - 1;
                if (iy < 0 || iy >= static_cast<int>(in_h)) {
                    continue;
                }
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    const int ix = static_cast<int>(ox + kx) - 1;
                    if (ix < 0 || ix >= static_cast<int>(in_w)) {
                        continue;
                    }
                    const float scalar = loadC4Scalar(input, in_c, in_h, in_w, ic, static_cast<uint32_t>(iy), static_cast<uint32_t>(ix));
                    const v128_t input_value = wasm_f32x4_splat(scalar);
                    const v128_t weight_value = use_packed
                        ? loadPackedOc4(packed_weights, in_c, 3, 3, oc_block, ic, ky, kx)
                        : loadOiHwOc4(weights, in_c, 3, 3, oc, ic, ky, kx);
                    sum = wasm_f32x4_add(sum, wasm_f32x4_mul(input_value, weight_value));
                }
            }
        }
        storeC4(output, out_c, out_h, out_w, oc_block, oy, ox, sum);
    });
}

void runConvC4Simd(
    const LayerDesc& layer,
    const float* input,
    const float* weights,
    const float* packed_weights,
    const float* bias,
    float* output,
    bool use_threads,
    ConvTileMode tile_mode
) {
    if (layer.kernel_y == 1 && layer.kernel_x == 1 && layer.padding_y == 0 && layer.padding_x == 0 &&
        tile_mode != ConvTileMode::Oc4x1) {
        if (tile_mode == ConvTileMode::Oc4x8) {
            runConv1x1C4Oc8xTileSimd<8>(layer, input, weights, packed_weights, bias, output, use_threads);
            return;
        }
        runConv1x1C4Oc8xTileSimd<4>(layer, input, weights, packed_weights, bias, output, use_threads);
        return;
    }
    if (layer.kernel_y == 3 && layer.kernel_x == 3 && layer.stride_y == 1 && layer.stride_x == 1 &&
        layer.padding_y == 1 && layer.padding_x == 1) {
        if (tile_mode == ConvTileMode::Oc4x8) {
            runConv3x3Stride1Pad1C4InteriorTiledSimd<8>(layer, input, weights, packed_weights, bias, output, use_threads);
            return;
        }
        if (tile_mode == ConvTileMode::Oc4x4) {
            runConv3x3Stride1Pad1C4InteriorTiledSimd<4>(layer, input, weights, packed_weights, bias, output, use_threads);
            return;
        }
    }
    runConvC4GenericSimd(layer, input, weights, packed_weights, bias, output, use_threads);
}
#endif

} // namespace

bool supportsSpecializedConv2d(const LayerDesc& layer) {
    if (layer.input_shape.dims.size() != 3 || layer.output_shape.dims.size() != 3) {
        return false;
    }
    return isSupportedShape(layer) && layer.output_shape.dims[0] >= 4;
}

std::uint64_t estimateConvMacs(const LayerDesc& layer) {
    if (layer.input_shape.dims.size() != 3 || layer.output_shape.dims.size() != 3) {
        return 0;
    }
    return static_cast<std::uint64_t>(layer.output_shape.dims[0]) *
           layer.output_shape.dims[1] *
           layer.output_shape.dims[2] *
           layer.input_shape.dims[0] *
           layer.kernel_y *
           layer.kernel_x;
}

std::vector<float> packWeightsOc4(const LayerDesc& layer, const std::vector<float>& weights) {
    if (!supportsSpecializedConv2d(layer)) {
        return {};
    }

    const uint32_t in_c = layer.input_shape.dims[0];
    const uint32_t out_c = layer.output_shape.dims[0];
    const uint32_t oc4 = out_c / 4;
    const std::size_t packed_count = static_cast<std::size_t>(oc4) * in_c * layer.kernel_y * layer.kernel_x * 4;
    std::vector<float> packed(packed_count, 0.0f);

    const std::size_t oc_stride = static_cast<std::size_t>(in_c) * layer.kernel_y * layer.kernel_x;
    for (uint32_t oc_block = 0; oc_block < oc4; ++oc_block) {
        const uint32_t oc = oc_block * 4;
        for (uint32_t ic = 0; ic < in_c; ++ic) {
            for (uint32_t ky = 0; ky < layer.kernel_y; ++ky) {
                for (uint32_t kx = 0; kx < layer.kernel_x; ++kx) {
                    float* dst = packed.data() + packedOc4Index(in_c, layer.kernel_y, layer.kernel_x, oc_block, ic, ky, kx);
                    const std::size_t inner = static_cast<std::size_t>(ic) * layer.kernel_y * layer.kernel_x +
                                              static_cast<std::size_t>(ky) * layer.kernel_x +
                                              kx;
                    dst[0] = weights[static_cast<std::size_t>(oc) * oc_stride + inner];
                    dst[1] = weights[static_cast<std::size_t>(oc + 1) * oc_stride + inner];
                    dst[2] = weights[static_cast<std::size_t>(oc + 2) * oc_stride + inner];
                    dst[3] = weights[static_cast<std::size_t>(oc + 3) * oc_stride + inner];
                }
            }
        }
    }
    return packed;
}

std::vector<float> nchwToC4(const std::vector<float>& input, const TensorShape& shape) {
    if (shape.dims.size() != 3) {
        return input;
    }
    const uint32_t channels = shape.dims[0];
    const uint32_t height = shape.dims[1];
    const uint32_t width = shape.dims[2];
    const uint32_t c4 = (channels + 3) / 4;
    std::vector<float> output(static_cast<std::size_t>(height) * width * c4 * 4, 0.0f);
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    for (uint32_t c = 0; c < channels; ++c) {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                output[c4Index(channels, height, width, c, y, x)] =
                    input[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * width + x];
            }
        }
    }
    return output;
}

std::vector<float> c4ToNchw(const std::vector<float>& input, const TensorShape& shape) {
    if (shape.dims.size() != 3) {
        return input;
    }
    const uint32_t channels = shape.dims[0];
    const uint32_t height = shape.dims[1];
    const uint32_t width = shape.dims[2];
    std::vector<float> output(shape.elementCount(), 0.0f);
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    for (uint32_t c = 0; c < channels; ++c) {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                output[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * width + x] =
                    input[c4Index(channels, height, width, c, y, x)];
            }
        }
    }
    return output;
}

bool runSpecializedConv2d(
    const LayerDesc& layer,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>* packed_oc4_weights,
    const std::vector<float>& bias,
    bool use_simd,
    bool use_threads,
    ConvTileMode tile_mode,
    std::vector<float>& output
) {
    if (!use_simd || !supportsSpecializedConv2d(layer)) {
        return false;
    }

#if defined(__wasm_simd128__)
    output.assign(layer.output_shape.elementCount(), 0.0f);
    const float* packed = packed_oc4_weights && !packed_oc4_weights->empty() ? packed_oc4_weights->data() : nullptr;
    runConvOc4TiledSimd(layer, input.data(), weights.data(), packed, bias.data(), output.data(), use_threads, tile_mode);
    return true;
#else
    (void)input;
    (void)weights;
    (void)packed_oc4_weights;
    (void)bias;
    (void)use_threads;
    (void)tile_mode;
    (void)output;
    return false;
#endif
}

bool runSpecializedConv2dC4(
    const LayerDesc& layer,
    const std::vector<float>& input_c4,
    const std::vector<float>& weights,
    const std::vector<float>* packed_oc4_weights,
    const std::vector<float>& bias,
    bool use_simd,
    bool use_threads,
    ConvTileMode tile_mode,
    std::vector<float>& output_c4
) {
    if (!use_simd || !supportsSpecializedConv2d(layer)) {
        return false;
    }

#if defined(__wasm_simd128__)
    const uint32_t out_c = layer.output_shape.dims[0];
    const uint32_t out_h = layer.output_shape.dims[1];
    const uint32_t out_w = layer.output_shape.dims[2];
    const uint32_t out_c4 = (out_c + 3) / 4;
    output_c4.assign(static_cast<std::size_t>(out_h) * out_w * out_c4 * 4, 0.0f);
    const float* packed = packed_oc4_weights && !packed_oc4_weights->empty() ? packed_oc4_weights->data() : nullptr;
    runConvC4Simd(layer, input_c4.data(), weights.data(), packed, bias.data(), output_c4.data(), use_threads, tile_mode);
    return true;
#else
    (void)layer;
    (void)input_c4;
    (void)weights;
    (void)packed_oc4_weights;
    (void)bias;
    (void)use_threads;
    (void)tile_mode;
    (void)output_c4;
    return false;
#endif
}

const char* selectedKernelName(const LayerDesc& layer, bool use_simd, ConvTileMode tile_mode) {
    if (use_simd && supportsSpecializedConv2d(layer)) {
        if (layer.kernel_y == 1 && layer.kernel_x == 1 && layer.padding_y == 0 && layer.padding_x == 0) {
            if (tile_mode == ConvTileMode::Oc4x8) {
                return "conv_oc4_1x1_x8";
            }
            if (tile_mode == ConvTileMode::Oc4x4) {
                return "conv_oc4_1x1_x4";
            }
        }
        if (layer.kernel_y == 3 && layer.kernel_x == 3 && layer.stride_y == 1 && layer.stride_x == 1 &&
            layer.padding_y == 1 && layer.padding_x == 1) {
            if (tile_mode == ConvTileMode::Oc4x8) {
                return "conv_oc4_3x3_s1p1_x8";
            }
            if (tile_mode == ConvTileMode::Oc4x4) {
                return "conv_oc4_3x3_s1p1_x4";
            }
        }
        return "conv_oc4";
    }
    return "conv_generic";
}

const char* selectedKernelNameC4(const LayerDesc& layer, bool use_simd, ConvTileMode tile_mode) {
    if (use_simd && supportsSpecializedConv2d(layer)) {
        if (layer.kernel_y == 1 && layer.kernel_x == 1 && layer.padding_y == 0 && layer.padding_x == 0 &&
            tile_mode != ConvTileMode::Oc4x1) {
            if (tile_mode == ConvTileMode::Oc4x8) {
                return "conv_c4_1x1_oc8x8";
            }
            return "conv_c4_1x1_oc8x4";
        }
        if (layer.kernel_y == 3 && layer.kernel_x == 3 && layer.stride_y == 1 && layer.stride_x == 1 &&
            layer.padding_y == 1 && layer.padding_x == 1) {
            if (tile_mode == ConvTileMode::Oc4x8) {
                return "conv_c4_3x3_s1p1_x8";
            }
            if (tile_mode == ConvTileMode::Oc4x4) {
                return "conv_c4_3x3_s1p1_x4";
            }
        }
        return "conv_c4_oc4";
    }
    return "conv_generic";
}

} // namespace cpu_conv
} // namespace network
