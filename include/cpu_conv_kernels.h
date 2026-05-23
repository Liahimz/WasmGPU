#pragma once

#include "model_loader.h"

#include <cstdint>
#include <vector>

namespace network {
namespace cpu_conv {

struct PackedConvWeights {
    int layer_index = -1;
    std::vector<float> oc4_weights;
};

enum class ConvTileMode : int {
    Oc4x1 = 1,
    Oc4x4 = 4,
    Oc4x8 = 8,
};

bool supportsSpecializedConv2d(const LayerDesc& layer);
std::uint64_t estimateConvMacs(const LayerDesc& layer);
std::vector<float> packWeightsOc4(const LayerDesc& layer, const std::vector<float>& weights);
std::vector<float> nchwToC4(const std::vector<float>& input, const TensorShape& shape);
std::vector<float> c4ToNchw(const std::vector<float>& input, const TensorShape& shape);

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
);

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
);

const char* selectedKernelName(const LayerDesc& layer, bool use_simd, ConvTileMode tile_mode);
const char* selectedKernelNameC4(const LayerDesc& layer, bool use_simd, ConvTileMode tile_mode);

} // namespace cpu_conv
} // namespace network
