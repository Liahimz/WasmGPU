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

bool supportsSpecializedConv2d(const LayerDesc& layer);
std::vector<float> packWeightsOc4(const LayerDesc& layer, const std::vector<float>& weights);

bool runSpecializedConv2d(
    const LayerDesc& layer,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>* packed_oc4_weights,
    const std::vector<float>& bias,
    bool use_simd,
    bool use_threads,
    std::vector<float>& output
);

} // namespace cpu_conv
} // namespace network
