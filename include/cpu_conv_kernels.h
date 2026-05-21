#pragma once

#include "model_loader.h"

#include <cstdint>
#include <vector>

namespace network {
namespace cpu_conv {

bool runSpecializedConv2d(
    const LayerDesc& layer,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    bool use_simd,
    bool use_threads,
    std::vector<float>& output
);

} // namespace cpu_conv
} // namespace network
