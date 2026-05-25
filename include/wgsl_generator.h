#pragma once

#include "model_loader.h"

#include <cstdint>
#include <string>

namespace network {

enum class Conv2dShapeClass {
    NotConv2D,
    Stem7x7Stride2,
    Bottleneck1x1Projection,
    Downsample1x1Stride2,
    Conv3x3Stride1Pad1,
    Conv3x3Stride2Pad1,
    GenericFallback,
};

struct GeneratedWgsl {
    std::string label;
    std::string source;
    std::string kernel_variant;
    Conv2dShapeClass conv_class = Conv2dShapeClass::NotConv2D;
    uint64_t estimated_macs = 0;
    uint32_t workgroup_x = 1;
    uint32_t workgroup_y = 1;
    uint32_t workgroup_z = 1;
    uint32_t dispatch_x = 1;
    uint32_t dispatch_y = 1;
    uint32_t dispatch_z = 1;
};

Conv2dShapeClass classifyConv2dShape(const LayerDesc& layer);
const char* conv2dShapeClassName(Conv2dShapeClass shape_class);
uint64_t estimateConv2dMacs(const LayerDesc& layer);
GeneratedWgsl generateLayerWgsl(const LayerDesc& layer);

} // namespace network
