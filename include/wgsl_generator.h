#pragma once

#include "model_loader.h"

#include <cstdint>
#include <string>

namespace network {

struct GeneratedWgsl {
    std::string label;
    std::string source;
    uint32_t workgroup_x = 1;
    uint32_t workgroup_y = 1;
    uint32_t workgroup_z = 1;
    uint32_t dispatch_x = 1;
    uint32_t dispatch_y = 1;
    uint32_t dispatch_z = 1;
};

GeneratedWgsl generateLayerWgsl(const LayerDesc& layer);

} // namespace network
