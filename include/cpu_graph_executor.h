#pragma once

#include "cpu_conv_kernels.h"
#include "model_loader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace network {

struct CpuGraphOptions {
    bool use_simd = false;
    bool use_threads = false;
    cpu_conv::ConvTileMode conv_tile_mode = cpu_conv::ConvTileMode::Oc4x4;
};

class CpuGraphExecutor {
public:
    bool configure(const ModelDesc& model);
    bool ready() const;

    std::vector<float> infer(const std::vector<float>& input, CpuGraphOptions options = {}) const;
    std::vector<float> inferBytes(const std::vector<uint8_t>& input, CpuGraphOptions options = {}) const;
    int inferClass(const std::vector<float>& input, CpuGraphOptions options = {}) const;
    int inferClassBytes(const std::vector<uint8_t>& input, CpuGraphOptions options = {}) const;

    const std::string& error() const;

private:
    const ModelDesc* model_ = nullptr;
    std::string error_;
    std::vector<cpu_conv::PackedConvWeights> packed_conv_weights_;
};

int argmax(const std::vector<float>& values);

} // namespace network
