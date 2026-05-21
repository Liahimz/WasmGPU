#pragma once

#include "model_loader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace network {

struct CpuGraphOptions {
    bool use_simd = false;
    bool use_threads = false;
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
};

int argmax(const std::vector<float>& values);

} // namespace network
