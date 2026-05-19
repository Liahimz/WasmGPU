#pragma once

#include "model_loader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace network {

class CpuGraphExecutor {
public:
    bool configure(const ModelDesc& model);
    bool ready() const;

    std::vector<float> infer(const std::vector<float>& input) const;
    std::vector<float> inferBytes(const std::vector<uint8_t>& input) const;
    int inferClass(const std::vector<float>& input) const;
    int inferClassBytes(const std::vector<uint8_t>& input) const;

    const std::string& error() const;

private:
    const ModelDesc* model_ = nullptr;
    std::string error_;
};

int argmax(const std::vector<float>& values);

} // namespace network
