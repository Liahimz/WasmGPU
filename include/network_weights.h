#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace network {

struct TinyLenetWeights {
    std::vector<float> conv_weights;
    std::vector<float> conv_bias;
    std::vector<float> linear_weights;
    std::vector<float> linear_bias;
    std::string error;

    bool valid() const;
};

TinyLenetWeights loadTinyLenetWeights();

} // namespace network
