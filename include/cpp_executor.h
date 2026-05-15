#pragma once

#include "network_weights.h"

#include <cstdint>
#include <vector>

enum class CppExecutorMode : int {
    Scalar = 0,
    Simd = 1,
    SimdThreads = 2,
};

class CppExecutor {
public:
    CppExecutor() = default;

    void configure(const network::TinyLenetWeights* weights);
    bool ready() const;
    int infer(const std::vector<uint8_t>& image, CppExecutorMode mode) const;

private:
    const network::TinyLenetWeights* weights_ = nullptr;
};
