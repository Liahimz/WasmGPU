#pragma once

#include "cpu_graph_executor.h"
#include "model_loader.h"
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
    void configure(const network::ModelDesc* model, const network::TinyLenetWeights* weights);
    bool ready() const;
    int infer(const std::vector<uint8_t>& image, CppExecutorMode mode) const;
    std::vector<float> infer(const std::vector<float>& input, CppExecutorMode mode) const;
    void prepareSyntheticLarge() const;
    int inferSyntheticLarge(CppExecutorMode mode, uint32_t input_seed) const;

private:
    const network::ModelDesc* model_ = nullptr;
    const network::TinyLenetWeights* weights_ = nullptr;
    network::CpuGraphExecutor graph_;
};
