// wasm_gpu_engine.h
#pragma once
#include "gpu_executor.h"
#include "network_weights.h"

#include <vector>
#include <cstdint>

struct ProcessResult {
    std::vector<uint8_t> image;
    int width;
    int height;
    int prediction = -1;
};

class WasmGpuEngine {
public:
    WasmGpuEngine();
    ~WasmGpuEngine();

    void configure(int target_size);

    // Accepts grayscale image (flat vector), width, height
    ProcessResult process(const std::vector<uint8_t>& data, int width, int height, int channels);

    int argmax(const std::vector<float>& data);
    bool webgpuReady() const;
    bool inferencePending() const;
    int latestPrediction() const;

private:
  int target_size = 0;
  network::TinyLenetWeights weights_;
  GpuExecutor gpu_;

  int runNetwork(const std::vector<uint8_t>& image);
};
