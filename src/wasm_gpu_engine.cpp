#include "wasm_gpu_engine.h"
#include "image_proc.h"

#include <algorithm>
#include <iostream>
#include <limits>

WasmGpuEngine::WasmGpuEngine() {}
WasmGpuEngine::~WasmGpuEngine() {}

void WasmGpuEngine::configure(int target_size_) {
    std::cout << "Configure" << std::endl;
    target_size = target_size_;

    weights_ = network::loadTinyLenetWeights();
    if (weights_.valid()) {
        std::cout << "Loaded embedded tiny_lenet weights: "
                  << weights_.conv_weights.size() << " conv weights, "
                  << weights_.linear_weights.size() << " linear weights"
                  << std::endl;
    } else {
        std::cerr << "Failed to load embedded tiny_lenet weights: " << weights_.error << std::endl;
    }

    gpu_.configure(&weights_);
}

ProcessResult WasmGpuEngine::process(const std::vector<uint8_t>& data, int width, int height, int channels) {
    int size = width * height;
    std::vector<uint8_t> gray(size);
    to_grayscale(data.data(), gray.data(), width, height, channels);

    int new_w = target_size;
    int new_h = target_size;
    std::vector<uint8_t> gray_scaled;

    if (new_w > width) {
        new_w = width;
        new_h = height;
        gray_scaled.resize(width * height);
        gray_scaled = std::move(gray);
    } else {
        gray_scaled.resize(new_w * new_h);
        rescale(gray.data(), gray_scaled.data(), width, height, new_w, new_h);
    }

    ProcessResult result;
    result.image = std::move(gray_scaled);
    result.width = new_w;
    result.height = new_h;
    result.prediction = runNetwork(result.image);
    return result;
}

int WasmGpuEngine::argmax(const std::vector<float>& data) {
    int result = -1;
    float max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] > max) {
            result = i;
            max = data[i];
        }
    }
    return result;
}

bool WasmGpuEngine::webgpuReady() const {
    return gpu_.ready();
}

bool WasmGpuEngine::inferencePending() const {
    return gpu_.inferencePending();
}

int WasmGpuEngine::latestPrediction() const {
    return gpu_.latestPrediction();
}

int WasmGpuEngine::runNetwork(const std::vector<uint8_t>& image) {
    if (!weights_.valid()) {
        return -1;
    }

    return gpu_.infer(image);
}
