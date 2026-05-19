#include "wasm_gpu_engine.h"
#include "image_proc.h"
#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

const char* cpuModeName(int mode) {
    switch (static_cast<CppExecutorMode>(mode)) {
        case CppExecutorMode::Scalar:
            return "scalar";
        case CppExecutorMode::Simd:
            return "simd";
        case CppExecutorMode::SimdThreads:
            return "simd_threads";
    }
    return "simd";
}

} // namespace

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
    cpu_.configure(&weights_);
    parallel::initialize();
}

ProcessResult WasmGpuEngine::process(const std::vector<uint8_t>& data, int width, int height, int channels) {
    const auto total_start = Clock::now();
    const auto preprocess_start = Clock::now();
    ProcessResult result = preprocess(data, width, height, channels);
    const auto preprocess_end = Clock::now();

    const auto inference_start = Clock::now();
    result.prediction = runNetwork(result.image);
    const auto inference_end = Clock::now();

#if defined(BUILD_WASM_WEBGPU_ASYNC)
    std::cout << "[timing] gpu_async_start preprocess=" << elapsedMs(preprocess_start, preprocess_end)
              << "ms submit=" << elapsedMs(inference_start, inference_end)
              << "ms total_to_start=" << elapsedMs(total_start, inference_end)
              << "ms"
              << std::endl;
#else
    std::cout << "[timing] gpu preprocess=" << elapsedMs(preprocess_start, preprocess_end)
              << "ms inference=" << elapsedMs(inference_start, inference_end)
              << "ms total=" << elapsedMs(total_start, inference_end)
              << "ms prediction=" << result.prediction
              << std::endl;
#endif
    return result;
}

ProcessResult WasmGpuEngine::processCpu(
    const std::vector<uint8_t>& data,
    int width,
    int height,
    int channels,
    int mode
) {
    const auto total_start = Clock::now();
    const auto preprocess_start = Clock::now();
    ProcessResult result = preprocess(data, width, height, channels);
    const auto preprocess_end = Clock::now();

    const auto inference_start = Clock::now();
    result.prediction = cpu_.infer(result.image, static_cast<CppExecutorMode>(mode));
    const auto inference_end = Clock::now();

    std::cout << "[timing] cpu mode=" << cpuModeName(mode)
              << " preprocess=" << elapsedMs(preprocess_start, preprocess_end)
              << "ms inference=" << elapsedMs(inference_start, inference_end)
              << "ms total=" << elapsedMs(total_start, inference_end)
              << "ms prediction=" << result.prediction
              << std::endl;
    return result;
}

int WasmGpuEngine::benchmarkCpuLarge(int mode, int input_seed) {
    cpu_.prepareSyntheticLarge();

    const auto inference_start = Clock::now();
    const int prediction = cpu_.inferSyntheticLarge(static_cast<CppExecutorMode>(mode), static_cast<uint32_t>(input_seed));
    const auto inference_end = Clock::now();

    std::cout << "[timing] synthetic_cpu_large mode=" << cpuModeName(mode)
              << " input=1000x500 kernel=5x3"
              << " seed=" << input_seed
              << " inference=" << elapsedMs(inference_start, inference_end)
              << "ms prediction=" << prediction
              << std::endl;
    return prediction;
}

void WasmGpuEngine::prepareSyntheticLargeData() {
    const auto start = Clock::now();
    cpu_.prepareSyntheticLarge();
    gpu_.prepareSyntheticLargeData();
    const auto end = Clock::now();

    std::cout << "[timing] synthetic_large_data_prepare"
              << " elapsed=" << elapsedMs(start, end)
              << "ms"
              << std::endl;
}

int WasmGpuEngine::benchmarkGpuLarge(int input_seed) {
    const auto prepare_start = Clock::now();
    gpu_.prepareSyntheticLarge();
    const auto prepare_end = Clock::now();

    const auto inference_start = Clock::now();
    const int prediction = gpu_.benchmarkSyntheticLarge(static_cast<uint32_t>(input_seed));
    const auto inference_end = Clock::now();

#if defined(BUILD_WASM_WEBGPU_ASYNC)
    std::cout << "[timing] synthetic_gpu_large_async_start"
              << " input=1000x500 kernel=5x3"
              << " seed=" << input_seed
              << " prepare=" << elapsedMs(prepare_start, prepare_end)
              << "ms"
              << " submit=" << elapsedMs(inference_start, inference_end)
              << "ms"
              << std::endl;
#else
    std::cout << "[timing] synthetic_gpu_large"
              << " input=1000x500 kernel=5x3"
              << " seed=" << input_seed
              << " prepare=" << elapsedMs(prepare_start, prepare_end)
              << "ms"
              << " inference=" << elapsedMs(inference_start, inference_end)
              << "ms prediction=" << prediction
              << std::endl;
#endif
    return prediction;
}

ProcessResult WasmGpuEngine::preprocess(
    const std::vector<uint8_t>& data,
    int width,
    int height,
    int channels
) const {
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
