#include "wasm_gpu_engine.h"
#include "image_proc.h"
#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

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

network::cpu_conv::ConvTileMode convTileMode(int tile_mode) {
    if (tile_mode == 8) {
        return network::cpu_conv::ConvTileMode::Oc4x8;
    }
    if (tile_mode == 1) {
        return network::cpu_conv::ConvTileMode::Oc4x1;
    }
    return network::cpu_conv::ConvTileMode::Oc4x4;
}

std::vector<std::string> parseStringArray(const std::string& text) {
    std::vector<std::string> result;
    std::string current;
    bool in_string = false;
    bool escape = false;

    for (char ch : text) {
        if (!in_string) {
            if (ch == '"') {
                in_string = true;
                current.clear();
            }
            continue;
        }

        if (escape) {
            current.push_back(ch);
            escape = false;
            continue;
        }
        if (ch == '\\') {
            escape = true;
            continue;
        }
        if (ch == '"') {
            result.push_back(current);
            in_string = false;
            continue;
        }
        current.push_back(ch);
    }
    return result;
}

} // namespace

WasmGpuEngine::WasmGpuEngine() {}
WasmGpuEngine::~WasmGpuEngine() {}

void WasmGpuEngine::configure(int target_size_) {
    std::cout << "Configure" << std::endl;
    target_size = target_size_;

#if defined(BUILD_RESNET50_MODE)
    weights_ = {};
    model_ = network::loadModelFromEmbedded("resnet50/resnet50_full_manifest.json");
    labels_ = parseStringArray(network::loadEmbeddedText("resnet50/imagenet_classes.json"));
#else
    weights_ = network::loadTinyLenetWeights();
    model_ = network::loadModelFromEmbedded("lenet/tiny_lenet_manifest.json");
#endif
    if (model_.valid()) {
        std::cout << "Loaded embedded model manifest: " << model_.name
                  << " input=" << model_.input_shape.toString()
                  << " layers=" << model_.layers.size()
                  << std::endl;
    } else {
        std::cerr << "Failed to load embedded model manifest: " << model_.error << std::endl;
    }

#if !defined(BUILD_RESNET50_MODE)
    if (weights_.valid()) {
        std::cout << "Loaded embedded tiny_lenet weights: "
                  << weights_.conv_weights.size() << " conv weights, "
                  << weights_.linear_weights.size() << " linear weights"
                  << std::endl;
    } else {
        std::cerr << "Failed to load embedded tiny_lenet weights: " << weights_.error << std::endl;
    }
#endif

    gpu_.configure(model_.valid() ? &model_ : nullptr, &weights_);
    cpu_.configure(model_.valid() ? &model_ : nullptr, &weights_);
    parallel::initialize();
}

ProcessResult WasmGpuEngine::processResnet(const std::vector<uint8_t>& data, int width, int height, int channels) {
    const auto total_start = Clock::now();
    const auto preprocess_start = Clock::now();
    std::vector<float> input = preprocessResnetInput(data, width, height, channels);
    std::vector<uint8_t> preview = preprocess_imagenet_rgb_preview(data.data(), width, height, channels, 256, 224);
    const auto preprocess_end = Clock::now();

    ProcessResult result;
    result.width = 224;
    result.height = 224;
    result.image = std::move(preview);
    const auto inference_start = Clock::now();
    result.prediction = runNetwork(input);
    result.gpu_backend = gpu_.latestBackend();
    result.class_label = classLabel(result.prediction);
    const auto inference_end = Clock::now();
#if !defined(BUILD_WASM_WEBGPU_ASYNC)
    result.top_k = topKText(gpu_.latestOutput(), 5);
#endif
    std::cout << "[timing] resnet_gpu preprocess=" << elapsedMs(preprocess_start, preprocess_end)
              << "ms submit_or_inference=" << elapsedMs(inference_start, inference_end)
              << "ms total_to_start=" << elapsedMs(total_start, inference_end)
              << "ms prediction=" << result.prediction
              << std::endl;
    return result;
}

ProcessResult WasmGpuEngine::processResnetCpu(
    const std::vector<uint8_t>& data,
    int width,
    int height,
    int channels,
    int mode
) {
    return processResnetCpuTiled(data, width, height, channels, mode, 4);
}

ProcessResult WasmGpuEngine::processResnetCpuTiled(
    const std::vector<uint8_t>& data,
    int width,
    int height,
    int channels,
    int mode,
    int tile_mode
) {
    return processResnetCpuProfiled(data, width, height, channels, mode, tile_mode, false);
}

ProcessResult WasmGpuEngine::processResnetCpuProfiled(
    const std::vector<uint8_t>& data,
    int width,
    int height,
    int channels,
    int mode,
    int tile_mode,
    bool log_layers
) {
    const auto total_start = Clock::now();
    const auto preprocess_start = Clock::now();
    std::vector<float> input = preprocessResnetInput(data, width, height, channels);
    const auto preprocess_end = Clock::now();

    const auto inference_start = Clock::now();
    std::vector<float> logits = cpu_.infer(
        input,
        static_cast<CppExecutorMode>(mode),
        convTileMode(tile_mode),
        log_layers
    );
    const auto inference_end = Clock::now();

    ProcessResult result;
    result.width = 224;
    result.height = 224;
    result.prediction = argmax(logits);
    result.class_label = classLabel(result.prediction);
    result.top_k = topKText(logits, 5);
    std::cout << "[timing] resnet_cpu mode=" << cpuModeName(mode)
              << " conv_tile=oc4x" << (tile_mode == 8 ? 8 : (tile_mode == 1 ? 1 : 4))
              << " preprocess=" << elapsedMs(preprocess_start, preprocess_end)
              << "ms inference=" << elapsedMs(inference_start, inference_end)
              << "ms total=" << elapsedMs(total_start, inference_end)
              << "ms prediction=" << result.prediction
              << std::endl;
    return result;
}

ProcessResult WasmGpuEngine::process(const std::vector<uint8_t>& data, int width, int height, int channels) {
    const auto total_start = Clock::now();
    const auto preprocess_start = Clock::now();
    ProcessResult result = preprocess(data, width, height, channels);
    const auto preprocess_end = Clock::now();

    const auto inference_start = Clock::now();
    result.prediction = runNetwork(result.image);
    result.gpu_backend = gpu_.latestBackend();
    result.class_label = classLabel(result.prediction);
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
    result.class_label = classLabel(result.prediction);
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

std::vector<float> WasmGpuEngine::preprocessResnetInput(
    const std::vector<uint8_t>& data,
    int width,
    int height,
    int channels
) const {
    return preprocess_imagenet_rgb_chw(data.data(), width, height, channels, 256, 224);
}

std::string WasmGpuEngine::topKText(const std::vector<float>& logits, int count) const {
    if (logits.empty() || count <= 0) {
        return "";
    }
    std::vector<int> indices(logits.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    const int kept = std::min<int>(count, static_cast<int>(indices.size()));
    std::partial_sort(indices.begin(), indices.begin() + kept, indices.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });

    std::ostringstream out;
    for (int i = 0; i < kept; ++i) {
        const int index = indices[i];
        if (i != 0) {
            out << "\n";
        }
        out << (i + 1) << ". " << index;
        if (index >= 0 && static_cast<std::size_t>(index) < labels_.size()) {
            out << " " << labels_[index];
        }
        out << " score=" << logits[index];
    }
    return out.str();
}

std::string WasmGpuEngine::classLabel(int index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= labels_.size()) {
        return "";
    }
    return labels_[index];
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

std::string WasmGpuEngine::latestClassLabel() const {
    return classLabel(gpu_.latestPrediction());
}

std::string WasmGpuEngine::latestTopK(int count) const {
    return topKText(gpu_.latestOutput(), count);
}

int WasmGpuEngine::runNetwork(const std::vector<uint8_t>& image) {
    if (!model_.valid() && !weights_.valid()) {
        return -1;
    }

    return gpu_.infer(image);
}

int WasmGpuEngine::runNetwork(const std::vector<float>& input) {
    if (!model_.valid()) {
        return -1;
    }

    return gpu_.infer(input);
}
