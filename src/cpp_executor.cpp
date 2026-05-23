#include "cpp_executor.h"
#include "cpu_simd_math.h"
#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace {

constexpr int InputWidth = 28;
constexpr int InputHeight = 28;
constexpr int ConvKernel = 3;
constexpr int ConvChannels = 4;
constexpr int ConvWidth = InputWidth - ConvKernel + 1;
constexpr int ConvHeight = InputHeight - ConvKernel + 1;
constexpr int InputValues = InputWidth * InputHeight;
constexpr int ConvPlaneValues = ConvWidth * ConvHeight;
constexpr int ConvValues = ConvPlaneValues * ConvChannels;
constexpr int LogitValues = 10;

constexpr int LargeInputWidth = 1000;
constexpr int LargeInputHeight = 500;
constexpr int LargeKernelWidth = 5;
constexpr int LargeKernelHeight = 3;
constexpr int LargeChannels = 4;
constexpr int LargeConvWidth = LargeInputWidth - LargeKernelWidth + 1;
constexpr int LargeConvHeight = LargeInputHeight - LargeKernelHeight + 1;
constexpr int LargeInputValues = LargeInputWidth * LargeInputHeight;
constexpr int LargeConvPlaneValues = LargeConvWidth * LargeConvHeight;
constexpr int LargeConvValues = LargeConvPlaneValues * LargeChannels;

void computeConvChannel(
    const float* input,
    const network::TinyLenetWeights& weights,
    float* conv_output,
    int channel
) {
    const int weight_offset = channel * ConvKernel * ConvKernel;
    const int output_offset = channel * ConvPlaneValues;

    for (int y = 0; y < ConvHeight; ++y) {
        for (int x = 0; x < ConvWidth; ++x) {
            float sum = weights.conv_bias[channel];
            for (int ky = 0; ky < ConvKernel; ++ky) {
                for (int kx = 0; kx < ConvKernel; ++kx) {
                    const int input_index = (y + ky) * InputWidth + (x + kx);
                    const int weight_index = weight_offset + ky * ConvKernel + kx;
                    sum += input[input_index] * weights.conv_weights[weight_index];
                }
            }
            conv_output[output_offset + y * ConvWidth + x] = std::max(sum, 0.0f);
        }
    }
}

void computeConvScalar(
    const float* input,
    const network::TinyLenetWeights& weights,
    float* conv_output
) {
    for (int channel = 0; channel < ConvChannels; ++channel) {
        computeConvChannel(input, weights, conv_output, channel);
    }
}

void computeConvThreaded(
    const float* input,
    const network::TinyLenetWeights& weights,
    float* conv_output
) {
    parallel::parallelFor(0, ConvChannels, [&](std::size_t channel) {
        computeConvChannel(input, weights, conv_output, static_cast<int>(channel));
    });
}

std::array<float, LogitValues> computeLinear(
    const float* conv_output,
    const network::TinyLenetWeights& weights,
    bool use_simd
) {
    std::array<float, LogitValues> logits{};
    for (int cls = 0; cls < LogitValues; ++cls) {
        const float* linear_weights = weights.linear_weights.data() + cls * ConvValues;
        const float dot = use_simd
            ? ::network::simd_math::dotProductSimd(conv_output, linear_weights, ConvValues)
            : ::network::simd_math::dotProductScalar(conv_output, linear_weights, ConvValues);
        logits[cls] = weights.linear_bias[cls] + dot;
    }
    return logits;
}

int argmax(const std::array<float, LogitValues>& logits) {
    int result = -1;
    float max_value = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > max_value) {
            result = static_cast<int>(i);
            max_value = logits[i];
        }
    }
    return result;
}

float nextWeight(uint32_t& state, float scale) {
    state = state * 1664525u + 1013904223u;
    const float normalized = static_cast<float>((state >> 8) & 0x00ffffffu) / 16777215.0f;
    return (normalized * 2.0f - 1.0f) * scale;
}

struct LargeSyntheticNetwork {
    std::vector<float> conv_weights;
    std::vector<float> conv_bias;
    std::vector<float> linear_weights;
    std::vector<float> linear_bias;

    LargeSyntheticNetwork()
        : conv_weights(LargeChannels * LargeKernelWidth * LargeKernelHeight),
          conv_bias(LargeChannels),
          linear_weights(LogitValues * LargeConvValues),
          linear_bias(LogitValues) {
        uint32_t state = 0x12345678u;
        for (float& value : conv_weights) {
            value = nextWeight(state, 0.05f);
        }
        for (float& value : conv_bias) {
            value = nextWeight(state, 0.01f);
        }
        for (float& value : linear_weights) {
            value = nextWeight(state, 0.005f);
        }
        for (float& value : linear_bias) {
            value = nextWeight(state, 0.01f);
        }
    }
};

const LargeSyntheticNetwork& largeSyntheticNetwork() {
    static const LargeSyntheticNetwork network;
    return network;
}

std::vector<float> makeLargeInput(uint32_t seed) {
    std::vector<float> input(LargeInputValues);
    uint32_t state = 0x87654321u ^ seed;
    for (float& value : input) {
        value = nextWeight(state, 1.0f);
    }
    return input;
}

void computeLargeConvChannel(
    const float* input,
    const LargeSyntheticNetwork& network,
    float* conv_output,
    int channel
) {
    const int weight_offset = channel * LargeKernelWidth * LargeKernelHeight;
    const int output_offset = channel * LargeConvPlaneValues;

    for (int y = 0; y < LargeConvHeight; ++y) {
        for (int x = 0; x < LargeConvWidth; ++x) {
            float sum = network.conv_bias[channel];
            for (int ky = 0; ky < LargeKernelHeight; ++ky) {
                for (int kx = 0; kx < LargeKernelWidth; ++kx) {
                    const int input_index = (y + ky) * LargeInputWidth + (x + kx);
                    const int weight_index = weight_offset + ky * LargeKernelWidth + kx;
                    sum += input[input_index] * network.conv_weights[weight_index];
                }
            }
            conv_output[output_offset + y * LargeConvWidth + x] = std::max(sum, 0.0f);
        }
    }
}

void computeLargeConvScalar(const float* input, const LargeSyntheticNetwork& network, float* conv_output) {
    for (int channel = 0; channel < LargeChannels; ++channel) {
        computeLargeConvChannel(input, network, conv_output, channel);
    }
}

void computeLargeConvThreaded(const float* input, const LargeSyntheticNetwork& network, float* conv_output) {
    parallel::parallelFor(0, LargeChannels, [&](std::size_t channel) {
        computeLargeConvChannel(input, network, conv_output, static_cast<int>(channel));
    });
}

std::array<float, LogitValues> computeLargeLinear(
    const LargeSyntheticNetwork& network,
    const float* conv_output,
    bool use_simd
) {
    std::array<float, LogitValues> logits{};
    for (int cls = 0; cls < LogitValues; ++cls) {
        const float* linear_weights = network.linear_weights.data() + cls * LargeConvValues;
        const float dot = use_simd
            ? ::network::simd_math::dotProductSimd(conv_output, linear_weights, LargeConvValues)
            : ::network::simd_math::dotProductScalar(conv_output, linear_weights, LargeConvValues);
        logits[cls] = network.linear_bias[cls] + dot;
    }
    return logits;
}

CppExecutorMode normalizeMode(CppExecutorMode mode) {
    if (mode == CppExecutorMode::Scalar ||
        mode == CppExecutorMode::Simd ||
        mode == CppExecutorMode::SimdThreads) {
        return mode;
    }
    return CppExecutorMode::Simd;
}

} // namespace

void CppExecutor::configure(const network::TinyLenetWeights* weights) {
    configure(nullptr, weights);
}

void CppExecutor::configure(const network::ModelDesc* model, const network::TinyLenetWeights* weights) {
    model_ = model;
    weights_ = weights;
    if (model_) {
        graph_.configure(*model_);
    }
}

bool CppExecutor::ready() const {
    return graph_.ready() || (weights_ && weights_->valid());
}

int CppExecutor::infer(const std::vector<uint8_t>& image, CppExecutorMode mode) const {
    return infer(image, mode, network::cpu_conv::ConvTileMode::Oc4x4);
}

int CppExecutor::infer(
    const std::vector<uint8_t>& image,
    CppExecutorMode mode,
    network::cpu_conv::ConvTileMode tile_mode
) const {
    return infer(image, mode, tile_mode, false);
}

int CppExecutor::infer(
    const std::vector<uint8_t>& image,
    CppExecutorMode mode,
    network::cpu_conv::ConvTileMode tile_mode,
    bool log_layers
) const {
    if (!ready()) {
        return -1;
    }

    mode = normalizeMode(mode);

    if (graph_.ready()) {
        network::CpuGraphOptions options;
        options.use_simd = mode == CppExecutorMode::Simd || mode == CppExecutorMode::SimdThreads;
        options.use_threads = mode == CppExecutorMode::SimdThreads;
        options.conv_tile_mode = tile_mode;
        options.log_layers = log_layers;
        return graph_.inferClassBytes(image, options);
    }

    if (!weights_ || !weights_->valid() || image.size() != InputValues) {
        return -1;
    }

    std::array<float, InputValues> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(image[i]) / 255.0f;
    }

    std::array<float, ConvValues> conv_output{};
    if (mode == CppExecutorMode::SimdThreads) {
        computeConvThreaded(input.data(), *weights_, conv_output.data());
    } else {
        computeConvScalar(input.data(), *weights_, conv_output.data());
    }

    const bool use_simd = mode == CppExecutorMode::Simd || mode == CppExecutorMode::SimdThreads;
    const std::array<float, LogitValues> logits = computeLinear(conv_output.data(), *weights_, use_simd);
    return argmax(logits);
}

std::vector<float> CppExecutor::infer(const std::vector<float>& input, CppExecutorMode mode) const {
    return infer(input, mode, network::cpu_conv::ConvTileMode::Oc4x4);
}

std::vector<float> CppExecutor::infer(
    const std::vector<float>& input,
    CppExecutorMode mode,
    network::cpu_conv::ConvTileMode tile_mode
) const {
    return infer(input, mode, tile_mode, false);
}

std::vector<float> CppExecutor::infer(
    const std::vector<float>& input,
    CppExecutorMode mode,
    network::cpu_conv::ConvTileMode tile_mode,
    bool log_layers
) const {
    if (!graph_.ready()) {
        return {};
    }

    mode = normalizeMode(mode);
    network::CpuGraphOptions options;
    options.use_simd = mode == CppExecutorMode::Simd || mode == CppExecutorMode::SimdThreads;
    options.use_threads = mode == CppExecutorMode::SimdThreads;
    options.conv_tile_mode = tile_mode;
    options.log_layers = log_layers;
    return graph_.infer(input, options);
}

void CppExecutor::prepareSyntheticLarge() const {
    (void)largeSyntheticNetwork();
}

int CppExecutor::inferSyntheticLarge(CppExecutorMode mode, uint32_t input_seed) const {
    mode = normalizeMode(mode);

    const LargeSyntheticNetwork& network = largeSyntheticNetwork();
    std::vector<float> input = makeLargeInput(input_seed);
    std::vector<float> conv_output(LargeConvValues);

    if (mode == CppExecutorMode::SimdThreads) {
        computeLargeConvThreaded(input.data(), network, conv_output.data());
    } else {
        computeLargeConvScalar(input.data(), network, conv_output.data());
    }

    const bool use_simd = mode == CppExecutorMode::Simd || mode == CppExecutorMode::SimdThreads;
    const std::array<float, LogitValues> logits = computeLargeLinear(network, conv_output.data(), use_simd);
    return argmax(logits);
}
