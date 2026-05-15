#include "cpp_executor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <pthread.h>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

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

float dotProductScalar(const float* left, const float* right, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += left[i] * right[i];
    }
    return sum;
}

float dotProductSimd(const float* left, const float* right, int count) {
#if defined(__wasm_simd128__)
    v128_t acc = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 4 <= count; i += 4) {
        const v128_t a = wasm_v128_load(left + i);
        const v128_t b = wasm_v128_load(right + i);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(a, b));
    }

    alignas(16) float lanes[4];
    wasm_v128_store(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (; i < count; ++i) {
        sum += left[i] * right[i];
    }
    return sum;
#else
    return dotProductScalar(left, right, count);
#endif
}

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

struct ConvThreadArgs {
    const float* input = nullptr;
    const network::TinyLenetWeights* weights = nullptr;
    float* conv_output = nullptr;
    int channel = 0;
};

void* convThreadMain(void* user_data) {
    auto* args = static_cast<ConvThreadArgs*>(user_data);
    computeConvChannel(args->input, *args->weights, args->conv_output, args->channel);
    return nullptr;
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
    std::array<pthread_t, ConvChannels> threads{};
    std::array<ConvThreadArgs, ConvChannels> args{};
    std::array<bool, ConvChannels> started{};

    for (int channel = 0; channel < ConvChannels; ++channel) {
        args[channel].input = input;
        args[channel].weights = &weights;
        args[channel].conv_output = conv_output;
        args[channel].channel = channel;

        started[channel] = pthread_create(&threads[channel], nullptr, convThreadMain, &args[channel]) == 0;
        if (!started[channel]) {
            computeConvChannel(input, weights, conv_output, channel);
        }
    }

    for (int channel = 0; channel < ConvChannels; ++channel) {
        if (started[channel]) {
            pthread_join(threads[channel], nullptr);
        }
    }
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
            ? dotProductSimd(conv_output, linear_weights, ConvValues)
            : dotProductScalar(conv_output, linear_weights, ConvValues);
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
    weights_ = weights;
}

bool CppExecutor::ready() const {
    return weights_ && weights_->valid();
}

int CppExecutor::infer(const std::vector<uint8_t>& image, CppExecutorMode mode) const {
    if (!ready() || image.size() != InputValues) {
        return -1;
    }

    mode = normalizeMode(mode);

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
