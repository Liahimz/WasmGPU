#include "cpu_graph_executor.h"

#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace network {
namespace {

const std::vector<float>* requireWeight(const ModelDesc& model, int index, std::string& error, const char* role) {
    const ModelWeight* weight = model.weight(index);
    if (!weight) {
        error = std::string("Missing ") + role + " weight";
        return nullptr;
    }
    return &weight->values;
}

template <typename Body>
void runRange(std::size_t begin, std::size_t end, std::size_t grain_size, bool use_threads, const Body& body) {
    if (use_threads) {
        parallel::parallelFor(begin, end, grain_size, [&](std::size_t index) {
            body(index);
        });
        return;
    }

    for (std::size_t index = begin; index < end; ++index) {
        body(index);
    }
}

float dotProductScalar(const float* left, const float* right, uint32_t count) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        sum += left[i] * right[i];
    }
    return sum;
}

float dotProductSimd(const float* left, const float* right, uint32_t count) {
#if defined(__wasm_simd128__)
    v128_t acc = wasm_f32x4_splat(0.0f);
    uint32_t i = 0;
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

std::vector<float> runConv2d(
    const ModelDesc& model,
    const LayerDesc& layer,
    const std::vector<float>& input,
    const CpuGraphOptions& options,
    std::string& error
) {
    const std::vector<float>* weights = requireWeight(model, layer.weights_index, error, "conv2d");
    if (!weights) {
        return {};
    }
    const std::vector<float>* bias = requireWeight(model, layer.bias_index, error, "conv2d bias");
    if (!bias) {
        return {};
    }

    const uint32_t in_c = layer.input_shape.dims[0];
    const uint32_t in_h = layer.input_shape.dims[1];
    const uint32_t in_w = layer.input_shape.dims[2];
    const uint32_t out_c = layer.output_shape.dims[0];
    const uint32_t out_h = layer.output_shape.dims[1];
    const uint32_t out_w = layer.output_shape.dims[2];

    std::vector<float> output(layer.output_shape.elementCount(), 0.0f);
    const std::size_t plane = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t total = static_cast<std::size_t>(out_c) * plane;

    runRange(0, total, 64, options.use_threads, [&](std::size_t index) {
        const uint32_t oc = static_cast<uint32_t>(index / plane);
        const uint32_t rem = static_cast<uint32_t>(index % plane);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;

        float sum = (*bias)[oc];
        for (uint32_t ic = 0; ic < in_c; ++ic) {
            for (uint32_t ky = 0; ky < layer.kernel_y; ++ky) {
                for (uint32_t kx = 0; kx < layer.kernel_x; ++kx) {
                    const int iy = static_cast<int>(oy * layer.stride_y + ky) - static_cast<int>(layer.padding_y);
                    const int ix = static_cast<int>(ox * layer.stride_x + kx) - static_cast<int>(layer.padding_x);
                    if (iy < 0 || ix < 0 || iy >= static_cast<int>(in_h) || ix >= static_cast<int>(in_w)) {
                        continue;
                    }

                    const std::size_t input_index =
                        static_cast<std::size_t>(ic) * in_h * in_w +
                        static_cast<std::size_t>(iy) * in_w +
                        static_cast<std::size_t>(ix);
                    const std::size_t weight_index =
                        static_cast<std::size_t>(oc) * in_c * layer.kernel_y * layer.kernel_x +
                        static_cast<std::size_t>(ic) * layer.kernel_y * layer.kernel_x +
                        static_cast<std::size_t>(ky) * layer.kernel_x +
                        kx;
                    sum += input[input_index] * (*weights)[weight_index];
                }
            }
        }

        output[index] = sum;
    });

    return output;
}

std::vector<float> runRelu(const LayerDesc& layer, const std::vector<float>& input, const CpuGraphOptions& options) {
    std::vector<float> output(layer.output_shape.elementCount(), 0.0f);
    runRange(0, output.size(), 256, options.use_threads, [&](std::size_t index) {
        output[index] = std::max(input[index], 0.0f);
    });
    return output;
}

std::vector<float> runMaxPool2d(const LayerDesc& layer, const std::vector<float>& input, const CpuGraphOptions& options) {
    const uint32_t channels = layer.input_shape.dims[0];
    const uint32_t in_h = layer.input_shape.dims[1];
    const uint32_t in_w = layer.input_shape.dims[2];
    const uint32_t out_h = layer.output_shape.dims[1];
    const uint32_t out_w = layer.output_shape.dims[2];

    std::vector<float> output(layer.output_shape.elementCount(), -std::numeric_limits<float>::infinity());
    const std::size_t plane = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t total = static_cast<std::size_t>(channels) * plane;

    runRange(0, total, 64, options.use_threads, [&](std::size_t index) {
        const uint32_t c = static_cast<uint32_t>(index / plane);
        const uint32_t rem = static_cast<uint32_t>(index % plane);
        const uint32_t oy = rem / out_w;
        const uint32_t ox = rem % out_w;

        float best = -std::numeric_limits<float>::infinity();
        for (uint32_t ky = 0; ky < layer.kernel_y; ++ky) {
            for (uint32_t kx = 0; kx < layer.kernel_x; ++kx) {
                const int iy = static_cast<int>(oy * layer.stride_y + ky) - static_cast<int>(layer.padding_y);
                const int ix = static_cast<int>(ox * layer.stride_x + kx) - static_cast<int>(layer.padding_x);
                if (iy < 0 || ix < 0 || iy >= static_cast<int>(in_h) || ix >= static_cast<int>(in_w)) {
                    continue;
                }

                const std::size_t input_index =
                    static_cast<std::size_t>(c) * in_h * in_w +
                    static_cast<std::size_t>(iy) * in_w +
                    static_cast<std::size_t>(ix);
                best = std::max(best, input[input_index]);
            }
        }

        output[index] = best;
    });

    return output;
}

std::vector<float> runAdd(
    const LayerDesc& layer,
    const std::vector<float>& left,
    const std::vector<float>& right,
    const CpuGraphOptions& options
) {
    const std::size_t count = layer.output_shape.elementCount();
    if (left.size() != count || right.size() != count) {
        return {};
    }

    std::vector<float> output(count, 0.0f);
    runRange(0, count, 256, options.use_threads, [&](std::size_t index) {
        output[index] = left[index] + right[index];
    });
    return output;
}

std::vector<float> runGlobalAvgPool2d(const LayerDesc& layer, const std::vector<float>& input, const CpuGraphOptions& options) {
    const uint32_t channels = layer.input_shape.dims[0];
    const uint32_t height = layer.input_shape.dims[1];
    const uint32_t width = layer.input_shape.dims[2];
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    if (input.size() != static_cast<std::size_t>(channels) * plane) {
        return {};
    }

    std::vector<float> output(channels, 0.0f);
    runRange(0, channels, 1, options.use_threads, [&](std::size_t channel) {
        const std::size_t base = channel * plane;
        float sum = 0.0f;
        for (std::size_t i = 0; i < plane; ++i) {
            sum += input[base + i];
        }
        output[channel] = sum / static_cast<float>(plane);
    });
    return output;
}

std::vector<float> runLinear(
    const ModelDesc& model,
    const LayerDesc& layer,
    const std::vector<float>& input,
    const CpuGraphOptions& options,
    std::string& error
) {
    const std::vector<float>* weights = requireWeight(model, layer.weights_index, error, "linear");
    if (!weights) {
        return {};
    }
    const std::vector<float>* bias = requireWeight(model, layer.bias_index, error, "linear bias");
    if (!bias) {
        return {};
    }

    std::vector<float> output(layer.out_features, 0.0f);
    runRange(0, layer.out_features, 1, options.use_threads, [&](std::size_t out_index) {
        const std::size_t weight_offset = out_index * layer.in_features;
        const float dot = options.use_simd
            ? dotProductSimd(input.data(), weights->data() + weight_offset, layer.in_features)
            : dotProductScalar(input.data(), weights->data() + weight_offset, layer.in_features);
        const float sum = (*bias)[out_index] + dot;
        output[out_index] = sum;
    });
    return output;
}

bool validateLayerWeights(const ModelDesc& model, const LayerDesc& layer, std::string& error) {
    if (layer.type == LayerType::Conv2D || layer.type == LayerType::Linear) {
        const ModelWeight* weights = model.weight(layer.weights_index);
        const ModelWeight* bias = model.weight(layer.bias_index);
        if (!weights || !bias) {
            error = layer.name + ": missing weights or bias";
            return false;
        }
    }
    return true;
}

} // namespace

bool CpuGraphExecutor::configure(const ModelDesc& model) {
    model_ = nullptr;
    error_.clear();

    if (!model.valid()) {
        error_ = model.error.empty() ? "Invalid model" : model.error;
        return false;
    }

    for (const LayerDesc& layer : model.layers) {
        if (!validateLayerWeights(model, layer, error_)) {
            return false;
        }
    }

    model_ = &model;
    return true;
}

bool CpuGraphExecutor::ready() const {
    return model_ != nullptr && error_.empty();
}

std::vector<float> CpuGraphExecutor::infer(const std::vector<float>& input, CpuGraphOptions options) const {
    if (!ready()) {
        return {};
    }
    if (input.size() != model_->input_shape.elementCount()) {
        return {};
    }

    std::map<std::string, std::vector<float>> tensors;
    tensors[model_->input_name] = input;
    std::string error;

    for (const LayerDesc& layer : model_->layers) {
        if (layer.input_names.empty() || layer.output_names.empty()) {
            return {};
        }

        auto input_it = tensors.find(layer.input_names[0]);
        if (input_it == tensors.end()) {
            return {};
        }

        const std::vector<float>& first_input = input_it->second;
        std::vector<float> output;
        switch (layer.type) {
            case LayerType::Conv2D:
                output = runConv2d(*model_, layer, first_input, options, error);
                break;
            case LayerType::Relu:
                output = runRelu(layer, first_input, options);
                break;
            case LayerType::MaxPool2D:
                output = runMaxPool2d(layer, first_input, options);
                break;
            case LayerType::Flatten:
                output = first_input;
                break;
            case LayerType::Add: {
                if (layer.input_names.size() != 2) {
                    return {};
                }
                auto right_it = tensors.find(layer.input_names[1]);
                if (right_it == tensors.end()) {
                    return {};
                }
                output = runAdd(layer, first_input, right_it->second, options);
                break;
            }
            case LayerType::GlobalAvgPool2D:
                output = runGlobalAvgPool2d(layer, first_input, options);
                break;
            case LayerType::Linear:
                output = runLinear(*model_, layer, first_input, options, error);
                break;
            case LayerType::Unknown:
                return {};
        }

        if (!error.empty() || output.empty()) {
            return {};
        }
        tensors[layer.output_names[0]] = std::move(output);
    }

    auto output_it = tensors.find(model_->output_name);
    return output_it == tensors.end() ? std::vector<float>{} : output_it->second;
}

std::vector<float> CpuGraphExecutor::inferBytes(const std::vector<uint8_t>& input, CpuGraphOptions options) const {
    std::vector<float> float_input(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        float_input[i] = static_cast<float>(input[i]) / 255.0f;
    }
    return infer(float_input, options);
}

int CpuGraphExecutor::inferClass(const std::vector<float>& input, CpuGraphOptions options) const {
    return argmax(infer(input, options));
}

int CpuGraphExecutor::inferClassBytes(const std::vector<uint8_t>& input, CpuGraphOptions options) const {
    return argmax(inferBytes(input, options));
}

const std::string& CpuGraphExecutor::error() const {
    return error_;
}

int argmax(const std::vector<float>& values) {
    int result = -1;
    float best = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] > best) {
            best = values[i];
            result = static_cast<int>(i);
        }
    }
    return result;
}

} // namespace network
