#include "cpu_graph_executor.h"

#include "cpu_conv_kernels.h"
#include "cpu_simd_math.h"
#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

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

std::vector<float> runConv2d(
    const ModelDesc& model,
    const std::vector<cpu_conv::PackedConvWeights>& packed_conv_weights,
    int layer_index,
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
    const std::vector<float>* packed_oc4_weights = nullptr;
    for (const cpu_conv::PackedConvWeights& packed : packed_conv_weights) {
        if (packed.layer_index == layer_index) {
            packed_oc4_weights = &packed.oc4_weights;
            break;
        }
    }
    if (cpu_conv::runSpecializedConv2d(
        layer,
        input,
        *weights,
        packed_oc4_weights,
        *bias,
        options.use_simd,
        options.use_threads,
        output
    )) {
        return output;
    }

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
    if (options.use_simd) {
        if (options.use_threads) {
            constexpr std::size_t ChunkSize = 1024;
            const std::size_t chunks = (output.size() + ChunkSize - 1) / ChunkSize;
            parallel::parallelFor(0, chunks, [&](std::size_t chunk) {
                const std::size_t begin = chunk * ChunkSize;
                const std::size_t end = std::min(begin + ChunkSize, output.size());
                simd_math::reluSimd(input.data() + begin, output.data() + begin, end - begin);
            });
        } else {
            simd_math::reluSimd(input.data(), output.data(), output.size());
        }
    } else {
        runRange(0, output.size(), 256, options.use_threads, [&](std::size_t index) {
            output[index] = std::max(input[index], 0.0f);
        });
    }
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

        const float* channel_input = input.data() + static_cast<std::size_t>(c) * in_h * in_w;
        output[index] = options.use_simd
            ? simd_math::maxPoolSimd(
                channel_input,
                in_h,
                in_w,
                layer.kernel_y,
                layer.kernel_x,
                layer.stride_y,
                layer.stride_x,
                layer.padding_y,
                layer.padding_x,
                oy,
                ox
            )
            : simd_math::maxPoolScalar(
                channel_input,
                in_h,
                in_w,
                layer.kernel_y,
                layer.kernel_x,
                layer.stride_y,
                layer.stride_x,
                layer.padding_y,
                layer.padding_x,
                oy,
                ox
            );
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
        const float sum = options.use_simd
            ? simd_math::sumSimd(input.data() + base, plane)
            : simd_math::sumScalar(input.data() + base, plane);
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
            ? simd_math::dotProductSimd(input.data(), weights->data() + weight_offset, layer.in_features)
            : simd_math::dotProductScalar(input.data(), weights->data() + weight_offset, layer.in_features);
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
    packed_conv_weights_.clear();

    if (!model.valid()) {
        error_ = model.error.empty() ? "Invalid model" : model.error;
        return false;
    }

    for (const LayerDesc& layer : model.layers) {
        if (!validateLayerWeights(model, layer, error_)) {
            return false;
        }
    }

    for (std::size_t layer_index = 0; layer_index < model.layers.size(); ++layer_index) {
        const LayerDesc& layer = model.layers[layer_index];
        if (layer.type != LayerType::Conv2D || !cpu_conv::supportsSpecializedConv2d(layer)) {
            continue;
        }
        const ModelWeight* weights = model.weight(layer.weights_index);
        if (!weights) {
            continue;
        }
        cpu_conv::PackedConvWeights packed;
        packed.layer_index = static_cast<int>(layer_index);
        packed.oc4_weights = cpu_conv::packWeightsOc4(layer, weights->values);
        if (!packed.oc4_weights.empty()) {
            packed_conv_weights_.push_back(std::move(packed));
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

    for (std::size_t layer_index = 0; layer_index < model_->layers.size(); ++layer_index) {
        const LayerDesc& layer = model_->layers[layer_index];
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
                output = runConv2d(*model_, packed_conv_weights_, static_cast<int>(layer_index), layer, first_input, options, error);
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
