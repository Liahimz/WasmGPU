#include "cpu_graph_executor.h"

#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

std::vector<float> runConv2d(const ModelDesc& model, const LayerDesc& layer, const std::vector<float>& input, std::string& error) {
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

    parallel::parallelFor(0, total, 64, [&](std::size_t index) {
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

std::vector<float> runRelu(const LayerDesc& layer, const std::vector<float>& input) {
    std::vector<float> output(layer.output_shape.elementCount(), 0.0f);
    parallel::parallelFor(0, output.size(), 256, [&](std::size_t index) {
        output[index] = std::max(input[index], 0.0f);
    });
    return output;
}

std::vector<float> runMaxPool2d(const LayerDesc& layer, const std::vector<float>& input) {
    const uint32_t channels = layer.input_shape.dims[0];
    const uint32_t in_h = layer.input_shape.dims[1];
    const uint32_t in_w = layer.input_shape.dims[2];
    const uint32_t out_h = layer.output_shape.dims[1];
    const uint32_t out_w = layer.output_shape.dims[2];

    std::vector<float> output(layer.output_shape.elementCount(), -std::numeric_limits<float>::infinity());
    const std::size_t plane = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t total = static_cast<std::size_t>(channels) * plane;

    parallel::parallelFor(0, total, 64, [&](std::size_t index) {
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

std::vector<float> runLinear(const ModelDesc& model, const LayerDesc& layer, const std::vector<float>& input, std::string& error) {
    const std::vector<float>* weights = requireWeight(model, layer.weights_index, error, "linear");
    if (!weights) {
        return {};
    }
    const std::vector<float>* bias = requireWeight(model, layer.bias_index, error, "linear bias");
    if (!bias) {
        return {};
    }

    std::vector<float> output(layer.out_features, 0.0f);
    parallel::parallelFor(0, layer.out_features, 1, [&](std::size_t out_index) {
        float sum = (*bias)[out_index];
        const std::size_t weight_offset = out_index * layer.in_features;
        for (uint32_t i = 0; i < layer.in_features; ++i) {
            sum += input[i] * (*weights)[weight_offset + i];
        }
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

std::vector<float> CpuGraphExecutor::infer(const std::vector<float>& input) const {
    if (!ready()) {
        return {};
    }
    if (input.size() != model_->input_shape.elementCount()) {
        return {};
    }

    std::vector<float> current = input;
    std::string error;

    for (const LayerDesc& layer : model_->layers) {
        switch (layer.type) {
            case LayerType::Conv2D:
                current = runConv2d(*model_, layer, current, error);
                break;
            case LayerType::Relu:
                current = runRelu(layer, current);
                break;
            case LayerType::MaxPool2D:
                current = runMaxPool2d(layer, current);
                break;
            case LayerType::Flatten:
                break;
            case LayerType::Linear:
                current = runLinear(*model_, layer, current, error);
                break;
            case LayerType::Unknown:
                return {};
        }

        if (!error.empty() || current.empty()) {
            return {};
        }
    }

    return current;
}

std::vector<float> CpuGraphExecutor::inferBytes(const std::vector<uint8_t>& input) const {
    std::vector<float> float_input(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        float_input[i] = static_cast<float>(input[i]) / 255.0f;
    }
    return infer(float_input);
}

int CpuGraphExecutor::inferClass(const std::vector<float>& input) const {
    return argmax(infer(input));
}

int CpuGraphExecutor::inferClassBytes(const std::vector<uint8_t>& input) const {
    return argmax(inferBytes(input));
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
