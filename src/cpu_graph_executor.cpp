#include "cpu_graph_executor.h"

#include "cpu_conv_kernels.h"
#include "cpu_simd_math.h"
#include "thread_tools/parallel_utils.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace network {
namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

const std::vector<float>* requireWeight(const ModelDesc& model, int index, std::string& error, const char* role) {
    const ModelWeight* weight = model.weight(index);
    if (!weight) {
        error = std::string("Missing ") + role + " weight";
        return nullptr;
    }
    return &weight->values;
}

std::string cpuModeName(CpuGraphOptions options) {
    if (options.use_simd && options.use_threads) {
        return "simd_threads";
    }
    if (options.use_simd) {
        return "simd";
    }
    if (options.use_threads) {
        return "threads";
    }
    return "scalar";
}

std::string shapeText(const TensorShape& shape) {
    return shape.toString();
}

double percent(std::uint64_t part, std::uint64_t total) {
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(part) * 100.0 / static_cast<double>(total);
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

struct TensorValue {
    std::vector<float> data;
    TensorShape shape;
    bool c4 = false;
};

bool canUseC4Graph(const ModelDesc& model, const CpuGraphOptions& options) {
    if (!options.use_simd || !options.use_c4_layout) {
        return false;
    }
#if !defined(__wasm_simd128__)
    return false;
#else
    for (const LayerDesc& layer : model.layers) {
        if (layer.type == LayerType::Conv2D && !cpu_conv::supportsSpecializedConv2d(layer)) {
            return false;
        }
    }
    return model.input_shape.dims.size() == 3;
#endif
}

const std::vector<float>* findPackedWeights(
    const std::vector<cpu_conv::PackedConvWeights>& packed_conv_weights,
    int layer_index
) {
    for (const cpu_conv::PackedConvWeights& packed : packed_conv_weights) {
        if (packed.layer_index == layer_index) {
            return &packed.oc4_weights;
        }
    }
    return nullptr;
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
    const std::vector<float>* packed_oc4_weights = findPackedWeights(packed_conv_weights, layer_index);
    if (cpu_conv::runSpecializedConv2d(
        layer,
        input,
        *weights,
        packed_oc4_weights,
        *bias,
        options.use_simd,
        options.use_threads,
        options.conv_tile_mode,
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

std::vector<float> runConv2dC4(
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

    std::vector<float> output;
    if (!cpu_conv::runSpecializedConv2dC4(
        layer,
        input,
        *weights,
        findPackedWeights(packed_conv_weights, layer_index),
        *bias,
        options.use_simd,
        options.use_threads,
        options.conv_tile_mode,
        output
    )) {
        error = layer.name + ": C4 conv fallback is not available";
        return {};
    }
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

std::vector<float> runReluC4(const LayerDesc& layer, const std::vector<float>& input, const CpuGraphOptions& options) {
    const uint32_t channels = layer.output_shape.dims[0];
    const uint32_t height = layer.output_shape.dims[1];
    const uint32_t width = layer.output_shape.dims[2];
    const uint32_t c4 = (channels + 3) / 4;
    const std::size_t count = static_cast<std::size_t>(height) * width * c4 * 4;
    if (input.size() != count) {
        return {};
    }

    std::vector<float> output(count, 0.0f);
    if (options.use_simd) {
        if (options.use_threads) {
            constexpr std::size_t ChunkSize = 4096;
            const std::size_t chunks = (count + ChunkSize - 1) / ChunkSize;
            parallel::parallelFor(0, chunks, [&](std::size_t chunk) {
                const std::size_t begin = chunk * ChunkSize;
                const std::size_t end = std::min(begin + ChunkSize, count);
                simd_math::reluSimd(input.data() + begin, output.data() + begin, end - begin);
            });
        } else {
            simd_math::reluSimd(input.data(), output.data(), count);
        }
    } else {
        runRange(0, count, 256, options.use_threads, [&](std::size_t index) {
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

std::vector<float> runMaxPool2dC4(const LayerDesc& layer, const std::vector<float>& input, const CpuGraphOptions& options) {
    const uint32_t channels = layer.input_shape.dims[0];
    const uint32_t in_h = layer.input_shape.dims[1];
    const uint32_t in_w = layer.input_shape.dims[2];
    const uint32_t out_h = layer.output_shape.dims[1];
    const uint32_t out_w = layer.output_shape.dims[2];
    const uint32_t c4 = (channels + 3) / 4;

    std::vector<float> output(static_cast<std::size_t>(out_h) * out_w * c4 * 4, -std::numeric_limits<float>::infinity());
    const std::size_t total = static_cast<std::size_t>(out_h) * out_w * c4;
    runRange(0, total, 64, options.use_threads, [&](std::size_t index) {
        const uint32_t block = static_cast<uint32_t>(index % c4);
        const uint32_t spatial = static_cast<uint32_t>(index / c4);
        const uint32_t oy = spatial / out_w;
        const uint32_t ox = spatial % out_w;
        float best[4] = {
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()
        };

        for (uint32_t ky = 0; ky < layer.kernel_y; ++ky) {
            const int iy = static_cast<int>(oy * layer.stride_y + ky) - static_cast<int>(layer.padding_y);
            if (iy < 0 || iy >= static_cast<int>(in_h)) {
                continue;
            }
            for (uint32_t kx = 0; kx < layer.kernel_x; ++kx) {
                const int ix = static_cast<int>(ox * layer.stride_x + kx) - static_cast<int>(layer.padding_x);
                if (ix < 0 || ix >= static_cast<int>(in_w)) {
                    continue;
                }
                const float* src = input.data() + ((static_cast<std::size_t>(iy) * in_w + static_cast<uint32_t>(ix)) * c4 + block) * 4;
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    best[lane] = std::max(best[lane], src[lane]);
                }
            }
        }

        float* dst = output.data() + ((static_cast<std::size_t>(oy) * out_w + ox) * c4 + block) * 4;
        dst[0] = best[0];
        dst[1] = best[1];
        dst[2] = best[2];
        dst[3] = best[3];
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

std::vector<float> runAddC4(const LayerDesc& layer, const std::vector<float>& left, const std::vector<float>& right, const CpuGraphOptions& options) {
    const uint32_t channels = layer.output_shape.dims[0];
    const uint32_t height = layer.output_shape.dims[1];
    const uint32_t width = layer.output_shape.dims[2];
    const uint32_t c4 = (channels + 3) / 4;
    const std::size_t count = static_cast<std::size_t>(height) * width * c4 * 4;
    if (left.size() != count || right.size() != count) {
        return {};
    }

    std::vector<float> output(count, 0.0f);
    runRange(0, count, 1024, options.use_threads, [&](std::size_t index) {
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

std::vector<float> runGlobalAvgPool2dC4(const LayerDesc& layer, const std::vector<float>& input, const CpuGraphOptions& options) {
    const uint32_t channels = layer.input_shape.dims[0];
    const uint32_t height = layer.input_shape.dims[1];
    const uint32_t width = layer.input_shape.dims[2];
    const uint32_t c4 = (channels + 3) / 4;
    const std::size_t expected = static_cast<std::size_t>(height) * width * c4 * 4;
    if (input.size() != expected) {
        return {};
    }

    std::vector<float> output(channels, 0.0f);
    runRange(0, channels, 4, options.use_threads, [&](std::size_t channel) {
        const uint32_t c = static_cast<uint32_t>(channel);
        const uint32_t block = c / 4;
        const uint32_t lane = c % 4;
        float sum = 0.0f;
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                sum += input[((static_cast<std::size_t>(y) * width + x) * c4 + block) * 4 + lane];
            }
        }
        output[channel] = sum / static_cast<float>(height * width);
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

    const bool use_c4 = canUseC4Graph(*model_, options);
    std::map<std::string, TensorValue> tensors;
    tensors[model_->input_name] = TensorValue{
        use_c4 ? cpu_conv::nchwToC4(input, model_->input_shape) : input,
        model_->input_shape,
        use_c4
    };
    std::string error;
    std::uint32_t conv_layers = 0;
    std::uint32_t specialized_conv_layers = 0;
    std::uint64_t conv_macs = 0;
    std::uint64_t specialized_conv_macs = 0;

    for (std::size_t layer_index = 0; layer_index < model_->layers.size(); ++layer_index) {
        const LayerDesc& layer = model_->layers[layer_index];
        if (layer.input_names.empty() || layer.output_names.empty()) {
            return {};
        }

        auto input_it = tensors.find(layer.input_names[0]);
        if (input_it == tensors.end()) {
            return {};
        }

        const TensorValue& first_input = input_it->second;
        std::vector<float> output;
        bool output_c4 = false;
        const char* kernel_name = layerTypeName(layer.type);
        if (layer.type == LayerType::Conv2D) {
            kernel_name = first_input.c4
                ? cpu_conv::selectedKernelNameC4(layer, options.use_simd, options.conv_tile_mode)
                : cpu_conv::selectedKernelName(layer, options.use_simd, options.conv_tile_mode);
            const std::uint64_t macs = cpu_conv::estimateConvMacs(layer);
            conv_layers += 1;
            conv_macs += macs;
            if (options.use_simd && cpu_conv::supportsSpecializedConv2d(layer)) {
                specialized_conv_layers += 1;
                specialized_conv_macs += macs;
            }
        }

        const auto layer_start = Clock::now();
        switch (layer.type) {
            case LayerType::Conv2D:
                if (first_input.c4) {
                    output = runConv2dC4(*model_, packed_conv_weights_, static_cast<int>(layer_index), layer, first_input.data, options, error);
                    output_c4 = true;
                } else {
                    output = runConv2d(*model_, packed_conv_weights_, static_cast<int>(layer_index), layer, first_input.data, options, error);
                }
                break;
            case LayerType::Relu:
                output = first_input.c4 ? runReluC4(layer, first_input.data, options) : runRelu(layer, first_input.data, options);
                output_c4 = first_input.c4;
                break;
            case LayerType::MaxPool2D:
                output = first_input.c4 ? runMaxPool2dC4(layer, first_input.data, options) : runMaxPool2d(layer, first_input.data, options);
                output_c4 = first_input.c4;
                break;
            case LayerType::Flatten:
                output = first_input.c4 ? cpu_conv::c4ToNchw(first_input.data, first_input.shape) : first_input.data;
                break;
            case LayerType::Add: {
                if (layer.input_names.size() != 2) {
                    return {};
                }
                auto right_it = tensors.find(layer.input_names[1]);
                if (right_it == tensors.end()) {
                    return {};
                }
                if (first_input.c4 != right_it->second.c4) {
                    return {};
                }
                output = first_input.c4
                    ? runAddC4(layer, first_input.data, right_it->second.data, options)
                    : runAdd(layer, first_input.data, right_it->second.data, options);
                output_c4 = first_input.c4;
                break;
            }
            case LayerType::GlobalAvgPool2D:
                output = first_input.c4
                    ? runGlobalAvgPool2dC4(layer, first_input.data, options)
                    : runGlobalAvgPool2d(layer, first_input.data, options);
                break;
            case LayerType::Linear:
                output = runLinear(*model_, layer, first_input.data, options, error);
                break;
            case LayerType::Unknown:
                return {};
        }
        const auto layer_end = Clock::now();

        if (options.log_layers) {
            std::cout << "[cpu_layer] mode=" << cpuModeName(options)
                      << " layout=" << (first_input.c4 ? "c4" : "nchw")
                      << " index=" << layer_index
                      << " name=" << layer.name
                      << " type=" << layerTypeName(layer.type)
                      << " input=" << shapeText(layer.input_shape)
                      << " output=" << shapeText(layer.output_shape)
                      << " kernel=" << kernel_name
                      << " ms=" << elapsedMs(layer_start, layer_end)
                      << std::endl;
        }

        if (!error.empty() || output.empty()) {
            return {};
        }
        tensors[layer.output_names[0]] = TensorValue{std::move(output), layer.output_shape, output_c4};
    }

    auto output_it = tensors.find(model_->output_name);
    if (options.log_layers) {
        std::cout << "[cpu_conv_coverage] mode=" << cpuModeName(options)
                  << " layout=" << (use_c4 ? "c4" : "nchw")
                  << " specialized_layers=" << specialized_conv_layers << "/" << conv_layers
                  << " specialized_macs=" << percent(specialized_conv_macs, conv_macs) << "%"
                  << " fallback_macs=" << percent(conv_macs - specialized_conv_macs, conv_macs) << "%"
                  << " total_macs=" << conv_macs
                  << std::endl;
    }
    if (output_it == tensors.end()) {
        return {};
    }
    return output_it->second.c4 ? cpu_conv::c4ToNchw(output_it->second.data, output_it->second.shape) : output_it->second.data;
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
