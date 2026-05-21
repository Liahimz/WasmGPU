#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace network {

struct TensorShape {
    std::vector<uint32_t> dims;

    std::size_t elementCount() const;
    std::string toString() const;
};

enum class LayerType {
    Conv2D,
    Linear,
    MaxPool2D,
    Relu,
    Flatten,
    Add,
    GlobalAvgPool2D,
    Unknown,
};

const char* layerTypeName(LayerType type);
LayerType layerTypeFromName(const std::string& name);

struct ModelWeight {
    std::string file;
    TensorShape shape;
    std::vector<float> values;
};

struct TensorDesc {
    std::string name;
    TensorShape shape;
};

struct LayerDesc {
    std::string name;
    LayerType type = LayerType::Unknown;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    TensorShape input_shape;
    TensorShape output_shape;

    uint32_t in_channels = 0;
    uint32_t out_channels = 0;
    uint32_t in_features = 0;
    uint32_t out_features = 0;

    uint32_t kernel_y = 1;
    uint32_t kernel_x = 1;
    uint32_t stride_y = 1;
    uint32_t stride_x = 1;
    uint32_t padding_y = 0;
    uint32_t padding_x = 0;

    int weights_index = -1;
    int bias_index = -1;
};

struct ModelDesc {
    std::string name;
    std::string dtype;
    std::string endianness;
    TensorShape input_shape;
    std::string input_name;
    std::string output_name;
    std::vector<TensorDesc> tensors;
    std::vector<LayerDesc> layers;
    std::vector<ModelWeight> weights;
    std::string error;

    bool valid() const;
    const ModelWeight* weight(int index) const;
    const TensorDesc* tensor(const std::string& name) const;
};

ModelDesc loadModelFromEmbedded(const std::string& manifest_name);
std::string loadEmbeddedText(const std::string& file_name);
std::vector<std::string> embeddedModelFiles();

} // namespace network
