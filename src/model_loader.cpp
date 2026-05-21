#include "model_loader.h"

#include "embedded_data.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>

namespace network {
namespace {

struct JsonValue {
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;

    const JsonValue* get(const std::string& key) const {
        if (type != Type::Object) {
            return nullptr;
        }
        auto it = object_value.find(key);
        return it == object_value.end() ? nullptr : &it->second;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    bool parse(JsonValue& out, std::string& error) {
        skipWhitespace();
        if (!parseValue(out, error)) {
            return false;
        }
        skipWhitespace();
        if (pos_ != text_.size()) {
            error = "Unexpected trailing data in JSON";
            return false;
        }
        return true;
    }

private:
    std::string text_;
    std::size_t pos_ = 0;

    void skipWhitespace() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    bool consume(char ch) {
        skipWhitespace();
        if (pos_ < text_.size() && text_[pos_] == ch) {
            ++pos_;
            return true;
        }
        return false;
    }

    bool parseValue(JsonValue& out, std::string& error) {
        skipWhitespace();
        if (pos_ >= text_.size()) {
            error = "Unexpected end of JSON";
            return false;
        }

        const char ch = text_[pos_];
        if (ch == '{') {
            return parseObject(out, error);
        }
        if (ch == '[') {
            return parseArray(out, error);
        }
        if (ch == '"') {
            out.type = JsonValue::Type::String;
            return parseString(out.string_value, error);
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            out.type = JsonValue::Type::Number;
            return parseNumber(out.number_value, error);
        }
        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out.type = JsonValue::Type::Bool;
            out.bool_value = true;
            return true;
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out.type = JsonValue::Type::Bool;
            out.bool_value = false;
            return true;
        }
        if (text_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            out.type = JsonValue::Type::Null;
            return true;
        }

        error = "Unexpected JSON token";
        return false;
    }

    bool parseObject(JsonValue& out, std::string& error) {
        if (!consume('{')) {
            error = "Expected JSON object";
            return false;
        }

        out.type = JsonValue::Type::Object;
        skipWhitespace();
        if (consume('}')) {
            return true;
        }

        while (true) {
            std::string key;
            if (!parseString(key, error)) {
                return false;
            }
            if (!consume(':')) {
                error = "Expected ':' after JSON object key";
                return false;
            }
            JsonValue value;
            if (!parseValue(value, error)) {
                return false;
            }
            out.object_value.emplace(std::move(key), std::move(value));

            if (consume('}')) {
                return true;
            }
            if (!consume(',')) {
                error = "Expected ',' or '}' in JSON object";
                return false;
            }
        }
    }

    bool parseArray(JsonValue& out, std::string& error) {
        if (!consume('[')) {
            error = "Expected JSON array";
            return false;
        }

        out.type = JsonValue::Type::Array;
        skipWhitespace();
        if (consume(']')) {
            return true;
        }

        while (true) {
            JsonValue value;
            if (!parseValue(value, error)) {
                return false;
            }
            out.array_value.emplace_back(std::move(value));

            if (consume(']')) {
                return true;
            }
            if (!consume(',')) {
                error = "Expected ',' or ']' in JSON array";
                return false;
            }
        }
    }

    bool parseString(std::string& out, std::string& error) {
        skipWhitespace();
        if (pos_ >= text_.size() || text_[pos_] != '"') {
            error = "Expected JSON string";
            return false;
        }
        ++pos_;

        out.clear();
        while (pos_ < text_.size()) {
            char ch = text_[pos_++];
            if (ch == '"') {
                return true;
            }
            if (ch == '\\') {
                if (pos_ >= text_.size()) {
                    error = "Unterminated JSON escape";
                    return false;
                }
                const char esc = text_[pos_++];
                switch (esc) {
                    case '"':
                    case '\\':
                    case '/':
                        out.push_back(esc);
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    default:
                        error = "Unsupported JSON string escape";
                        return false;
                }
            } else {
                out.push_back(ch);
            }
        }

        error = "Unterminated JSON string";
        return false;
    }

    bool parseNumber(double& out, std::string& error) {
        skipWhitespace();
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        out = std::strtod(begin, &end);
        if (end == begin) {
            error = "Expected JSON number";
            return false;
        }
        pos_ += static_cast<std::size_t>(end - begin);
        return true;
    }
};

const internal::EmbeddedNetworkBlob* findBlob(const std::string& name) {
    for (std::size_t i = 0; i < internal::NETWORK_BLOB_COUNT; ++i) {
        if (name == internal::NETWORK_BLOBS[i].name) {
            return &internal::NETWORK_BLOBS[i];
        }
    }
    return nullptr;
}

float readFloat32LittleEndian(const uint8_t* data) {
    uint32_t bits =
        static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16) |
        (static_cast<uint32_t>(data[3]) << 24);

    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool readString(const JsonValue& object, const std::string& key, std::string& out, std::string& error, bool required = true) {
    const JsonValue* value = object.get(key);
    if (!value) {
        if (required) {
            error = "Missing string field: " + key;
            return false;
        }
        return true;
    }
    if (value->type != JsonValue::Type::String) {
        error = "Expected string field: " + key;
        return false;
    }
    out = value->string_value;
    return true;
}

bool readStringArrayValue(const JsonValue& value, std::vector<std::string>& out, std::string& error) {
    if (value.type != JsonValue::Type::Array) {
        error = "Expected string array";
        return false;
    }
    out.clear();
    for (const JsonValue& item : value.array_value) {
        if (item.type != JsonValue::Type::String) {
            error = "String array contains a non-string value";
            return false;
        }
        out.push_back(item.string_value);
    }
    return true;
}

bool readStringArray(const JsonValue& object, const std::string& key, std::vector<std::string>& out, std::string& error, bool required = false) {
    const JsonValue* value = object.get(key);
    if (!value) {
        if (required) {
            error = "Missing string array field: " + key;
            return false;
        }
        return true;
    }
    if (!readStringArrayValue(*value, out, error)) {
        error = key + ": " + error;
        return false;
    }
    return true;
}

bool readUint(const JsonValue& object, const std::string& key, uint32_t& out, std::string& error, bool required = true) {
    const JsonValue* value = object.get(key);
    if (!value) {
        if (required) {
            error = "Missing integer field: " + key;
            return false;
        }
        return true;
    }
    if (value->type != JsonValue::Type::Number || value->number_value < 0.0) {
        error = "Expected non-negative integer field: " + key;
        return false;
    }
    out = static_cast<uint32_t>(value->number_value);
    return true;
}

bool readShapeValue(const JsonValue& value, TensorShape& out, std::string& error) {
    if (value.type != JsonValue::Type::Array) {
        error = "Expected shape array";
        return false;
    }
    out.dims.clear();
    for (const JsonValue& item : value.array_value) {
        if (item.type != JsonValue::Type::Number || item.number_value < 0.0) {
            error = "Shape contains a non-integer value";
            return false;
        }
        out.dims.push_back(static_cast<uint32_t>(item.number_value));
    }
    return true;
}

bool readShape(const JsonValue& object, const std::string& key, TensorShape& out, std::string& error, bool required = true) {
    const JsonValue* value = object.get(key);
    if (!value) {
        if (required) {
            error = "Missing shape field: " + key;
            return false;
        }
        return true;
    }
    if (!readShapeValue(*value, out, error)) {
        error = key + ": " + error;
        return false;
    }
    return true;
}

bool readPair(const JsonValue& object, const std::string& key, uint32_t& y, uint32_t& x, std::string& error, uint32_t default_value) {
    const JsonValue* value = object.get(key);
    if (!value) {
        y = default_value;
        x = default_value;
        return true;
    }
    TensorShape pair;
    if (!readShapeValue(*value, pair, error) || pair.dims.size() != 2) {
        error = "Expected two-element integer array field: " + key;
        return false;
    }
    y = pair.dims[0];
    x = pair.dims[1];
    return true;
}

bool loadWeightBlob(const std::string& file, const TensorShape& shape, ModelWeight& out, std::string& error) {
    const internal::EmbeddedNetworkBlob* blob = findBlob(file);
    if (!blob) {
        error = "Missing embedded weight blob: " + file;
        return false;
    }

    const std::size_t expected_count = shape.elementCount();
    const std::size_t expected_size = expected_count * sizeof(float);
    if (blob->size != expected_size) {
        std::ostringstream message;
        message << "Weight blob " << file << " has " << blob->size
                << " bytes, expected " << expected_size;
        error = message.str();
        return false;
    }

    out.file = file;
    out.shape = shape;
    out.values.resize(expected_count);
    for (std::size_t i = 0; i < expected_count; ++i) {
        out.values[i] = readFloat32LittleEndian(blob->data + i * sizeof(float));
    }
    return true;
}

uint32_t flattenedCount(const TensorShape& shape) {
    const std::size_t count = shape.elementCount();
    return count > std::numeric_limits<uint32_t>::max() ? 0 : static_cast<uint32_t>(count);
}

bool inferOutputShape(const std::vector<TensorShape>& inputs, LayerDesc& layer, std::string& error) {
    if (inputs.empty()) {
        error = "Layer has no input tensors";
        return false;
    }
    const TensorShape& current = inputs[0];
    layer.input_shape = current;
    switch (layer.type) {
        case LayerType::Conv2D:
            if (current.dims.size() != 3) {
                error = "conv2d expects CHW input shape";
                return false;
            }
            layer.in_channels = layer.in_channels == 0 ? current.dims[0] : layer.in_channels;
            if (layer.kernel_y == 0 || layer.kernel_x == 0 || layer.stride_y == 0 || layer.stride_x == 0) {
                error = "conv2d kernel and stride must be non-zero";
                return false;
            }
            layer.output_shape.dims = {
                layer.out_channels,
                (current.dims[1] + 2 * layer.padding_y - layer.kernel_y) / layer.stride_y + 1,
                (current.dims[2] + 2 * layer.padding_x - layer.kernel_x) / layer.stride_x + 1,
            };
            return true;
        case LayerType::MaxPool2D:
            if (current.dims.size() != 3) {
                error = "maxpool2d expects CHW input shape";
                return false;
            }
            if (layer.kernel_y == 0 || layer.kernel_x == 0 || layer.stride_y == 0 || layer.stride_x == 0) {
                error = "maxpool2d kernel and stride must be non-zero";
                return false;
            }
            layer.output_shape.dims = {
                current.dims[0],
                (current.dims[1] + 2 * layer.padding_y - layer.kernel_y) / layer.stride_y + 1,
                (current.dims[2] + 2 * layer.padding_x - layer.kernel_x) / layer.stride_x + 1,
            };
            return true;
        case LayerType::Relu:
            layer.output_shape = current;
            return true;
        case LayerType::Flatten:
            layer.output_shape.dims = {flattenedCount(current)};
            return true;
        case LayerType::Add:
            if (inputs.size() != 2) {
                error = "add expects two input tensors";
                return false;
            }
            if (inputs[0].dims != inputs[1].dims) {
                error = "add input shapes must match";
                return false;
            }
            layer.output_shape = current;
            return true;
        case LayerType::GlobalAvgPool2D:
            if (current.dims.size() != 3) {
                error = "global_avg_pool2d expects CHW input shape";
                return false;
            }
            layer.output_shape.dims = {current.dims[0]};
            return true;
        case LayerType::Linear:
            layer.in_features = layer.in_features == 0 ? flattenedCount(current) : layer.in_features;
            layer.output_shape.dims = {layer.out_features};
            return true;
        case LayerType::Unknown:
            error = "Unknown layer type";
            return false;
    }
    return false;
}

int appendWeight(ModelDesc& model, const std::string& file, const TensorShape& shape) {
    for (std::size_t i = 0; i < model.weights.size(); ++i) {
        if (model.weights[i].file == file) {
            return static_cast<int>(i);
        }
    }

    ModelWeight weight;
    if (!loadWeightBlob(file, shape, weight, model.error)) {
        return -1;
    }
    model.weights.emplace_back(std::move(weight));
    return static_cast<int>(model.weights.size() - 1);
}

void upsertTensor(ModelDesc& model, std::map<std::string, TensorShape>& tensor_shapes, const std::string& name, const TensorShape& shape) {
    tensor_shapes[name] = shape;
    for (TensorDesc& tensor : model.tensors) {
        if (tensor.name == name) {
            tensor.shape = shape;
            return;
        }
    }
    model.tensors.push_back({name, shape});
}

bool parseLayer(
    const JsonValue& json,
    ModelDesc& model,
    std::map<std::string, TensorShape>& tensor_shapes,
    std::string& current_tensor,
    std::size_t layer_index
) {
    LayerDesc layer;
    std::string type_name;
    if (!readString(json, "type", type_name, model.error)) {
        return false;
    }
    layer.type = layerTypeFromName(type_name);

    readString(json, "name", layer.name, model.error, false);
    if (!model.error.empty()) {
        return false;
    }
    if (layer.name.empty()) {
        layer.name = std::string(layerTypeName(layer.type)) + "_" + std::to_string(layer_index);
    }

    if (!readStringArray(json, "inputs", layer.input_names, model.error, false) ||
        !readStringArray(json, "outputs", layer.output_names, model.error, false)) {
        return false;
    }
    if (layer.input_names.empty()) {
        std::string input_name;
        if (readString(json, "input", input_name, model.error, false) && !input_name.empty()) {
            layer.input_names.push_back(input_name);
        }
    }
    if (!model.error.empty()) {
        return false;
    }
    if (layer.input_names.empty()) {
        std::string lhs;
        std::string rhs;
        readString(json, "lhs", lhs, model.error, false);
        readString(json, "rhs", rhs, model.error, false);
        if (!model.error.empty()) {
            return false;
        }
        if (!lhs.empty() || !rhs.empty()) {
            if (lhs.empty() || rhs.empty()) {
                model.error = layer.name + ": add lhs/rhs must both be present";
                return false;
            }
            layer.input_names.push_back(lhs);
            layer.input_names.push_back(rhs);
        }
    }
    if (layer.input_names.empty()) {
        layer.input_names.push_back(current_tensor);
    }
    if (layer.output_names.empty()) {
        std::string output_name;
        if (readString(json, "output", output_name, model.error, false) && !output_name.empty()) {
            layer.output_names.push_back(output_name);
        }
    }
    if (!model.error.empty()) {
        return false;
    }
    if (layer.output_names.empty()) {
        layer.output_names.push_back(layer.name);
    }

    readUint(json, "in_channels", layer.in_channels, model.error, false);
    readUint(json, "out_channels", layer.out_channels, model.error, false);
    readUint(json, "in_features", layer.in_features, model.error, false);
    readUint(json, "out_features", layer.out_features, model.error, false);
    if (!model.error.empty()) {
        return false;
    }

    if (!readPair(json, "kernel", layer.kernel_y, layer.kernel_x, model.error, 1) ||
        !readPair(json, "stride", layer.stride_y, layer.stride_x, model.error, 1) ||
        !readPair(json, "padding", layer.padding_y, layer.padding_x, model.error, 0)) {
        return false;
    }

    std::string weight_file;
    TensorShape weight_shape;
    if (readString(json, "weights", weight_file, model.error, false) && !weight_file.empty()) {
        if (!readShape(json, "weights_shape", weight_shape, model.error)) {
            return false;
        }
        layer.weights_index = appendWeight(model, weight_file, weight_shape);
        if (layer.weights_index < 0) {
            return false;
        }
    }
    if (!model.error.empty()) {
        return false;
    }

    std::string bias_file;
    TensorShape bias_shape;
    if (readString(json, "bias", bias_file, model.error, false) && !bias_file.empty()) {
        if (!readShape(json, "bias_shape", bias_shape, model.error)) {
            return false;
        }
        layer.bias_index = appendWeight(model, bias_file, bias_shape);
        if (layer.bias_index < 0) {
            return false;
        }
    }
    if (!model.error.empty()) {
        return false;
    }

    TensorShape explicit_shape;
    if (readShape(json, "shape", explicit_shape, model.error, false) && !explicit_shape.dims.empty()) {
        TensorShape& shape = tensor_shapes[layer.input_names[0]];
        shape = explicit_shape;
        upsertTensor(model, tensor_shapes, layer.input_names[0], explicit_shape);
    }
    if (!model.error.empty()) {
        return false;
    }

    std::vector<TensorShape> input_shapes;
    for (const std::string& input_name : layer.input_names) {
        auto it = tensor_shapes.find(input_name);
        if (it == tensor_shapes.end()) {
            model.error = layer.name + ": missing input tensor '" + input_name + "'";
            return false;
        }
        input_shapes.push_back(it->second);
    }

    if (!inferOutputShape(input_shapes, layer, model.error)) {
        model.error = layer.name + ": " + model.error;
        return false;
    }

    if (layer.output_names.size() != 1) {
        model.error = layer.name + ": exactly one output tensor is currently supported";
        return false;
    }
    upsertTensor(model, tensor_shapes, layer.output_names[0], layer.output_shape);
    current_tensor = layer.output_names[0];
    model.layers.emplace_back(std::move(layer));
    return true;
}

} // namespace

std::size_t TensorShape::elementCount() const {
    std::size_t total = 1;
    for (uint32_t dim : dims) {
        total *= dim;
    }
    return dims.empty() ? 0 : total;
}

std::string TensorShape::toString() const {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << dims[i];
    }
    out << "]";
    return out.str();
}

const char* layerTypeName(LayerType type) {
    switch (type) {
        case LayerType::Conv2D:
            return "conv2d";
        case LayerType::Linear:
            return "linear";
        case LayerType::MaxPool2D:
            return "maxpool2d";
        case LayerType::Relu:
            return "relu";
        case LayerType::Flatten:
            return "flatten";
        case LayerType::Add:
            return "add";
        case LayerType::GlobalAvgPool2D:
            return "global_avg_pool2d";
        case LayerType::Unknown:
            return "unknown";
    }
    return "unknown";
}

LayerType layerTypeFromName(const std::string& name) {
    if (name == "conv2d") {
        return LayerType::Conv2D;
    }
    if (name == "linear" || name == "gemm") {
        return LayerType::Linear;
    }
    if (name == "maxpool2d" || name == "maxpool") {
        return LayerType::MaxPool2D;
    }
    if (name == "relu") {
        return LayerType::Relu;
    }
    if (name == "flatten") {
        return LayerType::Flatten;
    }
    if (name == "add") {
        return LayerType::Add;
    }
    if (name == "global_avg_pool2d" || name == "global_avg_pool" || name == "globalaveragepool") {
        return LayerType::GlobalAvgPool2D;
    }
    return LayerType::Unknown;
}

bool ModelDesc::valid() const {
    return error.empty() && !name.empty() && !input_shape.dims.empty() && !layers.empty();
}

const ModelWeight* ModelDesc::weight(int index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= weights.size()) {
        return nullptr;
    }
    return &weights[static_cast<std::size_t>(index)];
}

const TensorDesc* ModelDesc::tensor(const std::string& name) const {
    for (const TensorDesc& tensor_desc : tensors) {
        if (tensor_desc.name == name) {
            return &tensor_desc;
        }
    }
    return nullptr;
}

ModelDesc loadModelFromEmbedded(const std::string& manifest_name) {
    ModelDesc model;

    const internal::EmbeddedNetworkBlob* manifest_blob = findBlob(manifest_name);
    if (!manifest_blob) {
        model.error = "Missing embedded manifest: " + manifest_name;
        return model;
    }

    std::string manifest_text(
        reinterpret_cast<const char*>(manifest_blob->data),
        reinterpret_cast<const char*>(manifest_blob->data + manifest_blob->size)
    );

    JsonValue root;
    JsonParser parser(std::move(manifest_text));
    if (!parser.parse(root, model.error)) {
        return model;
    }
    if (root.type != JsonValue::Type::Object) {
        model.error = "Manifest root must be an object";
        return model;
    }

    if (!readString(root, "name", model.name, model.error) ||
        !readString(root, "dtype", model.dtype, model.error) ||
        !readString(root, "endianness", model.endianness, model.error, false) ||
        !readShape(root, "input_shape", model.input_shape, model.error)) {
        return model;
    }
    readString(root, "input", model.input_name, model.error, false);
    if (!model.error.empty()) {
        return model;
    }
    if (model.input_name.empty()) {
        model.input_name = "input";
    }
    if (model.endianness.empty()) {
        model.endianness = "little";
    }
    if (model.dtype != "float32") {
        model.error = "Only float32 models are supported";
        return model;
    }
    if (model.endianness != "little") {
        model.error = "Only little-endian weight blobs are supported";
        return model;
    }

    const JsonValue* layers = root.get("layers");
    if (!layers || layers->type != JsonValue::Type::Array) {
        model.error = "Manifest must contain a layers array";
        return model;
    }

    std::map<std::string, TensorShape> tensor_shapes;
    std::string current_tensor = model.input_name;
    upsertTensor(model, tensor_shapes, current_tensor, model.input_shape);
    for (std::size_t i = 0; i < layers->array_value.size(); ++i) {
        if (layers->array_value[i].type != JsonValue::Type::Object) {
            model.error = "Layer entry must be an object";
            return model;
        }
        if (!parseLayer(layers->array_value[i], model, tensor_shapes, current_tensor, i)) {
            return model;
        }
    }
    model.output_name = current_tensor;

    return model;
}

std::vector<std::string> embeddedModelFiles() {
    std::vector<std::string> files;
    for (std::size_t i = 0; i < internal::NETWORK_BLOB_COUNT; ++i) {
        files.emplace_back(internal::NETWORK_BLOBS[i].name);
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace network
