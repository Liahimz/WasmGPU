#include "network_weights.h"

#include "embedded_data.h"

#include <cstring>

namespace network {
namespace {

const internal::EmbeddedNetworkBlob* findBlob(const char* name) {
    for (std::size_t i = 0; i < internal::NETWORK_BLOB_COUNT; ++i) {
        if (std::strcmp(internal::NETWORK_BLOBS[i].name, name) == 0) {
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

bool readFloatBlob(const char* name, std::size_t expected_count, std::vector<float>& out, std::string& error) {
    const internal::EmbeddedNetworkBlob* blob = findBlob(name);
    if (!blob) {
        error = std::string("Missing embedded network blob: ") + name;
        return false;
    }

    const std::size_t expected_size = expected_count * sizeof(float);
    if (blob->size != expected_size) {
        error = std::string("Embedded network blob has wrong size: ") + name;
        return false;
    }

    out.resize(expected_count);
    for (std::size_t i = 0; i < expected_count; ++i) {
        out[i] = readFloat32LittleEndian(blob->data + i * sizeof(float));
    }

    return true;
}

} // namespace

bool TinyLenetWeights::valid() const {
    return error.empty() &&
        conv_weights.size() == 4 * 3 * 3 &&
        conv_bias.size() == 4 &&
        linear_weights.size() == 10 * 2704 &&
        linear_bias.size() == 10;
}

TinyLenetWeights loadTinyLenetWeights() {
    TinyLenetWeights weights;

    if (!readFloatBlob("lenet/tiny_lenet_conv_weights_f32.bin", 4 * 3 * 3, weights.conv_weights, weights.error)) {
        return weights;
    }
    if (!readFloatBlob("lenet/tiny_lenet_conv_bias_f32.bin", 4, weights.conv_bias, weights.error)) {
        return weights;
    }
    if (!readFloatBlob("lenet/tiny_lenet_linear_weights_f32.bin", 10 * 2704, weights.linear_weights, weights.error)) {
        return weights;
    }
    if (!readFloatBlob("lenet/tiny_lenet_linear_bias_f32.bin", 10, weights.linear_bias, weights.error)) {
        return weights;
    }

    return weights;
}

} // namespace network
