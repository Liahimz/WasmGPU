#include "gpu_executor.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>

#ifdef __EMSCRIPTEN__
#include "embedded_shaders.h"
#include "lib_webgpu.h"

namespace {

constexpr std::size_t INPUT_VALUES = 28 * 28;
constexpr std::size_t CONV_VALUES = 26 * 26 * 4;
constexpr std::size_t LOGIT_VALUES = 10;

int argmax(const std::array<float, LOGIT_VALUES>& logits) {
    int result = -1;
    float max = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > max) {
            result = static_cast<int>(i);
            max = logits[i];
        }
    }
    return result;
}

WGpuBindGroupLayoutEntry storageLayoutEntry(uint32_t binding, WGPU_BUFFER_BINDING_TYPE type, uint64_t min_size) {
    WGpuBindGroupLayoutEntry entry = WGPU_BUFFER_BINDING_LAYOUT_ENTRY_DEFAULT_INITIALIZER;
    entry.binding = binding;
    entry.visibility = WGPU_SHADER_STAGE_COMPUTE;
    entry.type = WGPU_BIND_GROUP_LAYOUT_TYPE_BUFFER;
    entry.layout.buffer.type = type;
    entry.layout.buffer.minBindingSize = min_size;
    return entry;
}

WGpuBindGroupEntry bufferEntry(uint32_t binding, WGpuBuffer buffer, uint64_t size) {
    WGpuBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_DEFAULT_INITIALIZER;
    entry.binding = binding;
    entry.resource = buffer;
    entry.bufferBindSize = size;
    return entry;
}

} // namespace
#endif

GpuExecutor::GpuExecutor() = default;

GpuExecutor::~GpuExecutor() {
#ifdef __EMSCRIPTEN__
    wgpu_object_destroy(readback_buffer_);
    wgpu_object_destroy(logits_buffer_);
    wgpu_object_destroy(linear_bias_buffer_);
    wgpu_object_destroy(linear_weights_buffer_);
    wgpu_object_destroy(conv_output_buffer_);
    wgpu_object_destroy(conv_bias_buffer_);
    wgpu_object_destroy(conv_weights_buffer_);
    wgpu_object_destroy(input_buffer_);
    wgpu_object_destroy(linear_bind_group_);
    wgpu_object_destroy(conv_bind_group_);
    wgpu_object_destroy(linear_pipeline_layout_);
    wgpu_object_destroy(conv_pipeline_layout_);
    wgpu_object_destroy(linear_bind_group_layout_);
    wgpu_object_destroy(conv_bind_group_layout_);
    wgpu_object_destroy(linear_pipeline_);
    wgpu_object_destroy(conv_pipeline_);
    wgpu_object_destroy(queue_);
    wgpu_object_destroy(device_);
    wgpu_object_destroy(adapter_);
#endif
}

void GpuExecutor::configure(const network::TinyLenetWeights* weights) {
    weights_ = weights;
    requestWebGpuDevice();
}

bool GpuExecutor::ready() const {
    return webgpu_ready_ && network_ready_;
}

int GpuExecutor::infer(const std::vector<uint8_t>& image) {
#ifdef __EMSCRIPTEN__
    if (!ready() || !weights_ || !weights_->valid() || image.size() != INPUT_VALUES) {
        return -1;
    }

    std::array<float, INPUT_VALUES> input{};
    for (std::size_t i = 0; i < image.size(); ++i) {
        input[i] = static_cast<float>(image[i]) / 255.0f;
    }

    wgpu_queue_write_buffer(queue_, input_buffer_, 0, input.data(), input.size() * sizeof(float));

    WGpuCommandEncoder encoder = wgpu_device_create_command_encoder(device_, &WGPU_COMMAND_ENCODER_DESCRIPTOR_DEFAULT_INITIALIZER);

    WGpuComputePassEncoder conv_pass = wgpu_command_encoder_begin_compute_pass(encoder, 0);
    wgpu_compute_pass_encoder_set_pipeline(conv_pass, conv_pipeline_);
    wgpu_compute_pass_encoder_set_bind_group(conv_pass, 0, conv_bind_group_, 0, 0);
    wgpu_compute_pass_encoder_dispatch_workgroups(conv_pass, 4, 4, 4);
    wgpu_compute_pass_encoder_end(conv_pass);

    WGpuComputePassEncoder linear_pass = wgpu_command_encoder_begin_compute_pass(encoder, 0);
    wgpu_compute_pass_encoder_set_pipeline(linear_pass, linear_pipeline_);
    wgpu_compute_pass_encoder_set_bind_group(linear_pass, 0, linear_bind_group_, 0, 0);
    wgpu_compute_pass_encoder_dispatch_workgroups(linear_pass, 1, 1, 1);
    wgpu_compute_pass_encoder_end(linear_pass);

    wgpu_command_encoder_copy_buffer_to_buffer(
        encoder,
        logits_buffer_,
        0,
        readback_buffer_,
        0,
        LOGIT_VALUES * sizeof(float)
    );

    WGpuCommandBuffer command_buffer = wgpu_command_encoder_finish(encoder);
    wgpu_queue_submit_one_and_destroy(queue_, command_buffer);

    std::array<float, LOGIT_VALUES> logits{};
    inference_pending_ = true;
    wgpu_buffer_map_sync(readback_buffer_, WGPU_MAP_MODE_READ, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_get_mapped_range(readback_buffer_, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_read_mapped_range(readback_buffer_, 0, 0, logits.data(), LOGIT_VALUES * sizeof(float));
    wgpu_buffer_unmap(readback_buffer_);

    latest_prediction_ = argmax(logits);
    inference_pending_ = false;
    return latest_prediction_;
#else
    (void)image;
    return -1;
#endif
}

bool GpuExecutor::inferencePending() const {
    return inference_pending_;
}

int GpuExecutor::latestPrediction() const {
    return latest_prediction_;
}

void GpuExecutor::requestWebGpuDevice() {
    if (webgpu_requested_) {
        return;
    }
    webgpu_requested_ = true;

#ifdef __EMSCRIPTEN__
    if (!navigator_gpu_available()) {
        std::cerr << "WebGPU is not available in this browser context." << std::endl;
        return;
    }

    WGpuRequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_DEFAULT_INITIALIZER;
    options.powerPreference = WGPU_POWER_PREFERENCE_HIGH_PERFORMANCE;

    if (!navigator_gpu_request_adapter_async(&options, &GpuExecutor::onAdapter, this)) {
        std::cerr << "Failed to start WebGPU adapter request." << std::endl;
    }
#else
    std::cerr << "wasm_webgpu is only active in Emscripten builds." << std::endl;
#endif
}

#ifdef __EMSCRIPTEN__
void GpuExecutor::onAdapter(WGpuAdapter adapter, void* user_data) {
    auto* self = static_cast<GpuExecutor*>(user_data);
    if (!self || !adapter) {
        std::cerr << "WebGPU adapter request failed." << std::endl;
        return;
    }

    self->adapter_ = adapter;

    WGpuDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_DEFAULT_INITIALIZER;
    wgpu_adapter_request_device_async(adapter, &device_desc, &GpuExecutor::onDevice, self);
}

void GpuExecutor::onDevice(WGpuDevice device, void* user_data) {
    auto* self = static_cast<GpuExecutor*>(user_data);
    if (!self || !device) {
        std::cerr << "WebGPU device request failed." << std::endl;
        return;
    }

    self->device_ = device;
    self->queue_ = wgpu_device_get_queue(device);
    self->webgpu_ready_ = true;
    self->createNetworkResources();
    std::cout << "wasm_webgpu device ready" << std::endl;
}

WGpuBuffer GpuExecutor::createBuffer(std::size_t size, WGPU_BUFFER_USAGE_FLAGS usage) const {
    WGpuBufferDescriptor desc = {};
    desc.size = size;
    desc.usage = usage;
    return wgpu_device_create_buffer(device_, &desc);
}

void GpuExecutor::createNetworkResources() {
    if (network_ready_) {
        return;
    }
    if (!weights_ || !weights_->valid() || !device_ || !queue_) {
        return;
    }

    const uint64_t input_size = INPUT_VALUES * sizeof(float);
    const uint64_t conv_weights_size = weights_->conv_weights.size() * sizeof(float);
    const uint64_t conv_bias_size = weights_->conv_bias.size() * sizeof(float);
    const uint64_t conv_output_size = CONV_VALUES * sizeof(float);
    const uint64_t linear_weights_size = weights_->linear_weights.size() * sizeof(float);
    const uint64_t linear_bias_size = weights_->linear_bias.size() * sizeof(float);
    const uint64_t logits_size = LOGIT_VALUES * sizeof(float);

    input_buffer_ = createBuffer(input_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    conv_weights_buffer_ = createBuffer(conv_weights_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    conv_bias_buffer_ = createBuffer(conv_bias_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    conv_output_buffer_ = createBuffer(conv_output_size, WGPU_BUFFER_USAGE_STORAGE);
    linear_weights_buffer_ = createBuffer(linear_weights_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    linear_bias_buffer_ = createBuffer(linear_bias_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    logits_buffer_ = createBuffer(logits_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC);
    readback_buffer_ = createBuffer(logits_size, WGPU_BUFFER_USAGE_COPY_DST | WGPU_BUFFER_USAGE_MAP_READ);

    wgpu_queue_write_buffer(queue_, conv_weights_buffer_, 0, weights_->conv_weights.data(), conv_weights_size);
    wgpu_queue_write_buffer(queue_, conv_bias_buffer_, 0, weights_->conv_bias.data(), conv_bias_size);
    wgpu_queue_write_buffer(queue_, linear_weights_buffer_, 0, weights_->linear_weights.data(), linear_weights_size);
    wgpu_queue_write_buffer(queue_, linear_bias_buffer_, 0, weights_->linear_bias.data(), linear_bias_size);

    WGpuBindGroupLayoutEntry conv_layout_entries[4] = {
        storageLayoutEntry(0, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, input_size),
        storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, conv_weights_size),
        storageLayoutEntry(2, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, conv_bias_size),
        storageLayoutEntry(3, WGPU_BUFFER_BINDING_TYPE_STORAGE, conv_output_size),
    };
    conv_bind_group_layout_ = wgpu_device_create_bind_group_layout(device_, conv_layout_entries, 4);
    conv_pipeline_layout_ = wgpu_device_create_pipeline_layout(device_, &conv_bind_group_layout_, 1);

    WGpuBindGroupLayoutEntry linear_layout_entries[4] = {
        storageLayoutEntry(0, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, conv_output_size),
        storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, linear_weights_size),
        storageLayoutEntry(2, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, linear_bias_size),
        storageLayoutEntry(3, WGPU_BUFFER_BINDING_TYPE_STORAGE, logits_size),
    };
    linear_bind_group_layout_ = wgpu_device_create_bind_group_layout(device_, linear_layout_entries, 4);
    linear_pipeline_layout_ = wgpu_device_create_pipeline_layout(device_, &linear_bind_group_layout_, 1);

    WGpuShaderModuleDescriptor conv_shader_desc = {};
    conv_shader_desc.code = internal::CONV_RELU_WGSL;
    WGpuShaderModule conv_shader = wgpu_device_create_shader_module(device_, &conv_shader_desc);

    WGpuShaderModuleDescriptor linear_shader_desc = {};
    linear_shader_desc.code = internal::LINEAR_WGSL;
    WGpuShaderModule linear_shader = wgpu_device_create_shader_module(device_, &linear_shader_desc);

    conv_pipeline_ = wgpu_device_create_compute_pipeline(device_, conv_shader, "main", conv_pipeline_layout_, 0, 0);
    linear_pipeline_ = wgpu_device_create_compute_pipeline(device_, linear_shader, "main", linear_pipeline_layout_, 0, 0);

    wgpu_object_destroy(conv_shader);
    wgpu_object_destroy(linear_shader);

    WGpuBindGroupEntry conv_entries[4] = {
        bufferEntry(0, input_buffer_, input_size),
        bufferEntry(1, conv_weights_buffer_, conv_weights_size),
        bufferEntry(2, conv_bias_buffer_, conv_bias_size),
        bufferEntry(3, conv_output_buffer_, conv_output_size),
    };
    conv_bind_group_ = wgpu_device_create_bind_group(device_, conv_bind_group_layout_, conv_entries, 4);

    WGpuBindGroupEntry linear_entries[4] = {
        bufferEntry(0, conv_output_buffer_, conv_output_size),
        bufferEntry(1, linear_weights_buffer_, linear_weights_size),
        bufferEntry(2, linear_bias_buffer_, linear_bias_size),
        bufferEntry(3, logits_buffer_, logits_size),
    };
    linear_bind_group_ = wgpu_device_create_bind_group(device_, linear_bind_group_layout_, linear_entries, 4);

    network_ready_ = conv_pipeline_ && linear_pipeline_ && conv_bind_group_ && linear_bind_group_;
    if (network_ready_) {
        std::cout << "wasm_webgpu tiny_lenet resources ready" << std::endl;
    } else {
        std::cerr << "Failed to create wasm_webgpu tiny_lenet resources" << std::endl;
    }
}
#endif
