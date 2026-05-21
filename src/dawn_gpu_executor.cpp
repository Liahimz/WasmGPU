#include "gpu_executor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>

#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
#include "embedded_shaders.h"

namespace {

constexpr std::size_t INPUT_VALUES = 28 * 28;
constexpr std::size_t CONV_VALUES = 26 * 26 * 4;
constexpr std::size_t LOGIT_VALUES = 10;
constexpr std::size_t LARGE_INPUT_VALUES = 1000 * 500;
constexpr std::size_t LARGE_CONV_VALUES = 996 * 498 * 4;
constexpr std::size_t LARGE_CONV_WEIGHT_VALUES = 4 * 5 * 3;
constexpr std::size_t LARGE_LINEAR_WEIGHT_VALUES = LOGIT_VALUES * LARGE_CONV_VALUES;
constexpr std::size_t LARGE_PARTIAL_CHUNKS = 512;
constexpr std::size_t LARGE_PARTIAL_VALUES = LOGIT_VALUES * LARGE_PARTIAL_CHUNKS;
constexpr std::size_t TINY_TIMESTAMP_VALUES = 4;
constexpr std::size_t LARGE_TIMESTAMP_VALUES = 6;

using Clock = std::chrono::high_resolution_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double nowMs() {
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

WGPUStringView strView(const char* value) {
    return WGPUStringView{value, WGPU_STRLEN};
}

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

float nextWeight(uint32_t& state, float scale) {
    state = state * 1664525u + 1013904223u;
    const float normalized = static_cast<float>((state >> 8) & 0x00ffffffu) / 16777215.0f;
    return (normalized * 2.0f - 1.0f) * scale;
}

std::vector<float> makeSyntheticValues(std::size_t count, uint32_t& state, float scale) {
    std::vector<float> values(count);
    for (float& value : values) {
        value = nextWeight(state, scale);
    }
    return values;
}

std::vector<float> makeSyntheticInput(uint32_t input_seed) {
    uint32_t state = 0x87654321u ^ input_seed;
    return makeSyntheticValues(LARGE_INPUT_VALUES, state, 1.0f);
}

WGPUBindGroupLayoutEntry storageLayoutEntry(uint32_t binding, WGPUBufferBindingType type, uint64_t min_size) {
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = binding;
    entry.visibility = WGPUShaderStage_Compute;
    entry.buffer.type = type;
    entry.buffer.minBindingSize = min_size;
    return entry;
}

WGPUBindGroupEntry bufferEntry(uint32_t binding, WGPUBuffer buffer, uint64_t size) {
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = binding;
    entry.buffer = buffer;
    entry.offset = 0;
    entry.size = size;
    return entry;
}

WGPUComputePassDescriptor timestampPassDescriptor(
    WGPUPassTimestampWrites& writes,
    WGPUQuerySet query_set,
    uint32_t begin_index,
    uint32_t end_index
) {
    writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    writes.querySet = query_set;
    writes.beginningOfPassWriteIndex = begin_index;
    writes.endOfPassWriteIndex = end_index;

    WGPUComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    desc.timestampWrites = &writes;
    return desc;
}

double timestampDeltaMs(uint64_t begin, uint64_t end) {
    return static_cast<double>(end - begin) / 1000000.0;
}

template <typename T, std::size_t Count>
void readMappedArray(WGPUBuffer buffer, std::array<T, Count>& out) {
    const std::size_t byte_size = out.size() * sizeof(T);
    const void* mapped = wgpuBufferGetConstMappedRange(buffer, 0, byte_size);
    if (mapped) {
        std::memcpy(out.data(), mapped, byte_size);
    }
}

} // namespace

GpuExecutor::GpuExecutor() = default;

GpuExecutor::~GpuExecutor() {
    graph_.reset();
    wgpuBufferRelease(readback_buffer_);
    wgpuBufferRelease(timestamp_readback_buffer_);
    wgpuBufferRelease(timestamp_buffer_);
    wgpuQuerySetRelease(timestamp_query_set_);
    wgpuBufferRelease(large_readback_buffer_);
    wgpuBufferRelease(large_timestamp_readback_buffer_);
    wgpuBufferRelease(large_timestamp_buffer_);
    wgpuQuerySetRelease(large_timestamp_query_set_);
    wgpuBufferRelease(large_logits_buffer_);
    wgpuBufferRelease(large_partial_sums_buffer_);
    wgpuBufferRelease(large_linear_bias_buffer_);
    wgpuBufferRelease(large_linear_weights_buffer_);
    wgpuBufferRelease(large_conv_output_buffer_);
    wgpuBufferRelease(large_conv_bias_buffer_);
    wgpuBufferRelease(large_conv_weights_buffer_);
    wgpuBufferRelease(large_input_buffer_);
    wgpuBindGroupRelease(large_linear_reduce_bind_group_);
    wgpuBindGroupRelease(large_linear_partial_bind_group_);
    wgpuBindGroupRelease(large_conv_bind_group_);
    wgpuPipelineLayoutRelease(large_linear_reduce_pipeline_layout_);
    wgpuPipelineLayoutRelease(large_linear_partial_pipeline_layout_);
    wgpuPipelineLayoutRelease(large_conv_pipeline_layout_);
    wgpuBindGroupLayoutRelease(large_linear_reduce_bind_group_layout_);
    wgpuBindGroupLayoutRelease(large_linear_partial_bind_group_layout_);
    wgpuBindGroupLayoutRelease(large_conv_bind_group_layout_);
    wgpuComputePipelineRelease(large_linear_reduce_pipeline_);
    wgpuComputePipelineRelease(large_linear_partial_pipeline_);
    wgpuComputePipelineRelease(large_conv_pipeline_);
    wgpuBufferRelease(logits_buffer_);
    wgpuBufferRelease(linear_bias_buffer_);
    wgpuBufferRelease(linear_weights_buffer_);
    wgpuBufferRelease(conv_output_buffer_);
    wgpuBufferRelease(conv_bias_buffer_);
    wgpuBufferRelease(conv_weights_buffer_);
    wgpuBufferRelease(input_buffer_);
    wgpuBindGroupRelease(linear_bind_group_);
    wgpuBindGroupRelease(conv_bind_group_);
    wgpuPipelineLayoutRelease(linear_pipeline_layout_);
    wgpuPipelineLayoutRelease(conv_pipeline_layout_);
    wgpuBindGroupLayoutRelease(linear_bind_group_layout_);
    wgpuBindGroupLayoutRelease(conv_bind_group_layout_);
    wgpuComputePipelineRelease(linear_pipeline_);
    wgpuComputePipelineRelease(conv_pipeline_);
    wgpuQueueRelease(queue_);
    wgpuDeviceRelease(device_);
    wgpuAdapterRelease(adapter_);
    wgpuInstanceRelease(instance_);
}

void GpuExecutor::configure(const network::TinyLenetWeights* weights) {
    configure(nullptr, weights);
}

void GpuExecutor::configure(const network::ModelDesc* model, const network::TinyLenetWeights* weights) {
    model_ = model;
    weights_ = weights;
    if (model_ && !graph_.configure(*model_)) {
        std::cerr << "Failed to configure Dawn WebGPU graph executor: " << graph_.error() << std::endl;
    }
    requestWebGpuDevice();
}

bool GpuExecutor::ready() const {
    return webgpu_ready_ && network_ready_;
}

void GpuExecutor::prepareSyntheticLargeData() {
    if (large_synthetic_data_ready_) {
        return;
    }

    const auto start = Clock::now();
    uint32_t state = 0x12345678u;
    large_conv_weights_data_ = makeSyntheticValues(LARGE_CONV_WEIGHT_VALUES, state, 0.05f);
    large_conv_bias_data_ = makeSyntheticValues(4, state, 0.01f);
    large_linear_weights_data_ = makeSyntheticValues(LARGE_LINEAR_WEIGHT_VALUES, state, 0.005f);
    large_linear_bias_data_ = makeSyntheticValues(LOGIT_VALUES, state, 0.01f);
    large_synthetic_data_ready_ = true;
    const auto end = Clock::now();

    std::cout << "[timing] dawn_synthetic_gpu_large_data_prepare"
              << " elapsed=" << elapsedMs(start, end)
              << "ms"
              << std::endl;
}

void GpuExecutor::prepareSyntheticLarge() {
    if (webgpu_ready_) {
        createLargeNetworkResources();
    }
}

bool GpuExecutor::inferencePending() const {
    if (pending_kind_ == 2) {
        return inference_pending_;
    }
    if (pending_kind_ == 1 && graph_.ready()) {
        return graph_.inferencePending();
    }
    return inference_pending_;
}

int GpuExecutor::latestPrediction() const {
    if (pending_kind_ == 2) {
        return latest_prediction_;
    }
    if (pending_kind_ == 1 && graph_.ready()) {
        return graph_.latestPrediction();
    }
    return latest_prediction_;
}

const std::vector<float>& GpuExecutor::latestOutput() const {
    return graph_.latestOutput();
}

const char* GpuExecutor::latestBackend() const {
    return latest_backend_;
}

WGPUBuffer GpuExecutor::createBuffer(std::size_t size, WGPUBufferUsage usage) const {
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.size = size;
    desc.usage = usage;
    return wgpuDeviceCreateBuffer(device_, &desc);
}

void GpuExecutor::requestWebGpuDevice() {
    if (webgpu_requested_) {
        return;
    }
    webgpu_requested_ = true;

    WGPUInstanceDescriptor instance_desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
    instance_ = wgpuCreateInstance(&instance_desc);
    if (!instance_) {
        std::cerr << "emdawnwebgpu instance creation failed." << std::endl;
        return;
    }

    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter_request_attempt_ = 1;

    WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.callback = &GpuExecutor::onAdapter;
    callback.userdata1 = this;
    wgpuInstanceRequestAdapter(instance_, &options, callback);
}

void GpuExecutor::onAdapter(
    WGPURequestAdapterStatus status,
    WGPUAdapter adapter,
    WGPUStringView,
    void* userdata1,
    void*
) {
    auto* self = static_cast<GpuExecutor*>(userdata1);
    if (!self || status != WGPURequestAdapterStatus_Success || !adapter) {
        if (self && self->adapter_request_attempt_ == 1) {
            self->adapter_request_attempt_ = 2;
            std::cerr << "High-performance emdawnwebgpu adapter request failed; retrying with default adapter options." << std::endl;

            WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
            WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
            callback.mode = WGPUCallbackMode_AllowSpontaneous;
            callback.callback = &GpuExecutor::onAdapter;
            callback.userdata1 = self;
            wgpuInstanceRequestAdapter(self->instance_, &options, callback);
            return;
        }
        std::cerr << "emdawnwebgpu adapter request failed." << std::endl;
        return;
    }

    self->adapter_ = adapter;

    WGPUDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_INIT;
    WGPUFeatureName timestamp_feature = WGPUFeatureName_TimestampQuery;
    if (wgpuAdapterHasFeature(adapter, timestamp_feature)) {
        device_desc.requiredFeatureCount = 1;
        device_desc.requiredFeatures = &timestamp_feature;
    }

    WGPURequestDeviceCallbackInfo callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.callback = &GpuExecutor::onDevice;
    callback.userdata1 = self;
    wgpuAdapterRequestDevice(adapter, &device_desc, callback);
}

void GpuExecutor::onDevice(
    WGPURequestDeviceStatus status,
    WGPUDevice device,
    WGPUStringView,
    void* userdata1,
    void*
) {
    auto* self = static_cast<GpuExecutor*>(userdata1);
    if (!self || status != WGPURequestDeviceStatus_Success || !device) {
        std::cerr << "emdawnwebgpu device request failed." << std::endl;
        return;
    }

    self->device_ = device;
    self->queue_ = wgpuDeviceGetQueue(device);
    self->timestamp_query_enabled_ = wgpuDeviceHasFeature(device, WGPUFeatureName_TimestampQuery);
    self->webgpu_ready_ = true;
    self->createNetworkResources();

    if (self->timestamp_query_enabled_) {
        std::cout << "emdawnwebgpu timestamp-query ready" << std::endl;
    } else {
        std::cout << "emdawnwebgpu timestamp-query unavailable" << std::endl;
    }
    std::cout << "emdawnwebgpu device ready" << std::endl;
}

WGPUShaderModule createShaderModule(WGPUDevice device, const char* code) {
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = strView(code);

    WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    desc.nextInChain = &source.chain;
    return wgpuDeviceCreateShaderModule(device, &desc);
}

WGPUComputePipeline createComputePipeline(
    WGPUDevice device,
    WGPUShaderModule shader,
    WGPUPipelineLayout layout
) {
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.layout = layout;
    desc.compute.module = shader;
    desc.compute.entryPoint = strView("main");
    return wgpuDeviceCreateComputePipeline(device, &desc);
}

void GpuExecutor::createNetworkResources() {
    if (network_ready_) {
        return;
    }
    if (model_ && model_->valid() && device_ && queue_) {
        if (graph_.attach(device_, queue_) && graph_.prepare()) {
            network_ready_ = true;
            std::cout << "emdawnwebgpu graph resources ready for " << model_->name << std::endl;
            return;
        }
        std::cerr << "Failed to create emdawnwebgpu graph resources: " << graph_.error()
                  << "; falling back to fixed tiny_lenet resources"
                  << std::endl;
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
    const uint64_t timestamp_size = TINY_TIMESTAMP_VALUES * sizeof(uint64_t);

    input_buffer_ = createBuffer(input_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    conv_weights_buffer_ = createBuffer(conv_weights_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    conv_bias_buffer_ = createBuffer(conv_bias_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    conv_output_buffer_ = createBuffer(conv_output_size, WGPUBufferUsage_Storage);
    linear_weights_buffer_ = createBuffer(linear_weights_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    linear_bias_buffer_ = createBuffer(linear_bias_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    logits_buffer_ = createBuffer(logits_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    readback_buffer_ = createBuffer(logits_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    if (timestamp_query_enabled_) {
        WGPUQuerySetDescriptor timestamp_desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        timestamp_desc.type = WGPUQueryType_Timestamp;
        timestamp_desc.count = TINY_TIMESTAMP_VALUES;
        timestamp_query_set_ = wgpuDeviceCreateQuerySet(device_, &timestamp_desc);
        timestamp_buffer_ = createBuffer(timestamp_size, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
        timestamp_readback_buffer_ = createBuffer(timestamp_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    }

    wgpuQueueWriteBuffer(queue_, conv_weights_buffer_, 0, weights_->conv_weights.data(), conv_weights_size);
    wgpuQueueWriteBuffer(queue_, conv_bias_buffer_, 0, weights_->conv_bias.data(), conv_bias_size);
    wgpuQueueWriteBuffer(queue_, linear_weights_buffer_, 0, weights_->linear_weights.data(), linear_weights_size);
    wgpuQueueWriteBuffer(queue_, linear_bias_buffer_, 0, weights_->linear_bias.data(), linear_bias_size);

    WGPUBindGroupLayoutEntry conv_layout_entries[4] = {
        storageLayoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage, input_size),
        storageLayoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage, conv_weights_size),
        storageLayoutEntry(2, WGPUBufferBindingType_ReadOnlyStorage, conv_bias_size),
        storageLayoutEntry(3, WGPUBufferBindingType_Storage, conv_output_size),
    };
    WGPUBindGroupLayoutDescriptor conv_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    conv_bgl_desc.entryCount = 4;
    conv_bgl_desc.entries = conv_layout_entries;
    conv_bind_group_layout_ = wgpuDeviceCreateBindGroupLayout(device_, &conv_bgl_desc);

    WGPUBindGroupLayout conv_layouts[1] = {conv_bind_group_layout_};
    WGPUPipelineLayoutDescriptor conv_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    conv_layout_desc.bindGroupLayoutCount = 1;
    conv_layout_desc.bindGroupLayouts = conv_layouts;
    conv_pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device_, &conv_layout_desc);

    WGPUBindGroupLayoutEntry linear_layout_entries[4] = {
        storageLayoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage, conv_output_size),
        storageLayoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage, linear_weights_size),
        storageLayoutEntry(2, WGPUBufferBindingType_ReadOnlyStorage, linear_bias_size),
        storageLayoutEntry(3, WGPUBufferBindingType_Storage, logits_size),
    };
    WGPUBindGroupLayoutDescriptor linear_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    linear_bgl_desc.entryCount = 4;
    linear_bgl_desc.entries = linear_layout_entries;
    linear_bind_group_layout_ = wgpuDeviceCreateBindGroupLayout(device_, &linear_bgl_desc);

    WGPUBindGroupLayout linear_layouts[1] = {linear_bind_group_layout_};
    WGPUPipelineLayoutDescriptor linear_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    linear_layout_desc.bindGroupLayoutCount = 1;
    linear_layout_desc.bindGroupLayouts = linear_layouts;
    linear_pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device_, &linear_layout_desc);

    WGPUShaderModule conv_shader = createShaderModule(device_, internal::CONV_RELU_WGSL);
    WGPUShaderModule linear_shader = createShaderModule(device_, internal::LINEAR_WGSL);
    conv_pipeline_ = createComputePipeline(device_, conv_shader, conv_pipeline_layout_);
    linear_pipeline_ = createComputePipeline(device_, linear_shader, linear_pipeline_layout_);
    wgpuShaderModuleRelease(conv_shader);
    wgpuShaderModuleRelease(linear_shader);

    WGPUBindGroupEntry conv_entries[4] = {
        bufferEntry(0, input_buffer_, input_size),
        bufferEntry(1, conv_weights_buffer_, conv_weights_size),
        bufferEntry(2, conv_bias_buffer_, conv_bias_size),
        bufferEntry(3, conv_output_buffer_, conv_output_size),
    };
    WGPUBindGroupDescriptor conv_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    conv_bg_desc.layout = conv_bind_group_layout_;
    conv_bg_desc.entryCount = 4;
    conv_bg_desc.entries = conv_entries;
    conv_bind_group_ = wgpuDeviceCreateBindGroup(device_, &conv_bg_desc);

    WGPUBindGroupEntry linear_entries[4] = {
        bufferEntry(0, conv_output_buffer_, conv_output_size),
        bufferEntry(1, linear_weights_buffer_, linear_weights_size),
        bufferEntry(2, linear_bias_buffer_, linear_bias_size),
        bufferEntry(3, logits_buffer_, logits_size),
    };
    WGPUBindGroupDescriptor linear_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    linear_bg_desc.layout = linear_bind_group_layout_;
    linear_bg_desc.entryCount = 4;
    linear_bg_desc.entries = linear_entries;
    linear_bind_group_ = wgpuDeviceCreateBindGroup(device_, &linear_bg_desc);

    network_ready_ = conv_pipeline_ && linear_pipeline_ && conv_bind_group_ && linear_bind_group_;
    if (network_ready_) {
        std::cout << "emdawnwebgpu tiny_lenet resources ready" << std::endl;
    } else {
        std::cerr << "Failed to create emdawnwebgpu tiny_lenet resources" << std::endl;
    }
}

void GpuExecutor::createLargeNetworkResources() {
    if (large_network_ready_) {
        return;
    }
    if (!device_ || !queue_) {
        return;
    }
    prepareSyntheticLargeData();
    if (!large_synthetic_data_ready_) {
        return;
    }

    const auto total_start = Clock::now();
    const uint64_t input_size = LARGE_INPUT_VALUES * sizeof(float);
    const uint64_t conv_weights_size = LARGE_CONV_WEIGHT_VALUES * sizeof(float);
    const uint64_t conv_bias_size = 4 * sizeof(float);
    const uint64_t conv_output_size = LARGE_CONV_VALUES * sizeof(float);
    const uint64_t linear_weights_size = LARGE_LINEAR_WEIGHT_VALUES * sizeof(float);
    const uint64_t linear_bias_size = LOGIT_VALUES * sizeof(float);
    const uint64_t partial_sums_size = LARGE_PARTIAL_VALUES * sizeof(float);
    const uint64_t logits_size = LOGIT_VALUES * sizeof(float);
    const uint64_t timestamp_size = LARGE_TIMESTAMP_VALUES * sizeof(uint64_t);

    large_input_buffer_ = createBuffer(input_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    large_conv_weights_buffer_ = createBuffer(conv_weights_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    large_conv_bias_buffer_ = createBuffer(conv_bias_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    large_conv_output_buffer_ = createBuffer(conv_output_size, WGPUBufferUsage_Storage);
    large_linear_weights_buffer_ = createBuffer(linear_weights_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    large_linear_bias_buffer_ = createBuffer(linear_bias_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    large_partial_sums_buffer_ = createBuffer(partial_sums_size, WGPUBufferUsage_Storage);
    large_logits_buffer_ = createBuffer(logits_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    large_readback_buffer_ = createBuffer(logits_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    const auto buffers_end = Clock::now();

    if (timestamp_query_enabled_) {
        WGPUQuerySetDescriptor timestamp_desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        timestamp_desc.type = WGPUQueryType_Timestamp;
        timestamp_desc.count = LARGE_TIMESTAMP_VALUES;
        large_timestamp_query_set_ = wgpuDeviceCreateQuerySet(device_, &timestamp_desc);
        large_timestamp_buffer_ = createBuffer(timestamp_size, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
        large_timestamp_readback_buffer_ = createBuffer(timestamp_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    }
    const auto timestamp_end = Clock::now();

    const auto upload_start = Clock::now();
    wgpuQueueWriteBuffer(queue_, large_conv_weights_buffer_, 0, large_conv_weights_data_.data(), conv_weights_size);
    wgpuQueueWriteBuffer(queue_, large_conv_bias_buffer_, 0, large_conv_bias_data_.data(), conv_bias_size);
    wgpuQueueWriteBuffer(queue_, large_linear_weights_buffer_, 0, large_linear_weights_data_.data(), linear_weights_size);
    wgpuQueueWriteBuffer(queue_, large_linear_bias_buffer_, 0, large_linear_bias_data_.data(), linear_bias_size);
    const auto data_upload_end = Clock::now();

    WGPUBindGroupLayoutEntry conv_layout_entries[4] = {
        storageLayoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage, input_size),
        storageLayoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage, conv_weights_size),
        storageLayoutEntry(2, WGPUBufferBindingType_ReadOnlyStorage, conv_bias_size),
        storageLayoutEntry(3, WGPUBufferBindingType_Storage, conv_output_size),
    };
    WGPUBindGroupLayoutDescriptor conv_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    conv_bgl_desc.entryCount = 4;
    conv_bgl_desc.entries = conv_layout_entries;
    large_conv_bind_group_layout_ = wgpuDeviceCreateBindGroupLayout(device_, &conv_bgl_desc);
    WGPUBindGroupLayout conv_layouts[1] = {large_conv_bind_group_layout_};
    WGPUPipelineLayoutDescriptor conv_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    conv_layout_desc.bindGroupLayoutCount = 1;
    conv_layout_desc.bindGroupLayouts = conv_layouts;
    large_conv_pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device_, &conv_layout_desc);

    WGPUBindGroupLayoutEntry partial_layout_entries[3] = {
        storageLayoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage, conv_output_size),
        storageLayoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage, linear_weights_size),
        storageLayoutEntry(2, WGPUBufferBindingType_Storage, partial_sums_size),
    };
    WGPUBindGroupLayoutDescriptor partial_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    partial_bgl_desc.entryCount = 3;
    partial_bgl_desc.entries = partial_layout_entries;
    large_linear_partial_bind_group_layout_ = wgpuDeviceCreateBindGroupLayout(device_, &partial_bgl_desc);
    WGPUBindGroupLayout partial_layouts[1] = {large_linear_partial_bind_group_layout_};
    WGPUPipelineLayoutDescriptor partial_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    partial_layout_desc.bindGroupLayoutCount = 1;
    partial_layout_desc.bindGroupLayouts = partial_layouts;
    large_linear_partial_pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device_, &partial_layout_desc);

    WGPUBindGroupLayoutEntry reduce_layout_entries[3] = {
        storageLayoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage, partial_sums_size),
        storageLayoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage, linear_bias_size),
        storageLayoutEntry(2, WGPUBufferBindingType_Storage, logits_size),
    };
    WGPUBindGroupLayoutDescriptor reduce_bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    reduce_bgl_desc.entryCount = 3;
    reduce_bgl_desc.entries = reduce_layout_entries;
    large_linear_reduce_bind_group_layout_ = wgpuDeviceCreateBindGroupLayout(device_, &reduce_bgl_desc);
    WGPUBindGroupLayout reduce_layouts[1] = {large_linear_reduce_bind_group_layout_};
    WGPUPipelineLayoutDescriptor reduce_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    reduce_layout_desc.bindGroupLayoutCount = 1;
    reduce_layout_desc.bindGroupLayouts = reduce_layouts;
    large_linear_reduce_pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device_, &reduce_layout_desc);
    const auto layouts_end = Clock::now();

    WGPUShaderModule conv_shader = createShaderModule(device_, internal::LARGE_CONV_RELU_WGSL);
    WGPUShaderModule partial_shader = createShaderModule(device_, internal::LARGE_LINEAR_PARTIAL_WGSL);
    WGPUShaderModule reduce_shader = createShaderModule(device_, internal::LARGE_LINEAR_REDUCE_WGSL);
    large_conv_pipeline_ = createComputePipeline(device_, conv_shader, large_conv_pipeline_layout_);
    large_linear_partial_pipeline_ = createComputePipeline(device_, partial_shader, large_linear_partial_pipeline_layout_);
    large_linear_reduce_pipeline_ = createComputePipeline(device_, reduce_shader, large_linear_reduce_pipeline_layout_);
    wgpuShaderModuleRelease(conv_shader);
    wgpuShaderModuleRelease(partial_shader);
    wgpuShaderModuleRelease(reduce_shader);
    const auto pipelines_end = Clock::now();

    WGPUBindGroupEntry conv_entries[4] = {
        bufferEntry(0, large_input_buffer_, input_size),
        bufferEntry(1, large_conv_weights_buffer_, conv_weights_size),
        bufferEntry(2, large_conv_bias_buffer_, conv_bias_size),
        bufferEntry(3, large_conv_output_buffer_, conv_output_size),
    };
    WGPUBindGroupDescriptor conv_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    conv_bg_desc.layout = large_conv_bind_group_layout_;
    conv_bg_desc.entryCount = 4;
    conv_bg_desc.entries = conv_entries;
    large_conv_bind_group_ = wgpuDeviceCreateBindGroup(device_, &conv_bg_desc);

    WGPUBindGroupEntry partial_entries[3] = {
        bufferEntry(0, large_conv_output_buffer_, conv_output_size),
        bufferEntry(1, large_linear_weights_buffer_, linear_weights_size),
        bufferEntry(2, large_partial_sums_buffer_, partial_sums_size),
    };
    WGPUBindGroupDescriptor partial_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    partial_bg_desc.layout = large_linear_partial_bind_group_layout_;
    partial_bg_desc.entryCount = 3;
    partial_bg_desc.entries = partial_entries;
    large_linear_partial_bind_group_ = wgpuDeviceCreateBindGroup(device_, &partial_bg_desc);

    WGPUBindGroupEntry reduce_entries[3] = {
        bufferEntry(0, large_partial_sums_buffer_, partial_sums_size),
        bufferEntry(1, large_linear_bias_buffer_, linear_bias_size),
        bufferEntry(2, large_logits_buffer_, logits_size),
    };
    WGPUBindGroupDescriptor reduce_bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    reduce_bg_desc.layout = large_linear_reduce_bind_group_layout_;
    reduce_bg_desc.entryCount = 3;
    reduce_bg_desc.entries = reduce_entries;
    large_linear_reduce_bind_group_ = wgpuDeviceCreateBindGroup(device_, &reduce_bg_desc);

    large_network_ready_ =
        large_conv_pipeline_ &&
        large_linear_partial_pipeline_ &&
        large_linear_reduce_pipeline_ &&
        large_conv_bind_group_ &&
        large_linear_partial_bind_group_ &&
        large_linear_reduce_bind_group_;
    const auto resources_end = Clock::now();

    if (large_network_ready_) {
        std::cout << "emdawnwebgpu synthetic large resources ready" << std::endl;
        std::cout << "[timing] dawn_synthetic_gpu_large_resource_detail"
                  << " buffers=" << elapsedMs(total_start, buffers_end)
                  << "ms timestamp=" << elapsedMs(buffers_end, timestamp_end)
                  << "ms data_upload=" << elapsedMs(upload_start, data_upload_end)
                  << "ms layouts=" << elapsedMs(data_upload_end, layouts_end)
                  << "ms pipelines=" << elapsedMs(layouts_end, pipelines_end)
                  << "ms bind_groups=" << elapsedMs(pipelines_end, resources_end)
                  << "ms total=" << elapsedMs(total_start, resources_end)
                  << "ms"
                  << std::endl;

        large_conv_weights_data_.clear();
        large_conv_bias_data_.clear();
        large_linear_weights_data_.clear();
        large_linear_bias_data_.clear();
        large_conv_weights_data_.shrink_to_fit();
        large_conv_bias_data_.shrink_to_fit();
        large_linear_weights_data_.shrink_to_fit();
        large_linear_bias_data_.shrink_to_fit();
    } else {
        std::cerr << "Failed to create emdawnwebgpu synthetic large resources" << std::endl;
    }
}

int GpuExecutor::infer(const std::vector<uint8_t>& image) {
    if (graph_.ready()) {
        const auto inference_start = Clock::now();
        latest_backend_ = "graph";
        const int prediction = graph_.inferClassBytesAsync(image);
        const auto inference_end = Clock::now();
        pending_kind_ = 1;
        inference_pending_ = graph_.inferencePending();
        std::cout << "[timing] dawn_gpu_graph_async_start"
                  << " submit=" << elapsedMs(inference_start, inference_end)
                  << "ms"
                  << std::endl;
        return prediction;
    }

    if (!ready() || !weights_ || !weights_->valid() || image.size() != INPUT_VALUES) {
        latest_backend_ = "unavailable";
        return -1;
    }
    latest_backend_ = "fixed_lenet";

    const auto input_start = Clock::now();
    std::array<float, INPUT_VALUES> input{};
    for (std::size_t i = 0; i < image.size(); ++i) {
        input[i] = static_cast<float>(image[i]) / 255.0f;
    }
    const auto input_end = Clock::now();

    const auto upload_start = Clock::now();
    wgpuQueueWriteBuffer(queue_, input_buffer_, 0, input.data(), input.size() * sizeof(float));
    const auto upload_end = Clock::now();

    const auto submit_start = Clock::now();
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);

    WGPUPassTimestampWrites conv_writes;
    WGPUComputePassDescriptor conv_pass_desc =
        timestampPassDescriptor(conv_writes, timestamp_query_set_, 0, 1);
    WGPUComputePassEncoder conv_pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        timestamp_query_enabled_ ? &conv_pass_desc : nullptr
    );
    wgpuComputePassEncoderSetPipeline(conv_pass, conv_pipeline_);
    wgpuComputePassEncoderSetBindGroup(conv_pass, 0, conv_bind_group_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(conv_pass, 4, 4, 4);
    wgpuComputePassEncoderEnd(conv_pass);
    wgpuComputePassEncoderRelease(conv_pass);

    WGPUPassTimestampWrites linear_writes;
    WGPUComputePassDescriptor linear_pass_desc =
        timestampPassDescriptor(linear_writes, timestamp_query_set_, 2, 3);
    WGPUComputePassEncoder linear_pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        timestamp_query_enabled_ ? &linear_pass_desc : nullptr
    );
    wgpuComputePassEncoderSetPipeline(linear_pass, linear_pipeline_);
    wgpuComputePassEncoderSetBindGroup(linear_pass, 0, linear_bind_group_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(linear_pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(linear_pass);
    wgpuComputePassEncoderRelease(linear_pass);

    wgpuCommandEncoderCopyBufferToBuffer(encoder, logits_buffer_, 0, readback_buffer_, 0, LOGIT_VALUES * sizeof(float));
    if (timestamp_query_enabled_) {
        wgpuCommandEncoderResolveQuerySet(encoder, timestamp_query_set_, 0, TINY_TIMESTAMP_VALUES, timestamp_buffer_, 0);
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            timestamp_buffer_,
            0,
            timestamp_readback_buffer_,
            0,
            TINY_TIMESTAMP_VALUES * sizeof(uint64_t)
        );
    }

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);
    wgpuQueueSubmit(queue_, 1, &command_buffer);
    wgpuCommandBufferRelease(command_buffer);
    const auto submit_end = Clock::now();

    inference_pending_ = true;
    pending_encode_submit_ms_ = elapsedMs(submit_start, submit_end);
    pending_input_ms_ = elapsedMs(input_start, input_end);
    pending_upload_ms_ = elapsedMs(upload_start, upload_end);
    pending_sync_start_ms_ = nowMs();
    pending_kind_ = 1;

    WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.callback = &GpuExecutor::onTinyReadbackMapped;
    callback.userdata1 = this;
    wgpuBufferMapAsync(readback_buffer_, WGPUMapMode_Read, 0, LOGIT_VALUES * sizeof(float), callback);
    return -1;
}

int GpuExecutor::infer(const std::vector<float>& input) {
    if (graph_.ready()) {
        const auto inference_start = Clock::now();
        latest_backend_ = "graph";
        const int prediction = graph_.inferClassAsync(input);
        const auto inference_end = Clock::now();
        pending_kind_ = 1;
        inference_pending_ = graph_.inferencePending();
        std::cout << "[timing] dawn_gpu_graph_async_start"
                  << " submit=" << elapsedMs(inference_start, inference_end)
                  << "ms"
                  << std::endl;
        return prediction;
    }

    latest_backend_ = "unavailable";
    return -1;
}

int GpuExecutor::benchmarkSyntheticLarge(uint32_t input_seed) {
    if (!webgpu_ready_) {
        return -1;
    }
    createLargeNetworkResources();
    if (!large_network_ready_) {
        return -1;
    }

    const auto input_start = Clock::now();
    std::vector<float> input = makeSyntheticInput(input_seed);
    const auto input_end = Clock::now();

    const auto upload_start = Clock::now();
    wgpuQueueWriteBuffer(queue_, large_input_buffer_, 0, input.data(), input.size() * sizeof(float));
    const auto upload_end = Clock::now();

    const auto submit_start = Clock::now();
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);

    WGPUPassTimestampWrites conv_writes;
    WGPUComputePassDescriptor conv_pass_desc =
        timestampPassDescriptor(conv_writes, large_timestamp_query_set_, 0, 1);
    WGPUComputePassEncoder conv_pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        timestamp_query_enabled_ ? &conv_pass_desc : nullptr
    );
    wgpuComputePassEncoderSetPipeline(conv_pass, large_conv_pipeline_);
    wgpuComputePassEncoderSetBindGroup(conv_pass, 0, large_conv_bind_group_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(conv_pass, 63, 63, 4);
    wgpuComputePassEncoderEnd(conv_pass);
    wgpuComputePassEncoderRelease(conv_pass);

    WGPUPassTimestampWrites partial_writes;
    WGPUComputePassDescriptor partial_pass_desc =
        timestampPassDescriptor(partial_writes, large_timestamp_query_set_, 2, 3);
    WGPUComputePassEncoder partial_pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        timestamp_query_enabled_ ? &partial_pass_desc : nullptr
    );
    wgpuComputePassEncoderSetPipeline(partial_pass, large_linear_partial_pipeline_);
    wgpuComputePassEncoderSetBindGroup(partial_pass, 0, large_linear_partial_bind_group_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(partial_pass, 8, 10, 1);
    wgpuComputePassEncoderEnd(partial_pass);
    wgpuComputePassEncoderRelease(partial_pass);

    WGPUPassTimestampWrites reduce_writes;
    WGPUComputePassDescriptor reduce_pass_desc =
        timestampPassDescriptor(reduce_writes, large_timestamp_query_set_, 4, 5);
    WGPUComputePassEncoder reduce_pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        timestamp_query_enabled_ ? &reduce_pass_desc : nullptr
    );
    wgpuComputePassEncoderSetPipeline(reduce_pass, large_linear_reduce_pipeline_);
    wgpuComputePassEncoderSetBindGroup(reduce_pass, 0, large_linear_reduce_bind_group_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(reduce_pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(reduce_pass);
    wgpuComputePassEncoderRelease(reduce_pass);

    wgpuCommandEncoderCopyBufferToBuffer(encoder, large_logits_buffer_, 0, large_readback_buffer_, 0, LOGIT_VALUES * sizeof(float));
    if (timestamp_query_enabled_) {
        wgpuCommandEncoderResolveQuerySet(encoder, large_timestamp_query_set_, 0, LARGE_TIMESTAMP_VALUES, large_timestamp_buffer_, 0);
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            large_timestamp_buffer_,
            0,
            large_timestamp_readback_buffer_,
            0,
            LARGE_TIMESTAMP_VALUES * sizeof(uint64_t)
        );
    }

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);
    wgpuQueueSubmit(queue_, 1, &command_buffer);
    wgpuCommandBufferRelease(command_buffer);
    const auto submit_end = Clock::now();

    inference_pending_ = true;
    pending_encode_submit_ms_ = elapsedMs(submit_start, submit_end);
    pending_input_ms_ = elapsedMs(input_start, input_end);
    pending_upload_ms_ = elapsedMs(upload_start, upload_end);
    pending_sync_start_ms_ = nowMs();
    pending_kind_ = 2;

    WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.callback = &GpuExecutor::onLargeReadbackMapped;
    callback.userdata1 = this;
    wgpuBufferMapAsync(large_readback_buffer_, WGPUMapMode_Read, 0, LOGIT_VALUES * sizeof(float), callback);
    return -1;
}

void GpuExecutor::onTinyReadbackMapped(WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) {
    auto* self = static_cast<GpuExecutor*>(userdata1);
    if (self) {
        self->finishTinyAsyncReadback();
    }
}

void GpuExecutor::onTinyTimestampMapped(WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) {
    auto* self = static_cast<GpuExecutor*>(userdata1);
    if (self) {
        self->finishTinyAsyncTimestamp();
    }
}

void GpuExecutor::onLargeReadbackMapped(WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) {
    auto* self = static_cast<GpuExecutor*>(userdata1);
    if (self) {
        self->finishLargeAsyncReadback();
    }
}

void GpuExecutor::onLargeTimestampMapped(WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) {
    auto* self = static_cast<GpuExecutor*>(userdata1);
    if (self) {
        self->finishLargeAsyncTimestamp();
    }
}

void GpuExecutor::finishTinyAsyncReadback() {
    std::array<float, LOGIT_VALUES> logits{};
    readMappedArray(readback_buffer_, logits);
    wgpuBufferUnmap(readback_buffer_);
    latest_prediction_ = argmax(logits);

    if (timestamp_query_enabled_) {
        WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        callback.mode = WGPUCallbackMode_AllowSpontaneous;
        callback.callback = &GpuExecutor::onTinyTimestampMapped;
        callback.userdata1 = this;
        wgpuBufferMapAsync(timestamp_readback_buffer_, WGPUMapMode_Read, 0, TINY_TIMESTAMP_VALUES * sizeof(uint64_t), callback);
        return;
    }

    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] dawn_gpu_detail"
              << " input_convert=" << pending_input_ms_
              << "ms upload=" << pending_upload_ms_
              << "ms encode_submit=" << pending_encode_submit_ms_
              << "ms sync_readback=" << sync_readback_ms
              << "ms gpu_timestamp=unavailable"
              << std::endl;
    inference_pending_ = false;
}

void GpuExecutor::finishTinyAsyncTimestamp() {
    std::array<uint64_t, TINY_TIMESTAMP_VALUES> timestamps{};
    readMappedArray(timestamp_readback_buffer_, timestamps);
    wgpuBufferUnmap(timestamp_readback_buffer_);

    const double conv_ms = timestampDeltaMs(timestamps[0], timestamps[1]);
    const double linear_ms = timestampDeltaMs(timestamps[2], timestamps[3]);
    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] dawn_gpu_detail"
              << " input_convert=" << pending_input_ms_
              << "ms upload=" << pending_upload_ms_
              << "ms encode_submit=" << pending_encode_submit_ms_
              << "ms sync_readback=" << sync_readback_ms
              << "ms gpu_conv=" << conv_ms
              << "ms gpu_linear=" << linear_ms
              << "ms gpu_total=" << (conv_ms + linear_ms)
              << "ms"
              << std::endl;
    inference_pending_ = false;
}

void GpuExecutor::finishLargeAsyncReadback() {
    std::array<float, LOGIT_VALUES> logits{};
    readMappedArray(large_readback_buffer_, logits);
    wgpuBufferUnmap(large_readback_buffer_);
    latest_prediction_ = argmax(logits);

    if (timestamp_query_enabled_) {
        WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        callback.mode = WGPUCallbackMode_AllowSpontaneous;
        callback.callback = &GpuExecutor::onLargeTimestampMapped;
        callback.userdata1 = this;
        wgpuBufferMapAsync(large_timestamp_readback_buffer_, WGPUMapMode_Read, 0, LARGE_TIMESTAMP_VALUES * sizeof(uint64_t), callback);
        return;
    }

    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] dawn_synthetic_gpu_large_detail"
              << " input_generation=" << pending_input_ms_
              << "ms upload=" << pending_upload_ms_
              << "ms encode_submit=" << pending_encode_submit_ms_
              << "ms sync_readback=" << sync_readback_ms
              << "ms gpu_timestamp=unavailable"
              << std::endl;
    inference_pending_ = false;
}

void GpuExecutor::finishLargeAsyncTimestamp() {
    std::array<uint64_t, LARGE_TIMESTAMP_VALUES> timestamps{};
    readMappedArray(large_timestamp_readback_buffer_, timestamps);
    wgpuBufferUnmap(large_timestamp_readback_buffer_);

    const double conv_ms = timestampDeltaMs(timestamps[0], timestamps[1]);
    const double partial_ms = timestampDeltaMs(timestamps[2], timestamps[3]);
    const double reduce_ms = timestampDeltaMs(timestamps[4], timestamps[5]);
    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] dawn_synthetic_gpu_large_detail"
              << " input_generation=" << pending_input_ms_
              << "ms upload=" << pending_upload_ms_
              << "ms encode_submit=" << pending_encode_submit_ms_
              << "ms sync_readback=" << sync_readback_ms
              << "ms gpu_conv=" << conv_ms
              << "ms gpu_linear_partial=" << partial_ms
              << "ms gpu_linear_reduce=" << reduce_ms
              << "ms gpu_total=" << (conv_ms + partial_ms + reduce_ms)
              << "ms"
              << std::endl;
    inference_pending_ = false;
}

#else

GpuExecutor::GpuExecutor() = default;
GpuExecutor::~GpuExecutor() = default;
void GpuExecutor::configure(const network::TinyLenetWeights*) {}
void GpuExecutor::configure(const network::ModelDesc*, const network::TinyLenetWeights*) {}
bool GpuExecutor::ready() const { return false; }
int GpuExecutor::infer(const std::vector<uint8_t>&) { return -1; }
int GpuExecutor::infer(const std::vector<float>&) { return -1; }
void GpuExecutor::prepareSyntheticLargeData() {}
void GpuExecutor::prepareSyntheticLarge() {}
int GpuExecutor::benchmarkSyntheticLarge(uint32_t) { return -1; }
bool GpuExecutor::inferencePending() const { return false; }
int GpuExecutor::latestPrediction() const { return -1; }
const std::vector<float>& GpuExecutor::latestOutput() const {
    static const std::vector<float> empty;
    return empty;
}
const char* GpuExecutor::latestBackend() const { return "unavailable"; }

#endif
