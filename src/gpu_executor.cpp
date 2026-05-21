#include "gpu_executor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>

#ifdef __EMSCRIPTEN__
#include "embedded_shaders.h"
#include "lib_webgpu.h"

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

WGpuComputePassDescriptor timestampPassDescriptor(WGpuQuerySet query_set, int begin_index, int end_index) {
    WGpuComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_DEFAULT_INITIALIZER;
    desc.timestampWrites.querySet = query_set;
    desc.timestampWrites.beginningOfPassWriteIndex = begin_index;
    desc.timestampWrites.endOfPassWriteIndex = end_index;
    return desc;
}

double timestampDeltaMs(uint64_t begin, uint64_t end) {
    return static_cast<double>(end - begin) / 1000000.0;
}

} // namespace
#endif

GpuExecutor::GpuExecutor() = default;

GpuExecutor::~GpuExecutor() {
#ifdef __EMSCRIPTEN__
    graph_.reset();
    wgpu_object_destroy(readback_buffer_);
    wgpu_object_destroy(timestamp_readback_buffer_);
    wgpu_object_destroy(timestamp_buffer_);
    wgpu_object_destroy(timestamp_query_set_);
    wgpu_object_destroy(large_readback_buffer_);
    wgpu_object_destroy(large_timestamp_readback_buffer_);
    wgpu_object_destroy(large_timestamp_buffer_);
    wgpu_object_destroy(large_timestamp_query_set_);
    wgpu_object_destroy(large_logits_buffer_);
    wgpu_object_destroy(large_partial_sums_buffer_);
    wgpu_object_destroy(large_linear_bias_buffer_);
    wgpu_object_destroy(large_linear_weights_buffer_);
    wgpu_object_destroy(large_conv_output_buffer_);
    wgpu_object_destroy(large_conv_bias_buffer_);
    wgpu_object_destroy(large_conv_weights_buffer_);
    wgpu_object_destroy(large_input_buffer_);
    wgpu_object_destroy(large_linear_reduce_bind_group_);
    wgpu_object_destroy(large_linear_partial_bind_group_);
    wgpu_object_destroy(large_conv_bind_group_);
    wgpu_object_destroy(large_linear_reduce_pipeline_layout_);
    wgpu_object_destroy(large_linear_partial_pipeline_layout_);
    wgpu_object_destroy(large_conv_pipeline_layout_);
    wgpu_object_destroy(large_linear_reduce_bind_group_layout_);
    wgpu_object_destroy(large_linear_partial_bind_group_layout_);
    wgpu_object_destroy(large_conv_bind_group_layout_);
    wgpu_object_destroy(large_linear_reduce_pipeline_);
    wgpu_object_destroy(large_linear_partial_pipeline_);
    wgpu_object_destroy(large_conv_pipeline_);
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

int GpuExecutor::benchmarkSyntheticLarge(uint32_t input_seed) {
#ifdef __EMSCRIPTEN__
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
    wgpu_queue_write_buffer(queue_, large_input_buffer_, 0, input.data(), input.size() * sizeof(float));
    const auto upload_end = Clock::now();

    const auto submit_start = Clock::now();
    WGpuCommandEncoder encoder = wgpu_device_create_command_encoder(device_, &WGPU_COMMAND_ENCODER_DESCRIPTOR_DEFAULT_INITIALIZER);

    WGpuComputePassDescriptor conv_pass_desc = timestampPassDescriptor(large_timestamp_query_set_, 0, 1);
    WGpuComputePassEncoder conv_pass = wgpu_command_encoder_begin_compute_pass(
        encoder,
        timestamp_query_enabled_ ? &conv_pass_desc : 0
    );
    wgpu_compute_pass_encoder_set_pipeline(conv_pass, large_conv_pipeline_);
    wgpu_compute_pass_encoder_set_bind_group(conv_pass, 0, large_conv_bind_group_, 0, 0);
    wgpu_compute_pass_encoder_dispatch_workgroups(conv_pass, 63, 63, 4);
    wgpu_compute_pass_encoder_end(conv_pass);

    WGpuComputePassDescriptor linear_partial_pass_desc = timestampPassDescriptor(large_timestamp_query_set_, 2, 3);
    WGpuComputePassEncoder linear_partial_pass = wgpu_command_encoder_begin_compute_pass(
        encoder,
        timestamp_query_enabled_ ? &linear_partial_pass_desc : 0
    );
    wgpu_compute_pass_encoder_set_pipeline(linear_partial_pass, large_linear_partial_pipeline_);
    wgpu_compute_pass_encoder_set_bind_group(linear_partial_pass, 0, large_linear_partial_bind_group_, 0, 0);
    wgpu_compute_pass_encoder_dispatch_workgroups(linear_partial_pass, 8, 10, 1);
    wgpu_compute_pass_encoder_end(linear_partial_pass);

    WGpuComputePassDescriptor linear_reduce_pass_desc = timestampPassDescriptor(large_timestamp_query_set_, 4, 5);
    WGpuComputePassEncoder linear_reduce_pass = wgpu_command_encoder_begin_compute_pass(
        encoder,
        timestamp_query_enabled_ ? &linear_reduce_pass_desc : 0
    );
    wgpu_compute_pass_encoder_set_pipeline(linear_reduce_pass, large_linear_reduce_pipeline_);
    wgpu_compute_pass_encoder_set_bind_group(linear_reduce_pass, 0, large_linear_reduce_bind_group_, 0, 0);
    wgpu_compute_pass_encoder_dispatch_workgroups(linear_reduce_pass, 1, 1, 1);
    wgpu_compute_pass_encoder_end(linear_reduce_pass);

    wgpu_command_encoder_copy_buffer_to_buffer(
        encoder,
        large_logits_buffer_,
        0,
        large_readback_buffer_,
        0,
        LOGIT_VALUES * sizeof(float)
    );
    if (timestamp_query_enabled_) {
        wgpu_command_encoder_resolve_query_set(
            encoder,
            large_timestamp_query_set_,
            0,
            LARGE_TIMESTAMP_VALUES,
            large_timestamp_buffer_,
            0
        );
        wgpu_command_encoder_copy_buffer_to_buffer(
            encoder,
            large_timestamp_buffer_,
            0,
            large_timestamp_readback_buffer_,
            0,
            LARGE_TIMESTAMP_VALUES * sizeof(uint64_t)
        );
    }

    WGpuCommandBuffer command_buffer = wgpu_command_encoder_finish(encoder);
    wgpu_queue_submit_one_and_destroy(queue_, command_buffer);
    const auto submit_end = Clock::now();

    std::array<float, LOGIT_VALUES> logits{};
    inference_pending_ = true;
#if defined(BUILD_WASM_WEBGPU_ASYNC)
    pending_encode_submit_ms_ = elapsedMs(submit_start, submit_end);
    pending_input_ms_ = elapsedMs(input_start, input_end);
    pending_upload_ms_ = elapsedMs(upload_start, upload_end);
    pending_sync_start_ms_ = nowMs();
    pending_kind_ = 2;
    wgpu_buffer_map_async(
        large_readback_buffer_,
        &GpuExecutor::onLargeReadbackMapped,
        this,
        WGPU_MAP_MODE_READ,
        0,
        LOGIT_VALUES * sizeof(float)
    );
    return -1;
#else
    const auto sync_start = Clock::now();
    wgpu_buffer_map_sync(large_readback_buffer_, WGPU_MAP_MODE_READ, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_get_mapped_range(large_readback_buffer_, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_read_mapped_range(large_readback_buffer_, 0, 0, logits.data(), LOGIT_VALUES * sizeof(float));
    wgpu_buffer_unmap(large_readback_buffer_);
    std::array<uint64_t, LARGE_TIMESTAMP_VALUES> timestamps{};
    if (timestamp_query_enabled_) {
        wgpu_buffer_map_sync(large_timestamp_readback_buffer_, WGPU_MAP_MODE_READ, 0, LARGE_TIMESTAMP_VALUES * sizeof(uint64_t));
        wgpu_buffer_get_mapped_range(large_timestamp_readback_buffer_, 0, LARGE_TIMESTAMP_VALUES * sizeof(uint64_t));
        wgpu_buffer_read_mapped_range(large_timestamp_readback_buffer_, 0, 0, timestamps.data(), LARGE_TIMESTAMP_VALUES * sizeof(uint64_t));
        wgpu_buffer_unmap(large_timestamp_readback_buffer_);
    }
    const auto sync_end = Clock::now();

    latest_prediction_ = argmax(logits);
    inference_pending_ = false;
    std::cout << "[timing] synthetic_gpu_large_detail"
              << " input_generation=" << elapsedMs(input_start, input_end)
              << "ms upload=" << elapsedMs(upload_start, upload_end)
              << "ms encode_submit=" << elapsedMs(submit_start, submit_end)
              << "ms sync_readback=" << elapsedMs(sync_start, sync_end)
              << "ms";
    if (timestamp_query_enabled_) {
        const double conv_ms = timestampDeltaMs(timestamps[0], timestamps[1]);
        const double partial_ms = timestampDeltaMs(timestamps[2], timestamps[3]);
        const double reduce_ms = timestampDeltaMs(timestamps[4], timestamps[5]);
        std::cout << " gpu_conv=" << conv_ms
                  << "ms gpu_linear_partial=" << partial_ms
                  << "ms gpu_linear_reduce=" << reduce_ms
                  << "ms gpu_total=" << (conv_ms + partial_ms + reduce_ms)
                  << "ms";
    } else {
        std::cout << " gpu_timestamp=unavailable";
    }
    std::cout << std::endl;
    return latest_prediction_;
#endif
#else
    return -1;
#endif
}

void GpuExecutor::configure(const network::TinyLenetWeights* weights) {
    configure(nullptr, weights);
}

void GpuExecutor::configure(const network::ModelDesc* model, const network::TinyLenetWeights* weights) {
    model_ = model;
    weights_ = weights;
    if (model_ && !graph_.configure(*model_)) {
        std::cerr << "Failed to configure WebGPU graph executor: " << graph_.error() << std::endl;
    }
    requestWebGpuDevice();
}

bool GpuExecutor::ready() const {
    return webgpu_ready_ && network_ready_;
}

int GpuExecutor::infer(const std::vector<uint8_t>& image) {
#ifdef __EMSCRIPTEN__
#if !defined(BUILD_EMDAWN_WEBGPU)
    if (graph_.ready()) {
        const auto inference_start = Clock::now();
        latest_backend_ = "graph";
#if defined(BUILD_WASM_WEBGPU_ASYNC)
        const int prediction = graph_.inferClassBytesAsync(image);
#else
        latest_prediction_ = graph_.inferClassBytes(image);
#endif
        const auto inference_end = Clock::now();
#if defined(BUILD_WASM_WEBGPU_ASYNC)
        pending_kind_ = 1;
        inference_pending_ = graph_.inferencePending();
        std::cout << "[timing] gpu_graph_async_start"
                  << " submit=" << elapsedMs(inference_start, inference_end)
                  << "ms"
                  << std::endl;
        return prediction;
#else
        inference_pending_ = false;
        std::cout << "[timing] gpu_graph_detail"
                  << " inference=" << elapsedMs(inference_start, inference_end)
                  << "ms prediction=" << latest_prediction_
                  << std::endl;
        return latest_prediction_;
#endif
    }
#endif

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
    wgpu_queue_write_buffer(queue_, input_buffer_, 0, input.data(), input.size() * sizeof(float));
    const auto upload_end = Clock::now();

    const auto submit_start = Clock::now();
    WGpuCommandEncoder encoder = wgpu_device_create_command_encoder(device_, &WGPU_COMMAND_ENCODER_DESCRIPTOR_DEFAULT_INITIALIZER);

    WGpuComputePassDescriptor conv_pass_desc = timestampPassDescriptor(timestamp_query_set_, 0, 1);
    WGpuComputePassEncoder conv_pass = wgpu_command_encoder_begin_compute_pass(
        encoder,
        timestamp_query_enabled_ ? &conv_pass_desc : 0
    );
    wgpu_compute_pass_encoder_set_pipeline(conv_pass, conv_pipeline_);
    wgpu_compute_pass_encoder_set_bind_group(conv_pass, 0, conv_bind_group_, 0, 0);
    wgpu_compute_pass_encoder_dispatch_workgroups(conv_pass, 4, 4, 4);
    wgpu_compute_pass_encoder_end(conv_pass);

    WGpuComputePassDescriptor linear_pass_desc = timestampPassDescriptor(timestamp_query_set_, 2, 3);
    WGpuComputePassEncoder linear_pass = wgpu_command_encoder_begin_compute_pass(
        encoder,
        timestamp_query_enabled_ ? &linear_pass_desc : 0
    );
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
    if (timestamp_query_enabled_) {
        wgpu_command_encoder_resolve_query_set(
            encoder,
            timestamp_query_set_,
            0,
            TINY_TIMESTAMP_VALUES,
            timestamp_buffer_,
            0
        );
        wgpu_command_encoder_copy_buffer_to_buffer(
            encoder,
            timestamp_buffer_,
            0,
            timestamp_readback_buffer_,
            0,
            TINY_TIMESTAMP_VALUES * sizeof(uint64_t)
        );
    }

    WGpuCommandBuffer command_buffer = wgpu_command_encoder_finish(encoder);
    wgpu_queue_submit_one_and_destroy(queue_, command_buffer);
    const auto submit_end = Clock::now();

    std::array<float, LOGIT_VALUES> logits{};
    inference_pending_ = true;
#if defined(BUILD_WASM_WEBGPU_ASYNC)
    pending_encode_submit_ms_ = elapsedMs(submit_start, submit_end);
    pending_input_ms_ = elapsedMs(input_start, input_end);
    pending_upload_ms_ = elapsedMs(upload_start, upload_end);
    pending_sync_start_ms_ = nowMs();
    pending_kind_ = 1;
    wgpu_buffer_map_async(
        readback_buffer_,
        &GpuExecutor::onTinyReadbackMapped,
        this,
        WGPU_MAP_MODE_READ,
        0,
        LOGIT_VALUES * sizeof(float)
    );
    return -1;
#else
    const auto sync_start = Clock::now();
    wgpu_buffer_map_sync(readback_buffer_, WGPU_MAP_MODE_READ, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_get_mapped_range(readback_buffer_, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_read_mapped_range(readback_buffer_, 0, 0, logits.data(), LOGIT_VALUES * sizeof(float));
    wgpu_buffer_unmap(readback_buffer_);
    std::array<uint64_t, TINY_TIMESTAMP_VALUES> timestamps{};
    if (timestamp_query_enabled_) {
        wgpu_buffer_map_sync(timestamp_readback_buffer_, WGPU_MAP_MODE_READ, 0, TINY_TIMESTAMP_VALUES * sizeof(uint64_t));
        wgpu_buffer_get_mapped_range(timestamp_readback_buffer_, 0, TINY_TIMESTAMP_VALUES * sizeof(uint64_t));
        wgpu_buffer_read_mapped_range(timestamp_readback_buffer_, 0, 0, timestamps.data(), TINY_TIMESTAMP_VALUES * sizeof(uint64_t));
        wgpu_buffer_unmap(timestamp_readback_buffer_);
    }
    const auto sync_end = Clock::now();

    latest_prediction_ = argmax(logits);
    inference_pending_ = false;
    std::cout << "[timing] gpu_detail"
              << " input_convert=" << elapsedMs(input_start, input_end)
              << "ms upload=" << elapsedMs(upload_start, upload_end)
              << "ms encode_submit=" << elapsedMs(submit_start, submit_end)
              << "ms sync_readback=" << elapsedMs(sync_start, sync_end)
              << "ms";
    if (timestamp_query_enabled_) {
        const double conv_ms = timestampDeltaMs(timestamps[0], timestamps[1]);
        const double linear_ms = timestampDeltaMs(timestamps[2], timestamps[3]);
        std::cout << " gpu_conv=" << conv_ms
                  << "ms gpu_linear=" << linear_ms
                  << "ms gpu_total=" << (conv_ms + linear_ms)
                  << "ms";
    } else {
        std::cout << " gpu_timestamp=unavailable";
    }
    std::cout << std::endl;
    return latest_prediction_;
#endif
#else
    (void)image;
    return -1;
#endif
}

void GpuExecutor::prepareSyntheticLarge() {
#ifdef __EMSCRIPTEN__
    if (webgpu_ready_) {
        createLargeNetworkResources();
    }
#endif
}

void GpuExecutor::prepareSyntheticLargeData() {
#ifdef __EMSCRIPTEN__
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

    std::cout << "[timing] synthetic_gpu_large_data_prepare"
              << " elapsed=" << elapsedMs(start, end)
              << "ms"
              << std::endl;
#endif
}

bool GpuExecutor::inferencePending() const {
#if defined(__EMSCRIPTEN__) && defined(BUILD_WASM_WEBGPU_ASYNC) && !defined(BUILD_EMDAWN_WEBGPU)
    if (pending_kind_ == 2) {
        return inference_pending_;
    }
    if (pending_kind_ == 1 && graph_.ready()) {
        return graph_.inferencePending();
    }
#endif
    return inference_pending_;
}

int GpuExecutor::latestPrediction() const {
#if defined(__EMSCRIPTEN__) && defined(BUILD_WASM_WEBGPU_ASYNC) && !defined(BUILD_EMDAWN_WEBGPU)
    if (pending_kind_ == 2) {
        return latest_prediction_;
    }
    if (pending_kind_ == 1 && graph_.ready()) {
        return graph_.latestPrediction();
    }
#endif
    return latest_prediction_;
}

const char* GpuExecutor::latestBackend() const {
    return latest_backend_;
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
    adapter_request_attempt_ = 1;

    if (!navigator_gpu_request_adapter_async(&options, &GpuExecutor::onAdapter, this)) {
        std::cerr << "Failed to start WebGPU adapter request." << std::endl;
    }
#else
    std::cerr << "wasm_webgpu is only active in Emscripten builds." << std::endl;
#endif
}

#ifdef __EMSCRIPTEN__
#if defined(BUILD_WASM_WEBGPU_ASYNC)
void GpuExecutor::onTinyReadbackMapped(WGpuBuffer, void* user_data, WGPU_MAP_MODE_FLAGS, double_int53_t, double_int53_t) {
    auto* self = static_cast<GpuExecutor*>(user_data);
    if (self) {
        self->finishTinyAsyncReadback();
    }
}

void GpuExecutor::onTinyTimestampMapped(WGpuBuffer, void* user_data, WGPU_MAP_MODE_FLAGS, double_int53_t, double_int53_t) {
    auto* self = static_cast<GpuExecutor*>(user_data);
    if (self) {
        self->finishTinyAsyncTimestamp();
    }
}

void GpuExecutor::onLargeReadbackMapped(WGpuBuffer, void* user_data, WGPU_MAP_MODE_FLAGS, double_int53_t, double_int53_t) {
    auto* self = static_cast<GpuExecutor*>(user_data);
    if (self) {
        self->finishLargeAsyncReadback();
    }
}

void GpuExecutor::onLargeTimestampMapped(WGpuBuffer, void* user_data, WGPU_MAP_MODE_FLAGS, double_int53_t, double_int53_t) {
    auto* self = static_cast<GpuExecutor*>(user_data);
    if (self) {
        self->finishLargeAsyncTimestamp();
    }
}

void GpuExecutor::finishTinyAsyncReadback() {
    std::array<float, LOGIT_VALUES> logits{};
    wgpu_buffer_get_mapped_range(readback_buffer_, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_read_mapped_range(readback_buffer_, 0, 0, logits.data(), LOGIT_VALUES * sizeof(float));
    wgpu_buffer_unmap(readback_buffer_);
    latest_prediction_ = argmax(logits);

    if (timestamp_query_enabled_) {
        wgpu_buffer_map_async(
            timestamp_readback_buffer_,
            &GpuExecutor::onTinyTimestampMapped,
            this,
            WGPU_MAP_MODE_READ,
            0,
            TINY_TIMESTAMP_VALUES * sizeof(uint64_t)
        );
        return;
    }

    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] gpu_detail"
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
    wgpu_buffer_get_mapped_range(timestamp_readback_buffer_, 0, TINY_TIMESTAMP_VALUES * sizeof(uint64_t));
    wgpu_buffer_read_mapped_range(timestamp_readback_buffer_, 0, 0, timestamps.data(), TINY_TIMESTAMP_VALUES * sizeof(uint64_t));
    wgpu_buffer_unmap(timestamp_readback_buffer_);

    const double conv_ms = timestampDeltaMs(timestamps[0], timestamps[1]);
    const double linear_ms = timestampDeltaMs(timestamps[2], timestamps[3]);
    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] gpu_detail"
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
    wgpu_buffer_get_mapped_range(large_readback_buffer_, 0, LOGIT_VALUES * sizeof(float));
    wgpu_buffer_read_mapped_range(large_readback_buffer_, 0, 0, logits.data(), LOGIT_VALUES * sizeof(float));
    wgpu_buffer_unmap(large_readback_buffer_);
    latest_prediction_ = argmax(logits);

    if (timestamp_query_enabled_) {
        wgpu_buffer_map_async(
            large_timestamp_readback_buffer_,
            &GpuExecutor::onLargeTimestampMapped,
            this,
            WGPU_MAP_MODE_READ,
            0,
            LARGE_TIMESTAMP_VALUES * sizeof(uint64_t)
        );
        return;
    }

    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] synthetic_gpu_large_detail"
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
    wgpu_buffer_get_mapped_range(large_timestamp_readback_buffer_, 0, LARGE_TIMESTAMP_VALUES * sizeof(uint64_t));
    wgpu_buffer_read_mapped_range(large_timestamp_readback_buffer_, 0, 0, timestamps.data(), LARGE_TIMESTAMP_VALUES * sizeof(uint64_t));
    wgpu_buffer_unmap(large_timestamp_readback_buffer_);

    const double conv_ms = timestampDeltaMs(timestamps[0], timestamps[1]);
    const double partial_ms = timestampDeltaMs(timestamps[2], timestamps[3]);
    const double reduce_ms = timestampDeltaMs(timestamps[4], timestamps[5]);
    const double sync_readback_ms = nowMs() - pending_sync_start_ms_;
    std::cout << "[timing] synthetic_gpu_large_detail"
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
#endif

void GpuExecutor::onAdapter(WGpuAdapter adapter, void* user_data) {
    auto* self = static_cast<GpuExecutor*>(user_data);
    if (!self || !adapter) {
        if (self && self->adapter_request_attempt_ == 1) {
            self->adapter_request_attempt_ = 2;
            std::cerr << "High-performance WebGPU adapter request failed; retrying with default adapter options." << std::endl;
            WGpuRequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_DEFAULT_INITIALIZER;
            if (!navigator_gpu_request_adapter_async(&options, &GpuExecutor::onAdapter, self)) {
                std::cerr << "Failed to start fallback WebGPU adapter request." << std::endl;
            }
            return;
        }
        std::cerr << "WebGPU adapter request failed." << std::endl;
        return;
    }

    self->adapter_ = adapter;

    WGpuDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_DEFAULT_INITIALIZER;
    if (wgpu_adapter_or_device_supports_feature(adapter, WGPU_FEATURE_TIMESTAMP_QUERY)) {
        device_desc.requiredFeatures = WGPU_FEATURE_TIMESTAMP_QUERY;
    }
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
    self->timestamp_query_enabled_ = wgpu_adapter_or_device_supports_feature(device, WGPU_FEATURE_TIMESTAMP_QUERY);
    self->webgpu_ready_ = true;
    self->createNetworkResources();
    if (self->timestamp_query_enabled_) {
        std::cout << "wasm_webgpu timestamp-query ready" << std::endl;
    } else {
        std::cout << "wasm_webgpu timestamp-query unavailable" << std::endl;
    }
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
#if !defined(BUILD_EMDAWN_WEBGPU)
    if (model_ && model_->valid() && device_ && queue_) {
        if (graph_.attach(device_, queue_) && graph_.prepare()) {
            network_ready_ = true;
            std::cout << "wasm_webgpu graph resources ready for " << model_->name << std::endl;
            return;
        }
        std::cerr << "Failed to create wasm_webgpu graph resources: " << graph_.error()
                  << "; falling back to fixed tiny_lenet resources"
                  << std::endl;
    }
#endif

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

    input_buffer_ = createBuffer(input_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    conv_weights_buffer_ = createBuffer(conv_weights_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    conv_bias_buffer_ = createBuffer(conv_bias_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    conv_output_buffer_ = createBuffer(conv_output_size, WGPU_BUFFER_USAGE_STORAGE);
    linear_weights_buffer_ = createBuffer(linear_weights_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    linear_bias_buffer_ = createBuffer(linear_bias_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    logits_buffer_ = createBuffer(logits_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC);
    readback_buffer_ = createBuffer(logits_size, WGPU_BUFFER_USAGE_COPY_DST | WGPU_BUFFER_USAGE_MAP_READ);
    if (timestamp_query_enabled_) {
        WGpuQuerySetDescriptor timestamp_desc = {};
        timestamp_desc.type = WGPU_QUERY_TYPE_TIMESTAMP;
        timestamp_desc.count = TINY_TIMESTAMP_VALUES;
        timestamp_query_set_ = wgpu_device_create_query_set(device_, &timestamp_desc);
        timestamp_buffer_ = createBuffer(timestamp_size, WGPU_BUFFER_USAGE_QUERY_RESOLVE | WGPU_BUFFER_USAGE_COPY_SRC);
        timestamp_readback_buffer_ = createBuffer(timestamp_size, WGPU_BUFFER_USAGE_COPY_DST | WGPU_BUFFER_USAGE_MAP_READ);
    }

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

    large_input_buffer_ = createBuffer(input_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    large_conv_weights_buffer_ = createBuffer(conv_weights_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    large_conv_bias_buffer_ = createBuffer(conv_bias_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    large_conv_output_buffer_ = createBuffer(conv_output_size, WGPU_BUFFER_USAGE_STORAGE);
    large_linear_weights_buffer_ = createBuffer(linear_weights_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    large_linear_bias_buffer_ = createBuffer(linear_bias_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
    large_partial_sums_buffer_ = createBuffer(partial_sums_size, WGPU_BUFFER_USAGE_STORAGE);
    large_logits_buffer_ = createBuffer(logits_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC);
    large_readback_buffer_ = createBuffer(logits_size, WGPU_BUFFER_USAGE_COPY_DST | WGPU_BUFFER_USAGE_MAP_READ);
    const auto buffers_end = Clock::now();
    if (timestamp_query_enabled_) {
        WGpuQuerySetDescriptor timestamp_desc = {};
        timestamp_desc.type = WGPU_QUERY_TYPE_TIMESTAMP;
        timestamp_desc.count = LARGE_TIMESTAMP_VALUES;
        large_timestamp_query_set_ = wgpu_device_create_query_set(device_, &timestamp_desc);
        large_timestamp_buffer_ = createBuffer(timestamp_size, WGPU_BUFFER_USAGE_QUERY_RESOLVE | WGPU_BUFFER_USAGE_COPY_SRC);
        large_timestamp_readback_buffer_ = createBuffer(timestamp_size, WGPU_BUFFER_USAGE_COPY_DST | WGPU_BUFFER_USAGE_MAP_READ);
    }
    const auto timestamp_end = Clock::now();

    const auto prepare_synth_data = Clock::now();

    wgpu_queue_write_buffer(queue_, large_conv_weights_buffer_, 0, large_conv_weights_data_.data(), conv_weights_size);
    wgpu_queue_write_buffer(queue_, large_conv_bias_buffer_, 0, large_conv_bias_data_.data(), conv_bias_size);
    wgpu_queue_write_buffer(queue_, large_linear_weights_buffer_, 0, large_linear_weights_data_.data(), linear_weights_size);
    wgpu_queue_write_buffer(queue_, large_linear_bias_buffer_, 0, large_linear_bias_data_.data(), linear_bias_size);
    const auto data_upload_end = Clock::now();

    WGpuBindGroupLayoutEntry conv_layout_entries[4] = {
        storageLayoutEntry(0, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, input_size),
        storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, conv_weights_size),
        storageLayoutEntry(2, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, conv_bias_size),
        storageLayoutEntry(3, WGPU_BUFFER_BINDING_TYPE_STORAGE, conv_output_size),
    };
    large_conv_bind_group_layout_ = wgpu_device_create_bind_group_layout(device_, conv_layout_entries, 4);
    large_conv_pipeline_layout_ = wgpu_device_create_pipeline_layout(device_, &large_conv_bind_group_layout_, 1);

    WGpuBindGroupLayoutEntry partial_layout_entries[3] = {
        storageLayoutEntry(0, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, conv_output_size),
        storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, linear_weights_size),
        storageLayoutEntry(2, WGPU_BUFFER_BINDING_TYPE_STORAGE, partial_sums_size),
    };
    large_linear_partial_bind_group_layout_ = wgpu_device_create_bind_group_layout(device_, partial_layout_entries, 3);
    large_linear_partial_pipeline_layout_ = wgpu_device_create_pipeline_layout(device_, &large_linear_partial_bind_group_layout_, 1);

    WGpuBindGroupLayoutEntry reduce_layout_entries[3] = {
        storageLayoutEntry(0, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, partial_sums_size),
        storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, linear_bias_size),
        storageLayoutEntry(2, WGPU_BUFFER_BINDING_TYPE_STORAGE, logits_size),
    };
    large_linear_reduce_bind_group_layout_ = wgpu_device_create_bind_group_layout(device_, reduce_layout_entries, 3);
    large_linear_reduce_pipeline_layout_ = wgpu_device_create_pipeline_layout(device_, &large_linear_reduce_bind_group_layout_, 1);
    const auto layouts_end = Clock::now();

    WGpuShaderModuleDescriptor conv_shader_desc = {};
    conv_shader_desc.code = internal::LARGE_CONV_RELU_WGSL;
    WGpuShaderModule conv_shader = wgpu_device_create_shader_module(device_, &conv_shader_desc);

    WGpuShaderModuleDescriptor partial_shader_desc = {};
    partial_shader_desc.code = internal::LARGE_LINEAR_PARTIAL_WGSL;
    WGpuShaderModule partial_shader = wgpu_device_create_shader_module(device_, &partial_shader_desc);

    WGpuShaderModuleDescriptor reduce_shader_desc = {};
    reduce_shader_desc.code = internal::LARGE_LINEAR_REDUCE_WGSL;
    WGpuShaderModule reduce_shader = wgpu_device_create_shader_module(device_, &reduce_shader_desc);

    large_conv_pipeline_ = wgpu_device_create_compute_pipeline(device_, conv_shader, "main", large_conv_pipeline_layout_, 0, 0);
    large_linear_partial_pipeline_ = wgpu_device_create_compute_pipeline(device_, partial_shader, "main", large_linear_partial_pipeline_layout_, 0, 0);
    large_linear_reduce_pipeline_ = wgpu_device_create_compute_pipeline(device_, reduce_shader, "main", large_linear_reduce_pipeline_layout_, 0, 0);

    wgpu_object_destroy(conv_shader);
    wgpu_object_destroy(partial_shader);
    wgpu_object_destroy(reduce_shader);
    const auto pipelines_end = Clock::now();

    WGpuBindGroupEntry conv_entries[4] = {
        bufferEntry(0, large_input_buffer_, input_size),
        bufferEntry(1, large_conv_weights_buffer_, conv_weights_size),
        bufferEntry(2, large_conv_bias_buffer_, conv_bias_size),
        bufferEntry(3, large_conv_output_buffer_, conv_output_size),
    };
    large_conv_bind_group_ = wgpu_device_create_bind_group(device_, large_conv_bind_group_layout_, conv_entries, 4);

    WGpuBindGroupEntry partial_entries[3] = {
        bufferEntry(0, large_conv_output_buffer_, conv_output_size),
        bufferEntry(1, large_linear_weights_buffer_, linear_weights_size),
        bufferEntry(2, large_partial_sums_buffer_, partial_sums_size),
    };
    large_linear_partial_bind_group_ = wgpu_device_create_bind_group(device_, large_linear_partial_bind_group_layout_, partial_entries, 3);

    WGpuBindGroupEntry reduce_entries[3] = {
        bufferEntry(0, large_partial_sums_buffer_, partial_sums_size),
        bufferEntry(1, large_linear_bias_buffer_, linear_bias_size),
        bufferEntry(2, large_logits_buffer_, logits_size),
    };
    large_linear_reduce_bind_group_ = wgpu_device_create_bind_group(device_, large_linear_reduce_bind_group_layout_, reduce_entries, 3);

    large_network_ready_ =
        large_conv_pipeline_ &&
        large_linear_partial_pipeline_ &&
        large_linear_reduce_pipeline_ &&
        large_conv_bind_group_ &&
        large_linear_partial_bind_group_ &&
        large_linear_reduce_bind_group_;
    const auto resources_end = Clock::now();

    if (large_network_ready_) {
        std::cout << "wasm_webgpu synthetic large resources ready" << std::endl;
        std::cout << "[timing] synthetic_gpu_large_resource_detail"
                  << " buffers=" << elapsedMs(total_start, buffers_end)
                  << "ms timestamp=" << elapsedMs(buffers_end, timestamp_end)
                  << "ms synth_data=" << elapsedMs(timestamp_end, prepare_synth_data)
                  << "ms data_upload=" << elapsedMs(prepare_synth_data, data_upload_end)
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
        std::cerr << "Failed to create wasm_webgpu synthetic large resources" << std::endl;
    }
}
#endif
