#pragma once

#include "network_weights.h"

#include <cstdint>
#include <cstddef>
#include <vector>

#ifdef __EMSCRIPTEN__
#include "lib_webgpu_fwd.h"
#endif

class GpuExecutor {
public:
    GpuExecutor();
    ~GpuExecutor();

    void configure(const network::TinyLenetWeights* weights);
    bool ready() const;
    int infer(const std::vector<uint8_t>& image);
    void prepareSyntheticLargeData();
    void prepareSyntheticLarge();
    int benchmarkSyntheticLarge(uint32_t input_seed);
    bool inferencePending() const;
    int latestPrediction() const;

private:
    const network::TinyLenetWeights* weights_ = nullptr;
    bool webgpu_requested_ = false;
    bool webgpu_ready_ = false;
    bool network_ready_ = false;
    bool large_network_ready_ = false;
    bool timestamp_query_enabled_ = false;
    bool inference_pending_ = false;
    bool large_synthetic_data_ready_ = false;
    int latest_prediction_ = -1;

    void requestWebGpuDevice();

#ifdef __EMSCRIPTEN__
    WGpuAdapter adapter_ = 0;
    WGpuDevice device_ = 0;
    WGpuQueue queue_ = 0;
    WGpuComputePipeline conv_pipeline_ = 0;
    WGpuComputePipeline linear_pipeline_ = 0;
    WGpuBindGroupLayout conv_bind_group_layout_ = 0;
    WGpuBindGroupLayout linear_bind_group_layout_ = 0;
    WGpuPipelineLayout conv_pipeline_layout_ = 0;
    WGpuPipelineLayout linear_pipeline_layout_ = 0;
    WGpuBindGroup conv_bind_group_ = 0;
    WGpuBindGroup linear_bind_group_ = 0;
    WGpuBuffer input_buffer_ = 0;
    WGpuBuffer conv_weights_buffer_ = 0;
    WGpuBuffer conv_bias_buffer_ = 0;
    WGpuBuffer conv_output_buffer_ = 0;
    WGpuBuffer linear_weights_buffer_ = 0;
    WGpuBuffer linear_bias_buffer_ = 0;
    WGpuBuffer logits_buffer_ = 0;
    WGpuBuffer readback_buffer_ = 0;
    WGpuQuerySet timestamp_query_set_ = 0;
    WGpuBuffer timestamp_buffer_ = 0;
    WGpuBuffer timestamp_readback_buffer_ = 0;
    WGpuComputePipeline large_conv_pipeline_ = 0;
    WGpuComputePipeline large_linear_partial_pipeline_ = 0;
    WGpuComputePipeline large_linear_reduce_pipeline_ = 0;
    WGpuBindGroupLayout large_conv_bind_group_layout_ = 0;
    WGpuBindGroupLayout large_linear_partial_bind_group_layout_ = 0;
    WGpuBindGroupLayout large_linear_reduce_bind_group_layout_ = 0;
    WGpuPipelineLayout large_conv_pipeline_layout_ = 0;
    WGpuPipelineLayout large_linear_partial_pipeline_layout_ = 0;
    WGpuPipelineLayout large_linear_reduce_pipeline_layout_ = 0;
    WGpuBindGroup large_conv_bind_group_ = 0;
    WGpuBindGroup large_linear_partial_bind_group_ = 0;
    WGpuBindGroup large_linear_reduce_bind_group_ = 0;
    WGpuBuffer large_input_buffer_ = 0;
    WGpuBuffer large_conv_weights_buffer_ = 0;
    WGpuBuffer large_conv_bias_buffer_ = 0;
    WGpuBuffer large_conv_output_buffer_ = 0;
    WGpuBuffer large_linear_weights_buffer_ = 0;
    WGpuBuffer large_linear_bias_buffer_ = 0;
    WGpuBuffer large_partial_sums_buffer_ = 0;
    WGpuBuffer large_logits_buffer_ = 0;
    WGpuBuffer large_readback_buffer_ = 0;
    WGpuQuerySet large_timestamp_query_set_ = 0;
    WGpuBuffer large_timestamp_buffer_ = 0;
    WGpuBuffer large_timestamp_readback_buffer_ = 0;
    std::vector<float> large_conv_weights_data_;
    std::vector<float> large_conv_bias_data_;
    std::vector<float> large_linear_weights_data_;
    std::vector<float> large_linear_bias_data_;
    double pending_encode_submit_ms_ = 0.0;
    double pending_input_ms_ = 0.0;
    double pending_upload_ms_ = 0.0;
    double pending_sync_start_ms_ = 0.0;
    int pending_kind_ = 0;

    static void onAdapter(WGpuAdapter adapter, void* user_data);
    static void onDevice(WGpuDevice device, void* user_data);
#if defined(BUILD_WASM_WEBGPU_ASYNC)
    static void onTinyReadbackMapped(WGpuBuffer buffer, void* user_data, WGPU_MAP_MODE_FLAGS mode, double_int53_t offset, double_int53_t size);
    static void onTinyTimestampMapped(WGpuBuffer buffer, void* user_data, WGPU_MAP_MODE_FLAGS mode, double_int53_t offset, double_int53_t size);
    static void onLargeReadbackMapped(WGpuBuffer buffer, void* user_data, WGPU_MAP_MODE_FLAGS mode, double_int53_t offset, double_int53_t size);
    static void onLargeTimestampMapped(WGpuBuffer buffer, void* user_data, WGPU_MAP_MODE_FLAGS mode, double_int53_t offset, double_int53_t size);
    void finishTinyAsyncReadback();
    void finishTinyAsyncTimestamp();
    void finishLargeAsyncReadback();
    void finishLargeAsyncTimestamp();
#endif
    void createNetworkResources();
    void createLargeNetworkResources();
    WGpuBuffer createBuffer(std::size_t size, WGPU_BUFFER_USAGE_FLAGS usage) const;
#endif
};
