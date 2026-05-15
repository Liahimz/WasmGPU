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
    bool inferencePending() const;
    int latestPrediction() const;

private:
    const network::TinyLenetWeights* weights_ = nullptr;
    bool webgpu_requested_ = false;
    bool webgpu_ready_ = false;
    bool network_ready_ = false;
    bool inference_pending_ = false;
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

    static void onAdapter(WGpuAdapter adapter, void* user_data);
    static void onDevice(WGpuDevice device, void* user_data);
    void createNetworkResources();
    WGpuBuffer createBuffer(std::size_t size, WGPU_BUFFER_USAGE_FLAGS usage) const;
#endif
};
