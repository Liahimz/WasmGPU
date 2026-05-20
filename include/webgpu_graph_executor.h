#pragma once

#include "model_loader.h"
#include "wgsl_generator.h"

#include <cstdint>
#include <string>
#include <vector>

#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
#include <webgpu/webgpu.h>
#elif defined(__EMSCRIPTEN__)
#include "lib_webgpu_fwd.h"
#endif

namespace network {

class WebGpuGraphExecutor {
public:
    struct Impl;

    WebGpuGraphExecutor();
    ~WebGpuGraphExecutor();

    WebGpuGraphExecutor(const WebGpuGraphExecutor&) = delete;
    WebGpuGraphExecutor& operator=(const WebGpuGraphExecutor&) = delete;

    bool configure(const ModelDesc& model);
    void reset();

#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
    bool attach(WGPUDevice device, WGPUQueue queue);
#elif defined(__EMSCRIPTEN__)
    bool attach(WGpuDevice device, WGpuQueue queue);
#else
    bool attach();
#endif

    bool ready() const;
    bool prepare();
    int inferClassBytes(const std::vector<uint8_t>& input);
    int inferClassBytesAsync(const std::vector<uint8_t>& input);
    bool inferencePending() const;
    int latestPrediction() const;
    const std::string& error() const;
    const std::vector<GeneratedWgsl>& generatedShaders() const;

private:
    Impl* impl_ = nullptr;
};

} // namespace network
