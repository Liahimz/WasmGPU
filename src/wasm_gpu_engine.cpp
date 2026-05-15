#include "wasm_gpu_engine.h"
#include "image_proc.h"
#include <iostream>
#include <pthread.h>
#include <algorithm>
#include <limits>

#ifdef __EMSCRIPTEN__
#include "lib_webgpu.h"
#endif

WasmGpuEngine::WasmGpuEngine() {}
WasmGpuEngine::~WasmGpuEngine() {
#ifdef __EMSCRIPTEN__
    wgpu_object_destroy(queue_);
    wgpu_object_destroy(device_);
    wgpu_object_destroy(adapter_);
#endif
}

void WasmGpuEngine::configure(int target_size_) {
    std::cout << "Configure" << std::endl;
    this->target_size = target_size_;
    requestWebGpuDevice();
}

ProcessResult WasmGpuEngine::process(const std::vector<uint8_t>& data, int width, int height, int channels) {
    // 1. Grayscale conversion
    int size = width * height;
    std::vector<uint8_t> gray(size);
    to_grayscale(data.data(), gray.data(), width, height, channels);

    // 2. Rescale
    
    int new_w = this->target_size;
    int new_h = this->target_size;
    std::vector<uint8_t> gray_scaled;

    if (new_w > width) {
        new_w = width;
        new_h = height;
        gray_scaled.resize(width * height);
        gray_scaled = std::move(gray);
    } else {
        gray_scaled.resize(new_w * new_h);
        rescale(gray.data(), gray_scaled.data(), width, height, new_w, new_h);
    }

    ProcessResult result;
    result.image = std::move(gray_scaled);
    result.width = this->target_size;
    result.height = this->target_size;
    return result;
}


int WasmGpuEngine::argmax(const std::vector<float>& data) {
    int result = -1;
    float max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] > max) {
            result = i;
            max = data[i];
        }
    }
    return result;
}

bool WasmGpuEngine::webgpuReady() const {
    return webgpu_ready_;
}

void WasmGpuEngine::requestWebGpuDevice() {
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

    if (!navigator_gpu_request_adapter_async(&options, &WasmGpuEngine::onAdapter, this)) {
        std::cerr << "Failed to start WebGPU adapter request." << std::endl;
    }
#else
    std::cerr << "wasm_webgpu is only active in Emscripten builds." << std::endl;
#endif
}

#ifdef __EMSCRIPTEN__
void WasmGpuEngine::onAdapter(WGpuAdapter adapter, void* user_data) {
    auto* self = static_cast<WasmGpuEngine*>(user_data);
    if (!self || !adapter) {
        std::cerr << "WebGPU adapter request failed." << std::endl;
        return;
    }

    self->adapter_ = adapter;

    WGpuDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_DEFAULT_INITIALIZER;
    wgpu_adapter_request_device_async(adapter, &device_desc, &WasmGpuEngine::onDevice, self);
}

void WasmGpuEngine::onDevice(WGpuDevice device, void* user_data) {
    auto* self = static_cast<WasmGpuEngine*>(user_data);
    if (!self || !device) {
        std::cerr << "WebGPU device request failed." << std::endl;
        return;
    }

    self->device_ = device;
    self->queue_ = wgpu_device_get_queue(device);
    self->webgpu_ready_ = true;
    std::cout << "wasm_webgpu device ready" << std::endl;
}
#endif
