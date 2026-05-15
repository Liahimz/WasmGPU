// wasm_gpu_engine.h
#pragma once
#include <vector>
#include <cstdint>

#ifdef __EMSCRIPTEN__
#include "lib_webgpu_fwd.h"
#endif

struct ProcessResult {
    std::vector<uint8_t> image;
    int width;
    int height;
};

class WasmGpuEngine {
public:
    WasmGpuEngine();
    ~WasmGpuEngine();

    void configure(int target_size);

    // Accepts grayscale image (flat vector), width, height
    ProcessResult process(const std::vector<uint8_t>& data, int width, int height, int channels);

    int argmax(const std::vector<float>& data);
    bool webgpuReady() const;

private:
  int target_size = 0;
  bool webgpu_requested_ = false;
  bool webgpu_ready_ = false;

  void requestWebGpuDevice();

#ifdef __EMSCRIPTEN__
  WGpuAdapter adapter_ = 0;
  WGpuDevice device_ = 0;
  WGpuQueue queue_ = 0;

  static void onAdapter(WGpuAdapter adapter, void* user_data);
  static void onDevice(WGpuDevice device, void* user_data);
#endif
};
