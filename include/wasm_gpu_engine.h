// wasm_gpu_engine.h
#pragma once
#include "cpp_executor.h"
#include "cpu_graph_executor.h"
#include "gpu_executor.h"
#include "model_loader.h"
#include "network_weights.h"

#include <string>
#include <vector>
#include <cstdint>

struct ProcessResult {
    std::vector<uint8_t> image;
    int width;
    int height;
    int prediction = -1;
    std::string gpu_backend;
};

class WasmGpuEngine {
public:
    WasmGpuEngine();
    ~WasmGpuEngine();

    void configure(int target_size);

    // Accepts grayscale image (flat vector), width, height
    ProcessResult process(const std::vector<uint8_t>& data, int width, int height, int channels);
    ProcessResult processCpuGraph(const std::vector<uint8_t>& data, int width, int height, int channels);
    ProcessResult processCpu(const std::vector<uint8_t>& data, int width, int height, int channels, int mode);
    int benchmarkCpuLarge(int mode, int input_seed);
    void prepareSyntheticLargeData();
    int benchmarkGpuLarge(int input_seed);

    int argmax(const std::vector<float>& data);
    bool webgpuReady() const;
    bool inferencePending() const;
    int latestPrediction() const;

private:
  int target_size = 0;
  network::ModelDesc model_;
  network::TinyLenetWeights weights_;
  network::CpuGraphExecutor cpu_graph_;
  CppExecutor cpu_;
  GpuExecutor gpu_;

  ProcessResult preprocess(const std::vector<uint8_t>& data, int width, int height, int channels) const;
  int runNetwork(const std::vector<uint8_t>& image);
};
