#include "gpu_engine.h"
#include "image_proc.h"
#include <iostream>
#include <pthread.h>
#include <algorithm>
#include <limits>

GpuEngine::GpuEngine() {}
GpuEngine::~GpuEngine() {}

void GpuEngine::configure(int target_size_) {
    std::cout << "Configure" << std::endl;
    this->target_size = target_size_;
}

ProcessResult GpuEngine::process(const std::vector<uint8_t>& data, int width, int height, int channels) {
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


int GpuEngine::argmax(const std::vector<float>& data) {
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
