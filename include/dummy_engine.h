// dummy_engine.h
#pragma once
#include <vector>
#include <cstdint>

struct ProcessResult {
    std::vector<uint8_t> image;
    int width;
    int height;
};

class DummyEngine {
public:
    DummyEngine();
    ~DummyEngine();

    void configure(int nThreads);

    // Accepts grayscale image (flat vector), width, height
    ProcessResult process(const std::vector<uint8_t>& data, int width, int height, int channels, int targetWidth);
};
