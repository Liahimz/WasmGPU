#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>

// Apply binary threshold
void threshold(const uint8_t* src, uint8_t* dst, int width, int height, uint8_t thresh) {
    int size = width * height;
    for (int i = 0; i < size; ++i)
        dst[i] = src[i] > thresh ? 255 : 0;
}

// Erode (3x3)
void erode(const uint8_t* src, uint8_t* dst, int width, int height) {
    std::memset(dst, 0, width * height);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            bool all_one = true;
            for (int ky = -1; ky <= 1 && all_one; ++ky) {
                for (int kx = -1; kx <= 1 && all_one; ++kx) {
                    if (src[(y + ky) * width + (x + kx)] == 0)
                        all_one = false;
                }
            }
            dst[y * width + x] = all_one ? 255 : 0;
        }
    }
}

// Dilate (3x3)
void dilate(const uint8_t* src, uint8_t* dst, int width, int height) {
    std::memset(dst, 0, width * height);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            bool any_one = false;
            for (int ky = -1; ky <= 1 && !any_one; ++ky) {
                for (int kx = -1; kx <= 1 && !any_one; ++kx) {
                    if (src[(y + ky) * width + (x + kx)] != 0)
                        any_one = true;
                }
            }
            dst[y * width + x] = any_one ? 255 : 0;
        }
    }
}

// Morphological open: erode then dilate
void morph_open(const uint8_t* src, uint8_t* dst, int width, int height) {
    std::vector<uint8_t> tmp(width * height);
    erode(src, tmp.data(), width, height);
    dilate(tmp.data(), dst, width, height);
}

// Convert RGB/RGBA to grayscale
void to_grayscale(const uint8_t* src, uint8_t* dst, int width, int height, int channels) {
    int size = width * height;
//   if (channels == 1) {
//       std::memcpy(dst, src, size);
//       return;
//   }
//   for (int i = 0; i < size; ++i) {
//       uint8_t r = src[i*channels];
//       uint8_t g = src[i*channels+1];
//       uint8_t b = src[i*channels+2];
//       dst[i] = static_cast<uint8_t>(0.299*r + 0.587*g + 0.114*b);
//   }

    if (channels == 4) {
        for (int i = 0; i < width * height; ++i) {
            int idx = i * 4;
            uint8_t r = src[idx];
            uint8_t g = src[idx + 1];
            uint8_t b = src[idx + 2];
            // Ignore alpha, or use it for transparency if needed
            dst[i] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    } else if (channels == 3) {
        for (int i = 0; i < width * height; ++i) {
            int idx = i * 3;
            uint8_t r = src[idx];
            uint8_t g = src[idx + 1];
            uint8_t b = src[idx + 2];
            dst[i] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    } else if (channels == 1) {
        // Already grayscale, just copy
        std::memcpy(dst, src, size);
    }
}

void rescale(const uint8_t* src, uint8_t* dst, int src_width, int src_height, int dst_width, int dst_height) {
    for (int y = 0; y < dst_height; ++y) {
        // Find the y in the source image
        int src_y = static_cast<int>(y * (static_cast<float>(src_height) / dst_height));
        if (src_y >= src_height) src_y = src_height - 1;
        for (int x = 0; x < dst_width; ++x) {
            // Find the x in the source image
            int src_x = static_cast<int>(x * (static_cast<float>(src_width) / dst_width));
            if (src_x >= src_width) src_x = src_width - 1;
            dst[y * dst_width + x] = src[src_y * src_width + src_x];
        }
    }
}