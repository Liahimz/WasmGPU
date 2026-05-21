#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>

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

void rescale_rgb(
    const uint8_t* src,
    uint8_t* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int src_channels
) {
    for (int y = 0; y < dst_height; ++y) {
        int src_y = static_cast<int>(y * (static_cast<float>(src_height) / dst_height));
        if (src_y >= src_height) {
            src_y = src_height - 1;
        }
        for (int x = 0; x < dst_width; ++x) {
            int src_x = static_cast<int>(x * (static_cast<float>(src_width) / dst_width));
            if (src_x >= src_width) {
                src_x = src_width - 1;
            }

            const int dst_index = (y * dst_width + x) * 3;
            if (src_channels == 1) {
                const uint8_t value = src[src_y * src_width + src_x];
                dst[dst_index] = value;
                dst[dst_index + 1] = value;
                dst[dst_index + 2] = value;
            } else {
                const int src_index = (src_y * src_width + src_x) * src_channels;
                dst[dst_index] = src[src_index];
                dst[dst_index + 1] = src[src_index + 1];
                dst[dst_index + 2] = src[src_index + 2];
            }
        }
    }
}

void center_crop_rgb(
    const uint8_t* src_rgb,
    uint8_t* dst_rgb,
    int src_width,
    int src_height,
    int crop_width,
    int crop_height
) {
    const int offset_x = std::max((src_width - crop_width) / 2, 0);
    const int offset_y = std::max((src_height - crop_height) / 2, 0);

    for (int y = 0; y < crop_height; ++y) {
        const int src_y = std::min(offset_y + y, src_height - 1);
        for (int x = 0; x < crop_width; ++x) {
            const int src_x = std::min(offset_x + x, src_width - 1);
            const int src_index = (src_y * src_width + src_x) * 3;
            const int dst_index = (y * crop_width + x) * 3;
            dst_rgb[dst_index] = src_rgb[src_index];
            dst_rgb[dst_index + 1] = src_rgb[src_index + 1];
            dst_rgb[dst_index + 2] = src_rgb[src_index + 2];
        }
    }
}

void rgb_to_chw_normalized(
    const uint8_t* src_rgb,
    float* dst_chw,
    int width,
    int height,
    const float mean[3],
    const float stddev[3]
) {
    const int pixels = width * height;
    for (int i = 0; i < pixels; ++i) {
        const float r = static_cast<float>(src_rgb[i * 3]) / 255.0f;
        const float g = static_cast<float>(src_rgb[i * 3 + 1]) / 255.0f;
        const float b = static_cast<float>(src_rgb[i * 3 + 2]) / 255.0f;
        dst_chw[i] = (r - mean[0]) / stddev[0];
        dst_chw[pixels + i] = (g - mean[1]) / stddev[1];
        dst_chw[pixels * 2 + i] = (b - mean[2]) / stddev[2];
    }
}

std::vector<float> preprocess_imagenet_rgb_chw(
    const uint8_t* src,
    int width,
    int height,
    int channels,
    int resize_shorter_side = 256,
    int crop_size = 224
) {
    if (!src || width <= 0 || height <= 0 || channels <= 0 || resize_shorter_side <= 0 || crop_size <= 0) {
        return {};
    }

    int resized_width = resize_shorter_side;
    int resized_height = resize_shorter_side;
    if (width < height) {
        resized_width = resize_shorter_side;
        resized_height = static_cast<int>(std::round(static_cast<float>(height) * resize_shorter_side / width));
    } else {
        resized_height = resize_shorter_side;
        resized_width = static_cast<int>(std::round(static_cast<float>(width) * resize_shorter_side / height));
    }
    resized_width = std::max(resized_width, crop_size);
    resized_height = std::max(resized_height, crop_size);

    std::vector<uint8_t> resized_rgb(static_cast<std::size_t>(resized_width) * resized_height * 3);
    rescale_rgb(src, resized_rgb.data(), width, height, resized_width, resized_height, channels);

    std::vector<uint8_t> cropped_rgb(static_cast<std::size_t>(crop_size) * crop_size * 3);
    center_crop_rgb(resized_rgb.data(), cropped_rgb.data(), resized_width, resized_height, crop_size, crop_size);

    static constexpr float ImageNetMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float ImageNetStd[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> chw(static_cast<std::size_t>(3) * crop_size * crop_size);
    rgb_to_chw_normalized(cropped_rgb.data(), chw.data(), crop_size, crop_size, ImageNetMean, ImageNetStd);
    return chw;
}

std::vector<uint8_t> preprocess_imagenet_rgb_preview(
    const uint8_t* src,
    int width,
    int height,
    int channels,
    int resize_shorter_side = 256,
    int crop_size = 224
) {
    if (!src || width <= 0 || height <= 0 || channels <= 0 || resize_shorter_side <= 0 || crop_size <= 0) {
        return {};
    }

    int resized_width = resize_shorter_side;
    int resized_height = resize_shorter_side;
    if (width < height) {
        resized_width = resize_shorter_side;
        resized_height = static_cast<int>(std::round(static_cast<float>(height) * resize_shorter_side / width));
    } else {
        resized_height = resize_shorter_side;
        resized_width = static_cast<int>(std::round(static_cast<float>(width) * resize_shorter_side / height));
    }
    resized_width = std::max(resized_width, crop_size);
    resized_height = std::max(resized_height, crop_size);

    std::vector<uint8_t> resized_rgb(static_cast<std::size_t>(resized_width) * resized_height * 3);
    rescale_rgb(src, resized_rgb.data(), width, height, resized_width, resized_height, channels);

    std::vector<uint8_t> cropped_rgb(static_cast<std::size_t>(crop_size) * crop_size * 3);
    center_crop_rgb(resized_rgb.data(), cropped_rgb.data(), resized_width, resized_height, crop_size, crop_size);

    std::vector<uint8_t> preview_rgba(static_cast<std::size_t>(crop_size) * crop_size * 4);
    const int pixels = crop_size * crop_size;
    for (int i = 0; i < pixels; ++i) {
        preview_rgba[i * 4] = cropped_rgb[i * 3];
        preview_rgba[i * 4 + 1] = cropped_rgb[i * 3 + 1];
        preview_rgba[i * 4 + 2] = cropped_rgb[i * 3 + 2];
        preview_rgba[i * 4 + 3] = 255;
    }
    return preview_rgba;
}
