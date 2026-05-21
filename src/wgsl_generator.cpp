#include "wgsl_generator.h"

#include <sstream>

namespace network {
namespace {

uint32_t ceilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

GeneratedWgsl generateConv2d(const LayerDesc& layer) {
    GeneratedWgsl shader;
    shader.label = layer.name + "_wgsl";
    shader.workgroup_x = 8;
    shader.workgroup_y = 8;
    shader.workgroup_z = 1;
    shader.dispatch_x = ceilDiv(layer.output_shape.dims[2], shader.workgroup_x);
    shader.dispatch_y = ceilDiv(layer.output_shape.dims[1], shader.workgroup_y);
    shader.dispatch_z = layer.output_shape.dims[0];

    std::ostringstream wgsl;
    wgsl
        << "@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;\n"
        << "@group(0) @binding(1) var<storage, read> weights: array<f32>;\n"
        << "@group(0) @binding(2) var<storage, read> bias: array<f32>;\n"
        << "@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;\n\n"
        << "const IN_C: u32 = " << layer.input_shape.dims[0] << "u;\n"
        << "const IN_H: u32 = " << layer.input_shape.dims[1] << "u;\n"
        << "const IN_W: u32 = " << layer.input_shape.dims[2] << "u;\n"
        << "const OUT_C: u32 = " << layer.output_shape.dims[0] << "u;\n"
        << "const OUT_H: u32 = " << layer.output_shape.dims[1] << "u;\n"
        << "const OUT_W: u32 = " << layer.output_shape.dims[2] << "u;\n"
        << "const K_H: u32 = " << layer.kernel_y << "u;\n"
        << "const K_W: u32 = " << layer.kernel_x << "u;\n"
        << "const STRIDE_Y: u32 = " << layer.stride_y << "u;\n"
        << "const STRIDE_X: u32 = " << layer.stride_x << "u;\n"
        << "const PAD_Y: i32 = " << layer.padding_y << ";\n"
        << "const PAD_X: i32 = " << layer.padding_x << ";\n\n"
        << "@compute @workgroup_size(" << shader.workgroup_x << ", " << shader.workgroup_y << ", 1)\n"
        << "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        << "    let ox = id.x;\n"
        << "    let oy = id.y;\n"
        << "    let oc = id.z;\n"
        << "    if (ox >= OUT_W || oy >= OUT_H || oc >= OUT_C) { return; }\n"
        << "    var sum = bias[oc];\n"
        << "    for (var ic = 0u; ic < IN_C; ic = ic + 1u) {\n"
        << "        for (var ky = 0u; ky < K_H; ky = ky + 1u) {\n"
        << "            for (var kx = 0u; kx < K_W; kx = kx + 1u) {\n"
        << "                let iy = i32(oy * STRIDE_Y + ky) - PAD_Y;\n"
        << "                let ix = i32(ox * STRIDE_X + kx) - PAD_X;\n"
        << "                if (iy >= 0 && ix >= 0 && iy < i32(IN_H) && ix < i32(IN_W)) {\n"
        << "                    let input_index = ic * IN_H * IN_W + u32(iy) * IN_W + u32(ix);\n"
        << "                    let weight_index = oc * IN_C * K_H * K_W + ic * K_H * K_W + ky * K_W + kx;\n"
        << "                    sum = sum + input_tensor[input_index] * weights[weight_index];\n"
        << "                }\n"
        << "            }\n"
        << "        }\n"
        << "    }\n"
        << "    let output_index = oc * OUT_H * OUT_W + oy * OUT_W + ox;\n"
        << "    output_tensor[output_index] = sum;\n"
        << "}\n";
    shader.source = wgsl.str();
    return shader;
}

GeneratedWgsl generateRelu(const LayerDesc& layer) {
    GeneratedWgsl shader;
    shader.label = layer.name + "_wgsl";
    shader.workgroup_x = 64;
    shader.dispatch_x = ceilDiv(static_cast<uint32_t>(layer.output_shape.elementCount()), shader.workgroup_x);

    std::ostringstream wgsl;
    wgsl
        << "@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;\n"
        << "@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;\n\n"
        << "const VALUE_COUNT: u32 = " << layer.output_shape.elementCount() << "u;\n\n"
        << "@compute @workgroup_size(" << shader.workgroup_x << ", 1, 1)\n"
        << "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        << "    let index = id.x;\n"
        << "    if (index >= VALUE_COUNT) { return; }\n"
        << "    output_tensor[index] = max(input_tensor[index], 0.0);\n"
        << "}\n";
    shader.source = wgsl.str();
    return shader;
}

GeneratedWgsl generateMaxPool2d(const LayerDesc& layer) {
    GeneratedWgsl shader;
    shader.label = layer.name + "_wgsl";
    shader.workgroup_x = 8;
    shader.workgroup_y = 8;
    shader.workgroup_z = 1;
    shader.dispatch_x = ceilDiv(layer.output_shape.dims[2], shader.workgroup_x);
    shader.dispatch_y = ceilDiv(layer.output_shape.dims[1], shader.workgroup_y);
    shader.dispatch_z = layer.output_shape.dims[0];

    std::ostringstream wgsl;
    wgsl
        << "@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;\n"
        << "@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;\n\n"
        << "const C: u32 = " << layer.input_shape.dims[0] << "u;\n"
        << "const IN_H: u32 = " << layer.input_shape.dims[1] << "u;\n"
        << "const IN_W: u32 = " << layer.input_shape.dims[2] << "u;\n"
        << "const OUT_H: u32 = " << layer.output_shape.dims[1] << "u;\n"
        << "const OUT_W: u32 = " << layer.output_shape.dims[2] << "u;\n"
        << "const K_H: u32 = " << layer.kernel_y << "u;\n"
        << "const K_W: u32 = " << layer.kernel_x << "u;\n"
        << "const STRIDE_Y: u32 = " << layer.stride_y << "u;\n"
        << "const STRIDE_X: u32 = " << layer.stride_x << "u;\n"
        << "const PAD_Y: i32 = " << layer.padding_y << ";\n"
        << "const PAD_X: i32 = " << layer.padding_x << ";\n\n"
        << "@compute @workgroup_size(" << shader.workgroup_x << ", " << shader.workgroup_y << ", 1)\n"
        << "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        << "    let ox = id.x;\n"
        << "    let oy = id.y;\n"
        << "    let c = id.z;\n"
        << "    if (ox >= OUT_W || oy >= OUT_H || c >= C) { return; }\n"
        << "    var best = -3.4028234663852886e+38;\n"
        << "    for (var ky = 0u; ky < K_H; ky = ky + 1u) {\n"
        << "        for (var kx = 0u; kx < K_W; kx = kx + 1u) {\n"
        << "            let iy = i32(oy * STRIDE_Y + ky) - PAD_Y;\n"
        << "            let ix = i32(ox * STRIDE_X + kx) - PAD_X;\n"
        << "            if (iy >= 0 && ix >= 0 && iy < i32(IN_H) && ix < i32(IN_W)) {\n"
        << "                let input_index = c * IN_H * IN_W + u32(iy) * IN_W + u32(ix);\n"
        << "                best = max(best, input_tensor[input_index]);\n"
        << "            }\n"
        << "        }\n"
        << "    }\n"
        << "    let output_index = c * OUT_H * OUT_W + oy * OUT_W + ox;\n"
        << "    output_tensor[output_index] = best;\n"
        << "}\n";
    shader.source = wgsl.str();
    return shader;
}

GeneratedWgsl generateLinear(const LayerDesc& layer) {
    GeneratedWgsl shader;
    shader.label = layer.name + "_wgsl";
    shader.workgroup_x = 64;
    shader.dispatch_x = ceilDiv(layer.out_features, shader.workgroup_x);

    std::ostringstream wgsl;
    wgsl
        << "@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;\n"
        << "@group(0) @binding(1) var<storage, read> weights: array<f32>;\n"
        << "@group(0) @binding(2) var<storage, read> bias: array<f32>;\n"
        << "@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;\n\n"
        << "const IN_FEATURES: u32 = " << layer.in_features << "u;\n"
        << "const OUT_FEATURES: u32 = " << layer.out_features << "u;\n\n"
        << "@compute @workgroup_size(" << shader.workgroup_x << ", 1, 1)\n"
        << "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        << "    let out_index = id.x;\n"
        << "    if (out_index >= OUT_FEATURES) { return; }\n"
        << "    var sum = bias[out_index];\n"
        << "    for (var i = 0u; i < IN_FEATURES; i = i + 1u) {\n"
        << "        sum = sum + input_tensor[i] * weights[out_index * IN_FEATURES + i];\n"
        << "    }\n"
        << "    output_tensor[out_index] = sum;\n"
        << "}\n";
    shader.source = wgsl.str();
    return shader;
}

GeneratedWgsl generateAdd(const LayerDesc& layer) {
    GeneratedWgsl shader;
    shader.label = layer.name + "_wgsl";
    shader.workgroup_x = 64;
    shader.dispatch_x = ceilDiv(static_cast<uint32_t>(layer.output_shape.elementCount()), shader.workgroup_x);

    std::ostringstream wgsl;
    wgsl
        << "@group(0) @binding(0) var<storage, read> left_tensor: array<f32>;\n"
        << "@group(0) @binding(1) var<storage, read> right_tensor: array<f32>;\n"
        << "@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>;\n\n"
        << "const VALUE_COUNT: u32 = " << layer.output_shape.elementCount() << "u;\n\n"
        << "@compute @workgroup_size(" << shader.workgroup_x << ", 1, 1)\n"
        << "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        << "    let index = id.x;\n"
        << "    if (index >= VALUE_COUNT) { return; }\n"
        << "    output_tensor[index] = left_tensor[index] + right_tensor[index];\n"
        << "}\n";
    shader.source = wgsl.str();
    return shader;
}

GeneratedWgsl generateGlobalAvgPool2d(const LayerDesc& layer) {
    GeneratedWgsl shader;
    shader.label = layer.name + "_wgsl";
    shader.workgroup_x = 64;
    shader.dispatch_x = ceilDiv(layer.output_shape.dims[0], shader.workgroup_x);

    std::ostringstream wgsl;
    wgsl
        << "@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;\n"
        << "@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;\n\n"
        << "const C: u32 = " << layer.input_shape.dims[0] << "u;\n"
        << "const H: u32 = " << layer.input_shape.dims[1] << "u;\n"
        << "const W: u32 = " << layer.input_shape.dims[2] << "u;\n"
        << "const PLANE: u32 = H * W;\n\n"
        << "@compute @workgroup_size(" << shader.workgroup_x << ", 1, 1)\n"
        << "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        << "    let c = id.x;\n"
        << "    if (c >= C) { return; }\n"
        << "    var sum = 0.0;\n"
        << "    let base = c * PLANE;\n"
        << "    for (var i = 0u; i < PLANE; i = i + 1u) {\n"
        << "        sum = sum + input_tensor[base + i];\n"
        << "    }\n"
        << "    output_tensor[c] = sum / f32(PLANE);\n"
        << "}\n";
    shader.source = wgsl.str();
    return shader;
}

} // namespace

GeneratedWgsl generateLayerWgsl(const LayerDesc& layer) {
    switch (layer.type) {
        case LayerType::Conv2D:
            return generateConv2d(layer);
        case LayerType::Relu:
            return generateRelu(layer);
        case LayerType::MaxPool2D:
            return generateMaxPool2d(layer);
        case LayerType::Linear:
            return generateLinear(layer);
        case LayerType::Add:
            return generateAdd(layer);
        case LayerType::GlobalAvgPool2D:
            return generateGlobalAvgPool2d(layer);
        case LayerType::Flatten:
        case LayerType::Unknown:
            break;
    }

    GeneratedWgsl shader;
    shader.label = layer.name + "_unsupported";
    shader.source = "";
    return shader;
}

} // namespace network
