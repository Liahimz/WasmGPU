@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read> conv_weights: array<f32>;
@group(0) @binding(2) var<storage, read> conv_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_output: array<f32>;

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let out_channel = id.z;

    if (x >= 996u || y >= 498u || out_channel >= 4u) {
        return;
    }

    var sum = conv_bias[out_channel];

    for (var ky = 0u; ky < 3u; ky = ky + 1u) {
        for (var kx = 0u; kx < 5u; kx = kx + 1u) {
            let image_index = (y + ky) * 1000u + (x + kx);
            let weight_index = out_channel * 15u + ky * 5u + kx;
            sum = sum + input_image[image_index] * conv_weights[weight_index];
        }
    }

    let output_index = out_channel * 996u * 498u + y * 996u + x;
    conv_output[output_index] = max(sum, 0.0);
}
