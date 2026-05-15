@group(0) @binding(0) var<storage, read> conv_output: array<f32>;
@group(0) @binding(1) var<storage, read> linear_weights: array<f32>;
@group(0) @binding(2) var<storage, read> linear_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(10, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let class_index = id.x;

    if (class_index >= 10u) {
        return;
    }

    var sum = linear_bias[class_index];

    for (var i = 0u; i < 2704u; i = i + 1u) {
        let weight_index = class_index * 2704u + i;
        sum = sum + conv_output[i] * linear_weights[weight_index];
    }

    logits[class_index] = sum;
}
