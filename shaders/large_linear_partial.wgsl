@group(0) @binding(0) var<storage, read> conv_output: array<f32>;
@group(0) @binding(1) var<storage, read> linear_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let chunk_index = id.x;
    let class_index = id.y;

    if (chunk_index >= 512u || class_index >= 10u) {
        return;
    }

    let values_per_chunk = 3876u;
    let total_values = 1984032u;
    let begin = chunk_index * values_per_chunk;
    var end = begin + values_per_chunk;
    if (end > total_values) {
        end = total_values;
    }

    var sum = 0.0;
    for (var i = begin; i < end; i = i + 1u) {
        let weight_index = class_index * total_values + i;
        sum = sum + conv_output[i] * linear_weights[weight_index];
    }

    partial_sums[class_index * 512u + chunk_index] = sum;
}
