@group(0) @binding(0) var<storage, read> partial_sums: array<f32>;
@group(0) @binding(1) var<storage, read> linear_bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(10, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let class_index = id.x;

    if (class_index >= 10u) {
        return;
    }

    var sum = linear_bias[class_index];
    for (var chunk = 0u; chunk < 512u; chunk = chunk + 1u) {
        sum = sum + partial_sums[class_index * 512u + chunk];
    }

    logits[class_index] = sum;
}
