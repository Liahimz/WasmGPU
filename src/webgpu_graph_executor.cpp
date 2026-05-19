#include "webgpu_graph_executor.h"

#include "cpu_graph_executor.h"

#include <algorithm>
#include <iostream>
#include <limits>

#if defined(__EMSCRIPTEN__) && !defined(BUILD_EMDAWN_WEBGPU)
#include "lib_webgpu.h"
#endif

namespace network {

struct WebGpuGraphExecutor::Impl {
    const ModelDesc* model = nullptr;
    std::vector<GeneratedWgsl> shaders;
    std::string error;
    bool resources_ready = false;
    int latest_prediction = -1;

#if defined(__EMSCRIPTEN__) && !defined(BUILD_EMDAWN_WEBGPU)
    WGpuDevice device = 0;
    WGpuQueue queue = 0;

    struct GpuTensor {
        WGpuBuffer buffer = 0;
        uint64_t size = 0;
    };

    struct GpuLayer {
        LayerType type = LayerType::Unknown;
        GeneratedWgsl shader;
        WGpuBuffer weights = 0;
        WGpuBuffer bias = 0;
        WGpuBindGroupLayout bind_group_layout = 0;
        WGpuPipelineLayout pipeline_layout = 0;
        WGpuComputePipeline pipeline = 0;
        WGpuBindGroup bind_group = 0;
    };

    std::vector<GpuTensor> tensors;
    std::vector<GpuLayer> layers;
    WGpuBuffer readback = 0;
#endif
};

namespace {

#if defined(__EMSCRIPTEN__) && !defined(BUILD_EMDAWN_WEBGPU)
uint64_t byteSize(const TensorShape& shape) {
    return static_cast<uint64_t>(shape.elementCount() * sizeof(float));
}

WGpuBindGroupLayoutEntry storageLayoutEntry(uint32_t binding, WGPU_BUFFER_BINDING_TYPE type, uint64_t min_size) {
    WGpuBindGroupLayoutEntry entry = WGPU_BUFFER_BINDING_LAYOUT_ENTRY_DEFAULT_INITIALIZER;
    entry.binding = binding;
    entry.visibility = WGPU_SHADER_STAGE_COMPUTE;
    entry.type = WGPU_BIND_GROUP_LAYOUT_TYPE_BUFFER;
    entry.layout.buffer.type = type;
    entry.layout.buffer.minBindingSize = min_size;
    return entry;
}

WGpuBindGroupEntry bufferEntry(uint32_t binding, WGpuBuffer buffer, uint64_t size) {
    WGpuBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_DEFAULT_INITIALIZER;
    entry.binding = binding;
    entry.resource = buffer;
    entry.bufferBindSize = size;
    return entry;
}

WGpuBuffer createBuffer(WGpuDevice device, std::size_t size, WGPU_BUFFER_USAGE_FLAGS usage) {
    WGpuBufferDescriptor desc = {};
    desc.size = size;
    desc.usage = usage;
    return wgpu_device_create_buffer(device, &desc);
}

void destroyLayerObjects(
    WGpuBindGroup bind_group,
    WGpuComputePipeline pipeline,
    WGpuPipelineLayout pipeline_layout,
    WGpuBindGroupLayout bind_group_layout,
    WGpuBuffer bias,
    WGpuBuffer weights
) {
    wgpu_object_destroy(bind_group);
    wgpu_object_destroy(pipeline);
    wgpu_object_destroy(pipeline_layout);
    wgpu_object_destroy(bind_group_layout);
    wgpu_object_destroy(bias);
    wgpu_object_destroy(weights);
}
#endif

} // namespace

WebGpuGraphExecutor::WebGpuGraphExecutor()
    : impl_(new Impl()) {}

WebGpuGraphExecutor::~WebGpuGraphExecutor() {
    reset();
    delete impl_;
}

void WebGpuGraphExecutor::reset() {
#if defined(__EMSCRIPTEN__) && !defined(BUILD_EMDAWN_WEBGPU)
    for (auto& layer : impl_->layers) {
        destroyLayerObjects(
            layer.bind_group,
            layer.pipeline,
            layer.pipeline_layout,
            layer.bind_group_layout,
            layer.bias,
            layer.weights
        );
    }
    for (auto& tensor : impl_->tensors) {
        wgpu_object_destroy(tensor.buffer);
    }
    wgpu_object_destroy(impl_->readback);
    impl_->layers.clear();
    impl_->tensors.clear();
    impl_->readback = 0;
#endif
    impl_->resources_ready = false;
}

bool WebGpuGraphExecutor::configure(const ModelDesc& model) {
    reset();
    impl_->model = nullptr;
    impl_->error.clear();
    impl_->shaders.clear();
    impl_->resources_ready = false;

    if (!model.valid()) {
        impl_->error = model.error.empty() ? "Invalid model" : model.error;
        return false;
    }

    for (const LayerDesc& layer : model.layers) {
        GeneratedWgsl shader = generateLayerWgsl(layer);
        if (layer.type != LayerType::Flatten && shader.source.empty()) {
            impl_->error = layer.name + ": unsupported WebGPU layer";
            return false;
        }
        impl_->shaders.emplace_back(std::move(shader));
    }

    impl_->model = &model;
    return true;
}

#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
bool WebGpuGraphExecutor::attach(WGPUDevice, WGPUQueue) {
    impl_->error = "WebGpuGraphExecutor is not implemented for emdawnwebgpu yet";
    return false;
}
#elif defined(__EMSCRIPTEN__)
bool WebGpuGraphExecutor::attach(WGpuDevice device, WGpuQueue queue) {
    impl_->device = device;
    impl_->queue = queue;
    return device && queue;
}
#else
bool WebGpuGraphExecutor::attach() {
    impl_->error = "WebGpuGraphExecutor requires an Emscripten WebGPU build";
    return false;
}
#endif

bool WebGpuGraphExecutor::ready() const {
    return impl_->model != nullptr && impl_->resources_ready && impl_->error.empty();
}

bool WebGpuGraphExecutor::prepare() {
#if defined(__EMSCRIPTEN__) && !defined(BUILD_EMDAWN_WEBGPU)
    if (impl_->resources_ready) {
        return true;
    }
    if (!impl_->model || !impl_->device || !impl_->queue) {
        impl_->error = "WebGPU graph executor is missing model or device";
        return false;
    }

    impl_->tensors.clear();
    impl_->layers.clear();

    const ModelDesc& model = *impl_->model;
    impl_->tensors.push_back({
        createBuffer(impl_->device, byteSize(model.input_shape), WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST),
        byteSize(model.input_shape),
    });

    uint32_t input_tensor_index = 0;
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const LayerDesc& layer = model.layers[i];
        if (layer.type == LayerType::Flatten) {
            continue;
        }

        const uint64_t output_size = byteSize(layer.output_shape);
        const bool final_layer = i + 1 == model.layers.size();
        impl_->tensors.push_back({
            createBuffer(
                impl_->device,
                output_size,
                final_layer ? (WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC) : WGPU_BUFFER_USAGE_STORAGE
            ),
            output_size,
        });
        const uint32_t output_tensor_index = static_cast<uint32_t>(impl_->tensors.size() - 1);

        Impl::GpuLayer gpu_layer;
        gpu_layer.type = layer.type;
        gpu_layer.shader = impl_->shaders[i];

        std::vector<WGpuBindGroupLayoutEntry> layout_entries;
        std::vector<WGpuBindGroupEntry> bind_entries;
        layout_entries.push_back(storageLayoutEntry(0, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, impl_->tensors[input_tensor_index].size));
        bind_entries.push_back(bufferEntry(0, impl_->tensors[input_tensor_index].buffer, impl_->tensors[input_tensor_index].size));

        if (layer.type == LayerType::Conv2D || layer.type == LayerType::Linear) {
            const ModelWeight* weights = model.weight(layer.weights_index);
            const ModelWeight* bias = model.weight(layer.bias_index);
            if (!weights || !bias) {
                impl_->error = layer.name + ": missing weights or bias";
                return false;
            }

            const uint64_t weights_size = static_cast<uint64_t>(weights->values.size() * sizeof(float));
            const uint64_t bias_size = static_cast<uint64_t>(bias->values.size() * sizeof(float));
            gpu_layer.weights = createBuffer(impl_->device, weights_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
            gpu_layer.bias = createBuffer(impl_->device, bias_size, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
            wgpu_queue_write_buffer(impl_->queue, gpu_layer.weights, 0, weights->values.data(), weights_size);
            wgpu_queue_write_buffer(impl_->queue, gpu_layer.bias, 0, bias->values.data(), bias_size);

            layout_entries.push_back(storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, weights_size));
            layout_entries.push_back(storageLayoutEntry(2, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, bias_size));
            layout_entries.push_back(storageLayoutEntry(3, WGPU_BUFFER_BINDING_TYPE_STORAGE, output_size));
            bind_entries.push_back(bufferEntry(1, gpu_layer.weights, weights_size));
            bind_entries.push_back(bufferEntry(2, gpu_layer.bias, bias_size));
            bind_entries.push_back(bufferEntry(3, impl_->tensors[output_tensor_index].buffer, output_size));
        } else {
            layout_entries.push_back(storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_STORAGE, output_size));
            bind_entries.push_back(bufferEntry(1, impl_->tensors[output_tensor_index].buffer, output_size));
        }

        gpu_layer.bind_group_layout = wgpu_device_create_bind_group_layout(
            impl_->device,
            layout_entries.data(),
            static_cast<int>(layout_entries.size())
        );
        gpu_layer.pipeline_layout = wgpu_device_create_pipeline_layout(impl_->device, &gpu_layer.bind_group_layout, 1);

        WGpuShaderModuleDescriptor shader_desc = {};
        shader_desc.code = gpu_layer.shader.source.c_str();
        WGpuShaderModule shader_module = wgpu_device_create_shader_module(impl_->device, &shader_desc);
        gpu_layer.pipeline = wgpu_device_create_compute_pipeline(
            impl_->device,
            shader_module,
            "main",
            gpu_layer.pipeline_layout,
            0,
            0
        );
        wgpu_object_destroy(shader_module);

        gpu_layer.bind_group = wgpu_device_create_bind_group(
            impl_->device,
            gpu_layer.bind_group_layout,
            bind_entries.data(),
            static_cast<int>(bind_entries.size())
        );

        if (!gpu_layer.pipeline || !gpu_layer.bind_group) {
            impl_->error = layer.name + ": failed to create WebGPU pipeline or bind group";
            destroyLayerObjects(
                gpu_layer.bind_group,
                gpu_layer.pipeline,
                gpu_layer.pipeline_layout,
                gpu_layer.bind_group_layout,
                gpu_layer.bias,
                gpu_layer.weights
            );
            return false;
        }

        impl_->layers.emplace_back(gpu_layer);
        input_tensor_index = output_tensor_index;
    }

    if (impl_->tensors.empty()) {
        impl_->error = "WebGPU graph has no tensors";
        return false;
    }

    const uint64_t output_size = impl_->tensors.back().size;
    impl_->readback = createBuffer(impl_->device, output_size, WGPU_BUFFER_USAGE_COPY_DST | WGPU_BUFFER_USAGE_MAP_READ);
    impl_->resources_ready = true;
    return true;
#else
    if (impl_->error.empty()) {
        impl_->error = "WebGpuGraphExecutor prepare is unavailable in this build";
    }
    return false;
#endif
}

int WebGpuGraphExecutor::inferClassBytes(const std::vector<uint8_t>& input) {
#if defined(__EMSCRIPTEN__) && !defined(BUILD_EMDAWN_WEBGPU)
    if (!ready() && !prepare()) {
        return -1;
    }
    if (input.size() != impl_->model->input_shape.elementCount()) {
        impl_->error = "Input size does not match model input shape";
        return -1;
    }

    std::vector<float> float_input(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        float_input[i] = static_cast<float>(input[i]) / 255.0f;
    }
    wgpu_queue_write_buffer(impl_->queue, impl_->tensors.front().buffer, 0, float_input.data(), float_input.size() * sizeof(float));

    WGpuCommandEncoder encoder = wgpu_device_create_command_encoder(impl_->device, &WGPU_COMMAND_ENCODER_DESCRIPTOR_DEFAULT_INITIALIZER);
    for (const auto& layer : impl_->layers) {
        WGpuComputePassEncoder pass = wgpu_command_encoder_begin_compute_pass(encoder, 0);
        wgpu_compute_pass_encoder_set_pipeline(pass, layer.pipeline);
        wgpu_compute_pass_encoder_set_bind_group(pass, 0, layer.bind_group, 0, 0);
        wgpu_compute_pass_encoder_dispatch_workgroups(pass, layer.shader.dispatch_x, layer.shader.dispatch_y, layer.shader.dispatch_z);
        wgpu_compute_pass_encoder_end(pass);
    }
    wgpu_command_encoder_copy_buffer_to_buffer(
        encoder,
        impl_->tensors.back().buffer,
        0,
        impl_->readback,
        0,
        impl_->tensors.back().size
    );

    WGpuCommandBuffer command_buffer = wgpu_command_encoder_finish(encoder);
    wgpu_queue_submit_one_and_destroy(impl_->queue, command_buffer);

    std::vector<float> output(impl_->tensors.back().size / sizeof(float));
    wgpu_buffer_map_sync(impl_->readback, WGPU_MAP_MODE_READ, 0, impl_->tensors.back().size);
    wgpu_buffer_get_mapped_range(impl_->readback, 0, impl_->tensors.back().size);
    wgpu_buffer_read_mapped_range(impl_->readback, 0, 0, output.data(), impl_->tensors.back().size);
    wgpu_buffer_unmap(impl_->readback);

    impl_->latest_prediction = argmax(output);
    return impl_->latest_prediction;
#else
    (void)input;
    return -1;
#endif
}

int WebGpuGraphExecutor::latestPrediction() const {
    return impl_->latest_prediction;
}

const std::string& WebGpuGraphExecutor::error() const {
    return impl_->error;
}

const std::vector<GeneratedWgsl>& WebGpuGraphExecutor::generatedShaders() const {
    return impl_->shaders;
}

} // namespace network
