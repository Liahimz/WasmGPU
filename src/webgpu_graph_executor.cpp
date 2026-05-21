#include "webgpu_graph_executor.h"

#include "cpu_graph_executor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>

#if defined(__EMSCRIPTEN__) && !defined(BUILD_EMDAWN_WEBGPU)
#include "lib_webgpu.h"
#endif

namespace network {

struct WebGpuGraphExecutor::Impl {
    const ModelDesc* model = nullptr;
    std::vector<GeneratedWgsl> shaders;
    std::string error;
    bool resources_ready = false;
    bool inference_pending = false;
    int latest_prediction = -1;
    double pending_start_ms = 0.0;
    double pending_submit_ms = 0.0;
    std::map<std::string, uint32_t> tensor_indices;
    uint32_t output_tensor_index = 0;
    uint64_t output_size = 0;

#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;

    struct GpuTensor {
        WGPUBuffer buffer = nullptr;
        uint64_t size = 0;
    };

    struct GpuLayer {
        LayerType type = LayerType::Unknown;
        GeneratedWgsl shader;
        WGPUBuffer weights = nullptr;
        WGPUBuffer bias = nullptr;
        WGPUBindGroupLayout bind_group_layout = nullptr;
        WGPUPipelineLayout pipeline_layout = nullptr;
        WGPUComputePipeline pipeline = nullptr;
        WGPUBindGroup bind_group = nullptr;
    };

    std::vector<GpuTensor> tensors;
    std::vector<GpuLayer> layers;
    WGPUBuffer readback = nullptr;
#elif defined(__EMSCRIPTEN__)
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

#if defined(__EMSCRIPTEN__)
using Clock = std::chrono::high_resolution_clock;

double nowMs() {
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

uint64_t byteSize(const TensorShape& shape) {
    return static_cast<uint64_t>(shape.elementCount() * sizeof(float));
}
#endif

#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
WGPUStringView strView(const char* value) {
    return WGPUStringView{value, WGPU_STRLEN};
}

WGPUBindGroupLayoutEntry storageLayoutEntry(uint32_t binding, WGPUBufferBindingType type, uint64_t min_size) {
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = binding;
    entry.visibility = WGPUShaderStage_Compute;
    entry.buffer.type = type;
    entry.buffer.minBindingSize = min_size;
    return entry;
}

WGPUBindGroupEntry bufferEntry(uint32_t binding, WGPUBuffer buffer, uint64_t size) {
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = binding;
    entry.buffer = buffer;
    entry.offset = 0;
    entry.size = size;
    return entry;
}

WGPUBuffer createBuffer(WGPUDevice device, std::size_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.size = size;
    desc.usage = usage;
    return wgpuDeviceCreateBuffer(device, &desc);
}

WGPUShaderModule createShaderModule(WGPUDevice device, const char* code) {
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = strView(code);

    WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    desc.nextInChain = &source.chain;
    return wgpuDeviceCreateShaderModule(device, &desc);
}

WGPUComputePipeline createComputePipeline(WGPUDevice device, WGPUShaderModule shader, WGPUPipelineLayout layout) {
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.layout = layout;
    desc.compute.module = shader;
    desc.compute.entryPoint = strView("main");
    return wgpuDeviceCreateComputePipeline(device, &desc);
}

void destroyLayerObjects(
    WGPUBindGroup bind_group,
    WGPUComputePipeline pipeline,
    WGPUPipelineLayout pipeline_layout,
    WGPUBindGroupLayout bind_group_layout,
    WGPUBuffer bias,
    WGPUBuffer weights
) {
    wgpuBindGroupRelease(bind_group);
    wgpuComputePipelineRelease(pipeline);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupLayoutRelease(bind_group_layout);
    wgpuBufferRelease(bias);
    wgpuBufferRelease(weights);
}

int readPredictionFromMappedBuffer(WGPUBuffer buffer, uint64_t size) {
    std::vector<float> output(size / sizeof(float));
    const void* mapped = wgpuBufferGetConstMappedRange(buffer, 0, size);
    if (mapped) {
        std::memcpy(output.data(), mapped, size);
    }
    wgpuBufferUnmap(buffer);
    return argmax(output);
}

void encodeGraphDispatch(WebGpuGraphExecutor::Impl& impl, WGPUCommandEncoder encoder) {
    for (const auto& layer : impl.layers) {
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, layer.pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, layer.bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, layer.shader.dispatch_x, layer.shader.dispatch_y, layer.shader.dispatch_z);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        impl.tensors[impl.output_tensor_index].buffer,
        0,
        impl.readback,
        0,
        impl.output_size
    );
}

bool uploadInput(WebGpuGraphExecutor::Impl& impl, const std::vector<uint8_t>& input) {
    if (!impl.model || input.size() != impl.model->input_shape.elementCount()) {
        impl.error = "Input size does not match model input shape";
        return false;
    }

    std::vector<float> float_input(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        float_input[i] = static_cast<float>(input[i]) / 255.0f;
    }
    auto input_it = impl.tensor_indices.find(impl.model->input_name);
    if (input_it == impl.tensor_indices.end()) {
        impl.error = "Input tensor is not prepared";
        return false;
    }
    wgpuQueueWriteBuffer(impl.queue, impl.tensors[input_it->second].buffer, 0, float_input.data(), float_input.size() * sizeof(float));
    return true;
}

void onGraphReadbackMapped(WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) {
    auto* impl = static_cast<WebGpuGraphExecutor::Impl*>(userdata1);
    if (!impl) {
        return;
    }
    impl->latest_prediction = readPredictionFromMappedBuffer(impl->readback, impl->output_size);
    const double total_ms = nowMs() - impl->pending_start_ms;
    std::cout << "[timing] dawn_gpu_graph_detail"
              << " submit=" << impl->pending_submit_ms
              << "ms total_inference=" << total_ms
              << "ms prediction=" << impl->latest_prediction
              << std::endl;
    impl->inference_pending = false;
}
#elif defined(__EMSCRIPTEN__)
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

int readPredictionFromMappedBuffer(WGpuBuffer buffer, uint64_t size) {
    std::vector<float> output(size / sizeof(float));
    wgpu_buffer_get_mapped_range(buffer, 0, size);
    wgpu_buffer_read_mapped_range(buffer, 0, 0, output.data(), size);
    wgpu_buffer_unmap(buffer);
    return argmax(output);
}

void encodeGraphDispatch(WebGpuGraphExecutor::Impl& impl, WGpuCommandEncoder encoder) {
    for (const auto& layer : impl.layers) {
        WGpuComputePassEncoder pass = wgpu_command_encoder_begin_compute_pass(encoder, 0);
        wgpu_compute_pass_encoder_set_pipeline(pass, layer.pipeline);
        wgpu_compute_pass_encoder_set_bind_group(pass, 0, layer.bind_group, 0, 0);
        wgpu_compute_pass_encoder_dispatch_workgroups(pass, layer.shader.dispatch_x, layer.shader.dispatch_y, layer.shader.dispatch_z);
        wgpu_compute_pass_encoder_end(pass);
    }
    wgpu_command_encoder_copy_buffer_to_buffer(
        encoder,
        impl.tensors[impl.output_tensor_index].buffer,
        0,
        impl.readback,
        0,
        impl.output_size
    );
}

bool uploadInput(WebGpuGraphExecutor::Impl& impl, const std::vector<uint8_t>& input) {
    if (!impl.model || input.size() != impl.model->input_shape.elementCount()) {
        impl.error = "Input size does not match model input shape";
        return false;
    }

    std::vector<float> float_input(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        float_input[i] = static_cast<float>(input[i]) / 255.0f;
    }
    auto input_it = impl.tensor_indices.find(impl.model->input_name);
    if (input_it == impl.tensor_indices.end()) {
        impl.error = "Input tensor is not prepared";
        return false;
    }
    wgpu_queue_write_buffer(impl.queue, impl.tensors[input_it->second].buffer, 0, float_input.data(), float_input.size() * sizeof(float));
    return true;
}

#if defined(BUILD_WASM_WEBGPU_ASYNC)
void onGraphReadbackMapped(WGpuBuffer, void* user_data, WGPU_MAP_MODE_FLAGS, double_int53_t, double_int53_t) {
    auto* impl = static_cast<WebGpuGraphExecutor::Impl*>(user_data);
    if (!impl) {
        return;
    }
    impl->latest_prediction = readPredictionFromMappedBuffer(impl->readback, impl->output_size);
    const double total_ms = nowMs() - impl->pending_start_ms;
    std::cout << "[timing] gpu_graph_detail"
              << " submit=" << impl->pending_submit_ms
              << "ms total_inference=" << total_ms
              << "ms prediction=" << impl->latest_prediction
              << std::endl;
    impl->inference_pending = false;
}
#endif
#endif

} // namespace

WebGpuGraphExecutor::WebGpuGraphExecutor()
    : impl_(new Impl()) {}

WebGpuGraphExecutor::~WebGpuGraphExecutor() {
    reset();
    delete impl_;
}

void WebGpuGraphExecutor::reset() {
#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
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
        wgpuBufferRelease(tensor.buffer);
    }
    wgpuBufferRelease(impl_->readback);
    impl_->layers.clear();
    impl_->tensors.clear();
    impl_->tensor_indices.clear();
    impl_->output_tensor_index = 0;
    impl_->output_size = 0;
    impl_->readback = nullptr;
#elif defined(__EMSCRIPTEN__)
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
    impl_->tensor_indices.clear();
    impl_->output_tensor_index = 0;
    impl_->output_size = 0;
    impl_->readback = 0;
#endif
    impl_->resources_ready = false;
    impl_->inference_pending = false;
}

bool WebGpuGraphExecutor::configure(const ModelDesc& model) {
    reset();
    impl_->model = nullptr;
    impl_->error.clear();
    impl_->shaders.clear();
    impl_->resources_ready = false;
    impl_->inference_pending = false;

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
bool WebGpuGraphExecutor::attach(WGPUDevice device, WGPUQueue queue) {
    impl_->device = device;
    impl_->queue = queue;
    return device && queue;
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
#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
    if (impl_->resources_ready) {
        return true;
    }
    if (!impl_->model || !impl_->device || !impl_->queue) {
        impl_->error = "Dawn WebGPU graph executor is missing model or device";
        return false;
    }

    impl_->tensors.clear();
    impl_->layers.clear();
    impl_->tensor_indices.clear();

    const ModelDesc& model = *impl_->model;
    impl_->tensors.push_back({
        createBuffer(impl_->device, byteSize(model.input_shape), WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst),
        byteSize(model.input_shape),
    });
    impl_->tensor_indices[model.input_name] = 0;

    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const LayerDesc& layer = model.layers[i];
        if (layer.type == LayerType::Flatten) {
            if (layer.input_names.empty() || layer.output_names.empty()) {
                impl_->error = layer.name + ": missing tensor names";
                return false;
            }
            auto input_it = impl_->tensor_indices.find(layer.input_names[0]);
            if (input_it == impl_->tensor_indices.end()) {
                impl_->error = layer.name + ": missing input tensor '" + layer.input_names[0] + "'";
                return false;
            }
            impl_->tensor_indices[layer.output_names[0]] = input_it->second;
            continue;
        }
        if (layer.input_names.empty() || layer.output_names.empty()) {
            impl_->error = layer.name + ": missing tensor names";
            return false;
        }

        const uint64_t output_size = byteSize(layer.output_shape);
        impl_->tensors.push_back({
            createBuffer(
                impl_->device,
                output_size,
                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc
            ),
            output_size,
        });
        const uint32_t output_tensor_index = static_cast<uint32_t>(impl_->tensors.size() - 1);

        Impl::GpuLayer gpu_layer;
        gpu_layer.type = layer.type;
        gpu_layer.shader = impl_->shaders[i];

        std::vector<WGPUBindGroupLayoutEntry> layout_entries;
        std::vector<WGPUBindGroupEntry> bind_entries;
        auto first_input_it = impl_->tensor_indices.find(layer.input_names[0]);
        if (first_input_it == impl_->tensor_indices.end()) {
            impl_->error = layer.name + ": missing input tensor '" + layer.input_names[0] + "'";
            return false;
        }
        const uint32_t input_tensor_index = first_input_it->second;
        layout_entries.push_back(storageLayoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage, impl_->tensors[input_tensor_index].size));
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
            gpu_layer.weights = createBuffer(impl_->device, weights_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
            gpu_layer.bias = createBuffer(impl_->device, bias_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
            wgpuQueueWriteBuffer(impl_->queue, gpu_layer.weights, 0, weights->values.data(), weights_size);
            wgpuQueueWriteBuffer(impl_->queue, gpu_layer.bias, 0, bias->values.data(), bias_size);

            layout_entries.push_back(storageLayoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage, weights_size));
            layout_entries.push_back(storageLayoutEntry(2, WGPUBufferBindingType_ReadOnlyStorage, bias_size));
            layout_entries.push_back(storageLayoutEntry(3, WGPUBufferBindingType_Storage, output_size));
            bind_entries.push_back(bufferEntry(1, gpu_layer.weights, weights_size));
            bind_entries.push_back(bufferEntry(2, gpu_layer.bias, bias_size));
            bind_entries.push_back(bufferEntry(3, impl_->tensors[output_tensor_index].buffer, output_size));
        } else if (layer.type == LayerType::Add) {
            if (layer.input_names.size() != 2) {
                impl_->error = layer.name + ": add expects two inputs";
                return false;
            }
            auto second_input_it = impl_->tensor_indices.find(layer.input_names[1]);
            if (second_input_it == impl_->tensor_indices.end()) {
                impl_->error = layer.name + ": missing input tensor '" + layer.input_names[1] + "'";
                return false;
            }
            const uint32_t right_tensor_index = second_input_it->second;
            layout_entries.push_back(storageLayoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage, impl_->tensors[right_tensor_index].size));
            layout_entries.push_back(storageLayoutEntry(2, WGPUBufferBindingType_Storage, output_size));
            bind_entries.push_back(bufferEntry(1, impl_->tensors[right_tensor_index].buffer, impl_->tensors[right_tensor_index].size));
            bind_entries.push_back(bufferEntry(2, impl_->tensors[output_tensor_index].buffer, output_size));
        } else {
            layout_entries.push_back(storageLayoutEntry(1, WGPUBufferBindingType_Storage, output_size));
            bind_entries.push_back(bufferEntry(1, impl_->tensors[output_tensor_index].buffer, output_size));
        }

        WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bgl_desc.entryCount = layout_entries.size();
        bgl_desc.entries = layout_entries.data();
        gpu_layer.bind_group_layout = wgpuDeviceCreateBindGroupLayout(impl_->device, &bgl_desc);

        WGPUBindGroupLayout layouts[1] = {gpu_layer.bind_group_layout};
        WGPUPipelineLayoutDescriptor pipeline_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pipeline_layout_desc.bindGroupLayoutCount = 1;
        pipeline_layout_desc.bindGroupLayouts = layouts;
        gpu_layer.pipeline_layout = wgpuDeviceCreatePipelineLayout(impl_->device, &pipeline_layout_desc);

        WGPUShaderModule shader_module = createShaderModule(impl_->device, gpu_layer.shader.source.c_str());
        gpu_layer.pipeline = createComputePipeline(impl_->device, shader_module, gpu_layer.pipeline_layout);
        wgpuShaderModuleRelease(shader_module);

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.layout = gpu_layer.bind_group_layout;
        bg_desc.entryCount = bind_entries.size();
        bg_desc.entries = bind_entries.data();
        gpu_layer.bind_group = wgpuDeviceCreateBindGroup(impl_->device, &bg_desc);

        if (!gpu_layer.pipeline || !gpu_layer.bind_group) {
            impl_->error = layer.name + ": failed to create Dawn WebGPU pipeline or bind group";
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

        impl_->tensor_indices[layer.output_names[0]] = output_tensor_index;
        impl_->layers.emplace_back(gpu_layer);
    }

    auto output_it = impl_->tensor_indices.find(model.output_name);
    if (output_it == impl_->tensor_indices.end()) {
        impl_->error = "Dawn WebGPU graph is missing output tensor '" + model.output_name + "'";
        return false;
    }

    impl_->output_tensor_index = output_it->second;
    impl_->output_size = impl_->tensors[impl_->output_tensor_index].size;
    impl_->readback = createBuffer(impl_->device, impl_->output_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    impl_->resources_ready = true;
    return true;
#elif defined(__EMSCRIPTEN__)
    if (impl_->resources_ready) {
        return true;
    }
    if (!impl_->model || !impl_->device || !impl_->queue) {
        impl_->error = "WebGPU graph executor is missing model or device";
        return false;
    }

    impl_->tensors.clear();
    impl_->layers.clear();
    impl_->tensor_indices.clear();

    const ModelDesc& model = *impl_->model;
    impl_->tensors.push_back({
        createBuffer(impl_->device, byteSize(model.input_shape), WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST),
        byteSize(model.input_shape),
    });
    impl_->tensor_indices[model.input_name] = 0;

    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const LayerDesc& layer = model.layers[i];
        if (layer.type == LayerType::Flatten) {
            if (layer.input_names.empty() || layer.output_names.empty()) {
                impl_->error = layer.name + ": missing tensor names";
                return false;
            }
            auto input_it = impl_->tensor_indices.find(layer.input_names[0]);
            if (input_it == impl_->tensor_indices.end()) {
                impl_->error = layer.name + ": missing input tensor '" + layer.input_names[0] + "'";
                return false;
            }
            impl_->tensor_indices[layer.output_names[0]] = input_it->second;
            continue;
        }
        if (layer.input_names.empty() || layer.output_names.empty()) {
            impl_->error = layer.name + ": missing tensor names";
            return false;
        }

        const uint64_t output_size = byteSize(layer.output_shape);
        impl_->tensors.push_back({
            createBuffer(
                impl_->device,
                output_size,
                WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC
            ),
            output_size,
        });
        const uint32_t output_tensor_index = static_cast<uint32_t>(impl_->tensors.size() - 1);

        Impl::GpuLayer gpu_layer;
        gpu_layer.type = layer.type;
        gpu_layer.shader = impl_->shaders[i];

        std::vector<WGpuBindGroupLayoutEntry> layout_entries;
        std::vector<WGpuBindGroupEntry> bind_entries;
        auto first_input_it = impl_->tensor_indices.find(layer.input_names[0]);
        if (first_input_it == impl_->tensor_indices.end()) {
            impl_->error = layer.name + ": missing input tensor '" + layer.input_names[0] + "'";
            return false;
        }
        const uint32_t input_tensor_index = first_input_it->second;
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
        } else if (layer.type == LayerType::Add) {
            if (layer.input_names.size() != 2) {
                impl_->error = layer.name + ": add expects two inputs";
                return false;
            }
            auto second_input_it = impl_->tensor_indices.find(layer.input_names[1]);
            if (second_input_it == impl_->tensor_indices.end()) {
                impl_->error = layer.name + ": missing input tensor '" + layer.input_names[1] + "'";
                return false;
            }
            const uint32_t right_tensor_index = second_input_it->second;
            layout_entries.push_back(storageLayoutEntry(1, WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, impl_->tensors[right_tensor_index].size));
            layout_entries.push_back(storageLayoutEntry(2, WGPU_BUFFER_BINDING_TYPE_STORAGE, output_size));
            bind_entries.push_back(bufferEntry(1, impl_->tensors[right_tensor_index].buffer, impl_->tensors[right_tensor_index].size));
            bind_entries.push_back(bufferEntry(2, impl_->tensors[output_tensor_index].buffer, output_size));
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

        impl_->tensor_indices[layer.output_names[0]] = output_tensor_index;
        impl_->layers.emplace_back(gpu_layer);
    }

    auto output_it = impl_->tensor_indices.find(model.output_name);
    if (output_it == impl_->tensor_indices.end()) {
        impl_->error = "WebGPU graph is missing output tensor '" + model.output_name + "'";
        return false;
    }

    impl_->output_tensor_index = output_it->second;
    impl_->output_size = impl_->tensors[impl_->output_tensor_index].size;
    impl_->readback = createBuffer(impl_->device, impl_->output_size, WGPU_BUFFER_USAGE_COPY_DST | WGPU_BUFFER_USAGE_MAP_READ);
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
#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
    (void)input;
    impl_->error = "Synchronous Dawn graph inference is not supported";
    return -1;
#elif defined(__EMSCRIPTEN__)
    if (!ready() && !prepare()) {
        return -1;
    }
    if (impl_->inference_pending || !uploadInput(*impl_, input)) {
        return -1;
    }

    const double start_ms = nowMs();
    WGpuCommandEncoder encoder = wgpu_device_create_command_encoder(impl_->device, &WGPU_COMMAND_ENCODER_DESCRIPTOR_DEFAULT_INITIALIZER);
    encodeGraphDispatch(*impl_, encoder);

    WGpuCommandBuffer command_buffer = wgpu_command_encoder_finish(encoder);
    wgpu_queue_submit_one_and_destroy(impl_->queue, command_buffer);

    wgpu_buffer_map_sync(impl_->readback, WGPU_MAP_MODE_READ, 0, impl_->output_size);
    impl_->latest_prediction = readPredictionFromMappedBuffer(impl_->readback, impl_->output_size);
    std::cout << "[timing] gpu_graph_detail"
              << " total_inference=" << (nowMs() - start_ms)
              << "ms prediction=" << impl_->latest_prediction
              << std::endl;
    return impl_->latest_prediction;
#else
    (void)input;
    return -1;
#endif
}

int WebGpuGraphExecutor::inferClassBytesAsync(const std::vector<uint8_t>& input) {
#if defined(__EMSCRIPTEN__) && defined(BUILD_EMDAWN_WEBGPU)
    if (!ready() && !prepare()) {
        return -1;
    }
    if (impl_->inference_pending || !uploadInput(*impl_, input)) {
        return -1;
    }

    impl_->pending_start_ms = nowMs();
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(impl_->device, nullptr);
    encodeGraphDispatch(*impl_, encoder);

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);
    wgpuQueueSubmit(impl_->queue, 1, &command_buffer);
    wgpuCommandBufferRelease(command_buffer);

    impl_->inference_pending = true;
    impl_->pending_submit_ms = nowMs() - impl_->pending_start_ms;
    WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.callback = &onGraphReadbackMapped;
    callback.userdata1 = impl_;
    wgpuBufferMapAsync(impl_->readback, WGPUMapMode_Read, 0, impl_->output_size, callback);
    return -1;
#elif defined(__EMSCRIPTEN__) && defined(BUILD_WASM_WEBGPU_ASYNC)
    if (!ready() && !prepare()) {
        return -1;
    }
    if (impl_->inference_pending || !uploadInput(*impl_, input)) {
        return -1;
    }

    impl_->pending_start_ms = nowMs();
    WGpuCommandEncoder encoder = wgpu_device_create_command_encoder(impl_->device, &WGPU_COMMAND_ENCODER_DESCRIPTOR_DEFAULT_INITIALIZER);
    encodeGraphDispatch(*impl_, encoder);

    WGpuCommandBuffer command_buffer = wgpu_command_encoder_finish(encoder);
    wgpu_queue_submit_one_and_destroy(impl_->queue, command_buffer);

    impl_->inference_pending = true;
    impl_->pending_submit_ms = nowMs() - impl_->pending_start_ms;
    wgpu_buffer_map_async(
        impl_->readback,
        &onGraphReadbackMapped,
        impl_,
        WGPU_MAP_MODE_READ,
        0,
        impl_->output_size
    );
    return -1;
#else
    (void)input;
    return -1;
#endif
}

bool WebGpuGraphExecutor::inferencePending() const {
    return impl_->inference_pending;
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
