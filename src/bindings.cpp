// bindings.cpp
#include <emscripten/bind.h>

#if defined(BUILD_DUMMY_ENGINE)
#include "dummy_engine.h"
#elif defined(BUILD_WASM_WEBGPU_ENGINE)
#include "wasm_gpu_engine.h"
#else
#include "gpu_engine.h"
#endif

using namespace emscripten;

EMSCRIPTEN_BINDINGS(my_dummy_engine) {
    value_object<ProcessResult>("ProcessResult")
        .field("image", &ProcessResult::image)
        .field("width", &ProcessResult::width)
        .field("height", &ProcessResult::height)
        .field("prediction", &ProcessResult::prediction)
        .field("gpuBackend", &ProcessResult::gpu_backend)
        .field("classLabel", &ProcessResult::class_label)
        .field("topK", &ProcessResult::top_k);
#if defined(BUILD_DUMMY_ENGINE)
    class_<DummyEngine>("DummyEngine")
        .constructor<>()
        .function("configure", &DummyEngine::configure)
        .function("process", &DummyEngine::process)
        ;
#elif defined(BUILD_WASM_WEBGPU_ENGINE)
    class_<WasmGpuEngine>("GpuEngine")
        .constructor<>()
        .function("configure", &WasmGpuEngine::configure)
#if defined(BUILD_WASM_WEBGPU_ASYNC)
        .function("process", &WasmGpuEngine::process)
#else
        .function("process", &WasmGpuEngine::process, async())
#endif
        .function("processCpu", &WasmGpuEngine::processCpu)
#if defined(BUILD_WASM_WEBGPU_ASYNC)
        .function("processResnet", &WasmGpuEngine::processResnet)
#else
        .function("processResnet", &WasmGpuEngine::processResnet, async())
#endif
        .function("processResnetCpu", &WasmGpuEngine::processResnetCpu)
        .function("processResnetCpuTiled", &WasmGpuEngine::processResnetCpuTiled)
        .function("processResnetCpuProfiled", &WasmGpuEngine::processResnetCpuProfiled)
        .function("benchmarkCpuLarge", &WasmGpuEngine::benchmarkCpuLarge)
        .function("prepareSyntheticLargeData", &WasmGpuEngine::prepareSyntheticLargeData)
#if defined(BUILD_WASM_WEBGPU_ASYNC)
        .function("benchmarkGpuLarge", &WasmGpuEngine::benchmarkGpuLarge)
#else
        .function("benchmarkGpuLarge", &WasmGpuEngine::benchmarkGpuLarge, async())
#endif
        .function("argmax", &WasmGpuEngine::argmax)
        .function("webgpuReady", &WasmGpuEngine::webgpuReady)
        .function("inferencePending", &WasmGpuEngine::inferencePending)
        .function("latestPrediction", &WasmGpuEngine::latestPrediction)
        .function("latestClassLabel", &WasmGpuEngine::latestClassLabel)
        .function("latestTopK", &WasmGpuEngine::latestTopK)
        ;
#else
    class_<GpuEngine>("GpuEngine")
        .constructor<>()
        .function("configure", &GpuEngine::configure)
        .function("process", &GpuEngine::process)
        .function("argmax", &GpuEngine::argmax)
        ;
#endif

    register_vector<uint8_t>("Uint8Vector");
    register_vector<float>("FloatVector");
}
