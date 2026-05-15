// bindings.cpp
#include <emscripten/bind.h>
// #include "dummy_engine.h"
#include "gpu_engine.h"
using namespace emscripten;

EMSCRIPTEN_BINDINGS(my_dummy_engine) {
    value_object<ProcessResult>("ProcessResult")
        .field("image", &ProcessResult::image)
        .field("width", &ProcessResult::width)
        .field("height", &ProcessResult::height);
    class_<GpuEngine>("GpuEngine")
        .constructor<>()
        .function("configure", &GpuEngine::configure)
        .function("process", &GpuEngine::process)
        .function("argmax", &GpuEngine::argmax)
        ;
    register_vector<uint8_t>("Uint8Vector");
    register_vector<float>("FloatVector");
}
