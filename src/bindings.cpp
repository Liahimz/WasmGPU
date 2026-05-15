// bindings.cpp
#include <emscripten/bind.h>
#include "dummy_engine.h"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(my_dummy_engine) {
    value_object<ProcessResult>("ProcessResult")
        .field("image", &ProcessResult::image)
        .field("width", &ProcessResult::width)
        .field("height", &ProcessResult::height);
    class_<DummyEngine>("DummyEngine")
        .constructor<>()
        .function("configure", &DummyEngine::configure)
        .function("process", &DummyEngine::process)
        ;
    register_vector<uint8_t>("Uint8Vector");
}
