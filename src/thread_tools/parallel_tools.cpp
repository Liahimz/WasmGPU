#include "thread_tools/parallel_tools.h"

#include <atomic>
#include <iostream>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace {

std::atomic<bool> keep_alive{false};

void idleMainLoop() {}

} // namespace

extern "C" {

void start_parallel_session() {
    parallel::initialize();
    parallel::warmup();
    std::cout << "parallel backend=" << parallel::backendName(parallel::compileBackend())
              << " threads=" << parallel::maxConcurrency()
              << std::endl;
}

void start_tbb_session() {
    start_parallel_session();
}

void start_keepalive_mainloop() {
#if defined(__EMSCRIPTEN__)
    if (!keep_alive.exchange(true)) {
        emscripten_set_main_loop(idleMainLoop, 1, 0);
    }
#else
    keep_alive.store(true);
#endif
}

void stop_keepalive_mainloop() {
#if defined(__EMSCRIPTEN__)
    if (keep_alive.exchange(false)) {
        emscripten_cancel_main_loop();
    }
#else
    keep_alive.store(false);
#endif
}

} // extern "C"

int TbbInitializer::Initialize(int threads) {
    parallel::initialize({threads});
    return 0;
}

int TbbInitializer::ThreadWarmUp(int threads) {
    parallel::warmup({threads});
    return 0;
}
