#include "thread_tools/wasm_worker_pool.h"

#if !defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) || !defined(__EMSCRIPTEN__)
#include "thread_tools/thread_pool.hpp"
#endif

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
#include <emscripten/atomic.h>
#include <emscripten/eventloop.h>
#include <emscripten/threading_primitives.h>
#include <emscripten/wasm_worker.h>
#endif

namespace parallel {
namespace {

struct WasmWorkerTask {
    std::function<void()> function;
    std::atomic<std::uint32_t>* pending = nullptr;
    std::atomic<std::uint32_t>* failed = nullptr;
};

#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
void keepWasmWorkerRuntimeAlive() {
    emscripten_runtime_keepalive_push();
}

void runWasmWorkerTask(int task_address) {
    std::unique_ptr<WasmWorkerTask> task(reinterpret_cast<WasmWorkerTask*>(static_cast<std::intptr_t>(task_address)));
    try {
        task->function();
    } catch (...) {
        task->failed->store(1, std::memory_order_release);
    }

    task->pending->fetch_sub(1, std::memory_order_acq_rel);
    emscripten_futex_wake(reinterpret_cast<volatile void*>(task->pending), INT_MAX);
}
#endif

} // namespace

class WasmWorkerPool::Impl {
public:
    explicit Impl(std::size_t max_concurrency)
        : next_worker(0) {
        const std::size_t worker_count = std::max<std::size_t>(1, max_concurrency);

#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
        workers.reserve(worker_count);
        for (std::size_t index = 0; index < worker_count; ++index) {
            emscripten_wasm_worker_t worker = emscripten_malloc_wasm_worker(1024 * 1024);
            if (worker != 0) {
                workers.push_back(worker);
                emscripten_wasm_worker_post_function_v(worker, keepWasmWorkerRuntimeAlive);
            }
        }
#else
        fallback.reset(new ThreadPool(worker_count));
#endif
    }

    ~Impl() {
#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
        for (emscripten_wasm_worker_t worker : workers) {
            emscripten_terminate_wasm_worker(worker);
        }
#endif
    }

    void submit(
        std::function<void()> task,
        std::atomic<std::uint32_t>& pending,
        std::atomic<std::uint32_t>& failed
    ) {
        pending.fetch_add(1, std::memory_order_acq_rel);

#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
        if (workers.empty()) {
            try {
                task();
            } catch (...) {
                failed.store(1, std::memory_order_release);
            }
            pending.fetch_sub(1, std::memory_order_acq_rel);
            emscripten_futex_wake(reinterpret_cast<volatile void*>(&pending), INT_MAX);
            return;
        }

        const std::size_t index = next_worker.fetch_add(1, std::memory_order_relaxed) % workers.size();
        WasmWorkerTask* worker_task = new WasmWorkerTask{std::move(task), &pending, &failed};
        emscripten_wasm_worker_post_function_vi(
            workers[index],
            runWasmWorkerTask,
            static_cast<int>(reinterpret_cast<std::intptr_t>(worker_task))
        );
#else
        fallback->submit([task = std::move(task), &pending, &failed]() mutable {
            try {
                task();
            } catch (...) {
                failed.store(1, std::memory_order_release);
            }
            pending.fetch_sub(1, std::memory_order_acq_rel);
        });
#endif
    }

    void wait(std::atomic<std::uint32_t>& pending) {
#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
        while (pending.load(std::memory_order_acquire) != 0) {
            const std::uint32_t expected = pending.load(std::memory_order_acquire);
            if (expected != 0) {
                emscripten_futex_wait(reinterpret_cast<volatile void*>(&pending), expected, 1.0);
            }
        }
#else
        while (pending.load(std::memory_order_acquire) != 0) {
            fallback->runPendingTask();
        }
#endif
    }

    std::size_t maxConcurrency() const {
#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
        return workers.empty() ? 1 : workers.size();
#else
        return fallback->maxConcurrency();
#endif
    }

private:
    std::atomic<std::size_t> next_worker;

#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
    std::vector<emscripten_wasm_worker_t> workers;
#else
    std::unique_ptr<ThreadPool> fallback;
#endif
};

WasmWorkerPool::WasmWorkerPool(std::size_t max_concurrency)
    : impl_(new Impl(max_concurrency)) {}

WasmWorkerPool::~WasmWorkerPool() = default;

void WasmWorkerPool::submit(
    std::function<void()> task,
    std::atomic<std::uint32_t>& pending,
    std::atomic<std::uint32_t>& failed
) {
    impl_->submit(std::move(task), pending, failed);
}

void WasmWorkerPool::wait(std::atomic<std::uint32_t>& pending) {
    impl_->wait(pending);
}

std::size_t WasmWorkerPool::maxConcurrency() const {
    return impl_->maxConcurrency();
}

} // namespace parallel
