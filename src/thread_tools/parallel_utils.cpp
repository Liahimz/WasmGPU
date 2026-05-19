#include "thread_tools/parallel_utils.h"

#if !defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && \
    (defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD))
#include "thread_tools/thread_pool.hpp"
#endif
#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
#include "thread_tools/wasm_worker_pool.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#endif
#if !defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
#include <chrono>
#include <future>
#include <mutex>
#include <thread>
#endif
#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
#include <emscripten/wasm_worker.h>
#endif

namespace parallel {
namespace {

#if !defined(WASM_MAX_THREADS)
constexpr int DefaultMaxThreads = 4;
#else
constexpr int DefaultMaxThreads = WASM_MAX_THREADS;
#endif

#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
std::atomic<bool> initialized{false};
#else
std::once_flag init_flag;
#endif
Options init_options;

#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
std::unique_ptr<tbb::task_arena> tbb_arena;
std::unique_ptr<tbb::global_control> tbb_control;
#elif defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
std::unique_ptr<WasmWorkerPool> wasm_worker_pool;
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD)
std::unique_ptr<ThreadPool> custom_pool;
#endif

std::size_t normalizeGrain(std::size_t grain_size) {
    return std::max<std::size_t>(grain_size, 1);
}

int hardwareConcurrency() {
#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) && defined(__EMSCRIPTEN__)
    const int detected = emscripten_navigator_hardware_concurrency();
    if (detected <= 0) {
        return DefaultMaxThreads;
    }
    return detected;
#else
    const unsigned detected = std::thread::hardware_concurrency();
    if (detected == 0) {
        return DefaultMaxThreads;
    }
    return static_cast<int>(detected);
#endif
}

void runSerial(std::size_t begin, std::size_t end, const std::function<void(std::size_t)>& body) {
    for (std::size_t index = begin; index < end; ++index) {
        body(index);
    }
}

} // namespace

class TaskGroup::Impl {
public:
#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
    tbb::task_group group;
#elif defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    std::atomic<std::uint32_t> pending{0};
    std::atomic<std::uint32_t> failed{0};
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD)
    std::vector<std::future<void>> futures;
#else
    std::exception_ptr serial_exception;
#endif
};

const char* backendName(Backend backend) {
    switch (backend) {
        case Backend::Serial:
            return "serial";
        case Backend::StdThread:
            return "std_thread";
        case Backend::PThread:
            return "pthread";
        case Backend::WasmThread:
            return "wasm_thread";
        case Backend::Tbb:
            return "tbb";
    }
    return "unknown";
}

Backend compileBackend() {
#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
    return Backend::Tbb;
#elif defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    return Backend::WasmThread;
#elif defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD)
    return Backend::PThread;
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD)
    return Backend::StdThread;
#else
    return Backend::Serial;
#endif
}

int maxConcurrency(int requested_threads) {
    const int requested = requested_threads > 0 ? requested_threads : hardwareConcurrency();
    return std::max(1, std::min(requested, DefaultMaxThreads));
}

void initialize(Options options) {
#if defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    if (initialized.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    init_options = options;
    wasm_worker_pool.reset(new WasmWorkerPool(static_cast<std::size_t>(maxConcurrency(init_options.threads))));
#else
    std::call_once(init_flag, [&]() {
        init_options = options;
        const int threads = maxConcurrency(init_options.threads);

#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
        tbb_control.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, threads));
        tbb_arena.reset(new tbb::task_arena(threads));
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD)
        custom_pool.reset(new ThreadPool(static_cast<std::size_t>(threads)));
#endif
    });
#endif
}

void warmup(Options options) {
    initialize(options);
    parallelFor(0, static_cast<std::size_t>(maxConcurrency(options.threads)), [](std::size_t) {});
}

TaskGroup::TaskGroup()
    : impl_(new Impl()) {
    initialize();
}

TaskGroup::~TaskGroup() = default;

void TaskGroup::run(std::function<void()> task) {
#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
    tbb_arena->execute([&]() {
        impl_->group.run(std::move(task));
    });
#elif defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    wasm_worker_pool->submit(std::move(task), impl_->pending, impl_->failed);
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD)
    impl_->futures.push_back(custom_pool->submit(std::move(task)));
#else
    if (impl_->serial_exception) {
        return;
    }

    try {
        task();
    } catch (...) {
        impl_->serial_exception = std::current_exception();
    }
#endif
}

void TaskGroup::wait() {
#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
    tbb_arena->execute([&]() {
        impl_->group.wait();
    });
#elif defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    wasm_worker_pool->wait(impl_->pending);
    if (impl_->failed.load(std::memory_order_acquire) != 0) {
        throw std::runtime_error("A Wasm Worker task failed.");
    }
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD)
    for (std::future<void>& future : impl_->futures) {
        while (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            custom_pool->runPendingTask();
        }
        future.get();
    }
    impl_->futures.clear();
#else
    if (impl_->serial_exception) {
        std::rethrow_exception(impl_->serial_exception);
    }
#endif
}

void parallelFor(std::size_t begin, std::size_t end, const std::function<void(std::size_t)>& body) {
    parallelFor(begin, end, 1, body);
}

void parallelFor(
    std::size_t begin,
    std::size_t end,
    std::size_t grain_size,
    const std::function<void(std::size_t)>& body
) {
    if (end <= begin) {
        return;
    }

    initialize();
    grain_size = normalizeGrain(grain_size);

#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
    tbb_arena->execute([&]() {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(begin, end, grain_size),
            [&](const tbb::blocked_range<std::size_t>& range) {
                for (std::size_t index = range.begin(); index != range.end(); ++index) {
                    body(index);
                }
            }
        );
    });
#elif defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD)
    TaskGroup group;
    for (std::size_t block_begin = begin; block_begin < end; block_begin += grain_size) {
        const std::size_t block_end = std::min(end, block_begin + grain_size);
        group.run([block_begin, block_end, &body]() {
            runSerial(block_begin, block_end, body);
        });
    }
    group.wait();
#else
    runSerial(begin, end, body);
#endif
}

} // namespace parallel
