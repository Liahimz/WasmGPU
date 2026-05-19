#include "thread_tools/parallel_utils.h"

#include "thread_tools/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#endif

namespace parallel {
namespace {

#if !defined(WASM_MAX_THREADS)
constexpr int DefaultMaxThreads = 4;
#else
constexpr int DefaultMaxThreads = WASM_MAX_THREADS;
#endif

std::once_flag init_flag;
Options init_options;

#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
std::unique_ptr<tbb::task_arena> tbb_arena;
std::unique_ptr<tbb::global_control> tbb_control;
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
std::unique_ptr<ThreadPool> custom_pool;
#endif

std::size_t normalizeGrain(std::size_t grain_size) {
    return std::max<std::size_t>(grain_size, 1);
}

int hardwareConcurrency() {
    const unsigned detected = std::thread::hardware_concurrency();
    if (detected == 0) {
        return DefaultMaxThreads;
    }
    return static_cast<int>(detected);
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
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
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
    std::call_once(init_flag, [&]() {
        init_options = options;
        const int threads = maxConcurrency(init_options.threads);

#if defined(WASM_GPU_PARALLEL_BACKEND_TBB)
        tbb_control.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, threads));
        tbb_arena.reset(new tbb::task_arena(threads));
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
        custom_pool.reset(new ThreadPool(static_cast<std::size_t>(threads)));
#endif
    });
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
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
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
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
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
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
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
