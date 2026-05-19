#include "thread_tools/thread_pool.hpp"

#include <algorithm>

namespace parallel {

thread_local detail::WorkStealingQueue* ThreadPool::local_work_queue_ = nullptr;
thread_local std::uint8_t ThreadPool::local_task_depth_ = 0;
thread_local std::size_t ThreadPool::local_index_ = 0;

ThreadPool::ThreadPool(std::size_t max_concurrency) {
    const std::size_t thread_count = std::max<std::size_t>(1, max_concurrency);

    queues_.reserve(thread_count);
    for (std::size_t index = 0; index < thread_count; ++index) {
        queues_.push_back(std::unique_ptr<detail::WorkStealingQueue>(new detail::WorkStealingQueue()));
    }

#if defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    threads_.resize(thread_count);
    for (std::size_t index = 0; index < thread_count; ++index) {
        auto* args = new std::pair<ThreadPool*, std::size_t>(this, index);
        if (pthread_create(&threads_[index], nullptr, pthreadWorkerMain, args) != 0) {
            delete args;
            threads_.resize(index);
            break;
        }
    }
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD)
    threads_.reserve(thread_count);
    try {
        for (std::size_t index = 0; index < thread_count; ++index) {
            threads_.push_back(std::thread(&ThreadPool::workerThread, this, index));
        }
    } catch (...) {
        done_.store(true);
        throw;
    }
#else
    (void)thread_count;
#endif
}

ThreadPool::~ThreadPool() {
    done_.store(true);

#if defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    for (pthread_t& thread : threads_) {
        pthread_join(thread, nullptr);
    }
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD)
    for (std::thread& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
#endif
}

#if defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
void* ThreadPool::pthreadWorkerMain(void* user_data) {
    auto* args = static_cast<std::pair<ThreadPool*, std::size_t>*>(user_data);
    ThreadPool* pool = args->first;
    const std::size_t index = args->second;
    delete args;
    pool->workerThread(index);
    return nullptr;
}
#endif

void ThreadPool::workerThread(std::size_t index) {
    local_index_ = index;
    local_work_queue_ = queues_[index].get();

    while (!done_.load()) {
        runPendingTask();
    }
}

bool ThreadPool::popTaskFromLocalQueue(detail::FunctionWrapper& task) {
    return local_work_queue_ && local_work_queue_->tryPop(task);
}

bool ThreadPool::popTaskFromLocalQueue(detail::FunctionWrapper& task, std::uint8_t min_depth) {
    return local_work_queue_ && local_work_queue_->tryPop(task, min_depth);
}

bool ThreadPool::popTaskFromPoolQueue(detail::FunctionWrapper& task) {
    return pool_work_queue_.tryPop(task);
}

bool ThreadPool::popTaskFromPoolQueue(detail::FunctionWrapper& task, std::uint8_t min_depth) {
    return pool_work_queue_.tryPop(task, min_depth);
}

bool ThreadPool::popTaskFromOtherThreadQueue(detail::FunctionWrapper& task) {
    if (queues_.empty()) {
        return false;
    }

    for (std::size_t offset = 1; offset <= queues_.size(); ++offset) {
        const std::size_t index = (local_index_ + offset) % queues_.size();
        if (queues_[index]->trySteal(task)) {
            return true;
        }
    }
    return false;
}

void ThreadPool::runPendingTask() {
    detail::FunctionWrapper task;
    const std::uint8_t current_depth = localTaskDepth();

    if (current_depth > 0) {
        if (popTaskFromLocalQueue(task, current_depth) || popTaskFromPoolQueue(task, current_depth)) {
            setLocalTaskDepth(task.depth());
            task();
        } else {
            setLocalTaskDepth(static_cast<std::uint8_t>(current_depth - 1));
            std::this_thread::yield();
        }
        return;
    }

    if (popTaskFromLocalQueue(task) || popTaskFromPoolQueue(task) || popTaskFromOtherThreadQueue(task)) {
        setLocalTaskDepth(task.depth());
        task();
        setLocalTaskDepth(0);
    } else {
        std::this_thread::yield();
    }
}

std::size_t ThreadPool::maxConcurrency() const {
#if defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD) || defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD)
    return threads_.empty() ? 1 : threads_.size();
#else
    return 1;
#endif
}

std::uint8_t ThreadPool::localTaskDepth() {
    return local_task_depth_;
}

void ThreadPool::setLocalTaskDepth(std::uint8_t depth) {
    local_task_depth_ = depth;
}

} // namespace parallel
