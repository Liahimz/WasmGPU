#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
#include <pthread.h>
#endif

namespace parallel {

namespace detail {

class FunctionWrapper {
    struct ImplBase {
        virtual void call() = 0;
        virtual ~ImplBase() = default;
    };

    template<typename F>
    struct Impl final : ImplBase {
        explicit Impl(F&& f_) : f(std::move(f_)) {}
        void call() override { f(); }
        F f;
    };

public:
    FunctionWrapper() = default;

    template<typename F>
    explicit FunctionWrapper(F&& f, std::uint8_t depth = 1)
        : impl_(new Impl<F>(std::move(f))),
          depth_(depth) {}

    FunctionWrapper(FunctionWrapper&& other) noexcept = default;
    FunctionWrapper& operator=(FunctionWrapper&& other) noexcept = default;

    FunctionWrapper(const FunctionWrapper&) = delete;
    FunctionWrapper& operator=(const FunctionWrapper&) = delete;

    void operator()() {
        impl_->call();
    }

    std::uint8_t depth() const {
        return depth_;
    }

    explicit operator bool() const {
        return static_cast<bool>(impl_);
    }

private:
    std::unique_ptr<ImplBase> impl_;
    std::uint8_t depth_ = 1;
};

class WorkStealingQueue {
public:
    WorkStealingQueue() = default;
    WorkStealingQueue(const WorkStealingQueue&) = delete;
    WorkStealingQueue& operator=(const WorkStealingQueue&) = delete;

    void push(FunctionWrapper task) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_front(std::move(task));
    }

    bool tryPop(FunctionWrapper& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        task = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    bool tryPop(FunctionWrapper& task, std::uint8_t min_depth) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = queue_.begin(); it != queue_.end(); ++it) {
            if (it->depth() >= min_depth) {
                task = std::move(*it);
                queue_.erase(it);
                return true;
            }
        }
        return false;
    }

    bool trySteal(FunctionWrapper& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        task = std::move(queue_.back());
        queue_.pop_back();
        return true;
    }

private:
    std::deque<FunctionWrapper> queue_;
    mutable std::mutex mutex_;
};

class GlobalTaskQueue {
public:
    void push(FunctionWrapper task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(task));
        }
    }

    bool tryPop(FunctionWrapper& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        task = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool tryPop(FunctionWrapper& task, std::uint8_t min_depth) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }

        std::queue<FunctionWrapper> skipped;
        bool found = false;
        while (!queue_.empty()) {
            FunctionWrapper current = std::move(queue_.front());
            queue_.pop();
            if (!found && current.depth() >= min_depth) {
                task = std::move(current);
                found = true;
                break;
            }
            skipped.push(std::move(current));
        }

        while (!queue_.empty()) {
            skipped.push(std::move(queue_.front()));
            queue_.pop();
        }
        queue_ = std::move(skipped);
        return found;
    }

private:
    std::queue<FunctionWrapper> queue_;
    std::mutex mutex_;
};

} // namespace detail

class ThreadPool {
public:
    explicit ThreadPool(std::size_t max_concurrency);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template<typename Function>
    auto submit(Function&& function) -> std::future<typename std::invoke_result<Function>::type> {
        using Result = typename std::invoke_result<Function>::type;

        std::packaged_task<Result()> packaged(std::forward<Function>(function));
        std::future<Result> result = packaged.get_future();
        const std::uint8_t depth = static_cast<std::uint8_t>(localTaskDepth() + 1);

        if (local_work_queue_) {
            local_work_queue_->push(detail::FunctionWrapper(std::move(packaged), depth));
        } else {
            pool_work_queue_.push(detail::FunctionWrapper(std::move(packaged), depth));
        }
        return result;
    }

    void runPendingTask();
    std::size_t maxConcurrency() const;

private:
#if defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    static void* pthreadWorkerMain(void* user_data);
#endif

    void workerThread(std::size_t index);
    bool popTaskFromLocalQueue(detail::FunctionWrapper& task);
    bool popTaskFromLocalQueue(detail::FunctionWrapper& task, std::uint8_t min_depth);
    bool popTaskFromPoolQueue(detail::FunctionWrapper& task);
    bool popTaskFromPoolQueue(detail::FunctionWrapper& task, std::uint8_t min_depth);
    bool popTaskFromOtherThreadQueue(detail::FunctionWrapper& task);

    static std::uint8_t localTaskDepth();
    static void setLocalTaskDepth(std::uint8_t depth);

    std::atomic<bool> done_{false};
    detail::GlobalTaskQueue pool_work_queue_;
    std::vector<std::unique_ptr<detail::WorkStealingQueue>> queues_;

#if defined(WASM_GPU_PARALLEL_BACKEND_PTHREAD) || defined(WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
    std::vector<pthread_t> threads_;
#elif defined(WASM_GPU_PARALLEL_BACKEND_STD_THREAD)
    std::vector<std::thread> threads_;
#endif

    static thread_local detail::WorkStealingQueue* local_work_queue_;
    static thread_local std::uint8_t local_task_depth_;
    static thread_local std::size_t local_index_;
};

} // namespace parallel
