#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace parallel {

class WasmWorkerPool {
public:
    explicit WasmWorkerPool(std::size_t max_concurrency);
    ~WasmWorkerPool();

    WasmWorkerPool(const WasmWorkerPool&) = delete;
    WasmWorkerPool& operator=(const WasmWorkerPool&) = delete;

    void submit(
        std::function<void()> task,
        std::atomic<std::uint32_t>& pending,
        std::atomic<std::uint32_t>& failed
    );
    void wait(std::atomic<std::uint32_t>& pending);
    std::size_t maxConcurrency() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel
