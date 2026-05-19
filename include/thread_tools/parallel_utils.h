#pragma once

#include <cstddef>
#include <functional>
#include <memory>

namespace parallel {

enum class Backend {
    Serial,
    StdThread,
    PThread,
    WasmThread,
    Tbb,
};

struct Options {
    int threads = -1;
};

const char* backendName(Backend backend);
Backend compileBackend();
int maxConcurrency(int requested_threads = -1);

void initialize(Options options = {});
void warmup(Options options = {});

class TaskGroup {
public:
    TaskGroup();
    ~TaskGroup();

    TaskGroup(const TaskGroup&) = delete;
    TaskGroup& operator=(const TaskGroup&) = delete;

    void run(std::function<void()> task);
    void wait();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

void parallelFor(std::size_t begin, std::size_t end, const std::function<void(std::size_t)>& body);
void parallelFor(
    std::size_t begin,
    std::size_t end,
    std::size_t grain_size,
    const std::function<void(std::size_t)>& body
);

inline void parallel_for(std::size_t begin, std::size_t end, const std::function<void(std::size_t)>& body) {
    parallelFor(begin, end, body);
}

inline void parallel_for(
    std::size_t begin,
    std::size_t end,
    std::size_t grain_size,
    const std::function<void(std::size_t)>& body
) {
    parallelFor(begin, end, grain_size, body);
}

} // namespace parallel
