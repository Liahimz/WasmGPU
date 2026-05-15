#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <memory>
#include <algorithm>
#include <map>
#include <iostream>


#if !defined(WASM_MAX_THREADS)
  #define WASM_MAX_THREADS 4
#endif


class TbbInitializer {
  public:
      static int Initialize(int threads = -1);
      static int ThreadWarmUp(int threads = -1);
      static tbb::task_arena& GetArena();
  private:
      static std::once_flag warmupFlag;
      static std::once_flag initFlag;
      static std::unique_ptr<tbb::task_arena> task_arena_;

};