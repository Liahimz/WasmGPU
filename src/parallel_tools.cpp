#include "parallel_tools.h"
#include <emscripten/emscripten.h>


std::atomic<bool> keep_alive{false};

void idle_main_loop() {
  // No-op
}

extern "C" {
  void  start_tbb_session() {
    std::cout << "start_tbb_session" << std::endl;
   
  //  TbbInitializer::Initialize();
  //  TbbInitializer::ThreadWarmUp();
    
  }

  void  start_keepalive_mainloop() {
    std::cout << "start_keepalive_mainloop" << std::endl;
    if (!keep_alive.exchange(true)) {
      std::cout << "Started dummy main loop to keep WASM alive" << std::endl;
      emscripten_set_main_loop(idle_main_loop, 1, 1);
    }
  }

  void stop_keepalive_mainloop() {
    std::cout << "stop_keepalive_mainloop" << std::endl;
    if (keep_alive.exchange(false)) {
        emscripten_cancel_main_loop();
        std::cout << "Stopped dummy main loop" << std::endl;
    }
  }

}

int CurMaxTotalConcurrency() {
  return std::min(WASM_MAX_THREADS, tbb::this_task_arena::max_concurrency());
}


std::once_flag TbbInitializer::initFlag;
std::once_flag TbbInitializer::warmupFlag;
std::unique_ptr<tbb::task_arena> TbbInitializer::task_arena_;


int TbbInitMT(int threads) {
  
  int num_threads = threads == -1 ? CurMaxTotalConcurrency() : threads;
  std::atomic<int> barrier{num_threads};
 

  std::mutex mtx;
  std::condition_variable cv;
  std::atomic<bool> ready(false);
  auto start_time = std::chrono::steady_clock::now();
  auto timeout_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);

  TbbInitializer::GetArena().execute([&] {
    tbb::parallel_for(0, num_threads, [&](int) {
      if (--barrier == 0) {
        std::cout << "UP b" << barrier << std::endl;
        ready = true;
      } else {
        std::cout << "Down b" << barrier << std::endl;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (!ready && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        if (!ready) {
            std::cerr << "Timeout waiting for threads to synchronize" << std::endl;
        }
        // while (!ready) std::this_thread::yield();
      }
    }, tbb::static_partitioner{});
  });


  return 0;
}


int32_t TbbInitializer::ThreadWarmUp(int threads) {
  int32_t ret_code = 0;
  std::cout << "Thread WarmUp" << std::endl;
  std::call_once(warmupFlag, [&]() {
    std::cout << "Thread WarmUp Once" << std::endl;
    ret_code = TbbInitMT(threads);
  });
  return ret_code;
}


int TbbInitializer::Initialize(int threads) {
  std::cout << "Called Initialize " << std::endl;
  int num_threads = threads == -1 ? CurMaxTotalConcurrency() : threads;
  std::call_once(initFlag, [&]() {
    std::cout << "Initialize called once" << std::endl;
    task_arena_.reset(new tbb::task_arena(num_threads));
  });
  return 0;
}

tbb::task_arena& TbbInitializer::GetArena() {
  Initialize(); 
  return *task_arena_;
}