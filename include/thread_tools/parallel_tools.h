#pragma once

#include "thread_tools/parallel_utils.h"

class TbbInitializer {
public:
    static int Initialize(int threads = -1);
    static int ThreadWarmUp(int threads = -1);
};

extern "C" {
void start_tbb_session();
void start_parallel_session();
void start_keepalive_mainloop();
void stop_keepalive_mainloop();
}
