# WasmGPU

Small pet project for learning how WebAssembly, C++, JavaScript, pthreads, SIMD, and WebGPU fit together in the browser.

The current main demo preprocesses an image in C++/WASM, runs a tiny LeNet-like network, and compares several execution paths:

- C++ preprocessing + C++ WebGPU compute through `wasm_webgpu`
- C++ preprocessing + C++ WebGPU compute through Emscripten/Dawn `emdawnwebgpu`
- C++ preprocessing + CPU scalar inference
- C++ preprocessing + CPU SIMD inference
- C++ preprocessing + CPU SIMD plus pthreaded convolution
- Synthetic large CPU/GPU benchmarks for easier timing experiments

This is intentionally simple and experimental. The point is to make the data movement, synchronization, and timing visible.

## Build Modes

The build script supports five modes:

| Mode | Meaning | Sample folder |
| --- | --- | --- |
| `cpp-webgpu` | Default desktop benchmark mode. C++/WASM owns WebGPU through `thirdparty/wasm_webgpu` and uses JSPI for sync-looking readback. | `samples/CppWebGpu` |
| `cpp-webgpu-async` | Mobile-friendly C++ WebGPU mode. Uses async readback callbacks instead of JSPI / `WebAssembly.Suspending`. | `samples/CppWebGpuAsync` |
| `cpp-webgpu-dawn` | Experimental comparison mode. Uses Emscripten's Dawn-backed `--use-port=emdawnwebgpu` WebGPU binding with the same async benchmark UI and CPU paths. Requires Emscripten 4.0.10+. | `samples/CppWebGpuAsync` |
| `js-webgpu` | Older path where JavaScript owns WebGPU and calls WASM for preprocessing/postprocessing. | `samples/JsWebGpu` |
| `dummy` | Minimal CPU/sample mode for older experiments. | `samples/Dummy` |

Default:

```bash
python3 ./build.py
```

Explicit mode:

```bash
python3 ./build.py --mode cpp-webgpu
python3 ./build.py --mode cpp-webgpu-async
python3 ./build.py --mode cpp-webgpu-dawn
python3 ./build.py --mode js-webgpu
python3 ./build.py --mode dummy
```

CPU threaded paths use `wasm-thread` by default. You can switch the backend at build time:

```bash
python3 ./build.py --parallel-backend wasm-thread
python3 ./build.py --parallel-backend pthread
python3 ./build.py --parallel-backend std-thread
python3 ./build.py --parallel-backend tbb
python3 ./build.py --parallel-backend serial
```

For `cpp-webgpu-dawn`, Closure is disabled by default because the current Emscripten/Dawn combo can fail Closure on generated exception glue. To force it, add `--emdawn-closure`.

The script writes the runnable site into `build/`.

## Requirements

On macOS:

- Emscripten SDK activated in your shell
- CMake
- Python 3
- Node/npm, only for serving the built sample
- A browser with WebGPU support

Activate Emscripten before building:

```bash
source /path/to/emsdk/emsdk_env.sh
```

## Build And Run

Build the default C++ WebGPU sample:

```bash
python3 ./build.py --mode cpp-webgpu
```

Serve it:

```bash
cd build
npx serve
```

Open the shown local URL in a WebGPU-capable browser.

The app loads `sample.html`, starts `worker.js`, loads `wasm_gpu.js`, and runs the selected image through the engine.

## Project Layout

```text
.
├── build.py                  # Build helper for all modes
├── CMakeLists.txt            # Emscripten/CMake target setup
├── include/                  # C++ headers
├── src/                      # C++ implementation and embind bindings
├── shaders/                  # WGSL compute shaders embedded into C++
├── samples/                  # Browser sample files per build mode
├── network_data/             # Embedded trained/random network weights
├── network_trainer/          # PyTorch MNIST trainer/export helper
├── build_scripts/            # Generators for embedded headers
├── cmake/                    # Header templates used by build scripts
└── thirdparty/               # External dependencies, including wasm_webgpu and TBB
```

## Main C++ Pieces

| File | Purpose |
| --- | --- |
| `src/wasm_gpu_engine.cpp` | High-level engine. Owns preprocessing, CPU executor, GPU executor, and timing logs. |
| `src/gpu_executor.cpp` | C++ WebGPU executor using `wasm_webgpu`. Creates buffers, pipelines, bind groups, dispatches shaders, and reads results back. |
| `src/cpp_executor.cpp` | CPU implementation of the same tiny network plus synthetic large benchmark. Supports scalar, SIMD, and pthreaded modes. |
| `src/network_weights.cpp` | Loads embedded network weights from generated C++ data. |
| `src/bindings.cpp` | Emscripten embind exports to JavaScript. |
| `src/gpu_engine.cpp` | Older JS-WebGPU-facing engine. |
| `src/dummy_engine.cpp` | Minimal older sample engine. |

## Shader Layout

Tiny network shaders:

- `shaders/conv_relu.wgsl`
- `shaders/linear.wgsl`

Synthetic large benchmark shaders:

- `shaders/large_conv_relu.wgsl`
- `shaders/large_linear_partial.wgsl`
- `shaders/large_linear_reduce.wgsl`

The shaders are embedded at build time by:

```text
build_scripts/embed_shaders.py
```

That script generates:

```text
cmake-build-*/src/embedded_shaders.h
```

So C++ can refer to shader source as constants such as:

```cpp
internal::CONV_RELU_WGSL
internal::LARGE_LINEAR_PARTIAL_WGSL
```

## Network Weights

The tiny network is:

```text
28x28x1 input
3x3 conv, 4 channels
ReLU
flatten
linear to 10 logits
argmax in C++
```

Weights live in `network_data/` as float32 binary files. They are embedded into the WASM binary at build time by:

```text
build_scripts/embed_network_data.py
```

The generated header is:

```text
cmake-build-*/src/embedded_data.h
```

To train/export new MNIST weights:

```bash
cd network_trainer
python3 train_mnist.py
```

The trainer expects `torch` and `numpy`.

## Browser Flow

For `cpp-webgpu` mode:

```text
samples/CppWebGpu/app.js
  -> loads image in browser
  -> sends RGBA pixels to worker

samples/CppWebGpu/worker.js
  -> loads wasm_gpu.js
  -> creates Module.GpuEngine
  -> calls configure(28)
  -> calls process(...) for C++ WebGPU
  -> calls processCpu(...) for CPU modes
  -> calls benchmarkGpuLarge() / benchmarkCpuLarge(...)
  -> sends result back to app.js

C++ WasmGpuEngine
  -> preprocesses image to 28x28 grayscale
  -> runs GPU or CPU executor
  -> returns prediction and preprocessed image
```

## Timing Output

The C++ code prints timing lines to the browser console through `std::cout`.

Tiny real-image path:

```text
[timing] gpu preprocess=...ms inference=...ms total=...ms prediction=...
[timing] cpu mode=scalar preprocess=...ms inference=...ms total=...ms prediction=...
[timing] cpu mode=simd preprocess=...ms inference=...ms total=...ms prediction=...
[timing] cpu mode=simd_threads preprocess=...ms inference=...ms total=...ms prediction=...
```

Large synthetic path:

```text
[timing] synthetic_gpu_large input=1000x500 kernel=5x3 inference=...ms prediction=...
[timing] synthetic_cpu_large mode=scalar input=1000x500 kernel=5x3 inference=...ms prediction=...
[timing] synthetic_cpu_large mode=simd input=1000x500 kernel=5x3 inference=...ms prediction=...
[timing] synthetic_cpu_large mode=simd_threads input=1000x500 kernel=5x3 inference=...ms prediction=...
```

The synthetic large benchmark keeps weights and pipelines prepared, but regenerates the input tensor from a per-run seed. That models a frozen network receiving different activations/inputs each inference.

Synthetic benchmark data is generated once when the worker initializes:

```text
[timing] synthetic_gpu_large_data_prepare elapsed=...ms
[timing] synthetic_large_data_prepare elapsed=...ms
```

The first synthetic GPU use still has a preparation cost because it allocates WebGPU buffers, uploads the precomputed synthetic weights, creates shader modules, and creates compute pipelines. The benchmark reports that as `prepare=...ms` on the synthetic GPU timing line, plus a resource detail line. `synth_data` should be near zero when the worker precompute ran first.

```text
[timing] synthetic_gpu_large_async_start input=1000x500 kernel=5x3 seed=... prepare=...ms submit=...ms
[timing] synthetic_gpu_large_resource_detail buffers=...ms timestamp=...ms synth_data=...ms data_upload=...ms layouts=...ms pipelines=...ms bind_groups=...ms total=...ms
```

GPU detail timing:

```text
[timing] gpu_detail input_convert=...ms upload=...ms encode_submit=...ms sync_readback=...ms gpu_conv=...ms gpu_linear=...ms gpu_total=...ms
[timing] synthetic_gpu_large_detail input_generation=...ms upload=...ms encode_submit=...ms sync_readback=...ms gpu_conv=...ms gpu_linear_partial=...ms gpu_linear_reduce=...ms gpu_total=...ms
```

Meaning:

- `input_convert`: CPU-side conversion from 8-bit preprocessed pixels to float input for the tiny network.
- `input_generation`: CPU-side synthetic input generation for the large benchmark. This is part of end-to-end benchmark time, but not GPU utilization.
- `upload`: `wgpu_queue_write_buffer` time for the input buffer.
- `encode_submit`: CPU-side command recording and queue submit time.
- `sync_readback`: blocking wait at `wgpu_buffer_map_sync`. This includes waiting for GPU completion, buffer copies, mapping, and browser/WASM synchronization overhead.
- `gpu_*`: GPU timestamp-query measurements for actual compute passes.
- `gpu_total`: sum of GPU compute pass times. This is inside the `sync_readback` window, not added after it.

So this is expected:

```text
inference ~= input_convert/input_generation + upload + encode_submit + sync_readback + small CPU bookkeeping
```

Do not add `sync_readback + gpu_total`; that double-counts the GPU compute time.

## CPU Benchmark Modes

`processCpu(...)` and `benchmarkCpuLarge(...)` use integer modes:

| Mode | Name | What it does |
| --- | --- | --- |
| `0` | `scalar` | Plain CPU loops. |
| `1` | `simd` | SIMD dot products for the linear layer when WASM SIMD is enabled. |
| `2` | `simd_threads` | Parallel convolution plus SIMD linear layer. |

The parallel backend is selected at build time with `--parallel-backend`. `wasm-thread` is the default for browser/WASM builds; `tbb`, `pthread`, `std-thread`, and `serial` are available for comparisons.

Threaded CPU work goes through `thread_tools/parallel_utils.h`, which exposes `parallel::TaskGroup` and `parallel::parallelFor(...)`. TBB builds map those calls to TBB primitives, `std-thread` and `pthread` builds use the local custom thread pool in `thread_tools/thread_pool.hpp`, and `wasm-thread` builds use Emscripten Wasm Workers from `thread_tools/wasm_worker_pool.h`.

## Notes On WebGPU Synchronization

The C++ WebGPU path intentionally uses synchronous readback:

```cpp
wgpu_buffer_map_sync(...)
```

That makes the C++ control flow easy to understand:

```text
submit GPU work
wait for result
continue in C++
```

It is useful for learning and benchmarking synchronization cost, but it is not the fastest way to structure a real GPU pipeline. For larger systems, it is usually better to keep intermediate tensors on the GPU and only read back final results when needed.

For iPhone/mobile Safari, use:

```bash
python3 ./build.py --mode cpp-webgpu-async
```

That mode avoids JSPI and `WebAssembly.Suspending`. It starts GPU work from C++, returns to JavaScript, and completes through `wgpu_buffer_map_async` callbacks. The worker waits by polling `inferencePending()` and then reads `latestPrediction()`.

## Useful Commands

Build default mode:

```bash
python3 ./build.py
```

Build C++ WebGPU mode:

```bash
python3 ./build.py --mode cpp-webgpu
```

Build C++ WebGPU async mode:

```bash
python3 ./build.py --mode cpp-webgpu-async
```

Build JS WebGPU mode:

```bash
python3 ./build.py --mode js-webgpu
```

Build dummy mode:

```bash
python3 ./build.py --mode dummy
```

Serve:

```bash
cd build
npx serve
```
