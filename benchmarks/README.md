# ResNet50 Benchmark Collection

These benchmarks compare one ImageNet ResNet50 inference on `test_data/Image_net_dog.png`.
All non-WASM runners use the shared ONNX model and the shared preprocessed tensor in `benchmarks/artifacts/`.

## Prepare Shared Artifacts

Run this once, or rerun it after changing the image/model:

```bash
cd /Users/michael/Desktop/WasmGPU
python3 benchmarks/export_resnet50_onnx.py
```

This writes:

- `benchmarks/artifacts/resnet50_imagenet.onnx`
- `benchmarks/artifacts/resnet50_input.npy`
- `benchmarks/artifacts/resnet50_input_f32.bin`

## Torch Local CPU

```bash
cd /Users/michael/Desktop/WasmGPU
python3 benchmarks/bench_torch_resnet50.py --device cpu --runs 20 --warmup 5
```

Writes `benchmarks/results/torch_local_cpu.json`.

## Torch Local GPU

Use `auto` to select CUDA/MPS when available:

```bash
cd /Users/michael/Desktop/WasmGPU
python3 benchmarks/bench_torch_resnet50.py --device auto --runs 20 --warmup 5
```

Or force a backend:

```bash
python3 benchmarks/bench_torch_resnet50.py --device cuda --runs 20 --warmup 5
python3 benchmarks/bench_torch_resnet50.py --device mps --runs 20 --warmup 5
```

Writes `benchmarks/results/torch_local_gpu.json` when the selected device is not CPU.

## Node.js CPU

If packages are installed in this repo:

```bash
cd /Users/michael/Desktop/WasmGPU
npm install
node benchmarks/bench_node_onnx_cpu.mjs --runs 20 --warmup 5
```

If packages are installed in `/Users/michael/node_modules`:

```bash
cd /Users/michael/Desktop/WasmGPU
NODE_PATH=/Users/michael/node_modules node benchmarks/bench_node_onnx_cpu.mjs --runs 20 --warmup 5
```

Writes `benchmarks/results/nodejs_cpu.json`.

## ONNX WebGPU Runtime

Serve the repo root:

```bash
cd /Users/michael/Desktop/WasmGPU
python3 -m http.server 8787 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8787/benchmarks/onnx_webgpu_bench.html?v=2
```

Run the page, then write the displayed JSON to:

```text
benchmarks/results/onnx_webgpu_runtime_chrome.json
benchmarks/results/onnx_webgpu_runtime_safari.json
```

Use the browser name in the filename so `bench.md` keeps Chrome and Safari as separate rows.

## WASM GPU And WASM CPU SIMD Threads

Build the ResNet50 browser sample:

```bash
cd /Users/michael/Desktop/WasmGPU
python3 build.py --mode resnet50 --parallel-backend wasm-thread
cd build
python3 -m http.server 8789 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8789/sample.html
```

Use `test_data/Image_net_dog.png`, set the run count, and run the sample.

Create `benchmarks/results/wasm_gpu.json` from the displayed `gpu` row:

```json
{
  "name": "wasm_gpu",
  "backend": "wasm:webgpu",
  "warmup_summary": {
    "count": 1,
    "min_ms": 0,
    "median_ms": 0,
    "max_ms": 0
  },
  "summary": {
    "count": 20,
    "min_ms": 0,
    "median_ms": 0,
    "max_ms": 0
  },
  "prediction": 207,
  "class_label": "golden retriever"
}
```

Create `benchmarks/results/wasm_cpu_simd_threads.json` from the displayed `cpu_simd_threads_*` row:

```json
{
  "name": "wasm_cpu_simd_threads",
  "backend": "wasm:cpu_simd_threads",
  "warmup_summary": {
    "count": 1,
    "min_ms": 0,
    "median_ms": 0,
    "max_ms": 0
  },
  "summary": {
    "count": 20,
    "min_ms": 0,
    "median_ms": 0,
    "max_ms": 0
  },
  "prediction": 207,
  "class_label": "golden retriever"
}
```

Replace the zeroes with the measured values from the browser table.

## Regenerate bench.md

After collecting any result JSON:

```bash
cd /Users/michael/Desktop/WasmGPU
python3 benchmarks/collect_bench.py
```

`bench.md` is intentionally only the title, benchmark note, and summary table.
