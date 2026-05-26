# ResNet50 Benchmark

Single-image ImageNet ResNet50 inference using `test_data/Image_net_dog.png` and the shared preprocessed tensor in `benchmarks/artifacts/resnet50_input.npy`.

| Backend | warmup best ms | warmup median ms | warmup worst ms | warmup runs | ready best ms | ready median ms | ready worst ms | ready runs | prediction | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| WASM GPU | 96.260 | 96.260 | 96.260 | 1 | 74.960 | 81.040 | 85.740 | 5 | 207 golden retriever | wasm:webgpu |
| WASM CPU SIMD threads | 1312.600 | 1312.600 | 1312.600 | 1 | 1309.420 | 1347.640 | 1380.800 | 5 | 207 golden retriever | wasm:cpu_simd_threads |
| ONNX WebGPU runtime (Chrome) | 54.900 | 54.900 | 54.900 | 1 | 15.200 | 16.200 | 17.300 | 5 | 207 | onnxruntime-web:webgpu |
| ONNX WebGPU runtime (Safari) | 395.000 | 395.000 | 395.000 | 1 | 20.000 | 20.000 | 21.000 | 5 | 207 | onnxruntime-web:webgpu |
| Node.js CPU | 62.093 | 62.093 | 62.093 | 1 | 16.590 | 17.740 | 18.178 | 5 | 207 golden retriever | onnxruntime-node:cpu |
| Torch local GPU | - | - | - | - | - | - | - | - | - | missing `benchmarks/results/torch_local_gpu.json` |
| Torch local CPU | 21.384 | 21.384 | 21.384 | 1 | 13.722 | 13.979 | 14.793 | 5 | 207 golden retriever | torch:cpu |
