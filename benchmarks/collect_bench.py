#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from common import RESULTS_DIR, ROOT, format_ms


ROWS = [
    ("wasm_gpu", "WASM GPU"),
    ("wasm_cpu_simd_threads", "WASM CPU SIMD threads"),
    ("onnx_webgpu_runtime_chrome", "ONNX WebGPU runtime (Chrome)"),
    ("onnx_webgpu_runtime_safari", "ONNX WebGPU runtime (Safari)"),
    ("nodejs_cpu", "Node.js CPU"),
    ("torch_local_gpu", "Torch local GPU"),
    ("torch_local_cpu", "Torch local CPU"),
]


def load_result(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def row(name: str, label: str) -> str:
    data = load_result(name)
    if data is None:
        return f"| {label} | - | - | - | - | - | - | - | - | - | missing `benchmarks/results/{name}.json` |"
    ready = data.get("summary", {})
    warmup = data.get("warmup_summary", {})
    prediction = data.get("prediction", "-")
    class_label = data.get("class_label", "")
    pred = f"{prediction} {class_label}".strip()
    return (
        f"| {label} | {format_ms(warmup.get('min_ms'))} | {format_ms(warmup.get('median_ms'))} | "
        f"{format_ms(warmup.get('max_ms'))} | {warmup.get('count', '-')} | "
        f"{format_ms(ready.get('min_ms'))} | {format_ms(ready.get('median_ms'))} | "
        f"{format_ms(ready.get('max_ms'))} | {ready.get('count', '-')} | {pred} | {data.get('backend', '-')} |"
    )


def main() -> None:
    lines = [
        "# ResNet50 Benchmark",
        "",
        "Single-image ImageNet ResNet50 inference using `test_data/Image_net_dog.png` and the shared preprocessed tensor in `benchmarks/artifacts/resnet50_input.npy`.",
        "",
        "| Backend | warmup best ms | warmup median ms | warmup worst ms | warmup runs | ready best ms | ready median ms | ready worst ms | ready runs | prediction | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(row(name, label) for name, label in ROWS)
    lines.append("")
    (ROOT / "bench.md").write_text("\n".join(lines))
    print(ROOT / "bench.md")


if __name__ == "__main__":
    main()
