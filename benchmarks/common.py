#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "benchmarks" / "artifacts"
RESULTS_DIR = ROOT / "benchmarks" / "results"
DEFAULT_IMAGE = ROOT / "test_data" / "Image_net_dog.png"
DEFAULT_ONNX = ARTIFACT_DIR / "resnet50_imagenet.onnx"
DEFAULT_INPUT_NPY = ARTIFACT_DIR / "resnet50_input.npy"
DEFAULT_INPUT_BIN = ARTIFACT_DIR / "resnet50_input_f32.bin"
DEFAULT_LABELS = ROOT / "network_data" / "resnet50" / "imagenet_classes.json"


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_torchvision_resnet50(cache_dir: Path | None = None):
    if cache_dir is not None:
        os.environ.setdefault("TORCH_HOME", str(cache_dir))
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    labels = ResNet50_Weights.IMAGENET1K_V2.meta.get("categories", [])
    return model, labels


def _rescale_rgb_nearest(src: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
    src_height, src_width, channels = src.shape
    out = np.empty((dst_height, dst_width, 3), dtype=np.uint8)
    for y in range(dst_height):
        src_y = int(y * (float(src_height) / dst_height))
        src_y = min(src_y, src_height - 1)
        for x in range(dst_width):
            src_x = int(x * (float(src_width) / dst_width))
            src_x = min(src_x, src_width - 1)
            if channels == 1:
                out[y, x, :] = src[src_y, src_x, 0]
            else:
                out[y, x, :] = src[src_y, src_x, :3]
    return out


def preprocess_image_like_wasm(image_path: Path, resize_shorter_side: int = 256, crop_size: int = 224) -> np.ndarray:
    from PIL import Image

    image = Image.open(image_path)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA")
    src = np.asarray(image)
    if src.ndim == 2:
        src = src[:, :, None]

    height, width = src.shape[:2]
    if width < height:
        resized_width = resize_shorter_side
        resized_height = round(float(height) * resize_shorter_side / width)
    else:
        resized_height = resize_shorter_side
        resized_width = round(float(width) * resize_shorter_side / height)
    resized_width = max(resized_width, crop_size)
    resized_height = max(resized_height, crop_size)

    resized = _rescale_rgb_nearest(src, resized_width, resized_height)
    offset_x = max((resized_width - crop_size) // 2, 0)
    offset_y = max((resized_height - crop_size) // 2, 0)
    cropped = resized[offset_y : offset_y + crop_size, offset_x : offset_x + crop_size, :]

    rgb = cropped.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    chw = ((rgb - mean) / std).transpose(2, 0, 1)
    return chw[np.newaxis, :, :, :].astype(np.float32)


def load_input_tensor(input_npy: Path = DEFAULT_INPUT_NPY) -> torch.Tensor:
    return torch.from_numpy(np.load(input_npy)).contiguous()


def labels_from_file(path: Path = DEFAULT_LABELS) -> list[str]:
    if path.exists():
        return json.loads(path.read_text())
    _, labels = load_torchvision_resnet50(ROOT / "network_trainer" / "data" / "torch_cache")
    return list(labels)


def topk_text(logits: torch.Tensor, labels: list[str], count: int = 5) -> str:
    values, indices = torch.topk(logits.detach().float().cpu().reshape(-1), count)
    lines = []
    for value, index in zip(values.tolist(), indices.tolist()):
        label = labels[index] if 0 <= index < len(labels) else ""
        lines.append(f"{index} {label} {value:.6g}".strip())
    return "\n".join(lines)


def time_callable(fn, warmup: int, runs: int) -> tuple[list[float], list[float], torch.Tensor]:
    last = None
    warmup_timings = []
    for _ in range(warmup):
        start = time.perf_counter()
        last = fn()
        warmup_timings.append((time.perf_counter() - start) * 1000.0)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        last = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    if last is None:
        last = fn()
    return warmup_timings, timings, last


def summarize_ms(samples: list[float]) -> dict:
    if not samples:
        return {
            "count": 0,
            "min_ms": None,
            "median_ms": None,
            "max_ms": None,
        }
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "max_ms": ordered[-1],
    }


def write_result(name: str, payload: dict) -> Path:
    ensure_dirs()
    payload = {
        "name": name,
        "created_unix": time.time(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        **payload,
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def format_ms(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.3f}"
