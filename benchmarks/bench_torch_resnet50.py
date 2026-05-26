#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from common import (
    DEFAULT_INPUT_NPY,
    ROOT,
    labels_from_file,
    load_input_tensor,
    load_torchvision_resnet50,
    summarize_ms,
    time_callable,
    topk_text,
    write_result,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA is not available in this Python environment.")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS is not available in this Python environment.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark torchvision ResNet50 on local Torch CPU or GPU.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda", "mps"], default="auto")
    parser.add_argument("--input-npy", default=str(DEFAULT_INPUT_NPY))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, _ = load_torchvision_resnet50(ROOT / "network_trainer" / "data" / "torch_cache")
    model = model.to(device)
    x = load_input_tensor(args.input_npy).to(device)
    labels = labels_from_file()

    def run():
        with torch.inference_mode():
            y = model(x)
        synchronize(device)
        return y

    warmup_timings, timings, logits = time_callable(run, args.warmup, args.runs)
    prediction = int(torch.argmax(logits.detach().cpu()).item())
    result_name = f"torch_local_{'gpu' if device.type != 'cpu' else 'cpu'}"
    path = write_result(
        result_name,
        {
            "backend": f"torch:{device.type}",
            "warmup_summary": summarize_ms(warmup_timings),
            "summary": summarize_ms(timings),
            "prediction": prediction,
            "class_label": labels[prediction] if 0 <= prediction < len(labels) else "",
            "top5": topk_text(logits, labels),
            "torch": torch.__version__,
        },
    )
    print(path)
    print(f"{result_name}: median={summarize_ms(timings)['median_ms']:.3f} ms prediction={prediction}")


if __name__ == "__main__":
    main()
