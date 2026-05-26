#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import torch

from pathlib import Path

from common import (
    ARTIFACT_DIR,
    DEFAULT_IMAGE,
    DEFAULT_INPUT_BIN,
    DEFAULT_INPUT_NPY,
    DEFAULT_ONNX,
    ROOT,
    ensure_dirs,
    load_torchvision_resnet50,
    preprocess_image_like_wasm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ImageNet ResNet50 ONNX and a shared benchmark input tensor.")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX))
    parser.add_argument("--input-npy", type=str, default=str(DEFAULT_INPUT_NPY))
    parser.add_argument("--input-bin", type=str, default=str(DEFAULT_INPUT_BIN))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ensure_dirs()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    x = preprocess_image_like_wasm(image_path)
    np.save(args.input_npy, x)
    x.astype("<f4", copy=False).tofile(args.input_bin)

    model, _ = load_torchvision_resnet50(ROOT / "network_trainer" / "data" / "torch_cache")
    dummy = torch.from_numpy(x)
    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Wrote {args.onnx}")
    print(f"Wrote {args.input_npy}")
    print(f"Wrote {args.input_bin}")
    print(f"Artifacts directory: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
