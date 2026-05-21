#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "network_data" / "resnet50"
DEFAULT_CACHE_DIR = ROOT / "network_trainer" / "data" / "torch_cache"
RESNET50_LAYERS = ("layer1", "layer2", "layer3", "layer4")


def fold_conv_batchnorm(conv, bn):
    """Return Conv2d weights/bias with BatchNorm2d folded into the conv."""
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("fold_conv_batchnorm expects conv to be nn.Conv2d")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("fold_conv_batchnorm expects bn to be nn.BatchNorm2d")

    weight = conv.weight.detach().float()
    if conv.bias is None:
        bias = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
    else:
        bias = conv.bias.detach().float()

    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    running_mean = bn.running_mean.detach().float()
    running_var = bn.running_var.detach().float()
    scale = gamma / torch.sqrt(running_var + bn.eps)

    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = beta + (bias - running_mean) * scale
    return folded_weight.contiguous(), folded_bias.contiguous()


def resnet50_conv_bn_pairs(model):
    yield "conv1", model.conv1, model.bn1

    for layer_name in RESNET50_LAYERS:
        layer = getattr(model, layer_name)
        for block_index, block in enumerate(layer):
            block_name = f"{layer_name}_block{block_index}"
            yield f"{block_name}_conv1", block.conv1, block.bn1
            yield f"{block_name}_conv2", block.conv2, block.bn2
            yield f"{block_name}_conv3", block.conv3, block.bn3
            if block.downsample is not None:
                yield f"{block_name}_downsample", block.downsample[0], block.downsample[1]


def load_torchvision_resnet50(cache_dir, weights):
    os.environ.setdefault("TORCH_HOME", str(cache_dir))
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required to load pretrained ResNet50. "
            "Install it in this Python environment, then rerun this exporter."
        ) from exc

    selected_weights = None if weights == "random" else ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=selected_weights)
    model.eval()
    return model


def export_folded_conv_npz(out_dir, cache_dir, weights):
    model = load_torchvision_resnet50(cache_dir, weights)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    metadata = []
    for name, conv, bn in resnet50_conv_bn_pairs(model):
        folded_weight, folded_bias = fold_conv_batchnorm(conv, bn)
        weight_key = f"{name}_weights"
        bias_key = f"{name}_bias"
        arrays[weight_key] = folded_weight.cpu().numpy().astype("<f4")
        arrays[bias_key] = folded_bias.cpu().numpy().astype("<f4")
        arrays[weight_key].tofile(out_dir / f"{weight_key}.bin")
        arrays[bias_key].tofile(out_dir / f"{bias_key}.bin")
        metadata.append(
            {
                "name": name,
                "weights": f"{weight_key}.bin",
                "weights_shape": list(arrays[weight_key].shape),
                "bias": f"{bias_key}.bin",
                "bias_shape": list(arrays[bias_key].shape),
                "stride": list(conv.stride),
                "padding": list(conv.padding),
                "kernel": list(conv.kernel_size),
                "groups": conv.groups,
            }
        )

    suffix = "imagenet" if weights == "imagenet" else "random"
    npz_path = out_dir / f"resnet50_folded_conv_bn_{suffix}.npz"
    meta_path = out_dir / f"resnet50_folded_conv_bn_{suffix}.json"
    np.savez(npz_path, **arrays)
    meta_path.write_text(json.dumps({"folded_convs": metadata}, indent=2) + "\n")
    print(f"Exported {len(metadata)} folded conv+bn pairs")
    print(f"Wrote raw folded .bin tensors to {out_dir}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {meta_path}")


def network_data_name(path):
    try:
        return path.relative_to(ROOT / "network_data").as_posix()
    except ValueError:
        return path.name


def load_folded_metadata(out_dir, weights):
    suffix = "imagenet" if weights == "imagenet" else "random"
    meta_path = out_dir / f"resnet50_folded_conv_bn_{suffix}.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run with --export-folded-convs first, or use it in the same command."
        )
    data = json.loads(meta_path.read_text())
    return {item["name"]: item for item in data["folded_convs"]}


def conv_layer(metadata, name, input_name, output_name):
    item = metadata[name]
    return {
        "name": name,
        "type": "conv2d",
        "input": input_name,
        "output": output_name,
        "in_channels": item["weights_shape"][1],
        "out_channels": item["weights_shape"][0],
        "kernel": item["kernel"],
        "stride": item["stride"],
        "padding": item["padding"],
        "weights": item["weights_embedded"],
        "weights_shape": item["weights_shape"],
        "bias": item["bias_embedded"],
        "bias_shape": item["bias_shape"],
    }


def relu_layer(name, input_name, output_name):
    return {
        "name": name,
        "type": "relu",
        "input": input_name,
        "output": output_name,
    }


def add_layer(name, lhs, rhs, output_name):
    return {
        "name": name,
        "type": "add",
        "lhs": lhs,
        "rhs": rhs,
        "output": output_name,
    }


def maxpool_layer(name, input_name, output_name, kernel, stride, padding):
    return {
        "name": name,
        "type": "maxpool2d",
        "input": input_name,
        "output": output_name,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
    }


def prepare_metadata_for_manifest(metadata, out_dir):
    prepared = {}
    for name, item in metadata.items():
        copied = dict(item)
        copied["weights_embedded"] = network_data_name(out_dir / item["weights"])
        copied["bias_embedded"] = network_data_name(out_dir / item["bias"])
        prepared[name] = copied
    return prepared


def build_manifest_slice(metadata, slice_name):
    layers = [
        conv_layer(metadata, "conv1", "input", "stem_conv"),
        relu_layer("stem_relu", "stem_conv", "stem_relu"),
        maxpool_layer("stem_maxpool", "stem_relu", "stem_pool", [3, 3], [2, 2], [1, 1]),
    ]

    if slice_name == "stem-layer1-block0":
        block_input = "stem_pool"
        layers.extend(
            [
                conv_layer(metadata, "layer1_block0_conv1", block_input, "layer1_block0_conv1"),
                relu_layer("layer1_block0_relu1", "layer1_block0_conv1", "layer1_block0_relu1"),
                conv_layer(metadata, "layer1_block0_conv2", "layer1_block0_relu1", "layer1_block0_conv2"),
                relu_layer("layer1_block0_relu2", "layer1_block0_conv2", "layer1_block0_relu2"),
                conv_layer(metadata, "layer1_block0_conv3", "layer1_block0_relu2", "layer1_block0_main"),
                conv_layer(metadata, "layer1_block0_downsample", block_input, "layer1_block0_skip"),
                add_layer("layer1_block0_add", "layer1_block0_main", "layer1_block0_skip", "layer1_block0_add"),
                relu_layer("layer1_block0_relu3", "layer1_block0_add", "layer1_block0_out"),
            ]
        )
    elif slice_name != "stem":
        raise ValueError(f"Unsupported manifest slice: {slice_name}")

    return {
        "name": f"resnet50_{slice_name.replace('-', '_')}",
        "dtype": "float32",
        "endianness": "little",
        "input": "input",
        "input_shape": [3, 224, 224],
        "layers": layers,
    }


def export_manifest_slice(out_dir, weights, slice_name):
    metadata = prepare_metadata_for_manifest(load_folded_metadata(out_dir, weights), out_dir)
    manifest = build_manifest_slice(metadata, slice_name)
    manifest_path = out_dir / f"{manifest['name']}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


def verify_folding(seed, batches, channels, height, width):
    torch.manual_seed(seed)
    conv = nn.Conv2d(channels, 5, kernel_size=3, stride=2, padding=1, bias=False)
    bn = nn.BatchNorm2d(5)
    bn.running_mean = torch.randn(5)
    bn.running_var = torch.rand(5) + 0.1
    bn.weight.data = torch.randn(5)
    bn.bias.data = torch.randn(5)
    conv.eval()
    bn.eval()

    x = torch.randn(batches, channels, height, width)
    expected = bn(conv(x))
    folded_weight, folded_bias = fold_conv_batchnorm(conv, bn)
    folded = nn.Conv2d(
        channels,
        5,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    folded.weight.data.copy_(folded_weight)
    folded.bias.data.copy_(folded_bias)
    folded.eval()

    actual = folded(x)
    max_abs = torch.max(torch.abs(expected - actual)).item()
    mean_abs = torch.mean(torch.abs(expected - actual)).item()
    print(f"folding verification max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}")
    if max_abs > 1e-5:
        raise RuntimeError("Folded conv output differs from conv+batchnorm")


def main():
    parser = argparse.ArgumentParser(description="Export folded torchvision ResNet50 weights for the WasmGPU graph runtime.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Torch model cache directory.")
    parser.add_argument("--weights", choices=["imagenet", "random"], default="imagenet", help="Use pretrained ImageNet weights or random torchvision initialization.")
    parser.add_argument("--verify-folding", action="store_true", help="Run a deterministic Conv2d+BatchNorm folding self-check.")
    parser.add_argument("--export-folded-convs", "--export-folded-conv", action="store_true", help="Load pretrained torchvision ResNet50 and export folded conv weights to NPZ/raw bins.")
    parser.add_argument("--export-manifest-slice", choices=["stem", "stem-layer1-block0"], help="Write a static graph manifest for an incremental ResNet50 slice.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.verify_folding:
        verify_folding(args.seed, batches=2, channels=3, height=17, width=19)

    if args.export_folded_convs:
        export_folded_conv_npz(args.out_dir, args.cache_dir, args.weights)

    if args.export_manifest_slice:
        export_manifest_slice(args.out_dir, args.weights, args.export_manifest_slice)

    if not args.verify_folding and not args.export_folded_convs and not args.export_manifest_slice:
        parser.error("Choose --verify-folding, --export-folded-convs, and/or --export-manifest-slice")


if __name__ == "__main__":
    main()
