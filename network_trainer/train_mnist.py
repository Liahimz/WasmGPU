#!/usr/bin/env python3
import argparse
import gzip
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "network_trainer" / "data"
DEFAULT_OUT_DIR = ROOT / "network_data"

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


class TinyLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear0 = nn.Linear(16 * 14 * 14, 64)
        self.linear1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear0(x)
        return self.linear1(x)


class MnistDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index]).unsqueeze(0)
        label = int(self.labels[index])
        return image, label


def download_mnist(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        dst = data_dir / filename
        if dst.exists():
            continue

        errors = []
        for mirror in MNIST_MIRRORS:
            url = mirror + filename
            try:
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, dst)
                break
            except urllib.error.URLError as err:
                errors.append(f"{url}: {err}")
        else:
            raise RuntimeError("Could not download MNIST file:\n" + "\n".join(errors))


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} has invalid image magic {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(count, rows, cols).astype(np.float32) / 255.0
    return images


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} has invalid label magic {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError(f"{path} expected {count} labels, got {labels.shape[0]}")
    return labels.astype(np.int64)


def load_mnist(data_dir):
    download_mnist(data_dir)
    train_images = read_idx_images(data_dir / MNIST_FILES["train_images"])
    train_labels = read_idx_labels(data_dir / MNIST_FILES["train_labels"])
    test_images = read_idx_images(data_dir / MNIST_FILES["test_images"])
    test_labels = read_idx_labels(data_dir / MNIST_FILES["test_labels"])
    return train_images, train_labels, test_images, test_labels


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def export_weights(model, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()

    conv0_weights = model.conv0.weight.detach().numpy().astype("<f4")
    conv0_bias = model.conv0.bias.detach().numpy().astype("<f4")
    conv1_weights = model.conv1.weight.detach().numpy().astype("<f4")
    conv1_bias = model.conv1.bias.detach().numpy().astype("<f4")
    linear0_weights = model.linear0.weight.detach().numpy().astype("<f4")
    linear0_bias = model.linear0.bias.detach().numpy().astype("<f4")
    linear1_weights = model.linear1.weight.detach().numpy().astype("<f4")
    linear1_bias = model.linear1.bias.detach().numpy().astype("<f4")

    paths = {
        "conv0_weights": out_dir / "tiny_lenet_conv0_weights_f32.bin",
        "conv0_bias": out_dir / "tiny_lenet_conv0_bias_f32.bin",
        "conv1_weights": out_dir / "tiny_lenet_conv1_weights_f32.bin",
        "conv1_bias": out_dir / "tiny_lenet_conv1_bias_f32.bin",
        "linear0_weights": out_dir / "tiny_lenet_linear0_weights_f32.bin",
        "linear0_bias": out_dir / "tiny_lenet_linear0_bias_f32.bin",
        "linear1_weights": out_dir / "tiny_lenet_linear1_weights_f32.bin",
        "linear1_bias": out_dir / "tiny_lenet_linear1_bias_f32.bin",
        "npz": out_dir / "tiny_lenet_weights.npz",
        "manifest": out_dir / "tiny_lenet_manifest.json",
    }

    conv0_weights.tofile(paths["conv0_weights"])
    conv0_bias.tofile(paths["conv0_bias"])
    conv1_weights.tofile(paths["conv1_weights"])
    conv1_bias.tofile(paths["conv1_bias"])
    linear0_weights.tofile(paths["linear0_weights"])
    linear0_bias.tofile(paths["linear0_bias"])
    linear1_weights.tofile(paths["linear1_weights"])
    linear1_bias.tofile(paths["linear1_bias"])

    np.savez(
        paths["npz"],
        conv0_weights=conv0_weights,
        conv0_bias=conv0_bias,
        conv1_weights=conv1_weights,
        conv1_bias=conv1_bias,
        linear0_weights=linear0_weights,
        linear0_bias=linear0_bias,
        linear1_weights=linear1_weights,
        linear1_bias=linear1_bias,
    )

    manifest = {
        "name": "tiny_lenet",
        "dtype": "float32",
        "endianness": "little",
        "input_shape": [1, 28, 28],
        "layers": [
            {
                "name": "conv0",
                "type": "conv2d",
                "in_channels": 1,
                "out_channels": 8,
                "kernel": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "weights": paths["conv0_weights"].name,
                "weights_shape": [8, 1, 3, 3],
                "bias": paths["conv0_bias"].name,
                "bias_shape": [8],
            },
            {
                "name": "relu0",
                "type": "relu",
                "shape": [8, 28, 28],
            },
            {
                "name": "conv1",
                "type": "conv2d",
                "in_channels": 8,
                "out_channels": 16,
                "kernel": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "weights": paths["conv1_weights"].name,
                "weights_shape": [16, 8, 3, 3],
                "bias": paths["conv1_bias"].name,
                "bias_shape": [16],
            },
            {
                "name": "relu1",
                "type": "relu",
                "shape": [16, 28, 28],
            },
            {
                "name": "maxpool0",
                "type": "maxpool2d",
                "kernel": [2, 2],
                "stride": [2, 2],
                "padding": [0, 0],
            },
            {
                "name": "flatten0",
                "type": "flatten",
            },
            {
                "name": "linear0",
                "type": "linear",
                "in_features": 3136,
                "out_features": 64,
                "weights": paths["linear0_weights"].name,
                "weights_shape": [64, 3136],
                "bias": paths["linear0_bias"].name,
                "bias_shape": [64],
            },
            {
                "name": "linear1",
                "type": "linear",
                "in_features": 64,
                "out_features": 10,
                "weights": paths["linear1_weights"].name,
                "weights_shape": [10, 64],
                "bias": paths["linear1_bias"].name,
                "bias_shape": [10],
            },
        ],
    }

    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported weights to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train the tiny WebGPU MNIST network.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--train-limit", type=int, default=0, help="Use first N training samples; 0 means all.")
    parser.add_argument("--test-limit", type=int, default=0, help="Use first N test samples; 0 means all.")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    train_images, train_labels, test_images, test_labels = load_mnist(args.data_dir)

    if args.train_limit > 0:
        train_images = train_images[:args.train_limit]
        train_labels = train_labels[:args.train_limit]
    if args.test_limit > 0:
        test_images = test_images[:args.test_limit]
        test_labels = test_labels[:args.test_limit]

    train_loader = DataLoader(
        MnistDataset(train_images, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        MnistDataset(test_images, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TinyLeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size

        test_acc = accuracy(model, test_loader, device)
        elapsed = time.time() - start
        avg_loss = total_loss / max(total_seen, 1)
        print(f"epoch {epoch:02d} loss={avg_loss:.4f} test_acc={test_acc:.4f} time={elapsed:.1f}s")

    export_weights(model, args.out_dir)


if __name__ == "__main__":
    main()
