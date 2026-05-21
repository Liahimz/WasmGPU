# Network Trainer

This folder trains the tiny network used by the WebGPU demo:

```text
input:  28x28x1
conv0:  8 filters, 3x3, stride 1, padding 1
relu0:  28x28x8
conv1:  16 filters, 3x3, stride 1, padding 1
relu1:  28x28x16
pool:   2x2 max pool, stride 2
flatten
linear0: 3136 -> 64
linear1: 64 -> 10
```

Run:

```bash
python3 network_trainer/train_mnist.py --epochs 3
```

The script downloads MNIST IDX files into `network_trainer/data/` and writes weights into `network_data/lenet/`:

```text
network_data/lenet/tiny_lenet_conv0_weights_f32.bin
network_data/lenet/tiny_lenet_conv0_bias_f32.bin
network_data/lenet/tiny_lenet_conv1_weights_f32.bin
network_data/lenet/tiny_lenet_conv1_bias_f32.bin
network_data/lenet/tiny_lenet_linear0_weights_f32.bin
network_data/lenet/tiny_lenet_linear0_bias_f32.bin
network_data/lenet/tiny_lenet_linear1_weights_f32.bin
network_data/lenet/tiny_lenet_linear1_bias_f32.bin
network_data/lenet/tiny_lenet_manifest.json
network_data/lenet/tiny_lenet_weights.npz
```

The raw `.bin` files are little-endian `float32`, flattened in the same order expected by the WGSL shaders.
The manifest is static-shape and names each layer so the generic model loader can prepare CPU/WebGPU executor plans from the same exported data.

## ResNet50 Export Prep

The ResNet50 exporter starts with the BatchNorm folding step used by the graph runtime:

```bash
python3 network_trainer/export_resnet50.py --verify-folding
```

To load torchvision's pretrained ResNet50 and export folded `Conv2d + BatchNorm2d` weights into an NPZ:

```bash
python3 network_trainer/export_resnet50.py --export-folded-convs
```

Use `--weights random` to verify the export traversal without downloading ImageNet weights.
The script uses `network_trainer/data/torch_cache/` as the default Torch model cache so downloaded weights stay inside this repo workspace.
Generated ResNet50 weights and slice manifests are written to `network_data/resnet50/`, which is intentionally ignored by git.

To create the first static graph bring-up manifests:

```bash
python3 network_trainer/export_resnet50.py --export-manifest-slice stem
python3 network_trainer/export_resnet50.py --export-manifest-slice stem-layer1-block0
```
