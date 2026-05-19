# Network Trainer

This folder trains the tiny network used by the WebGPU demo:

```text
input:  28x28x1
conv:   4 filters, 3x3, stride 1, no padding
relu:   26x26x4
flatten
linear: 2704 -> 10
```

Run:

```bash
python3 network_trainer/train_mnist.py --epochs 3
```

The script downloads MNIST IDX files into `network_trainer/data/` and writes weights into `network_data/`:

```text
network_data/tiny_lenet_conv_weights_f32.bin
network_data/tiny_lenet_conv_bias_f32.bin
network_data/tiny_lenet_linear_weights_f32.bin
network_data/tiny_lenet_linear_bias_f32.bin
network_data/tiny_lenet_manifest.json
network_data/tiny_lenet_weights.npz
```

The raw `.bin` files are little-endian `float32`, flattened in the same order expected by the WGSL shaders.
The manifest is static-shape and names each layer so the generic model loader can prepare CPU/WebGPU executor plans from the same exported data.
