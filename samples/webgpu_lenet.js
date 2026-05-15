class TinyWebGpuLenet {
  static INPUT_WIDTH = 28;
  static INPUT_HEIGHT = 28;
  static CONV_SIZE = 3;
  static CONV_CHANNELS = 4;
  static CONV_WIDTH = 26;
  static CONV_HEIGHT = 26;
  static CONV_VALUES = 26 * 26 * 4;
  static CLASS_COUNT = 10;

  constructor(device, convPipeline, linearPipeline, weights) {
    this.device = device;
    this.convPipeline = convPipeline;
    this.linearPipeline = linearPipeline;
    this.weights = weights;

    this.inputBuffer = this.createStorageBuffer(28 * 28 * 4, GPUBufferUsage.COPY_DST);
    this.convWeightBuffer = this.createStorageBuffer(weights.convWeights.byteLength, GPUBufferUsage.COPY_DST);
    this.convBiasBuffer = this.createStorageBuffer(weights.convBias.byteLength, GPUBufferUsage.COPY_DST);
    this.convOutputBuffer = this.createStorageBuffer(TinyWebGpuLenet.CONV_VALUES * 4, GPUBufferUsage.COPY_SRC);

    this.linearWeightBuffer = this.createStorageBuffer(weights.linearWeights.byteLength, GPUBufferUsage.COPY_DST);
    this.linearBiasBuffer = this.createStorageBuffer(weights.linearBias.byteLength, GPUBufferUsage.COPY_DST);
    this.logitsBuffer = this.createStorageBuffer(TinyWebGpuLenet.CLASS_COUNT * 4, GPUBufferUsage.COPY_SRC);
    this.readbackBuffer = this.device.createBuffer({
      size: TinyWebGpuLenet.CLASS_COUNT * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.device.queue.writeBuffer(this.convWeightBuffer, 0, weights.convWeights);
    this.device.queue.writeBuffer(this.convBiasBuffer, 0, weights.convBias);
    this.device.queue.writeBuffer(this.linearWeightBuffer, 0, weights.linearWeights);
    this.device.queue.writeBuffer(this.linearBiasBuffer, 0, weights.linearBias);

    this.convBindGroup = this.device.createBindGroup({
      layout: this.convPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.convWeightBuffer } },
        { binding: 2, resource: { buffer: this.convBiasBuffer } },
        { binding: 3, resource: { buffer: this.convOutputBuffer } },
      ],
    });

    this.linearBindGroup = this.device.createBindGroup({
      layout: this.linearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.convOutputBuffer } },
        { binding: 1, resource: { buffer: this.linearWeightBuffer } },
        { binding: 2, resource: { buffer: this.linearBiasBuffer } },
        { binding: 3, resource: { buffer: this.logitsBuffer } },
      ],
    });
  }

  static async create() {
    if (!self.navigator?.gpu) {
      throw new Error("WebGPU is not available in this browser or worker context.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("Could not get a WebGPU adapter.");
    }

    const device = await adapter.requestDevice();
    const [convShader, linearShader] = await Promise.all([
      fetchShader("shaders/conv_relu.wgsl"),
      fetchShader("shaders/linear.wgsl"),
    ]);

    const convModule = device.createShaderModule({ code: convShader });
    const linearModule = device.createShaderModule({ code: linearShader });

    const convPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: convModule, entryPoint: "main" },
    });

    const linearPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: linearModule, entryPoint: "main" },
    });

    return new TinyWebGpuLenet(device, convPipeline, linearPipeline, createDeterministicWeights());
  }

  createStorageBuffer(size, extraUsage) {
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | extraUsage,
    });
  }

  async infer(gray28x28) {
    if (gray28x28.length !== 28 * 28) {
      throw new Error(`Expected 784 grayscale pixels, got ${gray28x28.length}.`);
    }

    const input = new Float32Array(28 * 28);
    for (let i = 0; i < gray28x28.length; ++i) {
      input[i] = gray28x28[i] / 255.0;
    }

    this.device.queue.writeBuffer(this.inputBuffer, 0, input);

    const encoder = this.device.createCommandEncoder();

    const convPass = encoder.beginComputePass();
    convPass.setPipeline(this.convPipeline);
    convPass.setBindGroup(0, this.convBindGroup);
    convPass.dispatchWorkgroups(Math.ceil(26 / 8), Math.ceil(26 / 8), 4);
    convPass.end();

    const linearPass = encoder.beginComputePass();
    linearPass.setPipeline(this.linearPipeline);
    linearPass.setBindGroup(0, this.linearBindGroup);
    linearPass.dispatchWorkgroups(1);
    linearPass.end();

    encoder.copyBufferToBuffer(
      this.logitsBuffer,
      0,
      this.readbackBuffer,
      0,
      TinyWebGpuLenet.CLASS_COUNT * 4
    );

    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = this.readbackBuffer.getMappedRange();
    const logits = new Float32Array(mapped).slice();
    this.readbackBuffer.unmap();

    return logits;
  }
}

function createDeterministicWeights() {
  const rng = mulberry32(0x1234abcd);

  const convWeights = new Float32Array(4 * 3 * 3);
  const convBias = new Float32Array(4);
  const linearWeights = new Float32Array(10 * 2704);
  const linearBias = new Float32Array(10);

  fillSmallRandom(convWeights, rng, 0.15);
  fillSmallRandom(convBias, rng, 0.02);
  fillSmallRandom(linearWeights, rng, 0.04);
  fillSmallRandom(linearBias, rng, 0.02);

  return { convWeights, convBias, linearWeights, linearBias };
}

async function fetchShader(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

function fillSmallRandom(values, rng, scale) {
  for (let i = 0; i < values.length; ++i) {
    values[i] = (rng() * 2.0 - 1.0) * scale;
  }
}

function mulberry32(seed) {
  return function nextRandom() {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
