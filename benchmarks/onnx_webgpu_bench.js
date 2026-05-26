import * as ort from "../node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs";

const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const runsEl = document.getElementById("runs");
const warmupEl = document.getElementById("warmup");
const startButton = document.getElementById("start");

ort.env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

function parseNpyF32(buffer) {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.slice(0, 6)) !== "\x93NUMPY") throw new Error("Expected .npy input");
  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(bytes.slice(headerStart, headerStart + headerLen));
  const shape = header.match(/\(([^)]*)\)/)[1].split(",").map((part) => part.trim()).filter(Boolean).map(Number);
  return { shape, data: new Float32Array(buffer, headerStart + headerLen) };
}

function summarize(samples) {
  if (samples.length === 0) {
    return {
      count: 0,
      min_ms: null,
      median_ms: null,
      max_ms: null,
    };
  }
  const ordered = [...samples].sort((a, b) => a - b);
  return {
    count: samples.length,
    min_ms: ordered[0],
    median_ms: ordered[Math.floor(ordered.length / 2)],
    max_ms: ordered[ordered.length - 1],
  };
}

function topK(logits, count = 5) {
  return Array.from(logits)
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, count);
}

startButton.onclick = async () => {
  try {
    startButton.disabled = true;
    statusEl.textContent = "Loading ONNX Runtime WebGPU...";
    const [inputResponse] = await Promise.all([
      fetch("./artifacts/resnet50_input.npy", { cache: "no-store" }),
    ]);
    const input = parseNpyF32(await inputResponse.arrayBuffer());

    const session = await ort.InferenceSession.create("./artifacts/resnet50_imagenet.onnx", {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    const feeds = {
      [session.inputNames[0]]: new ort.Tensor("float32", input.data, input.shape),
    };
    const warmup = Number.parseInt(warmupEl.value, 10) || 0;
    const runs = Number.parseInt(runsEl.value, 10) || 1;

    statusEl.textContent = "Warming up...";
    const warmupTimings = [];
    for (let i = 0; i < warmup; i += 1) {
      const start = performance.now();
      await session.run(feeds);
      warmupTimings.push(performance.now() - start);
    }

    statusEl.textContent = "Benchmarking...";
    const timings = [];
    let output;
    for (let i = 0; i < runs; i += 1) {
      const start = performance.now();
      output = await session.run(feeds);
      timings.push(performance.now() - start);
    }

    const logits = output[session.outputNames[0]].data;
    const top = topK(logits);
    const payload = {
      name: "onnx_webgpu_runtime",
      created_unix: Date.now() / 1000,
      backend: "onnxruntime-web:webgpu",
      warmup_summary: summarize(warmupTimings),
      summary: summarize(timings),
      prediction: top[0].index,
      top5: top.map((item) => `${item.index} ${item.value.toPrecision(6)}`).join("\n"),
      user_agent: navigator.userAgent,
    };
    statusEl.textContent = `Done. Copy this JSON into benchmarks/results/onnx_webgpu_runtime.json if you want collect_bench.py to include it.`;
    outputEl.textContent = JSON.stringify(payload, null, 2);
  } catch (err) {
    statusEl.textContent = "Failed";
    outputEl.textContent = String(err && err.stack ? err.stack : err);
  } finally {
    startButton.disabled = false;
  }
};
