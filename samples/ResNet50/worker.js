console.log("ResNet50 C++ WebGPU worker started");

importScripts("wasm_gpu.js");

let engineInstance = null;
let moduleObject = null;

let readyPromise = SmartIDEngine({
  mainScriptUrlOrBlob: "wasm_gpu.js",
}).then((Module) => {
  Module._start_keepalive_mainloop();
  moduleObject = Module;
  engineInstance = new Module.GpuEngine();
  engineInstance.configure(224);
  return Module;
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForGpuPrediction() {
  while (engineInstance.inferencePending()) {
    await sleep(1);
  }
  return engineInstance.latestPrediction();
}

async function waitForWebGpuReady() {
  const start = performance.now();
  while (!engineInstance.webgpuReady()) {
    if (performance.now() - start > 10000) {
      throw new Error("Timed out waiting for C++ WebGPU device/resources");
    }
    await sleep(10);
  }
}

function summarizeRuns(name, runs) {
  const sorted = [...runs].sort((a, b) => a.ms - b.ms);
  const median = sorted[Math.floor(sorted.length / 2)];
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  return {
    name,
    count: runs.length,
    minMs: min.ms,
    medianMs: median.ms,
    maxMs: max.ms,
    minRun: min.run,
    medianRun: median.run,
    maxRun: max.run,
  };
}

function copyUint8Vector(vec) {
  const out = new Uint8Array(vec.size());
  for (let i = 0; i < out.length; ++i) {
    out[i] = vec.get(i);
  }
  return out;
}

async function timedGpuRun(run, fn) {
  const start = performance.now();
  const result = fn();
  const prediction = await waitForGpuPrediction();
  const ms = performance.now() - start;
  return {
    run,
    ms,
    prediction,
    classLabel: String(engineInstance.latestClassLabel() || ""),
    gpuBackend: String(result.gpuBackend || "unknown"),
    topK: engineInstance.latestTopK(5),
    image: result.image && result.image.size ? copyUint8Vector(result.image) : null,
    width: Number(result.width || 0),
    height: Number(result.height || 0),
  };
}

function timedCpuRun(name, run, fn) {
  const start = performance.now();
  const result = fn();
  const ms = performance.now() - start;
  return {
    name,
    run,
    ms,
    prediction: Number(result.prediction),
    classLabel: String(result.classLabel || ""),
    topK: String(result.topK || ""),
  };
}

onmessage = async function(msg) {
  try {
    await readyPromise;
    await waitForWebGpuReady();

    if (msg.data.requestType !== "file") {
      return;
    }

    let arr = msg.data.imageData;
    if (!(arr instanceof Uint8Array)) arr = new Uint8Array(arr);
    const width = msg.data.width;
    const height = msg.data.height;
    const channels = msg.data.channels || 4;
    const repetitions = Math.max(1, Math.min(20, msg.data.repetitions || 1));
    const cpuMode = msg.data.cpuMode || "fast";

    const cppVec = new moduleObject.Uint8Vector();
    for (let i = 0; i < arr.length; ++i) {
      cppVec.push_back(arr[i]);
    }

    const gpuRuns = [];
    const cpuScalarRuns = [];
    const cpuSimdRuns = [];
    const cpuSimdThreadsRuns = [];
    let gpuTopK = "";
    let cpuTopK = "";
    let gpuPrediction = -1;
    let gpuClassLabel = "";
    let cpuPredictions = null;
    let cpuClassLabels = null;
    let gpuBackend = "unknown";
    let preprocessedImage = null;
    let preprocessedWidth = 0;
    let preprocessedHeight = 0;

    for (let run = 1; run <= repetitions; ++run) {
      const gpuRun = await timedGpuRun(run, () => engineInstance.processResnet(cppVec, width, height, channels));
      gpuRuns.push(gpuRun);
      gpuPrediction = gpuRun.prediction;
      gpuClassLabel = gpuRun.classLabel;
      gpuTopK = gpuRun.topK;
      gpuBackend = gpuRun.gpuBackend || gpuBackend;
      if (!preprocessedImage && gpuRun.image) {
        preprocessedImage = gpuRun.image;
        preprocessedWidth = gpuRun.width;
        preprocessedHeight = gpuRun.height;
      }

      let cpuScalar = null;
      let cpuSimd = null;
      let cpuSimdThreads = null;
      if (cpuMode === "full") {
        cpuScalar = timedCpuRun("cpu_scalar", run, () => engineInstance.processResnetCpu(cppVec, width, height, channels, 0));
        cpuSimd = timedCpuRun("cpu_simd", run, () => engineInstance.processResnetCpu(cppVec, width, height, channels, 1));
        cpuScalarRuns.push(cpuScalar);
        cpuSimdRuns.push(cpuSimd);
      }
      if (cpuMode !== "none") {
        cpuSimdThreads = timedCpuRun("cpu_simd_threads", run, () => engineInstance.processResnetCpu(cppVec, width, height, channels, 2));
        cpuSimdThreadsRuns.push(cpuSimdThreads);
      }
      cpuPredictions = {
        scalar: cpuScalar ? cpuScalar.prediction : null,
        simd: cpuSimd ? cpuSimd.prediction : null,
        simdThreads: cpuSimdThreads ? cpuSimdThreads.prediction : null,
      };
      cpuClassLabels = {
        scalar: cpuScalar ? cpuScalar.classLabel : "",
        simd: cpuSimd ? cpuSimd.classLabel : "",
        simdThreads: cpuSimdThreads ? cpuSimdThreads.classLabel : "",
      };
      cpuTopK = (cpuSimdThreads && cpuSimdThreads.topK) || (cpuSimd && cpuSimd.topK) || (cpuScalar && cpuScalar.topK) || "";
    }

    postMessage({
      requestType: "result",
      prediction: gpuPrediction,
      classLabel: gpuClassLabel,
      gpuBackend,
      gpuTopK,
      cpuTopK,
      cpuPredictions,
      cpuClassLabels,
      outImage: preprocessedImage,
      width: preprocessedWidth,
      height: preprocessedHeight,
      webgpuReady: engineInstance.webgpuReady(),
      benchmarkStats: [
        summarizeRuns("gpu", gpuRuns),
        cpuScalarRuns.length ? summarizeRuns("cpu_scalar", cpuScalarRuns) : null,
        cpuSimdRuns.length ? summarizeRuns("cpu_simd", cpuSimdRuns) : null,
        cpuSimdThreadsRuns.length ? summarizeRuns("cpu_simd_threads", cpuSimdThreadsRuns) : null,
      ].filter(Boolean),
    }, preprocessedImage ? [preprocessedImage.buffer] : []);
    cppVec.delete();
  } catch (err) {
    console.error("WASM/WebGPU worker error:", err);
    postMessage({ requestType: "error", error: String(err && err.stack ? err.stack : err) });
  }
};
