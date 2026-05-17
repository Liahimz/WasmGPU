console.log("C++ WebGPU async worker started");

importScripts("wasm_gpu.js");

console.log("SmartIDEngine in worker:", typeof SmartIDEngine);

let engineInstance = null;
let moduleObject = null;

async function logWebGpuAdapterDiagnostics() {
  if (!navigator.gpu) {
    console.warn("[webgpu] navigator.gpu is unavailable in this worker");
    return;
  }

  const probes = [
    ["default", undefined],
    ["high-performance", { powerPreference: "high-performance" }],
    ["low-power", { powerPreference: "low-power" }],
    ["fallback", { forceFallbackAdapter: true }],
  ];

  for (const [name, options] of probes) {
    try {
      const adapter = await navigator.gpu.requestAdapter(options);
      console.log(`[webgpu] JS adapter probe ${name}:`, adapter ? "available" : "null");
    } catch (err) {
      console.warn(`[webgpu] JS adapter probe ${name} threw:`, err);
    }
  }
}

let readyPromise = SmartIDEngine({
  mainScriptUrlOrBlob: "wasm_gpu.js",
}).then(async (Module) => {
  // WebGPU setup and buffer map callbacks can outlive the JS call that starts
  // them. Keep Emscripten's runtime alive for the worker lifetime; otherwise it
  // may call exit() after a callback returns and cancel later async GPU work.
  Module._start_keepalive_mainloop();
  moduleObject = Module;
  await logWebGpuAdapterDiagnostics();
  engineInstance = new Module.GpuEngine();
  engineInstance.configure(28);
  engineInstance.prepareSyntheticLargeData();
  return Module;
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const scrubBuffer = new Uint8Array(64 * 1024 * 1024);
let scrubSeed = 1;

function scrubCpuCache() {
  scrubSeed = (scrubSeed * 1664525 + 1013904223) >>> 0;
  const step = 64;
  for (let i = 0; i < scrubBuffer.length; i += step) {
    scrubBuffer[i] = (scrubBuffer[i] + scrubSeed + i) & 255;
  }
}

async function waitForGpuPrediction(label) {
  while (engineInstance.inferencePending()) {
    await sleep(1);
  }
  const prediction = engineInstance.latestPrediction();
  console.log(`${label} async prediction ready:`, prediction);
  return prediction;
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

async function timedGpuRun(name, run, cacheScrub, startFn) {
  if (cacheScrub) {
    scrubCpuCache();
  }
  const start = performance.now();
  startFn();
  const prediction = await waitForGpuPrediction(`${name} run ${run}`);
  const ms = performance.now() - start;
  console.log(`[bench] ${name} run=${run} ms=${ms} prediction=${prediction}`);
  return { run, ms, prediction };
}

async function warmupGpu(name, startFn) {
  const start = performance.now();
  startFn();
  const prediction = await waitForGpuPrediction(`${name} warmup`);
  const ms = performance.now() - start;
  console.log(`[warmup] ${name} ms=${ms} prediction=${prediction}`);
  return { name, ms, prediction };
}

function timedCpuRun(name, run, cacheScrub, fn) {
  if (cacheScrub) {
    scrubCpuCache();
  }
  const start = performance.now();
  const result = fn();
  const ms = performance.now() - start;
  const prediction = result && typeof result === "object" && "prediction" in result ? result.prediction : result;
  console.log(`[bench] ${name} run=${run} ms=${ms} prediction=${prediction}`);
  return { run, ms, prediction, result };
}

onmessage = async function(msg) {
  try {
    await readyPromise;
    await waitForWebGpuReady();

    if (msg.data.requestType === "file") {
      let arr = msg.data.imageData;
      if (!(arr instanceof Uint8Array)) arr = new Uint8Array(arr);

      const width = msg.data.width;
      const height = msg.data.height;
      const channels = msg.data.channels || 4;
      const repetitions = Math.max(1, Math.min(100, msg.data.repetitions || 1));
      const cacheScrub = msg.data.cacheScrub !== false;

      const cppVec = new moduleObject.Uint8Vector();
      for (let i = 0; i < arr.length; ++i) {
        cppVec.push_back(arr[i]);
      }

      const runs = {
        gpu: [],
        cpuScalar: [],
        cpuSimd: [],
        cpuSimdThreads: [],
        largeGpu: [],
        largeCpuScalar: [],
        largeCpuSimd: [],
        largeCpuSimdThreads: [],
      };

      let resultVec = null;
      let prediction = -1;
      let cpuPredictions = null;
      let largeGpuPrediction = -1;
      let largeCpuPredictions = null;

      const warmupStats = [
        await warmupGpu("gpu", () => {
          const warmupResult = engineInstance.process(cppVec, width, height, channels);
          if (!resultVec) {
            resultVec = warmupResult;
          }
        }),
        await warmupGpu("synthetic_gpu_large", () => engineInstance.benchmarkGpuLarge(0)),
      ];
      console.table(warmupStats);

      for (let run = 1; run <= repetitions; ++run) {
        const gpuRun = await timedGpuRun("gpu", run, cacheScrub, () => {
          const currentResult = engineInstance.process(cppVec, width, height, channels);
          if (!resultVec) {
            resultVec = currentResult;
          }
        });
        runs.gpu.push(gpuRun);
        prediction = gpuRun.prediction;

        const cpuScalarRun = timedCpuRun("cpu_scalar", run, cacheScrub, () =>
          engineInstance.processCpu(cppVec, width, height, channels, 0)
        );
        runs.cpuScalar.push(cpuScalarRun);

        const cpuSimdRun = timedCpuRun("cpu_simd", run, cacheScrub, () =>
          engineInstance.processCpu(cppVec, width, height, channels, 1)
        );
        runs.cpuSimd.push(cpuSimdRun);

        const cpuSimdThreadsRun = timedCpuRun("cpu_simd_threads", run, cacheScrub, () =>
          engineInstance.processCpu(cppVec, width, height, channels, 2)
        );
        runs.cpuSimdThreads.push(cpuSimdThreadsRun);
        cpuPredictions = {
          scalar: cpuScalarRun.prediction,
          simd: cpuSimdRun.prediction,
          simdThreads: cpuSimdThreadsRun.prediction,
        };

        const largeGpuRun = await timedGpuRun("synthetic_gpu_large", run, cacheScrub, () =>
          engineInstance.benchmarkGpuLarge(run)
        );
        runs.largeGpu.push(largeGpuRun);
        largeGpuPrediction = largeGpuRun.prediction;

        const largeCpuScalarRun = timedCpuRun("synthetic_cpu_large_scalar", run, cacheScrub, () =>
          engineInstance.benchmarkCpuLarge(0, run)
        );
        runs.largeCpuScalar.push(largeCpuScalarRun);

        const largeCpuSimdRun = timedCpuRun("synthetic_cpu_large_simd", run, cacheScrub, () =>
          engineInstance.benchmarkCpuLarge(1, run)
        );
        runs.largeCpuSimd.push(largeCpuSimdRun);

        const largeCpuSimdThreadsRun = timedCpuRun("synthetic_cpu_large_simd_threads", run, cacheScrub, () =>
          engineInstance.benchmarkCpuLarge(2, run)
        );
        runs.largeCpuSimdThreads.push(largeCpuSimdThreadsRun);
        largeCpuPredictions = {
          scalar: largeCpuScalarRun.prediction,
          simd: largeCpuSimdRun.prediction,
          simdThreads: largeCpuSimdThreadsRun.prediction,
        };
      }

      const benchmarkStats = [
        summarizeRuns("gpu", runs.gpu),
        summarizeRuns("cpu_scalar", runs.cpuScalar),
        summarizeRuns("cpu_simd", runs.cpuSimd),
        summarizeRuns("cpu_simd_threads", runs.cpuSimdThreads),
        summarizeRuns("synthetic_gpu_large", runs.largeGpu),
        summarizeRuns("synthetic_cpu_large_scalar", runs.largeCpuScalar),
        summarizeRuns("synthetic_cpu_large_simd", runs.largeCpuSimd),
        summarizeRuns("synthetic_cpu_large_simd_threads", runs.largeCpuSimdThreads),
      ];
      console.table(benchmarkStats);

      const outArr = [];
      for (let i = 0; i < resultVec.image.size(); ++i) {
        outArr.push(resultVec.image.get(i));
      }

      const outImage = new Uint8Array(outArr);
      console.log("Preprocessed 28x28 image:", outImage);
      console.log("C++ WebGPU ready:", engineInstance.webgpuReady());
      console.log("GPU prediction:", prediction);
      console.log("CPU predictions:", cpuPredictions);
      console.log("Synthetic large CPU predictions:", largeCpuPredictions);
      console.log("Synthetic large GPU prediction:", largeGpuPrediction);

      cppVec.delete();

      postMessage({
        requestType: "result",
        outImage,
        width: resultVec.width,
        height: resultVec.height,
        prediction,
        cpuPredictions,
        largeCpuPredictions,
        largeGpuPrediction,
        warmupStats,
        benchmarkStats,
        webgpuReady: engineInstance.webgpuReady(),
      }, [outImage.buffer]);
    }
  } catch (err) {
    console.error("WASM/WebGPU worker error:", err);
    postMessage({ requestType: "error", error: err && err.toString ? err.toString() : String(err) });
  }
};
