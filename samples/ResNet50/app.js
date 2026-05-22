const selectFile = document.getElementById("select_file");
const startButton = document.getElementById("button_start");
const outputDiv = document.getElementById("output");
const benchmarkRunsInput = document.getElementById("benchmark_runs");
const cpuBenchmarkModeSelect = document.getElementById("cpu_benchmark_mode");
const cpuKernelModeSelect = document.getElementById("cpu_kernel_mode");
let selectedFile = null;
let worker = null;
let mainRuntimePromise = null;
let wasmScriptPromise = null;
const configPromise = loadBuildConfig();

selectFile.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  outputDiv.textContent = "";
});

startButton.addEventListener("click", async () => {
  if (!selectedFile) {
    outputDiv.textContent = "Please select a file.";
    return;
  }

  try {
    const payload = await loadImagePayload(selectedFile);
    outputDiv.textContent = `Processing ResNet50 ${payload.repetitions} run(s)...`;

    const config = await configPromise;
    if (shouldRunOnMainThread(config)) {
      if (!self.crossOriginIsolated) {
        throw new Error(
          "Threaded WASM needs cross-origin isolation. Serve the build with `npx serve` and the sample serve.json headers."
        );
      }
      const result = await runOnMainThread(payload);
      renderResult(result);
    } else {
      runInAppWorker(payload);
    }
  } catch (err) {
    outputDiv.textContent = String(err && err.stack ? err.stack : err);
  }
});

async function loadBuildConfig() {
  try {
    const response = await fetch("build_config.json", { cache: "no-store" });
    if (!response.ok) {
      return {};
    }
    return await response.json();
  } catch (_) {
    return {};
  }
}

function shouldRunOnMainThread(config) {
  const params = new URLSearchParams(window.location.search);
  const forceMain = params.get("wasm_host") === "main" || params.get("threads") === "1";
  const forceWorker = params.get("wasm_host") === "worker" || params.get("threads") === "0";
  if (forceMain) {
    return true;
  }
  if (forceWorker) {
    return false;
  }
  return config.parallelBackend && config.parallelBackend !== "serial";
}

function loadImagePayload(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error || new Error("Failed to read image file."));
    reader.onload = (event) => {
      const img = new Image();
      img.onerror = () => reject(new Error("Failed to decode image file."));
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const raw = new Uint8Array(imageData.data);
        const channels = imageData.data.length / (img.width * img.height);
        const repetitions = Math.max(1, Math.min(20, Number.parseInt(benchmarkRunsInput.value, 10) || 1));
        resolve({
          requestType: "file",
          imageData: raw,
          width: img.width,
          height: img.height,
          channels,
          repetitions,
          cpuMode: cpuBenchmarkModeSelect ? cpuBenchmarkModeSelect.value : "fast",
          cpuKernelMode: cpuKernelModeSelect ? cpuKernelModeSelect.value : "4",
        });
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function runInAppWorker(payload) {
  if (!worker) {
    worker = new Worker("worker.js");
    worker.onmessage = function(e) {
      if (e.data.requestType === "result") {
        renderResult(e.data);
      } else if (e.data.requestType === "error") {
        outputDiv.textContent = e.data.error;
      }
    };
  }

  const imageData = new Uint8Array(payload.imageData);
  worker.postMessage({
    ...payload,
    imageData,
  }, [imageData.buffer]);
}

function loadWasmScript() {
  if (wasmScriptPromise) {
    return wasmScriptPromise;
  }
  wasmScriptPromise = new Promise((resolve, reject) => {
    if (typeof SmartIDEngine === "function") {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = "wasm_gpu.js";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load wasm_gpu.js"));
    document.head.appendChild(script);
  });
  return wasmScriptPromise;
}

async function getMainRuntime() {
  if (mainRuntimePromise) {
    return mainRuntimePromise;
  }

  mainRuntimePromise = (async () => {
    await loadWasmScript();
    const Module = await SmartIDEngine({
      mainScriptUrlOrBlob: "wasm_gpu.js",
    });
    Module._start_keepalive_mainloop();
    const engine = new Module.GpuEngine();
    engine.configure(224);
    await waitForWebGpuReady(engine);
    return { Module, engine };
  })();
  return mainRuntimePromise;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForGpuPrediction(engine) {
  while (engine.inferencePending()) {
    await sleep(1);
  }
  return engine.latestPrediction();
}

async function waitForWebGpuReady(engine) {
  const start = performance.now();
  while (!engine.webgpuReady()) {
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

async function timedGpuRun(engine, run, fn) {
  const start = performance.now();
  const result = fn();
  const prediction = await waitForGpuPrediction(engine);
  const ms = performance.now() - start;
  return {
    run,
    ms,
    prediction,
    classLabel: String(engine.latestClassLabel() || ""),
    gpuBackend: String(result.gpuBackend || "unknown"),
    topK: engine.latestTopK(5),
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

function shouldRunCpuMode(payload, mode) {
  if (payload.cpuMode === "none") {
    return false;
  }
  if (payload.cpuMode === "full") {
    return true;
  }
  return mode === "simd_threads";
}

function cpuKernelTiles(payload) {
  if (payload.cpuKernelMode === "compare") {
    return [4, 8];
  }
  const tile = Number.parseInt(payload.cpuKernelMode || "4", 10);
  return tile === 8 ? [8] : [4];
}

function processResnetCpuTiled(engine, cppVec, width, height, channels, mode, tile) {
  if (typeof engine.processResnetCpuTiled === "function") {
    return engine.processResnetCpuTiled(cppVec, width, height, channels, mode, tile);
  }
  return engine.processResnetCpu(cppVec, width, height, channels, mode);
}

async function runOnMainThread(payload) {
  const { Module, engine } = await getMainRuntime();
  const cppVec = new Module.Uint8Vector();
  for (let i = 0; i < payload.imageData.length; ++i) {
    cppVec.push_back(payload.imageData[i]);
  }

  try {
    const gpuRuns = [];
    const cpuScalarRuns = [];
    const cpuSimdRuns = [];
    const cpuSimdThreadsRunsByTile = new Map();
    for (const tile of cpuKernelTiles(payload)) {
      cpuSimdThreadsRunsByTile.set(tile, []);
    }
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

    for (let run = 1; run <= payload.repetitions; ++run) {
      const gpuRun = await timedGpuRun(engine, run, () =>
        engine.processResnet(cppVec, payload.width, payload.height, payload.channels)
      );
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
      const primaryTile = cpuKernelTiles(payload)[0];
      if (shouldRunCpuMode(payload, "scalar")) {
        cpuScalar = timedCpuRun("cpu_scalar", run, () =>
          processResnetCpuTiled(engine, cppVec, payload.width, payload.height, payload.channels, 0, primaryTile)
        );
        cpuScalarRuns.push(cpuScalar);
      }
      if (shouldRunCpuMode(payload, "simd")) {
        cpuSimd = timedCpuRun("cpu_simd", run, () =>
          processResnetCpuTiled(engine, cppVec, payload.width, payload.height, payload.channels, 1, primaryTile)
        );
        cpuSimdRuns.push(cpuSimd);
      }
      if (shouldRunCpuMode(payload, "simd_threads")) {
        for (const tile of cpuKernelTiles(payload)) {
          const name = `cpu_simd_threads_oc4x${tile}`;
          cpuSimdThreads = timedCpuRun(name, run, () =>
            processResnetCpuTiled(engine, cppVec, payload.width, payload.height, payload.channels, 2, tile)
          );
          cpuSimdThreadsRunsByTile.get(tile).push(cpuSimdThreads);
        }
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

    return {
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
      webgpuReady: engine.webgpuReady(),
      benchmarkStats: [
        summarizeRuns("gpu", gpuRuns),
        cpuScalarRuns.length ? summarizeRuns("cpu_scalar", cpuScalarRuns) : null,
        cpuSimdRuns.length ? summarizeRuns("cpu_simd", cpuSimdRuns) : null,
        ...Array.from(cpuSimdThreadsRunsByTile.entries()).map(([tile, runs]) =>
          runs.length ? summarizeRuns(`cpu_simd_threads_oc4x${tile}`, runs) : null
        ),
      ].filter(Boolean),
    };
  } finally {
    cppVec.delete();
  }
}

function appendText(label, value) {
  const div = document.createElement("div");
  div.textContent = `${label}: ${value}`;
  outputDiv.appendChild(div);
}

function predictionText(id, label) {
  if (id === null || id === undefined) {
    return "skipped";
  }
  return label ? `${id} ${label}` : `${id}`;
}

function appendPre(label, value) {
  const heading = document.createElement("h2");
  heading.textContent = label;
  outputDiv.appendChild(heading);
  const pre = document.createElement("pre");
  pre.textContent = value || "";
  outputDiv.appendChild(pre);
}

function appendImage(label, image, width, height) {
  if (!image || !width || !height) {
    return;
  }

  const heading = document.createElement("h2");
  heading.textContent = label;
  outputDiv.appendChild(heading);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.style.imageRendering = "pixelated";
  canvas.style.border = "1px solid #ccc";

  const ctx = canvas.getContext("2d");
  const rgba = new Uint8ClampedArray(image.buffer, image.byteOffset, image.byteLength);
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  outputDiv.appendChild(canvas);
}

function appendBenchmarkTable(rows) {
  if (!rows) {
    return;
  }
  const heading = document.createElement("h2");
  heading.textContent = "Benchmark stats";
  outputDiv.appendChild(heading);

  const table = document.createElement("table");
  table.style.borderCollapse = "collapse";
  table.innerHTML = `
    <thead>
      <tr>
        <th style="text-align:left;padding:4px 8px;">name</th>
        <th style="text-align:right;padding:4px 8px;">min ms</th>
        <th style="text-align:right;padding:4px 8px;">median ms</th>
        <th style="text-align:right;padding:4px 8px;">max ms</th>
        <th style="text-align:right;padding:4px 8px;">min run</th>
        <th style="text-align:right;padding:4px 8px;">median run</th>
        <th style="text-align:right;padding:4px 8px;">max run</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="padding:4px 8px;">${row.name}</td>
      <td style="text-align:right;padding:4px 8px;">${row.minMs.toFixed(3)}</td>
      <td style="text-align:right;padding:4px 8px;">${row.medianMs.toFixed(3)}</td>
      <td style="text-align:right;padding:4px 8px;">${row.maxMs.toFixed(3)}</td>
      <td style="text-align:right;padding:4px 8px;">${row.minRun}</td>
      <td style="text-align:right;padding:4px 8px;">${row.medianRun}</td>
      <td style="text-align:right;padding:4px 8px;">${row.maxRun}</td>
    `;
    tbody.appendChild(tr);
  }
  outputDiv.appendChild(table);
}

function renderResult(data) {
  outputDiv.innerHTML = "";
  appendText("C++ WebGPU ready", data.webgpuReady);
  appendText("GPU prediction", predictionText(data.prediction, data.classLabel));
  appendText("GPU network", data.gpuBackend);
  if (data.cpuPredictions) {
    appendText(
      "CPU scalar/simd/simd_threads",
      `${predictionText(data.cpuPredictions.scalar, data.cpuClassLabels?.scalar)} / ` +
        `${predictionText(data.cpuPredictions.simd, data.cpuClassLabels?.simd)} / ` +
        `${predictionText(data.cpuPredictions.simdThreads, data.cpuClassLabels?.simdThreads)}`
    );
  }
  appendImage("Preprocessed image", data.outImage, data.width, data.height);
  appendPre("GPU top-5", data.gpuTopK);
  appendPre("CPU top-5", data.cpuTopK);
  appendBenchmarkTable(data.benchmarkStats);
}
