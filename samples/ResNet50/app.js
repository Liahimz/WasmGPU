const selectFile = document.getElementById("select_file");
const startButton = document.getElementById("button_start");
const outputDiv = document.getElementById("output");
const benchmarkRunsInput = document.getElementById("benchmark_runs");
let selectedFile = null;

let worker = new Worker("worker.js");

selectFile.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  outputDiv.textContent = "";
});

startButton.addEventListener("click", () => {
  if (!selectedFile) {
    outputDiv.textContent = "Please select a file.";
    return;
  }

  let reader = new FileReader();
  reader.onload = function(e) {
    let img = new Image();
    img.onload = function() {
      let canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      let ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      let imageData = ctx.getImageData(0, 0, img.width, img.height);
      let raw = new Uint8Array(imageData.data.buffer);
      let channels = imageData.data.length / (img.width * img.height);
      const repetitions = Math.max(1, Math.min(20, Number.parseInt(benchmarkRunsInput.value, 10) || 1));

      worker.postMessage({
        requestType: "file",
        imageData: raw,
        width: img.width,
        height: img.height,
        channels,
        repetitions,
      }, [raw.buffer]);
      outputDiv.textContent = `Processing ResNet50 ${repetitions} run(s)...`;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(selectedFile);
});

function appendText(label, value) {
  const div = document.createElement("div");
  div.textContent = `${label}: ${value}`;
  outputDiv.appendChild(div);
}

function predictionText(id, label) {
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

worker.onmessage = function(e) {
  if (e.data.requestType === "result") {
    outputDiv.innerHTML = "";
    appendText("C++ WebGPU ready", e.data.webgpuReady);
    appendText("GPU prediction", predictionText(e.data.prediction, e.data.classLabel));
    appendText("GPU network", e.data.gpuBackend);
    if (e.data.cpuPredictions) {
      appendText(
        "CPU scalar/simd/simd_threads",
        `${predictionText(e.data.cpuPredictions.scalar, e.data.cpuClassLabels?.scalar)} / ` +
          `${predictionText(e.data.cpuPredictions.simd, e.data.cpuClassLabels?.simd)} / ` +
          `${predictionText(e.data.cpuPredictions.simdThreads, e.data.cpuClassLabels?.simdThreads)}`
      );
    }
    appendImage("Preprocessed image", e.data.outImage, e.data.width, e.data.height);
    appendPre("GPU top-5", e.data.gpuTopK);
    appendPre("CPU top-5", e.data.cpuTopK);
    appendBenchmarkTable(e.data.benchmarkStats);
  } else if (e.data.requestType === "error") {
    outputDiv.textContent = e.data.error;
  }
};
