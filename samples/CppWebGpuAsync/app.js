const selectFile = document.getElementById("select_file");
const startButton = document.getElementById("button_start");
const outputDiv = document.getElementById("output");
const benchmarkRunsInput = document.getElementById("benchmark_runs");
const cacheScrubInput = document.getElementById("cache_scrub");
let selectedFile = null;

let worker = new Worker("worker.js");

selectFile.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  outputDiv.textContent = "";
});

startButton.addEventListener("click", () => {
  if (!selectedFile) {
    outputDiv.textContent = "Please select a file!";
    return;
  }
  let reader = new FileReader();
  reader.onload = function(e) {
    // Load into an Image object to get dimensions and draw to canvas for grayscale
    let img = new Image();
    img.onload = function() {
      console.log("Image loaded, size:", img.width, img.height);
      let canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      let ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      // Get RGBA, convert to grayscale
      let imageData = ctx.getImageData(0, 0, img.width, img.height);
      let raw = new Uint8Array(imageData.data.buffer);

      for (let i = 0; i < 40; i += 4) {
        console.log(`Pixel ${i/4}: R=${raw[i]}, G=${raw[i+1]}, B=${raw[i+2]}, A=${raw[i+3]}`);
      }
      // let gray = new Uint8Array(img.width * img.height);
      // let count = 0;
      // for (let y = 0; y < img.height; ++y) {
      //   for (let x = 0; x < img.width; ++x) {

          
      //     let idx = (y * img.width + x) * 4;
      //     let r = imageData.data[idx];
      //     let g = imageData.data[idx + 1];
      //     let b = imageData.data[idx + 2];

      //     if (count < 10) {
      //       console.log(r, g, b);
      //     }
      //     count += 1;
      //     // Optionally, you can check alpha with imageData.data[idx+3]
      //     gray[y * img.width + x] = Math.round(0.299*r + 0.587*g + 0.114*b);
      //   }
      // }
      console.log(raw);
      let deduce_channels = imageData.data.length / (img.width * img.height);
      const repetitions = Math.max(1, Math.min(100, Number.parseInt(benchmarkRunsInput.value, 10) || 1));
      worker.postMessage({
        requestType: "file",
        imageData: raw,
        width: img.width,
        height: img.height,
        channels: deduce_channels, // or 3 for RGB, or 1 for grayscale
        targetWidth: 600,
        repetitions,
        cacheScrub: cacheScrubInput.checked,
      }, [raw.buffer]);
      outputDiv.textContent = `Processing ${repetitions} run(s)...`;
    };
    // img.src = URL.createObjectURL(selectedFile);
    img.src = e.target.result;
  };
  // reader.readAsArrayBuffer(selectedFile);
  reader.readAsDataURL(selectedFile);
});

worker.onmessage = function(e) {
  if (e.data.requestType === "result") {
    // Display processed image
    console.log("Grayscale output:", e.data.outImage.slice(0, 20));
    if (e.data.logits) {
      console.log("WebGPU logits:", e.data.logits);
    }
    if (e.data.prediction !== undefined) {
      console.log("GPU prediction:", e.data.prediction);
    }
    if (e.data.cpuPredictions) {
      console.log("CPU predictions:", e.data.cpuPredictions);
    }
    if (e.data.largeCpuPredictions) {
      console.log("Synthetic large CPU predictions:", e.data.largeCpuPredictions);
    }
    if (e.data.largeGpuPrediction !== undefined) {
      console.log("Synthetic large GPU prediction:", e.data.largeGpuPrediction);
    }
    if (e.data.benchmarkStats) {
      console.table(e.data.benchmarkStats);
    }
    if (e.data.warmupStats) {
      console.table(e.data.warmupStats);
    }
    console.log("C++ WebGPU ready:", e.data.webgpuReady);
    
    let { outImage, width, height, prediction, gpuBackend, cpuPredictions, largeCpuPredictions, largeGpuPrediction, warmupStats, benchmarkStats, webgpuReady } = e.data;
    // width = 400;
    // height = outImage.length / width;;
    let canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    let ctx = canvas.getContext("2d");
    // OutImage is Uint8Array (grayscale)
    let out = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; ++i) {
      out[i*4 + 0] = out[i*4 + 1] = out[i*4 + 2] = e.data.outImage[i];
      out[i*4 + 3] = 255;
    }
    let imageData = new ImageData(out, width, height);
    ctx.putImageData(imageData, 0, 0);
    outputDiv.innerHTML = "";
    let statusText = document.createElement("div");
    statusText.textContent = `C++ WebGPU ready: ${webgpuReady}`;
    outputDiv.appendChild(statusText);
    if (prediction !== undefined) {
      let predictionText = document.createElement("div");
      predictionText.textContent = `GPU prediction: ${prediction}`;
      outputDiv.appendChild(predictionText);
    }
    if (gpuBackend) {
      let backendText = document.createElement("div");
      backendText.textContent = `GPU network: ${gpuBackend}`;
      outputDiv.appendChild(backendText);
    }
    if (cpuPredictions) {
      let cpuPredictionText = document.createElement("div");
      cpuPredictionText.textContent =
        `CPU scalar/simd/simd_threads: ${cpuPredictions.scalar} / ${cpuPredictions.simd} / ${cpuPredictions.simdThreads}`;
      outputDiv.appendChild(cpuPredictionText);
    }
    if (largeCpuPredictions) {
      let largePredictionText = document.createElement("div");
      largePredictionText.textContent =
        `Large CPU scalar/simd/simd_threads: ${largeCpuPredictions.scalar} / ${largeCpuPredictions.simd} / ${largeCpuPredictions.simdThreads}`;
      outputDiv.appendChild(largePredictionText);
    }
    if (largeGpuPrediction !== undefined) {
      let largeGpuPredictionText = document.createElement("div");
      largeGpuPredictionText.textContent = `Large GPU: ${largeGpuPrediction}`;
      outputDiv.appendChild(largeGpuPredictionText);
    }
    if (warmupStats) {
      let warmupHeading = document.createElement("h2");
      warmupHeading.textContent = "Warmup";
      outputDiv.appendChild(warmupHeading);

      let warmupTable = document.createElement("table");
      warmupTable.style.borderCollapse = "collapse";
      warmupTable.innerHTML = `
        <thead>
          <tr>
            <th style="text-align:left;padding:4px 8px;">name</th>
            <th style="text-align:right;padding:4px 8px;">time ms</th>
            <th style="text-align:right;padding:4px 8px;">prediction</th>
          </tr>
        </thead>
        <tbody></tbody>
      `;
      const tbody = warmupTable.querySelector("tbody");
      for (const row of warmupStats) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="padding:4px 8px;">${row.name}</td>
          <td style="text-align:right;padding:4px 8px;">${row.ms.toFixed(3)}</td>
          <td style="text-align:right;padding:4px 8px;">${row.prediction}</td>
        `;
        tbody.appendChild(tr);
      }
      outputDiv.appendChild(warmupTable);
    }
    if (benchmarkStats) {
      let heading = document.createElement("h2");
      heading.textContent = "Benchmark stats";
      outputDiv.appendChild(heading);

      let table = document.createElement("table");
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
      for (const row of benchmarkStats) {
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
    outputDiv.appendChild(canvas);
  } else if (e.data.requestType === "error") {
    outputDiv.textContent = e.data.error;
  }
};
