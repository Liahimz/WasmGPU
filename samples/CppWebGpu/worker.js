console.log("C++ WebGPU worker started");

importScripts("wasm_gpu.js");

console.log("SmartIDEngine in worker:", typeof SmartIDEngine);

let engineInstance = null;
let moduleObject = null;

let readyPromise = SmartIDEngine({
  mainScriptUrlOrBlob: "wasm_gpu.js",
}).then((Module) => {
  engineInstance = new Module.GpuEngine();
  engineInstance.configure(28);
  moduleObject = Module;
  return Module;
});

onmessage = async function(msg) {
  try {
    await readyPromise;

    if (msg.data.requestType === "file") {
      let arr = msg.data.imageData;
      if (!(arr instanceof Uint8Array)) arr = new Uint8Array(arr);

      const width = msg.data.width;
      const height = msg.data.height;
      const channels = msg.data.channels || 4;

      const cppVec = new moduleObject.Uint8Vector();
      for (let i = 0; i < arr.length; ++i) {
        cppVec.push_back(arr[i]);
      }

      const resultVec = await engineInstance.process(cppVec, width, height, channels);
      const prediction = resultVec.prediction;

      const cpuScalarResult = engineInstance.processCpu(cppVec, width, height, channels, 0);
      const cpuSimdResult = engineInstance.processCpu(cppVec, width, height, channels, 1);
      const cpuSimdThreadsResult = engineInstance.processCpu(cppVec, width, height, channels, 2);
      const cpuPredictions = {
        scalar: cpuScalarResult.prediction,
        simd: cpuSimdResult.prediction,
        simdThreads: cpuSimdThreadsResult.prediction,
      };

      const outArr = [];
      for (let i = 0; i < resultVec.image.size(); ++i) {
        outArr.push(resultVec.image.get(i));
      }

      const outImage = new Uint8Array(outArr);
      console.log("Preprocessed 28x28 image:", outImage);
      console.log("C++ WebGPU ready:", engineInstance.webgpuReady());
      console.log("GPU prediction:", prediction);
      console.log("CPU predictions:", cpuPredictions);

      cppVec.delete();

      postMessage({
        requestType: "result",
        outImage,
        width: resultVec.width,
        height: resultVec.height,
        prediction,
        cpuPredictions,
        webgpuReady: engineInstance.webgpuReady(),
      }, [outImage.buffer]);
    }
  } catch (err) {
    console.error("WASM/WebGPU worker error:", err);
    postMessage({ requestType: "error", error: err && err.toString ? err.toString() : String(err) });
  }
};
