console.log("Worker.js started")


importScripts("wasm_gpu.js");
importScripts("webgpu_lenet.js");

// if (typeof Module !== "undefined") {
//   // Worker-specific: set mainScriptUrlOrBlob to current script
//   console.log("Worker.js in undefined")
//   Module['mainScriptUrlOrBlob'] = 'wasm_gpu.js';
// }

console.log("SmartIDEngine in worker:", typeof SmartIDEngine);
let engineInstance = null;
let moduleObject = null;
let gpuNet = null;
let readyPromise = Promise.all([
  SmartIDEngine({
    mainScriptUrlOrBlob: 'wasm_gpu.js',
    // you can also add locateFile if needed
  }),
  TinyWebGpuLenet.create(),
]).then(([Module, net]) => {
  engineInstance = new Module.GpuEngine();
  engineInstance.configure(28);
  moduleObject = Module;
  gpuNet = net;

  // Module._start_tbb_session();

  return Module;
});

onmessage = async function(msg) {
  try {
    await readyPromise;

    if (msg.data.requestType === "file") {
      let arr = msg.data.imageData;
      if (!(arr instanceof Uint8Array)) arr = new Uint8Array(arr);
      let width = msg.data.width;
      let height = msg.data.height;
      let channels = msg.data.channels || 4;

      let cppVec = new moduleObject.Uint8Vector();
      for (let i = 0; i < arr.length; ++i) {
          cppVec.push_back(arr[i]);
      }

      // Start keepalive *before* calling C++/WASM code
      
      moduleObject._stop_keepalive_mainloop();

      let resultVec = engineInstance.process(cppVec, width, height, channels);

      let outArr = [];
      for (let i = 0; i < resultVec.image.size(); ++i) {
          outArr.push(resultVec.image.get(i));
      }
      let outImage = new Uint8Array(outArr);
      console.log("Preprocessed 28x28 image:", outImage);

      let logits = await gpuNet.infer(outImage);
      console.log("WebGPU logits:", logits);

      let logitsVec = new moduleObject.FloatVector();
      for (let i = 0; i < logits.length; ++i) {
        logitsVec.push_back(logits[i]);
      }

      let prediction = engineInstance.argmax(logitsVec);
      console.log("Prediction:", prediction);

      cppVec.delete();
      logitsVec.delete();

      postMessage({
        requestType: "result",
        outImage,
        width: resultVec.width,
        height: resultVec.height,
        logits,
        prediction,
      }, [outImage.buffer, logits.buffer]);
      
      // Stop keepalive *after* postMessage (and after all work is done)
      // moduleObject._start_keepalive_mainloop();
    }
  } catch (err) {
    // Log the error, post error message, and be sure to clean up runtime state
    console.error("WASM/WebGPU worker error:", err);
    try {
      moduleObject?._start_keepalive_mainloop();
    } catch (e) {}
    postMessage({ requestType: "error", error: err && err.toString ? err.toString() : String(err) });
  }
};
