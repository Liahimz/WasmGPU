console.log("Worker.js started")


importScripts("idengine_wasm.js");

// if (typeof Module !== "undefined") {
//   // Worker-specific: set mainScriptUrlOrBlob to current script
//   console.log("Worker.js in undefined")
//   Module['mainScriptUrlOrBlob'] = 'idengine_wasm.js';
// }

console.log("SmartIDEngine in worker:", typeof SmartIDEngine);
let engineInstance = null;
let readyPromise = SmartIDEngine({
  mainScriptUrlOrBlob: 'idengine_wasm.js',
  // you can also add locateFile if needed
}).then(Module => {
  engineInstance = new Module.DummyEngine();
  engineInstance.configure(4);
  moduleObject = Module;

  // Module._start_tbb_session();

  return Module;
});

onmessage = async function(msg) {
  await readyPromise;
  try {
    if (msg.data.requestType === "file") {
      let arr = msg.data.imageData;
      if (!(arr instanceof Uint8Array)) arr = new Uint8Array(arr);
      let width = msg.data.width;
      let height = msg.data.height;
      let channels = msg.data.channels || 4;
      let targetWidth = msg.data.targetWidth || width;

      let cppVec = new moduleObject.Uint8Vector();
      for (let i = 0; i < arr.length; ++i) {
          cppVec.push_back(arr[i]);
      }

      // Start keepalive *before* calling C++/WASM code
      
      moduleObject._stop_keepalive_mainloop();

      let resultVec = engineInstance.process(cppVec, width, height, channels, targetWidth);

      let outArr = [];
      for (let i = 0; i < resultVec.image.size(); ++i) {
          outArr.push(resultVec.image.get(i));
      }
      let outImage = new Uint8Array(outArr);

      postMessage({ requestType: "result", outImage, width: resultVec.width, height: resultVec.height }, [outImage.buffer]);
      
      // Stop keepalive *after* postMessage (and after all work is done)
      moduleObject._start_keepalive_mainloop();
    }
  } catch (err) {
    // Log the error, post error message, and be sure to clean up runtime state
    // console.error("WASM worker error:", err);
    // try { moduleObject._stop_keepalive_mainloop(); } catch (e) {}
    // postMessage({ requestType: "error", error: err && err.toString ? err.toString() : String(err) });
  }
};