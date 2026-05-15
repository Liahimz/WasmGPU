console.log("Dummy worker started");

importScripts("wasm_gpu.js");

console.log("SmartIDEngine in worker:", typeof SmartIDEngine);

let engineInstance = null;
let moduleObject = null;

let readyPromise = SmartIDEngine({
  mainScriptUrlOrBlob: "wasm_gpu.js",
}).then((Module) => {
  engineInstance = new Module.DummyEngine();
  engineInstance.configure(4);
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
      const targetWidth = msg.data.targetWidth || width;

      const cppVec = new moduleObject.Uint8Vector();
      for (let i = 0; i < arr.length; ++i) {
        cppVec.push_back(arr[i]);
      }

      const resultVec = engineInstance.process(cppVec, width, height, channels, targetWidth);

      const outArr = [];
      for (let i = 0; i < resultVec.image.size(); ++i) {
        outArr.push(resultVec.image.get(i));
      }

      const outImage = new Uint8Array(outArr);
      cppVec.delete();

      postMessage({
        requestType: "result",
        outImage,
        width: resultVec.width,
        height: resultVec.height,
      }, [outImage.buffer]);
    }
  } catch (err) {
    console.error("WASM worker error:", err);
    postMessage({ requestType: "error", error: err && err.toString ? err.toString() : String(err) });
  }
};
