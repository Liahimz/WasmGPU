const selectFile = document.getElementById("select_file");
const startButton = document.getElementById("button_start");
const outputDiv = document.getElementById("output");
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
      worker.postMessage({
        requestType: "file",
        imageData: raw,
        width: img.width,
        height: img.height,
        channels: deduce_channels, // or 3 for RGB, or 1 for grayscale
        targetWidth: 600
      }, [raw.buffer]);
      outputDiv.textContent = "Processing...";
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
    
    let { outImage, width, height } = e.data;
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
    outputDiv.appendChild(canvas);
  }
};
