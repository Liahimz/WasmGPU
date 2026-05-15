import os
import shutil
import subprocess
import sys

# Settings
BUILD_DIR = "build"
SRC_FILES = [
    "samples/index.html",
    "samples/app.js",
    "samples/worker.js",
    "samples/serve.json",
]
WASM_JS = "wasm_gpu.js"
WASM_BIN = "wasm_gpu.wasm"

TBB_DIR = "thirdparty/tbb"
TBB_BUILD = os.path.join(TBB_DIR, "build-emscripten")

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print("Error running command:", ' '.join(cmd))
        sys.exit(res.returncode)
        
def check_and_build_tbb():
    # Check if build-emscripten exists and is non-empty
    if os.path.isdir(TBB_BUILD) and os.listdir(TBB_BUILD):
        print(f"TBB already built in {TBB_BUILD}. Skipping TBB build.")
        return

    print(f"Building TBB in {TBB_BUILD} ...")
    if not os.path.exists(TBB_BUILD):
        os.makedirs(TBB_BUILD)
    run_cmd([
        "emcmake", "cmake", "..",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTBB_TEST=OFF", "-DTBB_STRICT=OFF",
        "-DTBB_EXAMPLES=OFF", "-DTBB4PY_BUILD=OFF"
    ], cwd=TBB_BUILD)
    run_cmd(["emmake", "make", "-j", "8"], cwd=TBB_BUILD)
    print("TBB build complete.\n")      


def main():
    #build tbb for emiscripten
    check_and_build_tbb()
    
    # Clean build dir
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    # Build with emcmake/cmake
    # Build out-of-source in a 'cmake-build' directory
    CMAKE_BUILD = "cmake-build"
    if os.path.exists(CMAKE_BUILD):
        shutil.rmtree(CMAKE_BUILD)
    os.makedirs(CMAKE_BUILD)

    # Configure and build with Emscripten
    run_cmd(["emcmake", "cmake", ".."], cwd=CMAKE_BUILD)
    run_cmd(["cmake", "--build", ".", "--config", "Release"], cwd=CMAKE_BUILD)

    # Move WASM outputs
    wasm_js_path = os.path.join(CMAKE_BUILD, WASM_JS)
    wasm_bin_path = os.path.join(CMAKE_BUILD, WASM_BIN)
    if not os.path.isfile(wasm_js_path) or not os.path.isfile(wasm_bin_path):
        print("Build failed: Missing wasm/js files")
        sys.exit(1)

    shutil.copy2(wasm_js_path, os.path.join(BUILD_DIR, WASM_JS))
    shutil.copy2(wasm_bin_path, os.path.join(BUILD_DIR, WASM_BIN))

    # Copy JS, HTML, and serve.json files
    SAMPLES_DIR = "samples"
    if os.path.exists(SAMPLES_DIR):
        for item in os.listdir(SAMPLES_DIR):
            s = os.path.join(SAMPLES_DIR, item)
            d = os.path.join(BUILD_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    print("\n✅ Build complete. To serve, run:")
    print(f"  cd {BUILD_DIR} && npx serve\n")

if __name__ == "__main__":
    main()
