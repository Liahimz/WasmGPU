import os
import shutil
import subprocess
import sys
import argparse

# Settings
BUILD_DIR = "build"
WASM_JS = "wasm_gpu.js"
WASM_BIN = "wasm_gpu.wasm"
BUILD_MODES = {
    "cpp-webgpu": "samples/CppWebGpu",
    "cpp-webgpu-async": "samples/CppWebGpuAsync",
    "js-webgpu": "samples/JsWebGpu",
    "dummy": "samples/Dummy",
}

TBB_DIR = "thirdparty/tbb"
TBB_BUILD = os.path.join(TBB_DIR, "build-emscripten")

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print("Error running command:", ' '.join(cmd))
        sys.exit(res.returncode)

def clean_dir_contents(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and not os.path.islink(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

def find_tbb_static_lib():
    if not os.path.isdir(TBB_BUILD):
        return None

    for root, _, files in os.walk(TBB_BUILD):
        if "libtbb.a" in files:
            return os.path.join(root, "libtbb.a")

    return None
        
def check_and_build_tbb():
    tbb_lib = find_tbb_static_lib()
    if tbb_lib:
        print(f"TBB already built: {tbb_lib}. Skipping TBB build.")
        return

    print(f"Building TBB in {TBB_BUILD} ...")
    if os.path.exists(TBB_BUILD):
        shutil.rmtree(TBB_BUILD)
    os.makedirs(TBB_BUILD)

    run_cmd([
        "emcmake", "cmake", "..",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DTBB_TEST=OFF",
        "-DTBB_STRICT=OFF",
        "-DTBB_EXAMPLES=OFF",
        "-DTBB4PY_BUILD=OFF",
        "-DTBBMALLOC_BUILD=OFF",
        "-DTBBMALLOC_PROXY_BUILD=OFF",
    ], cwd=TBB_BUILD)
    run_cmd(["emmake", "make", "-j", "8"], cwd=TBB_BUILD)
    print("TBB build complete.\n")      

def parse_args():
    parser = argparse.ArgumentParser(description="Build the WasmGPU demo.")
    parser.add_argument(
        "--mode",
        choices=BUILD_MODES.keys(),
        default="cpp-webgpu",
        help="Build mode. Default: cpp-webgpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_dir = BUILD_MODES[args.mode]
    print(f"Build mode: {args.mode}")
    print(f"Sample dir: {sample_dir}")

    #build tbb for emiscripten
    check_and_build_tbb()
    
    # Clean build output without deleting the directory itself. This keeps
    # terminals already inside build/ from ending up with a removed cwd.
    clean_dir_contents(BUILD_DIR)

    # Build with emcmake/cmake
    # Build out-of-source in a 'cmake-build' directory
    CMAKE_BUILD = f"cmake-build-{args.mode}"
    if os.path.exists(CMAKE_BUILD):
        shutil.rmtree(CMAKE_BUILD)
    os.makedirs(CMAKE_BUILD)

    # Configure and build with Emscripten
    run_cmd(["emcmake", "cmake", "..", f"-DWASM_GPU_BUILD_MODE={args.mode}"], cwd=CMAKE_BUILD)
    run_cmd(["cmake", "--build", ".", "--config", "Release"], cwd=CMAKE_BUILD)

    # Move WASM outputs
    wasm_js_path = os.path.join(CMAKE_BUILD, WASM_JS)
    wasm_bin_path = os.path.join(CMAKE_BUILD, WASM_BIN)
    if not os.path.isfile(wasm_js_path) or not os.path.isfile(wasm_bin_path):
        print("Build failed: Missing wasm/js files")
        sys.exit(1)

    shutil.copy2(wasm_js_path, os.path.join(BUILD_DIR, WASM_JS))
    shutil.copy2(wasm_bin_path, os.path.join(BUILD_DIR, WASM_BIN))

    # Copy selected sample files
    if os.path.exists(sample_dir):
        for item in os.listdir(sample_dir):
            s = os.path.join(sample_dir, item)
            d = os.path.join(BUILD_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
    else:
        print(f"Build failed: Missing sample directory {sample_dir}")
        sys.exit(1)

    SHADERS_DIR = "shaders"
    if args.mode == "js-webgpu" and os.path.exists(SHADERS_DIR):
        shutil.copytree(SHADERS_DIR, os.path.join(BUILD_DIR, SHADERS_DIR))

    print("\n✅ Build complete. To serve, run:")
    print(f"  cd {BUILD_DIR} && npx serve\n")

if __name__ == "__main__":
    main()
