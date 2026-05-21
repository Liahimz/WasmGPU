cmake_minimum_required(VERSION 3.13)
project(dummy_engine_wasm)

set(CMAKE_CXX_STANDARD 17)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(WASM_GPU_BUILD_MODE "cpp-webgpu" CACHE STRING "Build mode: cpp-webgpu, cpp-webgpu-async, cpp-webgpu-dawn, js-webgpu, or dummy")
set_property(CACHE WASM_GPU_BUILD_MODE PROPERTY STRINGS cpp-webgpu cpp-webgpu-async cpp-webgpu-dawn resnet50 js-webgpu dummy)
set(WASM_GPU_PARALLEL_BACKEND "wasm-thread" CACHE STRING "Parallel backend: wasm-thread, pthread, std-thread, tbb, or serial")
set_property(CACHE WASM_GPU_PARALLEL_BACKEND PROPERTY STRINGS wasm-thread pthread std-thread tbb serial)

set(TBB_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/tbb)
set(WASM_WEBGPU_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/wasm_webgpu)
set(WASM_WEBGPU_LIB_DIR ${WASM_WEBGPU_ROOT}/lib)
set(BUILD_SCRIPTS_DIR ${CMAKE_SOURCE_DIR}/build_scripts)
set(NETWORK_DATA_DIR ${CMAKE_SOURCE_DIR}/network_data)
set(SHADER_DATA_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(EMBEDDED_DATA_TEMPLATE ${CMAKE_SOURCE_DIR}/cmake/embedded_data.h.cmakein)
set(EMBEDDED_DATA_HEADER ${CMAKE_CURRENT_BINARY_DIR}/src/embedded_data.h)
set(EMBEDDED_SHADER_TEMPLATE ${CMAKE_SOURCE_DIR}/cmake/embedded_shaders.h.cmakein)
set(EMBEDDED_SHADER_HEADER ${CMAKE_CURRENT_BINARY_DIR}/src/embedded_shaders.h)
# set(TBB_LIB_DIR ${CMAKE_SOURCE_DIR}/thirdparty/tbb/build-emscripten/clang_20.0_cxx11_32_release)
# link_directories(${TBB_LIB_DIR})

if(WASM_GPU_PARALLEL_BACKEND STREQUAL "tbb")
    set(TBB_BUILD_BASE "${CMAKE_SOURCE_DIR}/thirdparty/tbb/build-emscripten")
    file(GLOB TBB_BUILD_DIRS "${TBB_BUILD_BASE}/clang_*")

    list(LENGTH TBB_BUILD_DIRS _tbb_dir_count)
    if(_tbb_dir_count EQUAL 0)
        message(FATAL_ERROR "No TBB build directory found in ${TBB_BUILD_BASE}")
    endif()

    list(GET TBB_BUILD_DIRS 0 TBB_LIB_DIR)
    message(STATUS "Using TBB library dir: ${TBB_LIB_DIR}")
endif()



set(CMAKE_EXECUTABLE_SUFFIX ".js")

include_directories(include)

option(WASM_GPU_INCLUDE_RESNET50_DATA "Embed generated network_data/resnet50 files" OFF)
if(WASM_GPU_BUILD_MODE STREQUAL "resnet50")
    set(WASM_GPU_INCLUDE_RESNET50_DATA ON)
endif()

file(GLOB_RECURSE NETWORK_DATA_FILES CONFIGURE_DEPENDS
    "${NETWORK_DATA_DIR}/*.bin"
    "${NETWORK_DATA_DIR}/*.json"
)
set(EMBED_NETWORK_DATA_ARGS --recursive)
if(NOT WASM_GPU_INCLUDE_RESNET50_DATA)
    list(FILTER NETWORK_DATA_FILES EXCLUDE REGEX "/resnet50/")
    list(APPEND EMBED_NETWORK_DATA_ARGS --exclude-dir resnet50)
endif()
file(GLOB SHADER_DATA_FILES CONFIGURE_DEPENDS
    "${SHADER_DATA_DIR}/*.wgsl"
)

add_custom_command(
    OUTPUT ${EMBEDDED_DATA_HEADER}
    COMMAND ${Python3_EXECUTABLE}
        ${BUILD_SCRIPTS_DIR}/embed_network_data.py
        --input-dir ${NETWORK_DATA_DIR}
        --template ${EMBEDDED_DATA_TEMPLATE}
        --output ${EMBEDDED_DATA_HEADER}
        ${EMBED_NETWORK_DATA_ARGS}
    DEPENDS
        ${BUILD_SCRIPTS_DIR}/embed_network_data.py
        ${EMBEDDED_DATA_TEMPLATE}
        ${NETWORK_DATA_FILES}
    COMMENT "Embedding network_data files"
    VERBATIM
)

add_custom_command(
    OUTPUT ${EMBEDDED_SHADER_HEADER}
    COMMAND ${Python3_EXECUTABLE}
        ${BUILD_SCRIPTS_DIR}/embed_shaders.py
        --input-dir ${SHADER_DATA_DIR}
        --template ${EMBEDDED_SHADER_TEMPLATE}
        --output ${EMBEDDED_SHADER_HEADER}
    DEPENDS
        ${BUILD_SCRIPTS_DIR}/embed_shaders.py
        ${EMBEDDED_SHADER_TEMPLATE}
        ${SHADER_DATA_FILES}
    COMMENT "Embedding WGSL shader files"
    VERBATIM
)

set(WASM_GPU_ENGINE_SOURCE src/wasm_gpu_engine.cpp)
set(WASM_GPU_ENGINE_DEFINE BUILD_WASM_WEBGPU_ENGINE)
set(WASM_GPU_ENGINE_EXTRA_DEFINES)
set(WASM_GPU_MODEL_SOURCES
    src/model_loader.cpp
    src/wgsl_generator.cpp
    src/cpu_conv_kernels.cpp
    src/cpu_graph_executor.cpp
    src/webgpu_graph_executor.cpp
)
set(WASM_GPU_EXTRA_SOURCES src/gpu_executor.cpp src/cpp_executor.cpp ${EMBEDDED_SHADER_HEADER})
set(WASM_GPU_USE_WASM_WEBGPU ON)
set(WASM_GPU_USE_EMDAWN_WEBGPU OFF)
set(WASM_GPU_EMDAWN_PORT "emdawnwebgpu" CACHE STRING "Emscripten Dawn WebGPU port spec for cpp-webgpu-dawn")
option(WASM_GPU_EMDAWN_USE_CLOSURE "Run Closure compiler for cpp-webgpu-dawn builds" OFF)
set(WASM_GPU_USE_JSPI ON)

if(WASM_GPU_BUILD_MODE STREQUAL "js-webgpu")
    set(WASM_GPU_ENGINE_SOURCE src/gpu_engine.cpp)
    set(WASM_GPU_ENGINE_DEFINE BUILD_JS_WEBGPU_ENGINE)
    set(WASM_GPU_EXTRA_SOURCES)
    set(WASM_GPU_USE_WASM_WEBGPU OFF)
    set(WASM_GPU_USE_JSPI OFF)
elseif(WASM_GPU_BUILD_MODE STREQUAL "dummy")
    set(WASM_GPU_ENGINE_SOURCE src/dummy_engine.cpp)
    set(WASM_GPU_ENGINE_DEFINE BUILD_DUMMY_ENGINE)
    set(WASM_GPU_EXTRA_SOURCES)
    set(WASM_GPU_USE_WASM_WEBGPU OFF)
    set(WASM_GPU_USE_JSPI OFF)
elseif(WASM_GPU_BUILD_MODE STREQUAL "resnet50")
    set(WASM_GPU_ENGINE_EXTRA_DEFINES BUILD_WASM_WEBGPU_ASYNC BUILD_RESNET50_MODE)
    set(WASM_GPU_USE_JSPI OFF)
    set(WASM_GPU_INCLUDE_RESNET50_DATA ON)
elseif(WASM_GPU_BUILD_MODE STREQUAL "cpp-webgpu-async")
    set(WASM_GPU_ENGINE_EXTRA_DEFINES BUILD_WASM_WEBGPU_ASYNC)
    set(WASM_GPU_USE_JSPI OFF)
elseif(WASM_GPU_BUILD_MODE STREQUAL "cpp-webgpu-dawn")
    set(WASM_GPU_ENGINE_EXTRA_DEFINES BUILD_WASM_WEBGPU_ASYNC BUILD_EMDAWN_WEBGPU)
    set(WASM_GPU_EXTRA_SOURCES src/dawn_gpu_executor.cpp src/cpp_executor.cpp ${EMBEDDED_SHADER_HEADER})
    set(WASM_GPU_USE_WASM_WEBGPU OFF)
    set(WASM_GPU_USE_EMDAWN_WEBGPU ON)
    set(WASM_GPU_USE_JSPI OFF)
elseif(NOT WASM_GPU_BUILD_MODE STREQUAL "cpp-webgpu")
    message(FATAL_ERROR "Unknown WASM_GPU_BUILD_MODE='${WASM_GPU_BUILD_MODE}'. Use cpp-webgpu, cpp-webgpu-async, cpp-webgpu-dawn, resnet50, js-webgpu, or dummy.")
endif()

message(STATUS "WASM GPU build mode: ${WASM_GPU_BUILD_MODE}")
message(STATUS "WASM GPU parallel backend: ${WASM_GPU_PARALLEL_BACKEND}")

set(WASM_GPU_THREAD_TOOL_SOURCES
    src/thread_tools/parallel_tools.cpp
    src/thread_tools/parallel_utils.cpp
)
if(WASM_GPU_PARALLEL_BACKEND STREQUAL "wasm-thread")
    list(APPEND WASM_GPU_THREAD_TOOL_SOURCES src/thread_tools/wasm_worker_pool.cpp)
elseif(WASM_GPU_PARALLEL_BACKEND STREQUAL "pthread" OR WASM_GPU_PARALLEL_BACKEND STREQUAL "std-thread")
    list(APPEND WASM_GPU_THREAD_TOOL_SOURCES src/thread_tools/thread_pool.cpp)
endif()

add_executable(wasm_gpu
    ${WASM_GPU_ENGINE_SOURCE}
    ${WASM_GPU_EXTRA_SOURCES}
    src/bindings.cpp
    src/network_weights.cpp
    ${WASM_GPU_MODEL_SOURCES}
    ${WASM_GPU_THREAD_TOOL_SOURCES}
    ${EMBEDDED_DATA_HEADER}
)

# Pthread and Emscripten flags
# add_compile_options(-pthread)
# add_compile_options(-sSHARED_MEMORY=1)

# add_link_options(-pthread)
# add_link_options(-sPTHREAD_POOL_SIZE=4)
# add_link_options(-sSHARED_MEMORY=1)
# add_link_options(-sPTHREADS_DEBUG=1)
# add_link_options(-sEXIT_RUNTIME=0)
# add_link_options(-sEXPORTED_RUNTIME_METHODS=PThread)
# add_link_options(-sMODULARIZE=1)
# add_link_options(-sEXPORT_NAME=SmartIDEngine)
# add_link_options(-sENVIRONMENT=worker)
# add_link_options(-sALLOW_MEMORY_GROWTH=1)
# # add_link_options(-sFORCE_FILESYSTEM=1)
# add_link_options(-O3)
# add_link_options(-lembind)

target_link_libraries(wasm_gpu PRIVATE embind)
if(WASM_GPU_PARALLEL_BACKEND STREQUAL "tbb")
    target_include_directories(wasm_gpu PRIVATE ${TBB_ROOT}/include)
    target_link_libraries(wasm_gpu PRIVATE ${TBB_LIB_DIR}/libtbb.a)
    target_compile_definitions(wasm_gpu PRIVATE WASM_GPU_PARALLEL_BACKEND_TBB)
elseif(WASM_GPU_PARALLEL_BACKEND STREQUAL "wasm-thread")
    target_compile_definitions(wasm_gpu PRIVATE WASM_GPU_PARALLEL_BACKEND_WASM_THREAD)
elseif(WASM_GPU_PARALLEL_BACKEND STREQUAL "pthread")
    target_compile_definitions(wasm_gpu PRIVATE WASM_GPU_PARALLEL_BACKEND_PTHREAD)
elseif(WASM_GPU_PARALLEL_BACKEND STREQUAL "std-thread")
    target_compile_definitions(wasm_gpu PRIVATE WASM_GPU_PARALLEL_BACKEND_STD_THREAD)
elseif(WASM_GPU_PARALLEL_BACKEND STREQUAL "serial")
    target_compile_definitions(wasm_gpu PRIVATE WASM_GPU_PARALLEL_BACKEND_SERIAL)
else()
    message(FATAL_ERROR "Unknown WASM_GPU_PARALLEL_BACKEND='${WASM_GPU_PARALLEL_BACKEND}'. Use wasm-thread, pthread, std-thread, tbb, or serial.")
endif()

target_include_directories(wasm_gpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
if(WASM_GPU_PARALLEL_BACKEND STREQUAL "wasm-thread")
    target_compile_options(wasm_gpu PRIVATE -sWASM_WORKERS -pthread -sSHARED_MEMORY=1)
else()
    target_compile_options(wasm_gpu PRIVATE -pthread -sSHARED_MEMORY=1)
endif()
target_compile_definitions(wasm_gpu PRIVATE ${WASM_GPU_ENGINE_DEFINE} ${WASM_GPU_ENGINE_EXTRA_DEFINES})

if(WASM_GPU_USE_WASM_WEBGPU)
    if(NOT EXISTS "${WASM_WEBGPU_LIB_DIR}/lib_webgpu.h")
        message(FATAL_ERROR "cpp-webgpu mode requires wasm_webgpu at ${WASM_WEBGPU_ROOT}")
    endif()

    target_include_directories(wasm_gpu PRIVATE ${WASM_WEBGPU_LIB_DIR})
    target_sources(wasm_gpu PRIVATE
        ${WASM_WEBGPU_LIB_DIR}/lib_webgpu.cpp
        ${WASM_WEBGPU_LIB_DIR}/lib_webgpu_cpp11.cpp
    )

    if(EMSCRIPTEN)
        target_compile_options(wasm_gpu PRIVATE -msimd128)
        target_link_options(wasm_gpu PRIVATE
            -msimd128
            "--js-library=${WASM_WEBGPU_LIB_DIR}/lib_webgpu.js"
        )
    endif()
endif()

if(WASM_GPU_USE_EMDAWN_WEBGPU)
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR "cpp-webgpu-dawn mode requires an Emscripten build.")
    endif()

    target_compile_options(wasm_gpu PRIVATE -msimd128 "--use-port=${WASM_GPU_EMDAWN_PORT}")
    target_link_options(wasm_gpu PRIVATE -msimd128 "--use-port=${WASM_GPU_EMDAWN_PORT}")
    if(WASM_GPU_EMDAWN_USE_CLOSURE)
        target_link_options(wasm_gpu PRIVATE --closure=1)
    endif()
endif()

set(WASM_GPU_LINK_FLAGS "-sMODULARIZE -sEXPORT_NAME=SmartIDEngine -lembind -sEXIT_RUNTIME=0 -sALLOW_MEMORY_GROWTH=1 -O3 -sENVIRONMENT=web,worker -sNO_FILESYSTEM -sNO_USE_SDL -sNO_USE_SDL_MIXER -sDYNAMIC_EXECUTION=1 -sSTACK_SIZE=16mb -sINITIAL_MEMORY=512mb -sNO_ERROR_ON_UNDEFINED_SYMBOLS -sINCOMING_MODULE_JS_API=['locateFile','mainScriptUrlOrBlob','wasmMemory','buffer','instantiateWasm'] -sEXPORTED_FUNCTIONS=_malloc,_free,_start_tbb_session,_start_parallel_session,_start_keepalive_mainloop,_stop_keepalive_mainloop")
if(WASM_GPU_PARALLEL_BACKEND STREQUAL "wasm-thread")
    set(WASM_GPU_LINK_FLAGS "${WASM_GPU_LINK_FLAGS} -sWASM_WORKERS -sPTHREAD_POOL_SIZE=4 -pthread -sSHARED_MEMORY=1 -sPTHREADS_DEBUG=1 -sEXPORTED_RUNTIME_METHODS=PThread")
elseif(NOT WASM_GPU_PARALLEL_BACKEND STREQUAL "serial")
    set(WASM_GPU_LINK_FLAGS "${WASM_GPU_LINK_FLAGS} -sPTHREAD_POOL_SIZE=4 -pthread -sSHARED_MEMORY=1 -sPTHREADS_DEBUG=1 -sEXPORTED_RUNTIME_METHODS=PThread")
endif()
if(WASM_GPU_USE_JSPI)
    set(WASM_GPU_LINK_FLAGS "${WASM_GPU_LINK_FLAGS} -sJSPI -Wno-experimental")
endif()

set_target_properties(wasm_gpu PROPERTIES LINK_FLAGS "${WASM_GPU_LINK_FLAGS}")
