# Emscripten module for the Moonshine WASM binding.
#
# This is added from core/CMakeLists.txt only when building with Emscripten
# (emcmake cmake ...). It compiles the embind bridge and links it against the
# `moonshine` static library (built by the core) plus the vendored ORT-wasm
# static archive, emitting moonshine.mjs + moonshine.wasm as an ES6 module.
#
# NOTE: the exact emscripten link flags (exceptions / pthread / memory) may need
# tuning against the real ORT-wasm archive on first compile.

add_executable(moonshine_wasm ${CMAKE_CURRENT_LIST_DIR}/moonshine-embind.cpp)

set_target_properties(moonshine_wasm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    OUTPUT_NAME "moonshine"
    # Emit an ES module (moonshine.mjs) rather than the default moonshine.js, so
    # `import ... from './moonshine.mjs'` in the TS layer resolves.
    SUFFIX ".mjs"
)

target_include_directories(moonshine_wasm PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../core
)

# The core library (libmoonshine.a) is built with TTS enabled for Emscripten
# (MOONSHINE_TTS_BUILD_ONNX defaults ON), so expose the TTS/G2P embind bindings.
# This mirrors the PRIVATE define set on the `moonshine` target in
# core/CMakeLists.txt, which doesn't propagate to this translation unit.
target_compile_definitions(moonshine_wasm PRIVATE MOONSHINE_C_API_MOONSHINE_TTS=1)

# `moonshine` (defined in core/CMakeLists.txt) transitively pulls in ort-utils,
# bin-tokenizer, the TTS libs, and the vendored ORT-wasm archive
# (ONNXRUNTIME_LIB_PATH), so linking it is enough.
target_link_libraries(moonshine_wasm PRIVATE moonshine ${ONNXRUNTIME_LIB_PATH})

# Common link flags: emit an ES6 module that works on the main thread and in
# Web Workers, grows memory on demand, and exposes embind + the heap views the
# TS layer needs for zero-copy Float32Array marshalling.
set(_MOONSHINE_WASM_LINK_FLAGS
    "-lembind"
    "-fexceptions"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sEXPORT_NAME=createMoonshineModule"
    "-sALLOW_MEMORY_GROWTH=1"
    # Static data (models + ORT) needs a larger initial heap than the 16MB
    # default; it still grows on demand up to MAXIMUM_MEMORY.
    "-sINITIAL_MEMORY=64MB"
    "-sMAXIMUM_MEMORY=4GB"
    "-sWASM_BIGINT=1"
    # node is included so the Node test suite (scripts/test-wasm.sh) can load
    # the same module the browser uses.
    "-sENVIRONMENT=web,worker,node"
    # FS is exported (and forced in) so the intent recognizer — which only has a
    # from-files C API — can be fed models staged into the in-memory MEMFS.
    "-sFORCE_FILESYSTEM=1"
    "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','HEAPF32','FS']"
    "-sSTACK_SIZE=5MB"
)

# The default vendored ORT archive is built with SIMD + pthreads, so match it
# unless the SIMD-only fallback was selected (-DMOONSHINE_WASM_SINGLE_THREAD=ON).
if (NOT MOONSHINE_WASM_SINGLE_THREAD)
    list(APPEND _MOONSHINE_WASM_LINK_FLAGS
        "-pthread"
        "-sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency"
    )
endif()

list(JOIN _MOONSHINE_WASM_LINK_FLAGS " " _MOONSHINE_WASM_LINK_FLAGS_STR)
set_target_properties(moonshine_wasm PROPERTIES LINK_FLAGS "${_MOONSHINE_WASM_LINK_FLAGS_STR}")
