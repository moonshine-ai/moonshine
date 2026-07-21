if (EMSCRIPTEN)
    # WebAssembly build: link the static ORT archive produced by
    # scripts/build-ort-wasm.sh (Microsoft does not publish a prebuilt one).
    # The default archive is built with SIMD + multithreading. Set
    # -DMOONSHINE_WASM_SINGLE_THREAD=ON to link the SIMD-only fallback
    # (for pages that can't be cross-origin isolated / lack SharedArrayBuffer).
    if (MOONSHINE_WASM_SINGLE_THREAD)
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/wasm/libonnxruntime_webassembly_singlethread.a" CACHE INTERNAL "")
    else()
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/wasm/libonnxruntime_webassembly.a" CACHE INTERNAL "")
    endif()
    if (NOT EXISTS "${ONNXRUNTIME_LIB_PATH}")
        message(FATAL_ERROR
            "ORT-wasm static library not found at ${ONNXRUNTIME_LIB_PATH}.\n"
            "Build and vendor it first: scripts/build-ort-wasm.sh"
            "$<$<BOOL:${MOONSHINE_WASM_SINGLE_THREAD}>: single-thread>")
    endif()
elseif (ANDROID)
    # Detect Android ABI and map to library directory name
    if(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(ONNXRUNTIME_ABI_DIR "armeabi-v7a")
    elseif(ANDROID_ABI STREQUAL "arm64-v8a")
        set(ONNXRUNTIME_ABI_DIR "arm64")
    elseif(ANDROID_ABI STREQUAL "x86")
        set(ONNXRUNTIME_ABI_DIR "x86")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(ONNXRUNTIME_ABI_DIR "x86_64")
    else()
        # Default to arm64 if ABI is not specified
        set(ONNXRUNTIME_ABI_DIR "arm64")
    endif()
    set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/android/${ONNXRUNTIME_ABI_DIR}/libonnxruntime.so"  CACHE INTERNAL "")
elseif(IOS OR MOONSHINE_BUILD_SWIFT)
    if (MOONSHINE_BUILD_SWIFT)
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/macos/arm64/libonnxruntime.a" CACHE INTERNAL "")
    elseif (CMAKE_OSX_SYSROOT STREQUAL "iphonesimulator")
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/ios/simulator/libonnxruntime.a" CACHE INTERNAL "")
    else()
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/ios/arm64/libonnxruntime.a" CACHE INTERNAL "")
    endif()
elseif(APPLE)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/macos/x86_64/libonnxruntime.1.23.2.dylib" CACHE INTERNAL "")
    else()
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/macos/arm64/libonnxruntime.1.23.2.dylib" CACHE INTERNAL "")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR (UNIX AND NOT APPLE))
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/linux/aarch64/libonnxruntime.so.1" CACHE INTERNAL "")
    else()
        set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/linux/x86_64/libonnxruntime.so.1" CACHE INTERNAL "")
    endif()
elseif(WIN32)
    set(ONNXRUNTIME_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/windows/x86_64/onnxruntime.lib" CACHE INTERNAL "")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

function(copy_onnxruntime_dll target_name)
if(WIN32)
    string(REPLACE ".lib" ".dll" ONNXRUNTIME_DLL_PATH "${ONNXRUNTIME_LIB_PATH}")
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL_PATH}"
        $<TARGET_FILE_DIR:${target_name}>
        COMMENT "Copying onnxruntime DLL to executable directory to prevent loading from system libraries"
    )
endif()
endfunction()