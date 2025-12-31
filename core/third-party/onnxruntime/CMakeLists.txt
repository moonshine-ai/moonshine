cmake_minimum_required(VERSION 3.22.1)

project("onnxruntime")

if (IOS)
    add_library(onnxruntime STATIC IMPORTED)
else()
    add_library(onnxruntime SHARED IMPORTED)
endif()

if (ANDROID)
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
    set(ONNXRUNTIME_LIB_PATH_LOCAL "${CMAKE_CURRENT_LIST_DIR}/lib/android/${ONNXRUNTIME_ABI_DIR}/libonnxruntime.so")
    set(ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_LIB_PATH_LOCAL}" PARENT_SCOPE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH_LOCAL}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
elseif(IOS)
    if (CMAKE_OSX_SYSROOT STREQUAL "iphonesimulator")
        set(ONNXRUNTIME_LIB_PATH_LOCAL "${CMAKE_CURRENT_LIST_DIR}/lib/ios/simulator/libonnxruntime.a")
    else()
        set(ONNXRUNTIME_LIB_PATH_LOCAL "${CMAKE_CURRENT_LIST_DIR}/lib/ios/arm64/libonnxruntime.a")
    endif()
    set(ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_LIB_PATH_LOCAL}" PARENT_SCOPE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH_LOCAL}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
    # Link iOS frameworks required by onnxruntime
    target_link_libraries(onnxruntime INTERFACE
        "-framework CoreFoundation"
        "-framework Foundation"
    )
elseif(APPLE)
    set(ONNXRUNTIME_LIB_PATH_LOCAL "${CMAKE_CURRENT_LIST_DIR}/lib/macos/arm64/libonnxruntime.1.23.2.dylib")
    set(ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_LIB_PATH_LOCAL}" PARENT_SCOPE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH_LOCAL}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
elseif(LINUX)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ONNXRUNTIME_LIB_PATH_LOCAL "${CMAKE_CURRENT_LIST_DIR}/lib/linux/aarch64/libonnxruntime.so.1")
    else()
        set(ONNXRUNTIME_LIB_PATH_LOCAL "${CMAKE_CURRENT_LIST_DIR}/lib/linux/x86_64/libonnxruntime.so.1")
    endif()
    set(ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_LIB_PATH_LOCAL}" PARENT_SCOPE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH_LOCAL}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" PARENT_SCOPE)

add_executable(onnxruntime-test onnxruntime-test.cpp)

target_include_directories(onnxruntime-test PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
)

if (IOS)
    set_target_properties(onnxruntime-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.onnxruntime-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
    target_link_libraries(onnxruntime-test PRIVATE
        onnxruntime
        "-framework CoreFoundation"
        "-framework Foundation"
    )
else()
    target_link_libraries(onnxruntime-test
        onnxruntime
    )
endif()