cmake_minimum_required(VERSION 3.22.1)

project("onnxruntime")

include(${CMAKE_CURRENT_LIST_DIR}/find-ort-library-path.cmake)

if (IOS OR MOONSHINE_BUILD_SWIFT)
    add_library(onnxruntime STATIC IMPORTED)
else()
    add_library(onnxruntime SHARED IMPORTED)
endif()

if (ANDROID)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
elseif(IOS OR MOONSHINE_BUILD_SWIFT)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
    # Link iOS frameworks required by onnxruntime
    target_link_libraries(onnxruntime INTERFACE
        "-framework CoreFoundation"
        "-framework Foundation"
    )
elseif(APPLE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
elseif(LINUX)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
elseif(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_PATH}"
        IMPORTED_IMPLIB "${ONNXRUNTIME_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/"
    )
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")

add_executable(onnxruntime-test onnxruntime-test.cpp)

target_include_directories(onnxruntime-test PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
)

if (IOS OR MOONSHINE_BUILD_SWIFT)
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
elseif(WIN32)
    target_link_libraries(onnxruntime-test PRIVATE
        onnxruntime
        psapi
    )
    copy_onnxruntime_dll(onnxruntime-test)
else()
    target_link_libraries(onnxruntime-test
        onnxruntime
    )
endif()