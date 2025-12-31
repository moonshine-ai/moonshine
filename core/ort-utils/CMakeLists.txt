cmake_minimum_required(VERSION 3.22.1)

project("ort-utils")

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/../third-party/onnxruntime/
    ${CMAKE_CURRENT_LIST_DIR}/../third-party/onnxruntime/build
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/../moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/../moonshine-utils/build
)

add_library(ort-utils STATIC
    ort-utils.cpp
    moonshine-ort-allocator.cpp
    moonshine-tensor-view.cpp
    moonshine-tensor.cpp
)

target_include_directories(ort-utils PRIVATE
    ${MOONSHINE_UTILS_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

if (ANDROID)
target_link_libraries(ort-utils PUBLIC
    moonshine-utils
    ${ONNXRUNTIME_LIB_PATH}
    android
    log
)
else()
target_link_libraries(ort-utils PUBLIC
    moonshine-utils
    ${ONNXRUNTIME_LIB_PATH}
)
endif()
