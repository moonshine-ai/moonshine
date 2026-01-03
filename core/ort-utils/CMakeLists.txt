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

target_compile_options(ort-utils PRIVATE
    -fPIC
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
elseif(IOS OR MOONSHINE_BUILD_SWIFT)
target_link_libraries(ort-utils PUBLIC
    moonshine-utils
    ${ONNXRUNTIME_LIB_PATH}
)
add_custom_command(TARGET ort-utils POST_BUILD
COMMAND ${CMAKE_COMMAND} -E echo "Merging static libraries into framework..."
COMMAND libtool -static -o "$<TARGET_FILE:ort-utils>.merged"
    "$<TARGET_FILE:moonshine-utils>"
    "$ONNXRUNTIME_LIB_PATH"
    "$<TARGET_FILE:ort-utils>"
COMMAND ${CMAKE_COMMAND} -E remove "$<TARGET_FILE:ort-utils>"
COMMAND ${CMAKE_COMMAND} -E rename "$<TARGET_FILE:ort-utils>.merged" "$<TARGET_FILE:ort-utils>"
COMMENT "Merging static dependencies into framework binary"
)

else()
target_link_libraries(ort-utils PUBLIC
    moonshine-utils
    ${ONNXRUNTIME_LIB_PATH}
)
endif()
