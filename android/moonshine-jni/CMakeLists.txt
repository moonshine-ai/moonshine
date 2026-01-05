cmake_minimum_required(VERSION 3.22.1)

project("moonshine-jni")

add_library(moonshine-jni SHARED
    moonshine-jni.cpp
)

target_include_directories(moonshine-jni PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../core
    ${CMAKE_CURRENT_LIST_DIR}/../../core/third-party/utf-8
)

target_link_libraries(moonshine-jni PUBLIC
    moonshine
    android
)