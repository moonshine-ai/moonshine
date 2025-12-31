cmake_minimum_required(VERSION 3.22.1)

project("moonshine-utils")

include_directories(${CMAKE_CURRENT_LIST_DIR}/../third-party/doctest)

add_library(moonshine-utils STATIC
    string-utils.cpp
    debug-utils.cpp
)

target_compile_options(moonshine-utils PRIVATE
    -fPIC
)

set_target_properties(moonshine-utils PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if (ANDROID)
  target_link_libraries(moonshine-utils PUBLIC
    android
    log
  )
endif()

set(MOONSHINE_UTILS_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}" PARENT_SCOPE)
set(MOONSHINE_UTILS_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/build" PARENT_SCOPE)

add_executable(string-utils-test string-utils-test.cpp)
target_include_directories(string-utils-test PRIVATE
    ${MOONSHINE_UTILS_INCLUDE_DIR}
    ${DOCTEST_INCLUDE_DIR}
)

if (IOS)
    set_target_properties(string-utils-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.string-utils-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
endif()

target_link_libraries(string-utils-test
    moonshine-utils
)

add_executable(debug-utils-test debug-utils-test.cpp)
target_include_directories(debug-utils-test PRIVATE
    ${MOONSHINE_UTILS_INCLUDE_DIR}
    ${DOCTEST_INCLUDE_DIR}
)

if (IOS)
    set_target_properties(debug-utils-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.debug-utils-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
endif()

set_target_properties(debug-utils-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(debug-utils-test
    moonshine-utils
)