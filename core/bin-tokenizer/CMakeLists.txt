cmake_minimum_required(VERSION 3.22.1)

project("bin-tokenizer")

add_library(bin-tokenizer STATIC
    bin-tokenizer.cpp
)

target_compile_options(bin-tokenizer PRIVATE
    -fPIC
)

target_include_directories(bin-tokenizer PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../moonshine-utils
)

if (ANDROID)
    target_link_libraries(bin-tokenizer PUBLIC
        moonshine-utils
        android
        log
    )
else()
    target_link_libraries(bin-tokenizer PUBLIC
        moonshine-utils
    )
endif()

add_executable(bin-tokenizer-test bin-tokenizer-test.cpp)

target_include_directories(bin-tokenizer-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/../third-party/doctest
)

set_target_properties(bin-tokenizer-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(bin-tokenizer-test
    bin-tokenizer
    moonshine-utils
)

set(BIN_TOKENIZER_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}" PARENT_SCOPE)
