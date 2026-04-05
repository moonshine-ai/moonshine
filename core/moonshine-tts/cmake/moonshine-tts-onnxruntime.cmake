# ONNX Runtime for moonshine-tts. When built inside moonshine core, reuses
# ONNXRUNTIME_LIB_PATH from third-party/onnxruntime/find-ort-library-path.cmake.

set(_MOONSHINE_TTS_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(_MOONSHINE_TTS_CORE_THIRD_PARTY "${_MOONSHINE_TTS_CMAKE_DIR}/../../third-party/onnxruntime")

if(DEFINED ONNXRUNTIME_LIB_PATH AND EXISTS "${ONNXRUNTIME_LIB_PATH}")
  set(MOONSHINE_TTS_ORT_INCLUDE "${_MOONSHINE_TTS_CORE_THIRD_PARTY}/include")
  set(MOONSHINE_TTS_ORT_LIB "${ONNXRUNTIME_LIB_PATH}")
else()
  set(_MOONSHINE_TTS_ORT_DEFAULT_ROOT "${_MOONSHINE_TTS_CORE_THIRD_PARTY}")
  set(ONNXRUNTIME_ROOT "${_MOONSHINE_TTS_ORT_DEFAULT_ROOT}" CACHE PATH
      "Root of ONNX Runtime distribution (default: core/third-party/onnxruntime)")

  set(MOONSHINE_TTS_ORT_INCLUDE "${ONNXRUNTIME_ROOT}/include")

  set(_ort_linux_x64 "${ONNXRUNTIME_ROOT}/lib/linux/x86_64/libonnxruntime.so.1")

  if(WIN32)
    find_library(
      MOONSHINE_TTS_ORT_LIB
      NAMES onnxruntime
      PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib/windows/x86_64"
      NO_DEFAULT_PATH
      REQUIRED
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${_ort_linux_x64}")
    set(MOONSHINE_TTS_ORT_LIB "${_ort_linux_x64}")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(
      MOONSHINE_TTS_ORT_LIB
      NAMES onnxruntime libonnxruntime.so.1
      PATHS "${ONNXRUNTIME_ROOT}/lib/linux/x86_64" "${ONNXRUNTIME_ROOT}/lib/linux/aarch64"
            "${ONNXRUNTIME_ROOT}/lib"
      NO_DEFAULT_PATH
      REQUIRED
    )
  elseif(APPLE)
    set(_moonshine_tts_ort_mac_lib_dirs
      "${ONNXRUNTIME_ROOT}/lib/macos/arm64"
      "${ONNXRUNTIME_ROOT}/lib/macos/x86_64"
      "${ONNXRUNTIME_ROOT}/lib/arm64"
      "${ONNXRUNTIME_ROOT}/lib/x86_64"
      "${ONNXRUNTIME_ROOT}/lib"
    )
    foreach(_d IN LISTS _moonshine_tts_ort_mac_lib_dirs)
      if(IS_DIRECTORY "${_d}")
        file(GLOB _moonshine_tts_ort_dylibs LIST_DIRECTORIES false "${_d}/libonnxruntime*.dylib")
        if(_moonshine_tts_ort_dylibs)
          list(GET _moonshine_tts_ort_dylibs 0 MOONSHINE_TTS_ORT_LIB)
          break()
        endif()
      endif()
    endforeach()
    if(NOT MOONSHINE_TTS_ORT_LIB)
      find_library(
        MOONSHINE_TTS_ORT_LIB
        NAMES onnxruntime
        PATHS ${_moonshine_tts_ort_mac_lib_dirs}
        NO_DEFAULT_PATH
      )
    endif()
    if(NOT MOONSHINE_TTS_ORT_LIB)
      message(FATAL_ERROR
        "Could not find ONNX Runtime under ${ONNXRUNTIME_ROOT}/lib. "
        "Set -DONNXRUNTIME_ROOT=... See core/third-party/onnxruntime/README.md.")
    endif()
  else()
    find_library(
      MOONSHINE_TTS_ORT_LIB
      NAMES onnxruntime libonnxruntime.so.1
      PATHS "${ONNXRUNTIME_ROOT}/lib"
      NO_DEFAULT_PATH
      REQUIRED
    )
  endif()
endif()

if(NOT EXISTS "${MOONSHINE_TTS_ORT_INCLUDE}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "onnxruntime_cxx_api.h not found under ${MOONSHINE_TTS_ORT_INCLUDE}")
endif()

function(moonshine_tts_link_onnxruntime target_name)
  target_include_directories(${target_name} SYSTEM PUBLIC "${MOONSHINE_TTS_ORT_INCLUDE}")
  target_link_libraries(${target_name} PUBLIC "${MOONSHINE_TTS_ORT_LIB}")
  if(UNIX AND NOT APPLE AND NOT ANDROID)
    target_link_libraries(${target_name} PUBLIC pthread dl)
  endif()
endfunction()

function(moonshine_tts_set_ort_rpath target_name)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MOONSHINE_TTS_ORT_LIB MATCHES "\\.so")
    get_filename_component(_moonshine_tts_ort_rpath "${MOONSHINE_TTS_ORT_LIB}" DIRECTORY)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${_moonshine_tts_ort_rpath}"
      INSTALL_RPATH "${_moonshine_tts_ort_rpath}"
    )
  endif()
endfunction()
