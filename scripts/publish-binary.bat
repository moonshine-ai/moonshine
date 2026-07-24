@echo off
setlocal enabledelayedexpansion

set VERSION=0.0.73
set REPO=moonshine-ai/moonshine

REM Get the directory where this script is located
set SCRIPTS_DIR=%~dp0
REM Remove trailing backslash
set SCRIPTS_DIR=!SCRIPTS_DIR:~0,-1!

REM Get the parent directory (repo root)
for %%I in ("!SCRIPTS_DIR!") do set REPO_ROOT_DIR=%%~dpI
REM Remove trailing backslash
set REPO_ROOT_DIR=!REPO_ROOT_DIR:~0,-1!

set CORE_DIR=!REPO_ROOT_DIR!\core
set BUILD_DIR=!CORE_DIR!\build

REM Clean and create build directory
if exist !BUILD_DIR! rmdir /s /q !BUILD_DIR!
if not exist !BUILD_DIR! mkdir !BUILD_DIR!
cd /d !BUILD_DIR!
REM Pin the redistributed libraries to the VS2022 (v143) toolset. The release
REM box may also have a newer Visual Studio (e.g. VS2026/v145) installed, whose
REM STL emits calls to helpers (__std_find_first_not_of_trivial_pos_1, etc.)
REM that older VS2022 installs don't provide, producing LNK2001 unresolved
REM externals for consumers. Building with v143 keeps the shipped .libs
REM linkable by VS2022 users. See github.com/moonshine-ai/moonshine/issues/125.
cmake .. -G "Visual Studio 17 2022" -A x64 -T v143
cmake --build . --config Release --target clean
cmake --build . --config Release

REM Create temporary directory
set TMP_DIR=%TEMP%\moonshine-build-%RANDOM%
md !TMP_DIR!

set FOLDER_NAME=moonshine-voice-windows-x86_64
set BINARY_DIR=!TMP_DIR!\!FOLDER_NAME!
md !BINARY_DIR!

set INCLUDE_DIR=!BINARY_DIR!\include
md !INCLUDE_DIR!
copy /Y !CORE_DIR!\moonshine-c-api.h !INCLUDE_DIR!\
copy /Y !CORE_DIR!\moonshine-cpp.h !INCLUDE_DIR!\

set LIB_DIR=!BINARY_DIR!\lib
md !LIB_DIR!

copy /Y !BUILD_DIR!\Release\moonshine.lib !LIB_DIR!\
copy /Y !BUILD_DIR!\..\bin-tokenizer\build\Release\bin-tokenizer.lib !LIB_DIR!\
copy /Y !BUILD_DIR!\..\ort-utils\build\Release\ort-utils.lib !LIB_DIR!\
copy /Y !BUILD_DIR!\..\moonshine-utils\build\Release\moonshine-utils.lib !LIB_DIR!\
copy /Y !CORE_DIR!\third-party\onnxruntime\lib\windows\x86_64\onnxruntime.lib !LIB_DIR!\
copy /Y !CORE_DIR!\third-party\onnxruntime\lib\windows\x86_64\onnxruntime.dll !LIB_DIR!\

cd /d !TMP_DIR!
set TAR_NAME=!FOLDER_NAME!.tar.gz
tar -czf !TAR_NAME! !FOLDER_NAME!
copy /Y !TAR_NAME! !REPO_ROOT_DIR!\

cd /d !REPO_ROOT_DIR!

if /I "%~1"=="upload" (
    REM Check if the GitHub release exists; create it if missing
    gh release view v!VERSION! >nul 2>&1
    if errorlevel 1 (
        gh release create v!VERSION! --title "v!VERSION!" --notes "Release v!VERSION!"
    )

    REM gh release upload has no client-side timeout, so a stalled connection
    REM to GitHub can hang the release forever. Wrap it with a per-attempt
    REM timeout + retry helper. See scripts/gh-upload-retry.ps1.
    powershell -NoProfile -ExecutionPolicy Bypass -File "!SCRIPTS_DIR!\gh-upload-retry.ps1" -Version !VERSION! -Asset "!TAR_NAME!"
    if errorlevel 1 exit /b 1
)

REM Cleanup temporary directory
rmdir /s /q !TMP_DIR!

endlocal
