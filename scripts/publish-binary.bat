@echo off
setlocal enabledelayedexpansion

set VERSION=0.0.48
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
cmake ..
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

REM Check if the GitHub release exists; create it if missing
gh release view v!VERSION! >nul 2>&1
if errorlevel 1 (
    gh release create v!VERSION! --title "v!VERSION!" --notes "Release v!VERSION!"
)

gh release upload v!VERSION! !TAR_NAME! --clobber

REM Cleanup temporary directory
rmdir /s /q !TMP_DIR!

endlocal
