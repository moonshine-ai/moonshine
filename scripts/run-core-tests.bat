@echo off
setlocal enabledelayedexpansion
REM Stop batch script execution on error
set ERRORS=0
set ERRORLEVEL=0
REM Enable error checking after every command in the script
set EXIT_ON_ERROR=1
REM The following line will cause the script to exit if any command returns an error
set "ON_ERROR_GO_TO=handle_error"
goto :main

:check_error
if errorlevel 1 (
    goto :handle_error
)
goto :eof

:handle_error
echo Error encountered. Exiting...
exit /b 1

:main
REM Script continues here
setlocal enabledelayedexpansion

set "SCRIPTS_DIR=%~dp0"
set "SCRIPTS_DIR=!SCRIPTS_DIR:~0,-1!"
for %%I in ("!SCRIPTS_DIR!") do set "REPO_ROOT_DIR=%%~dpI"
set "REPO_ROOT_DIR=!REPO_ROOT_DIR:~0,-1!"
set "BUILD_DIR=!REPO_ROOT_DIR!\core\build"

if exist "!BUILD_DIR!" (
    rmdir /s /q "!BUILD_DIR!"
)
mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"

set "BUILD_TYPE=Debug"
cmake ..
cmake --build . --config !BUILD_TYPE!

cd /d "!REPO_ROOT_DIR!\test-assets"

set "PATH=!REPO_ROOT_DIR!\core\third-party\onnxruntime\lib\windows\x64;%PATH%"

"!REPO_ROOT_DIR!\core\bin-tokenizer\build\!BUILD_TYPE!\bin-tokenizer-test.exe"
"!REPO_ROOT_DIR!\core\third-party\onnxruntime\build\!BUILD_TYPE!\onnxruntime-test.exe"
"!REPO_ROOT_DIR!\core\moonshine-utils\build\!BUILD_TYPE!\debug-utils-test.exe"
"!REPO_ROOT_DIR!\core\moonshine-utils\build\!BUILD_TYPE!\string-utils-test.exe"
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\resampler-test.exe"
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\voice-activity-detector-test.exe"
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\transcriber-test.exe"
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\moonshine-c-api-test.exe"

echo All tests passed

