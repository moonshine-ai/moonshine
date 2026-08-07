@echo off
setlocal enabledelayedexpansion

REM Builds the core C++ library and runs its unit tests. This is the Windows
REM counterpart to scripts\test-core.sh.

set "SCRIPTS_DIR=%~dp0"
set "SCRIPTS_DIR=!SCRIPTS_DIR:~0,-1!"
for %%I in ("!SCRIPTS_DIR!") do set "REPO_ROOT_DIR=%%~dpI"
set "REPO_ROOT_DIR=!REPO_ROOT_DIR:~0,-1!"
set "BUILD_DIR=!REPO_ROOT_DIR!\core\build"

REM Git for Windows ships bash, but it is often not on PATH for cmd.exe /
REM pwsh-launched batch files. Prefer PATH, then the usual install locations.
set "BASH_EXE="
where bash >nul 2>&1 && for /f "delims=" %%B in ('where bash') do (
    if not defined BASH_EXE set "BASH_EXE=%%B"
)
if not defined BASH_EXE if exist "C:\Program Files\Git\bin\bash.exe" (
    set "BASH_EXE=C:\Program Files\Git\bin\bash.exe"
)
if not defined BASH_EXE if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
    set "BASH_EXE=C:\Program Files (x86)\Git\bin\bash.exe"
)

REM Always run the fetch when bash is available: the script is idempotent and
REM repairs partial CDN trees. Fall back to the old sentinel check only if bash
REM cannot be found (should not happen on the CI image).
if defined BASH_EXE (
    echo Fetching voice assets from CDN/HF...
    "!BASH_EXE!" "!SCRIPTS_DIR!\fetch-voice-assets.sh" all
    if errorlevel 1 exit /b 1
) else (
    echo WARNING: bash not found; cannot run scripts\fetch-voice-assets.sh
    if not exist "!REPO_ROOT_DIR!\test-assets\tiny-en\encoder_model.ort" (
        echo error: missing test-assets and no bash to fetch them >&2
        exit /b 1
    )
    if not exist "!REPO_ROOT_DIR!\core\moonshine-tts\data\kokoro\model.ort" (
        echo error: missing kokoro TTS assets and no bash to fetch them >&2
        exit /b 1
    )
)

if exist "!BUILD_DIR!" (
    rmdir /s /q "!BUILD_DIR!"
)
mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"
if errorlevel 1 exit /b 1

set "BUILD_TYPE=Debug"
REM Pin to the VS2022 (v143) toolset so the tests exercise the same toolchain we
REM ship to users, even when a newer Visual Studio (e.g. VS2026/v145) is also
REM installed on the box. See github.com/moonshine-ai/moonshine/issues/125.
cmake .. -G "Visual Studio 17 2022" -A x64 -T v143
if errorlevel 1 exit /b 1
cmake --build . --config !BUILD_TYPE!
if errorlevel 1 exit /b 1

cd /d "!REPO_ROOT_DIR!\test-assets"
if errorlevel 1 exit /b 1

set "PATH=!REPO_ROOT_DIR!\core\third-party\onnxruntime\lib\windows\x64;%PATH%"

"!REPO_ROOT_DIR!\core\bin-tokenizer\build\!BUILD_TYPE!\bin-tokenizer-test.exe"
if errorlevel 1 exit /b 1
"!REPO_ROOT_DIR!\core\third-party\onnxruntime\build\!BUILD_TYPE!\onnxruntime-test.exe"
if errorlevel 1 exit /b 1
"!REPO_ROOT_DIR!\core\moonshine-utils\build\!BUILD_TYPE!\debug-utils-test.exe"
if errorlevel 1 exit /b 1
"!REPO_ROOT_DIR!\core\moonshine-utils\build\!BUILD_TYPE!\string-utils-test.exe"
if errorlevel 1 exit /b 1
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\resampler-test.exe"
if errorlevel 1 exit /b 1
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\voice-activity-detector-test.exe"
if errorlevel 1 exit /b 1
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\transcriber-test.exe"
if errorlevel 1 exit /b 1
"!REPO_ROOT_DIR!\core\build\!BUILD_TYPE!\moonshine-c-api-test.exe"
if errorlevel 1 exit /b 1

echo All tests passed
