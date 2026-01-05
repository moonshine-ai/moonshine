@echo off
setlocal enabledelayedexpansion

REM Get script directory and resolve to absolute path
pushd "%~dp0"
set "SCRIPTS_DIR=%CD%"
popd

REM Get repository root (parent of scripts directory)
for %%i in ("%SCRIPTS_DIR%\..") do set "REPO_ROOT_DIR=%%~fi"
set "PYTHON_DIR=%REPO_ROOT_DIR%\python"

set "CORE_DIR=%REPO_ROOT_DIR%\core"
set "CORE_BUILD_DIR=%CORE_DIR%\build"

if exist "%CORE_BUILD_DIR%" rmdir /s /q "%CORE_BUILD_DIR%"
if errorlevel 1 exit /b 1

mkdir "%CORE_BUILD_DIR%"
if errorlevel 1 exit /b 1

cd /d "%CORE_BUILD_DIR%"
if errorlevel 1 exit /b 1

cmake .. -DMOONSHINE_BUILD_SHARED=YES
if errorlevel 1 exit /b 1

cmake --build . --config Release
if errorlevel 1 exit /b 1

copy /Y "%CORE_BUILD_DIR%\Release\*.dll" "%PYTHON_DIR%\src\moonshine_voice\"
if errorlevel 1 exit /b 1

REM Detect platform and copy appropriate ONNX Runtime library
if "%OS%"=="Windows_NT" (
    if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
        copy /Y "%CORE_DIR%\third-party\onnxruntime\lib\windows\x86_64\onnxruntime*.dll" "%PYTHON_DIR%\src\moonshine_voice\"
        if errorlevel 1 exit /b 1
    ) else if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
        copy /Y "%CORE_DIR%\third-party\onnxruntime\lib\windows\arm64\onnxruntime*.dll" "%PYTHON_DIR%\src\moonshine_voice\"
        if errorlevel 1 exit /b 1
    ) else (
        echo Unsupported Windows architecture: %PROCESSOR_ARCHITECTURE%
        echo You'll need to manually copy the ONNX Runtime library to the python/src/moonshine_voice/ directory.
        exit /b 1
    )
) else (
    echo Unsupported platform
    echo You'll need to manually copy the ONNX Runtime library to the python/src/moonshine_voice/ directory.
    exit /b 1
)

cd /d "%PYTHON_DIR%"
if errorlevel 1 exit /b 1

if exist ".venv" rmdir /s /q ".venv"
uv venv
if errorlevel 1 exit /b 1

call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1

uv pip install -r build-requirements.txt
if errorlevel 1 exit /b 1

REM Determine platform tag for wheel name (Windows)
set "PLAT_NAME=win_amd64"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    set "PLAT_NAME=win_arm64"
)

REM Build platform-specific wheel
if exist "dist" rmdir /s /q "dist"
if exist "wheelhouse" rmdir /s /q "wheelhouse"
mkdir "dist" 2>nul
mkdir "wheelhouse" 2>nul

uv run setup.py bdist_wheel
if errorlevel 1 exit /b 1

REM Upload to PyPI if "upload" argument is provided
if "%1"=="upload" (
    twine upload dist\*
    if errorlevel 1 exit /b 1
)

endlocal

