@echo off
setlocal enabledelayedexpansion

REM Verify the Windows cli-transcriber example builds using only the prebuilt
REM library bundle from the latest GitHub release (same as download-lib.bat /
REM README). Fails if the published binary is missing headers or libs.

set "SCRIPTS_DIR=%~dp0"
set "SCRIPTS_DIR=!SCRIPTS_DIR:~0,-1!"
for %%I in ("!SCRIPTS_DIR!") do set "REPO_ROOT_DIR=%%~dpI"
set "REPO_ROOT_DIR=!REPO_ROOT_DIR:~0,-1!"
set "EXAMPLE_DIR=!REPO_ROOT_DIR!\examples\windows\cli-transcriber"

echo [test-examples] Downloading latest moonshine-voice-windows-x86_64 release...
cd /d "!EXAMPLE_DIR!"
call download-lib.bat
if errorlevel 1 exit /b 1

echo [test-examples] Building cli-transcriber (Release^|x64)...
msbuild cli-transcriber.sln /t:Clean /p:Configuration=Release /p:Platform=x64 /nologo
if errorlevel 1 exit /b 1
msbuild cli-transcriber.sln /p:Configuration=Release /p:Platform=x64 /nologo
if errorlevel 1 exit /b 1

echo [test-examples] Windows cli-transcriber example build succeeded
endlocal
