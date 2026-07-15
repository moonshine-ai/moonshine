@echo off
setlocal enabledelayedexpansion

set VERSION=0.0.69
set REPO=moonshine-ai/moonshine

set "SCRIPTS_DIR=%~dp0"
set "SCRIPTS_DIR=!SCRIPTS_DIR:~0,-1!"
for %%I in ("!SCRIPTS_DIR!") do set "REPO_ROOT_DIR=%%~dpI"
set "REPO_ROOT_DIR=!REPO_ROOT_DIR:~0,-1!"

set "SOURCE_DIR=!REPO_ROOT_DIR!\examples\windows\cli-transcriber"
set "STAGE_ROOT=!REPO_ROOT_DIR!\build\windows-cli-transcriber-stage"
if exist "!STAGE_ROOT!" rmdir /s /q "!STAGE_ROOT!"
set "STAGE_DIR=!STAGE_ROOT!\cli-transcriber"
set "TAR_NAME=windows-cli-transcriber.tar.gz"
set "MODEL_URL=https://download.moonshine.ai/model/medium-streaming-en/quantized"
set "BECKETT_URL=https://github.com/moonshine-ai/moonshine/raw/refs/heads/main/test-assets/beckett.wav"

echo [publish-examples] Building Windows library bundle...
call "!SCRIPTS_DIR!\publish-binary.bat"
if errorlevel 1 exit /b 1

echo [publish-examples] Staging cli-transcriber example...
mkdir "!STAGE_DIR!"
robocopy "!SOURCE_DIR!" "!STAGE_DIR!" /E /XD moonshine-voice-windows-x86_64 x64 .vs /XF *.user >nul
if errorlevel 8 exit /b 1

tar -xzf "!REPO_ROOT_DIR!\moonshine-voice-windows-x86_64.tar.gz" -C "!STAGE_DIR!"
if errorlevel 1 exit /b 1

set "MODEL_DIR=!STAGE_DIR!\models\medium-streaming-en"
mkdir "!MODEL_DIR!"
for %%F in (
  adapter.ort
  cross_kv.ort
  decoder_kv.ort
  encoder.ort
  frontend.ort
  streaming_config.json
  tokenizer.bin
  decoder_kv_with_attention.ort
) do (
  echo [publish-examples] Downloading model file %%F...
  curl -fL "!MODEL_URL!/%%F" -o "!MODEL_DIR!\%%F"
  if errorlevel 1 exit /b 1
)

echo [publish-examples] Downloading beckett.wav...
curl -fL "!BECKETT_URL!" -o "!STAGE_DIR!\beckett.wav"
if errorlevel 1 exit /b 1

echo [publish-examples] Verifying staged example...
call "!SCRIPTS_DIR!\test-examples.bat" --local "!STAGE_DIR!"
if errorlevel 1 (
  rmdir /s /q "!STAGE_ROOT!"
  exit /b 1
)

cd /d "!STAGE_ROOT!"
tar -czf "!TAR_NAME!" cli-transcriber
if errorlevel 1 exit /b 1
copy /Y "!TAR_NAME!" "!REPO_ROOT_DIR!\" >nul
if errorlevel 1 exit /b 1

if exist "!STAGE_ROOT!" (
  rmdir /s /q "!STAGE_ROOT!" 2>nul
)

if /I "%~1"=="upload" (
  cd /d "!REPO_ROOT_DIR!"
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

echo [publish-examples] Windows example archive ready: !TAR_NAME!
endlocal
