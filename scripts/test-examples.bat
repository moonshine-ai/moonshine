@echo off
setlocal enabledelayedexpansion

REM Verify the Windows cli-transcriber example from the GitHub release archive
REM (or a local staged copy when --local PATH is passed by publish-examples.bat).

set "SCRIPTS_DIR=%~dp0"
set "SCRIPTS_DIR=!SCRIPTS_DIR:~0,-1!"
for %%I in ("!SCRIPTS_DIR!") do set "REPO_ROOT_DIR=%%~dpI"
set "REPO_ROOT_DIR=!REPO_ROOT_DIR:~0,-1!"

set "EXAMPLE_DIR="
set "WORK_ROOT="
set "DOWNLOADED=0"

if /I "%~1"=="--local" (
  set "EXAMPLE_DIR=%~2"
  if not exist "!EXAMPLE_DIR!\cli-transcriber.sln" (
    echo [test-examples] ERROR: missing cli-transcriber.sln in !EXAMPLE_DIR!
    exit /b 1
  )
  goto build_example
)

set "WORK_ROOT=%TEMP%\moonshine-test-examples-%RANDOM%"
mkdir "!WORK_ROOT!"
set "EXAMPLE_DIR=!WORK_ROOT!\cli-transcriber"
set "DOWNLOADED=1"

echo [test-examples] Downloading latest windows-cli-transcriber release...
curl -fL https://github.com/moonshine-ai/moonshine/releases/latest/download/windows-cli-transcriber.tar.gz -o "!WORK_ROOT!\windows-cli-transcriber.tar.gz"
if errorlevel 1 exit /b 1
tar -xzf "!WORK_ROOT!\windows-cli-transcriber.tar.gz" -C "!WORK_ROOT!"
if errorlevel 1 exit /b 1

:build_example
echo [test-examples] Building cli-transcriber (Release^|x64)...
cd /d "!EXAMPLE_DIR!"
msbuild cli-transcriber.sln /t:Clean /p:Configuration=Release /p:Platform=x64 /nologo
if errorlevel 1 goto fail
msbuild cli-transcriber.sln /p:Configuration=Release /p:Platform=x64 /nologo
if errorlevel 1 goto fail

if not exist "models\medium-streaming-en\streaming_config.json" (
  echo [test-examples] ERROR: bundled medium-streaming-en model is missing
  goto fail
)
if not exist "beckett.wav" (
  echo [test-examples] ERROR: bundled beckett.wav is missing
  goto fail
)

echo [test-examples] Running cli-transcriber on beckett.wav...
set "OUTPUT_FILE=!EXAMPLE_DIR!\test-output.txt"
del /q "!OUTPUT_FILE!" 2>nul
"x64\Release\cli-transcriber.exe" --wav-path beckett.wav > "!OUTPUT_FILE!" 2>&1
if errorlevel 1 (
  type "!OUTPUT_FILE!"
  goto fail
)

findstr /I /C:"ever tried" "!OUTPUT_FILE!" >nul
if errorlevel 1 (
  echo [test-examples] ERROR: expected transcript to contain "ever tried"
  type "!OUTPUT_FILE!"
  goto fail
)
findstr /I /C:"fail better" "!OUTPUT_FILE!" >nul
if errorlevel 1 (
  echo [test-examples] ERROR: expected transcript to contain "fail better"
  type "!OUTPUT_FILE!"
  goto fail
)

echo [test-examples] Windows cli-transcriber example build and run succeeded
if "!DOWNLOADED!"=="1" rmdir /s /q "!WORK_ROOT!"
endlocal
exit /b 0

:fail
if "!DOWNLOADED!"=="1" rmdir /s /q "!WORK_ROOT!"
exit /b 1
