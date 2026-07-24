@echo off
setlocal enabledelayedexpansion

REM Runs the Python module tests (python\tests\test_modules.py), which drive the
REM __main__ sections of the most significant moonshine_voice modules, and the
REM CLI tests (python\tests\test_cli.py), which exercise the installed
REM moonshine-voice console script -- both against a freshly built wheel. This
REM is the Windows counterpart to scripts\test-python.sh.
REM
REM Usage:
REM   scripts\test-python.bat                Build the wheel first, then test.
REM   scripts\test-python.bat --skip-build   Reuse the wheel already in
REM                                          python\dist\ (used by the Windows CI
REM                                          orchestrator, which builds the wheel
REM                                          in the preceding build-pip step).

REM Resolve script dir and repo root.
pushd "%~dp0"
set "SCRIPTS_DIR=%CD%"
popd
for %%i in ("%SCRIPTS_DIR%\..") do set "REPO_ROOT_DIR=%%~fi"
set "PYTHON_DIR=%REPO_ROOT_DIR%\python"

REM Build the wheel unless the caller says to reuse the existing one.
if /I not "%1"=="--skip-build" (
    call "%SCRIPTS_DIR%\build-pip.bat"
    if errorlevel 1 exit /b 1
)

REM Locate the wheel produced by build-pip (last match wins if several exist).
set "WHEEL="
for %%f in ("%PYTHON_DIR%\dist\moonshine_voice-*.whl") do set "WHEEL=%%f"
if not defined WHEEL (
    echo No wheel found in %PYTHON_DIR%\dist\. Run scripts\build-pip.bat first.
    exit /b 1
)
echo Testing against wheel: !WHEEL!

cd /d "%PYTHON_DIR%"
if errorlevel 1 exit /b 1

REM A throwaway venv so the modules are tested against exactly the wheel that
REM will be uploaded, not whatever is already installed on this machine.
if exist ".venv-pytest" rmdir /s /q ".venv-pytest"
uv venv .venv-pytest
if errorlevel 1 exit /b 1

REM Install the built wheel (which pulls in the runtime deps: numpy,
REM sounddevice, ...) plus the test-only requirements (pytest, nbmake).
uv pip install --python ".venv-pytest\Scripts\python.exe" "!WHEEL!" -r "tests\requirements.txt"
if errorlevel 1 exit /b 1

REM Activate the venv so its Scripts\ directory is on PATH. test_cli.py resolves
REM the moonshine-voice / moonshine console scripts via shutil.which(), which
REM needs them on PATH (on Windows they are moonshine-voice.exe, so the bare
REM Path(sys.executable).parent / name check in the test misses them).
call ".venv-pytest\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pytest -v "tests\test_modules.py" "tests\test_cli.py"
if errorlevel 1 exit /b 1

echo All Python tests passed

endlocal
