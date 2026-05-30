curl -L https://github.com/moonshine-ai/moonshine/releases/latest/download/moonshine-voice-windows-x86_64.tar.gz -o moonshine-voice-windows-x86_64.tar.gz
if errorlevel 1 exit /b 1
if exist moonshine-voice-windows-x86_64 rmdir /s /q moonshine-voice-windows-x86_64
tar -xzf moonshine-voice-windows-x86_64.tar.gz
if errorlevel 1 exit /b 1
del /q moonshine-voice-windows-x86_64.tar.gz
