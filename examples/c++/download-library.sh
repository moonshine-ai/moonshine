#!/bin/bash

ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    ARCH=arm64
elif [[ "$ARCH" == "x86_64" ]]; then
    ARCH=x86_64
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM=macos-${ARCH}
else
    PLATFORM=linux-${ARCH}
fi

# Download the library for the current platform
rm -rf moonshine-voice-${PLATFORM}
curl -O -L https://github.com/moonshine-ai/moonshine/releases/download/v0.0.52/moonshine-voice-${PLATFORM}.tar.gz
tar xzf moonshine-voice-${PLATFORM}.tar.gz
rm moonshine-voice-${PLATFORM}.tar.gz

echo "Library downloaded and extracted to moonshine-voice-${PLATFORM}"