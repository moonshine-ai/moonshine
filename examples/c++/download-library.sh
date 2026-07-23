#!/bin/bash

ARCH=$(uname -m)
case "$ARCH" in
    arm64 | aarch64)
        ARCH=arm64
        ;;
    x86_64 | amd64)
        ARCH=x86_64
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM=macos-${ARCH}
else
    PLATFORM=linux-${ARCH}
fi

# Download the library for the current platform and extract it into a folder
# named simply "moonshine-voice" (stripping the platform-specific top-level
# directory from the archive), so the compile command is the same regardless of
# operating system or architecture.
rm -rf moonshine-voice
curl -O -L https://github.com/moonshine-ai/moonshine/releases/download/v0.0.71/moonshine-voice-${PLATFORM}.tar.gz
mkdir -p moonshine-voice
tar xzf moonshine-voice-${PLATFORM}.tar.gz -C moonshine-voice --strip-components=1
rm moonshine-voice-${PLATFORM}.tar.gz

echo "Library downloaded and extracted to moonshine-voice"

# Also fetch the Medium Streaming English model and a sample recording so the
# transcriber example can run straight out of the box (see transcriber.cpp's
# default paths, which use the MEDIUM_STREAMING architecture).
MODEL_DIR=medium-streaming-en
MODEL_BASE_URL=https://download.moonshine.ai/model/medium-streaming-en/quantized
mkdir -p ${MODEL_DIR}
for MODEL_FILE in \
    adapter.ort \
    cross_kv.ort \
    decoder_kv.ort \
    decoder_kv_with_attention.ort \
    encoder.ort \
    frontend.ort \
    streaming_config.json \
    tokenizer.bin; do
    curl -o ${MODEL_DIR}/${MODEL_FILE} -L ${MODEL_BASE_URL}/${MODEL_FILE}
done
curl -o two_cities.wav -L https://github.com/moonshine-ai/moonshine/raw/main/test-assets/two_cities.wav

echo "Model downloaded to ${MODEL_DIR} and sample audio saved to two_cities.wav"