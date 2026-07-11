#!/usr/bin/env bash
# Builds moonshine-tts-streaming-cli.
#
# Mirrors the manual-download-then-g++ pattern used by
# examples/c++/download-library.sh + examples/c++/README.md - there is no
# CMake project for the C++ examples in this repo, so this is a plain
# shell script rather than a build system integration.
#
# Usage:
#   ./build.sh [release-version]
#
# If release-version is omitted, downloads the latest release.

set -euo pipefail

cd "$(dirname "$0")"

ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ]; then
  echo "This script only knows about the linux-x86_64 prebuilt library." >&2
  echo "If you're on a different Linux architecture, download the matching" >&2
  echo "moonshine-voice-linux-* archive from the releases page and adjust" >&2
  echo "LIB_DIR below." >&2
  exit 1
fi

LIB_ARCHIVE="moonshine-voice-linux-x86_64.tar.gz"
LIB_DIR="moonshine-voice-linux-x86_64"
VERSION="${1:-latest}"

if [ ! -d "$LIB_DIR" ]; then
  echo "Downloading prebuilt Moonshine library ($VERSION)..."
  if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/moonshine-ai/moonshine/releases/latest/download/${LIB_ARCHIVE}"
  else
    URL="https://github.com/moonshine-ai/moonshine/releases/download/${VERSION}/${LIB_ARCHIVE}"
  fi
  curl -L -f -O "$URL"
  tar xzf "$LIB_ARCHIVE"
else
  echo "Found existing $LIB_DIR, skipping download."
fi

if ! pkg-config --exists libpipewire-0.3; then
  echo "libpipewire-0.3 dev headers not found." >&2
  echo "Install them first, e.g. on Debian/Ubuntu:" >&2
  echo "  sudo apt-get install libpipewire-0.3-dev pkg-config" >&2
  exit 1
fi

echo "Building moonshine-tts-streaming-cli..."
g++ -std=c++17 -O2 moonshine-tts-streaming-cli.cpp \
  -I"${LIB_DIR}/include" \
  -L"${LIB_DIR}/lib" \
  $(pkg-config --cflags libpipewire-0.3) \
  -lmoonshine \
  $(pkg-config --libs libpipewire-0.3) \
  -lpthread \
  -o moonshine-tts-streaming-cli

echo "Built ./moonshine-tts-streaming-cli"
echo ""
echo "Before running, make the shared library discoverable:"
echo "  export LD_LIBRARY_PATH=\$(pwd)/${LIB_DIR}/lib"
echo ""
echo "Then run, e.g.:"
echo "  ./moonshine-tts-streaming-cli -r ../../../core/moonshine-tts/data -l en_us"