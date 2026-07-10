#!/usr/bin/env bash
# One-time setup of the Android toolchain needed to run the moonshine-voice
# instrumentation tests (scripts/test-android.sh) on an x86_64 Linux host with
# KVM. Everything installs under $HOME (no sudo), EXCEPT KVM group membership,
# which must be granted separately once:
#
#   sudo usermod -aG kvm "$USER"   # then log out/in (or reboot)
#
# Installs: Temurin JDK 17, Android command-line tools, platform-tools, emulator,
# the compileSdk platform + build-tools, the NDK + CMake pinned by
# build.gradle.kts, an x86_64 system image, and an AVD for it.
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
JDK_DIR="$HOME/android-ci/jdk"
CLT_URL="https://dl.google.com/android/repository/commandlinetools-linux-14742923_latest.zip"
JDK_URL="https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"

# Keep these in sync with build.gradle.kts (ndkVersion / compileSdk / cmake).
NDK_VER="28.2.13676358"
CMAKE_VER="3.22.1"
PLATFORM_PKG="platforms;android-35"
BUILDTOOLS_PKG="build-tools;35.0.0"
SYSIMG_PKG="system-images;android-26;google_apis;x86_64"
AVD_NAME="${ANDROID_X86_64_AVD:-moonshine_api26_x86_64}"

log() { echo "[setup-android-ci] $*"; }

mkdir -p "$HOME/android-ci"

# --- JDK 17 (Temurin) ---
if [ ! -x "$JDK_DIR/bin/java" ]; then
	log "downloading Temurin JDK 17..."
	mkdir -p "$JDK_DIR"
	curl -fsSL "$JDK_URL" -o /tmp/jdk17.tar.gz
	tar -xzf /tmp/jdk17.tar.gz -C "$JDK_DIR" --strip-components=1
	rm -f /tmp/jdk17.tar.gz
fi
export JAVA_HOME="$JDK_DIR"
export PATH="$JAVA_HOME/bin:$PATH"
java -version

# --- Android command-line tools ---
CLT_DIR="$SDK_ROOT/cmdline-tools/latest"
if [ ! -x "$CLT_DIR/bin/sdkmanager" ]; then
	log "downloading Android command-line tools..."
	mkdir -p "$SDK_ROOT/cmdline-tools"
	curl -fsSL "$CLT_URL" -o /tmp/clt.zip
	rm -rf /tmp/cmdline-tools "$CLT_DIR"
	unzip -q /tmp/clt.zip -d /tmp
	mkdir -p "$CLT_DIR"
	mv /tmp/cmdline-tools/* "$CLT_DIR/"
	rm -rf /tmp/cmdline-tools /tmp/clt.zip
fi

export ANDROID_HOME="$SDK_ROOT"
export ANDROID_SDK_ROOT="$SDK_ROOT"
SDKMANAGER="$CLT_DIR/bin/sdkmanager"
AVDMANAGER="$CLT_DIR/bin/avdmanager"

log "accepting SDK licenses..."
yes | "$SDKMANAGER" --licenses >/dev/null 2>&1 || true

log "installing SDK packages (this downloads a few GB: NDK + system image)..."
"$SDKMANAGER" --install \
	"platform-tools" "emulator" \
	"$PLATFORM_PKG" "$BUILDTOOLS_PKG" \
	"ndk;$NDK_VER" "cmake;$CMAKE_VER" \
	"$SYSIMG_PKG"

log "creating AVD $AVD_NAME..."
echo "no" | "$AVDMANAGER" create avd -n "$AVD_NAME" -k "$SYSIMG_PKG" -d pixel --force

log "DONE. SDK at $SDK_ROOT, JDK at $JDK_DIR, AVD '$AVD_NAME'."
