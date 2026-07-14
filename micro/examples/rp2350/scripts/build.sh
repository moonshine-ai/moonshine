#!/usr/bin/env bash
# Configure + build the moonshine-micro RP2350 example firmware.
#
# Why this exists: PICO_BOARD is a build-WIDE, configure-time setting, so one
# build tree produces firmware for exactly one board. Conveniently, the RP2350
# in a Pico 2 and a Pico 2 W is identical -- the "W" just adds a CYW43 wireless
# module -- so a `pico2` build runs on BOTH boards. Only the wifi app actually
# touches the radio and therefore needs a `pico2_w` build (and it can't run on
# a plain Pico 2 anyway). This script encodes that split:
#
#   build/    pico2     every non-wifi app; runs on any RP2350 board   (default)
#   build-w/  pico2_w   the wifi app (built only when you pass --wifi)
#
# Usage:
#   examples/rp2350/scripts/build.sh [--wifi] [--clean] [-- <extra cmake args>]
#
#   --wifi    Also configure build-w/ for pico2_w and build the wifi target.
#   --clean   Remove the relevant build dir(s) first (forces a fresh configure;
#             needed if you ever change PICO_BOARD or the toolchain in place).
#   --        Everything after this is passed verbatim to the `cmake -B ...`
#             configure step (e.g. -DSPELLING_TINY_VAD=ON).
#
# Toolchain / SDK:
#   Needs Arm GNU toolchain 13.3.Rel1 and the Pico SDK. This script honors the
#   PICO_SDK_PATH and PICO_TOOLCHAIN_PATH environment variables if set, passing
#   them through to CMake; otherwise it relies on whatever CMake/the SDK can
#   find (or a value already cached in an existing build dir). Typical setup:
#
#     export PICO_SDK_PATH="$HOME/projects/pico-sdk"
#     export PICO_TOOLCHAIN_PATH="$HOME/projects/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin"
#
# Examples:
#   examples/rp2350/scripts/build.sh                 # all non-wifi apps (pico2)
#   examples/rp2350/scripts/build.sh --wifi          # the above + wifi (pico2_w)
#   examples/rp2350/scripts/build.sh --clean --wifi  # fresh build of both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

JOBS="${JOBS:-8}"
WANT_WIFI=0
WANT_CLEAN=0
EXTRA_CMAKE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wifi)  WANT_WIFI=1; shift ;;
    --clean) WANT_CLEAN=1; shift ;;
    --)      shift; EXTRA_CMAKE_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "ERROR: unknown argument '$1'." >&2
      echo "       Usage: build.sh [--wifi] [--clean] [-- <extra cmake args>]" >&2
      exit 64 ;;
  esac
done

# Pass PICO_SDK_PATH / PICO_TOOLCHAIN_PATH through to CMake only when set, so
# this works both for people who export them and for those who rely on the
# SDK's own discovery (or a previously-cached value).
COMMON_CMAKE_ARGS=()
[[ -n "${PICO_SDK_PATH:-}" ]]       && COMMON_CMAKE_ARGS+=("-DPICO_SDK_PATH=$PICO_SDK_PATH")
[[ -n "${PICO_TOOLCHAIN_PATH:-}" ]] && COMMON_CMAKE_ARGS+=("-DPICO_TOOLCHAIN_PATH=$PICO_TOOLCHAIN_PATH")

# configure_and_build <build-dir> <pico-board> [build-target...]
# Any trailing <build-target> args restrict the build step to those targets;
# none builds everything in the tree.
configure_and_build() {
  local build_dir="$1"; shift
  local board="$1"; shift
  local targets=("$@")  # remaining args = targets (empty => build all)

  if (( WANT_CLEAN )) && [[ -d "$build_dir" ]]; then
    echo "==> Removing $build_dir (--clean)"
    rm -rf "$build_dir"
  fi

  # Note the ${arr[@]+...} guards: macOS's stock bash 3.2 treats "${arr[@]}"
  # on an empty array as an unbound-variable error under `set -u`.
  echo "==> Configuring $build_dir for PICO_BOARD=$board"
  cmake -B "$build_dir" -S "$REPO_DIR" \
    -DPICO_BOARD="$board" \
    ${COMMON_CMAKE_ARGS[@]+"${COMMON_CMAKE_ARGS[@]}"} \
    ${EXTRA_CMAKE_ARGS[@]+"${EXTRA_CMAKE_ARGS[@]}"}

  echo "==> Building $build_dir (-j $JOBS)${targets[*]:+ --target ${targets[*]}}"
  if (( ${#targets[@]} > 0 )); then
    cmake --build "$build_dir" -j "$JOBS" --target "${targets[@]}"
  else
    cmake --build "$build_dir" -j "$JOBS"
  fi
}

# Default board: every non-wifi app, runs on any RP2350 board.
configure_and_build "$REPO_DIR/build" pico2

if (( WANT_WIFI )); then
  # pico2_w build: only the wifi apps need the radio. We build just those
  # targets to keep it quick; the rest already exist in build/.
  configure_and_build "$REPO_DIR/build-w" pico2_w \
    moonshine_micro_echo_wifi moonshine_micro_echo_wifi_hardware
fi

echo ""
echo "Done."
echo "  Non-wifi apps:  $REPO_DIR/build/examples/rp2350/*.uf2"
if (( WANT_WIFI )); then
  echo "  Wifi apps:      $REPO_DIR/build-w/examples/rp2350/moonshine_micro_echo_wifi.uf2"
  echo "                  $REPO_DIR/build-w/examples/rp2350/moonshine_micro_echo_wifi_hardware.uf2"
fi
echo ""
echo "Flash, e.g.:"
echo "  examples/rp2350/scripts/flash.sh echo"
if (( WANT_WIFI )); then
  echo "  examples/rp2350/scripts/flash.sh wifi            # USB bridge; auto-uses build-w/"
  echo "  examples/rp2350/scripts/flash.sh wifi_hardware   # I2S mic + I2S amp"
fi
