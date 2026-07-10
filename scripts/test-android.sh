#!/usr/bin/env bash
# Run the moonshine-voice Android library's instrumentation tests on a real
# device or emulator (./gradlew connectedAndroidTest). Unlike scripts/build-
# android.sh (which only builds the AAR) and scripts/test-examples.sh (which only
# compiles the example apps), this actually installs and runs the library on an
# Android runtime, so it is the real validation of the library on a device: a
# successful link does not prove the bundled ONNX Runtime's symbols resolve on an
# older OS, but running the tests on a matching-API emulator does. The library's
# minSdk comes from build.gradle.kts; boot an emulator at (or above) that API to
# exercise the supported floor.
#
# Usage:
#   ./scripts/test-android.sh [--avd NAME] [--serial SERIAL]
#
# Options:
#   --avd NAME      Boot this AVD headless, wait for it, run the tests, then shut
#                   it down on exit. Without --avd, a device/emulator must already
#                   be connected (see --serial).
#   --serial SERIAL Target a specific adb device/emulator serial.
#
# Environment:
#   ANDROID_HOME or ANDROID_SDK_ROOT — required.
#
# ABI note: the library is built for arm64-v8a only (see build.gradle.kts
# abiFilters), so the device/emulator must be arm64-v8a. On Apple Silicon use an
# "arm64-v8a" system image; an x86_64 emulator will not have the native libs.

set -euo pipefail

AVD=""
SERIAL="${ANDROID_SERIAL:-}"
EMU_PID=""
STARTED_EMULATOR=0

log() { echo "[test-android] $*"; }
die() {
	echo "[test-android] ERROR: $*" >&2
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--avd)
		AVD="$2"
		shift 2
		;;
	--serial)
		SERIAL="$2"
		shift 2
		;;
	-h | --help)
		sed -n '2,26p' "$0"
		exit 0
		;;
	*)
		die "unknown option: $1 (try --help)"
		;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
if [[ -z "${SDK}" && -d "${HOME}/Library/Android/sdk" ]]; then
	SDK="${HOME}/Library/Android/sdk"
fi
[[ -n "${SDK}" ]] || die "ANDROID_HOME (or ANDROID_SDK_ROOT) is not set."
export ANDROID_HOME="${SDK}"
ADB="${SDK}/platform-tools/adb"
EMULATOR="${SDK}/emulator/emulator"
command -v "${ADB}" >/dev/null 2>&1 || [[ -x "${ADB}" ]] || die "adb not found at ${ADB}"

cleanup() {
	local exit_code=$?
	set +e
	if [[ "${STARTED_EMULATOR}" -eq 1 ]]; then
		if [[ -n "${SERIAL}" ]]; then
			log "shutting down emulator ${SERIAL}"
			"${ADB}" -s "${SERIAL}" emu kill >/dev/null 2>&1
		fi
		if [[ -n "${EMU_PID}" ]] && kill -0 "${EMU_PID}" >/dev/null 2>&1; then
			kill "${EMU_PID}" >/dev/null 2>&1
		fi
	fi
	exit ${exit_code}
}
trap cleanup EXIT

# Boot the requested AVD headless and wait for it to finish booting.
if [[ -n "${AVD}" ]]; then
	[[ -x "${EMULATOR}" ]] || die "emulator not found at ${EMULATOR}"
	if ! "${EMULATOR}" -list-avds | grep -qx "${AVD}"; then
		die "AVD not found: ${AVD} (see: ${EMULATOR} -list-avds)"
	fi
	log "starting emulator for AVD ${AVD} (headless)"
	"${ADB}" start-server >/dev/null 2>&1 || true
	# Space-delimited list of emulator serials already present, so we can spot the
	# one we start. Pure-bash membership below avoids grep, whose non-zero "no
	# match" exit would trip set -e/pipefail while polling.
	before_serials=" $("${ADB}" devices | awk '/emulator-/{print $1}' | tr '\n' ' ')"
	"${EMULATOR}" -avd "${AVD}" -no-window -no-boot-anim -no-audio -no-snapshot \
		-gpu swiftshader_indirect >/dev/null 2>&1 &
	EMU_PID=$!
	STARTED_EMULATOR=1

	log "waiting for emulator to appear"
	for _ in $(seq 1 60); do
		for cand in $("${ADB}" devices | awk '/emulator-/{print $1}'); do
			case "${before_serials}" in
			*" ${cand} "*) : ;;
			*) SERIAL="${cand}" ;;
			esac
		done
		[[ -n "${SERIAL}" ]] && break
		sleep 2
	done
	[[ -n "${SERIAL}" ]] || die "emulator did not appear via adb"
	log "emulator serial: ${SERIAL}"

	"${ADB}" -s "${SERIAL}" wait-for-device
	log "waiting for sys.boot_completed"
	for _ in $(seq 1 180); do
		booted="$("${ADB}" -s "${SERIAL}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)"
		[[ "${booted}" == "1" ]] && break
		sleep 2
	done
	[[ "${booted:-}" == "1" ]] || die "emulator failed to finish booting"
	log "emulator booted"
fi

# Ensure we have a target device (portable enumeration; macOS ships bash 3.2
# without mapfile).
if [[ -z "${SERIAL}" ]]; then
	devices=""
	device_count=0
	while IFS= read -r dev; do
		[[ -z "${dev}" ]] && continue
		devices="${devices}${dev} "
		device_count=$((device_count + 1))
	done < <("${ADB}" devices | awk 'NR>1 && $2=="device"{print $1}')
	if [[ "${device_count}" -eq 0 ]]; then
		die "no connected device/emulator (use --avd NAME or --serial SERIAL, or connect one)"
	elif [[ "${device_count}" -gt 1 ]]; then
		die "multiple devices connected; pass --serial (found: ${devices})"
	fi
	SERIAL="${devices%% }"
	SERIAL="${SERIAL// /}"
fi
export ANDROID_SERIAL="${SERIAL}"
log "running instrumentation tests on ${SERIAL}"

cd "${REPO_ROOT}"
./gradlew -Pandroid.useAndroidX=true connectedAndroidTest --no-daemon --stacktrace

log "connectedAndroidTest passed on ${SERIAL}"
