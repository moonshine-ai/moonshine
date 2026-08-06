#!/usr/bin/env bash
# Measure English Tiny Streaming end-of-phrase latency on this Mac and on
# connected mobile devices (Android via adb, iOS via an on-device XCTest host).
#
# Uses the same metric as core/benchmark / the README table: average
# lastTranscriptionLatencyMs over completed lines while feeding two_cities.wav
# in small chunks as fast as the device can process.
#
# Usage:
#   ./scripts/test-mobile-latency.sh
#   ./scripts/test-mobile-latency.sh --android-serial SERIAL --ios-udid UDID
#
# Options:
#   --android-serial SERIAL   adb serial (default: sole connected device, or
#                             ANDROID_SERIAL / Pixel serial if present)
#   --ios-udid UDID           Physical iOS device UDID (default: sole connected
#                             CoreDevice iPad/iPhone, or IOS_UDID)
#   --macos-only              Only run the MacBook Pro (host) stage
#   --android-only            Only run the Android stage
#   --ios-only                Only run the iOS stage
#   --skip-macos              Skip the MacBook Pro stage
#   --skip-build-swift        Do not run build-swift.sh before iOS/macOS tests
#                             (assumes swift/Moonshine.xcframework is current)
#   --update-readme           Rewrite MacBook Pro / Pixel / iPad Streaming
#                             latency cells in README.md from measured results
#                             (only when a cell changes by more than 5%)
#
# Environment:
#   ANDROID_HOME / ANDROID_SDK_ROOT
#   DEVELOPMENT_TEAM          Apple team id for on-device signing (default:
#                             S3AJ7B7ZCG, Pete Warden personal team / Apple
#                             Development cert). Override for a company team.
#   MOBILE_LATENCY_OPTIONAL=1 Exit 0 with a skip message when a required mobile
#                             device is missing (useful for unattended hosts
#                             without hardware). build-all-platforms.sh does
#                             not set this. The macOS stage still runs.
#
# Prerequisites:
#   scripts/fetch-voice-assets.sh   # tiny-streaming-en + two_cities.wav
#   Connected Pixel (or other arm64 Android) with USB debugging authorized
#   Connected iPad/iPhone trusted for development, Developer Mode on

set -euo pipefail

ANDROID_SERIAL="${ANDROID_SERIAL:-}"
IOS_UDID="${IOS_UDID:-}"
ANDROID_ONLY=0
IOS_ONLY=0
MACOS_ONLY=0
SKIP_MACOS=0
SKIP_BUILD_SWIFT=0
UPDATE_README=0
DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM:-S3AJ7B7ZCG}"

log() { echo "[mobile-latency] $*"; }
die() {
	echo "[mobile-latency] ERROR: $*" >&2
	exit 1
}
skip_or_die() {
	if [[ -n "${MOBILE_LATENCY_OPTIONAL:-}" ]]; then
		log "SKIP: $* (MOBILE_LATENCY_OPTIONAL=1)"
		exit 0
	fi
	die "$*"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--android-serial)
		ANDROID_SERIAL="$2"
		shift 2
		;;
	--ios-udid)
		IOS_UDID="$2"
		shift 2
		;;
	--macos-only)
		MACOS_ONLY=1
		shift
		;;
	--android-only)
		ANDROID_ONLY=1
		shift
		;;
	--ios-only)
		IOS_ONLY=1
		shift
		;;
	--skip-macos)
		SKIP_MACOS=1
		shift
		;;
	--skip-build-swift)
		SKIP_BUILD_SWIFT=1
		shift
		;;
	--update-readme)
		UPDATE_README=1
		shift
		;;
	-h | --help)
		sed -n '2,45p' "$0"
		exit 0
		;;
	*)
		die "unknown option: $1 (try --help)"
		;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/.mobile-latency"
mkdir -p "${RESULTS_DIR}"

ensure_assets() {
	# Models download from the CDN at test time. two_cities.wav is optional
	# locally (tests will fetch it if missing).
	:
}

resolve_android_sdk() {
	local sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
	if [[ -z "${sdk}" ]]; then
		for cand in "${HOME}/Library/Android/sdk" "${HOME}/Android/Sdk"; do
			if [[ -d "${cand}" ]]; then
				sdk="${cand}"
				break
			fi
		done
	fi
	[[ -n "${sdk}" ]] || die "ANDROID_HOME (or ANDROID_SDK_ROOT) is not set"
	export ANDROID_HOME="${sdk}"
	ADB="${sdk}/platform-tools/adb"
	command -v "${ADB}" >/dev/null 2>&1 || [[ -x "${ADB}" ]] || die "adb not found at ${ADB}"
}

pick_android_serial() {
	if [[ -n "${ANDROID_SERIAL}" ]]; then
		return 0
	fi
	local devices=()
	local line serial state
	while IFS= read -r line; do
		serial="$(awk '{print $1}' <<<"${line}")"
		state="$(awk '{print $2}' <<<"${line}")"
		[[ "${state}" == "device" ]] || continue
		# Prefer physical Pixel when several devices are attached.
		if [[ "${line}" == *"model:Pixel"* ]] || [[ "${line}" == *"device:stallion"* ]]; then
			ANDROID_SERIAL="${serial}"
			return 0
		fi
		devices+=("${serial}")
	done < <("${ADB}" devices -l | awk 'NR>1 && $2=="device"{print}')
	if [[ -n "${ANDROID_SERIAL}" ]]; then
		return 0
	fi
	if [[ ${#devices[@]} -eq 0 ]]; then
		skip_or_die "no Android device connected (adb)"
	elif [[ ${#devices[@]} -gt 1 ]]; then
		die "multiple Android devices; pass --android-serial (found: ${devices[*]})"
	fi
	ANDROID_SERIAL="${devices[0]}"
}

pick_ios_udid() {
	if [[ -n "${IOS_UDID}" ]]; then
		return 0
	fi
	local json="/tmp/moonshine-ios-devices-$$.json"
	xcrun devicectl list devices --json-output "${json}" >/dev/null
	# Prefer a connected physical iPad; else any connected iPhone/iPad.
	IOS_UDID="$(
		python3 - "${json}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
devices = data.get("result", {}).get("devices", [])
connected = []
for d in devices:
    props = d.get("deviceProperties", {})
    hw = d.get("hardwareProperties", {})
    conn = d.get("connectionProperties", {})
    state = (d.get("state") or "").lower()
    tunnel = (conn.get("tunnelState") or "").lower()
    udid = hw.get("udid") or ""
    dtype = (hw.get("deviceType") or "").lower()
    reality = (hw.get("reality") or "").lower()
    if reality and reality != "physical":
        continue
    if dtype not in ("ipad", "iphone"):
        continue
    online = state in ("connected", "available") or tunnel in ("connected", "available")
    if props.get("ddiServicesAvailable") is True:
        online = True
    if not online or not udid:
        continue
    name = props.get("name") or ""
    connected.append((dtype == "ipad", name, udid))
connected.sort(key=lambda t: (not t[0], t[1]))
if connected:
    print(connected[0][2])
PY
	)"
	rm -f "${json}"
	if [[ -z "${IOS_UDID}" ]]; then
		skip_or_die "no connected physical iOS device (devicectl)"
	fi
}

parse_latency_results() {
	# $1=prefix $2=env-out-path $3=log-path
	local prefix="$1"
	local out_env="$2"
	local log_path="$3"
	python3 - "${prefix}" "${out_env}" "${log_path}" <<'PY'
import sys
prefix, out_path, log_path = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log_path).read()
device = ""
avgs = {}
for line in text.splitlines():
    if "MOONSHINE_LATENCY" not in line:
        continue
    parts = {k: v for k, v in (p.split("=", 1) for p in line.split() if "=" in p)}
    model = parts.get("model", "")
    avg = parts.get("avg_ms", "")
    if parts.get("device"):
        device = parts["device"]
    key = None
    if model.startswith("tiny-streaming"):
        key = "tiny"
    elif model.startswith("small-streaming"):
        key = "small"
    elif model.startswith("medium-streaming"):
        key = "medium"
    if key and avg:
        avgs[key] = avg
with open(out_path, "w") as f:
    f.write(f"device={device}\n")
    for k in ("tiny", "small", "medium"):
        f.write(f"{prefix}_{k}_ms={avgs.get(k, '')}\n")
missing = [k for k in ("tiny", "small", "medium") if k not in avgs]
if missing:
    sys.exit(f"missing MOONSHINE_LATENCY for models: {', '.join(missing)}")
print(
    f"{prefix}: device={device} "
    f"tiny={avgs['tiny']}ms small={avgs['small']}ms medium={avgs['medium']}ms"
)
PY
}

load_env_file() {
	# shellcheck disable=SC1090
	source "$1"
}

run_android() {
	resolve_android_sdk
	pick_android_serial
	export ANDROID_SERIAL
	log "Android target: ${ANDROID_SERIAL}"
	local out="${RESULTS_DIR}/android.log"
	(
		cd "${REPO_ROOT}"
		./gradlew -Pandroid.useAndroidX=true \
			connectedAndroidTest \
			-Pandroid.testInstrumentationRunnerArguments.class=ai.moonshine.voice.StreamingLatencyTest \
			--no-daemon --stacktrace
	) 2>&1 | tee "${out}"

	local logcat_hits
	logcat_hits="$(find "${REPO_ROOT}/build/outputs/androidTest-results" \
		-name 'logcat-*StreamingLatency*' -type f 2>/dev/null | sort | tail -n1 || true)"
	if [[ -n "${logcat_hits}" ]]; then
		cat "${logcat_hits}" >>"${out}"
	fi
	"${ADB}" -s "${ANDROID_SERIAL}" logcat -d -s MoonshineLatency:I 2>/dev/null >>"${out}" || true

	local envf="${RESULTS_DIR}/android.env"
	parse_latency_results android "${envf}" "${out}"
	load_env_file "${envf}"
	ANDROID_DEVICE="${device}"
	ANDROID_TINY_MS="${android_tiny_ms}"
	ANDROID_SMALL_MS="${android_small_ms}"
	ANDROID_MEDIUM_MS="${android_medium_ms}"
	log "Android Tiny/Small/Medium: ${ANDROID_TINY_MS}/${ANDROID_SMALL_MS}/${ANDROID_MEDIUM_MS}ms (${ANDROID_DEVICE})"
}

ensure_swift_xcframework() {
	if [[ "${SKIP_BUILD_SWIFT}" -eq 1 ]]; then
		[[ -d "${REPO_ROOT}/swift/Moonshine.xcframework" ]] \
			|| die "swift/Moonshine.xcframework missing; omit --skip-build-swift"
	else
		log "building Swift xcframework (needed for on-device / macOS tests)"
		"${SCRIPT_DIR}/build-swift.sh"
	fi
	# Optional local two_cities.wav for faster macOS runs; models download from CDN.
	if [[ -f "${REPO_ROOT}/test-assets/two_cities.wav" ]]; then
		mkdir -p "${REPO_ROOT}/swift/Tests/MoonshineVoiceTests/test-assets"
		cp -f "${REPO_ROOT}/test-assets/two_cities.wav" \
			"${REPO_ROOT}/swift/Tests/MoonshineVoiceTests/test-assets/two_cities.wav"
	fi
}

run_macos() {
	log "MacBook Pro (host) target"
	ensure_swift_xcframework

	local out="${RESULTS_DIR}/macos.log"
	(
		cd "${REPO_ROOT}/swift"
		swift test --filter MoonshineVoiceTests.StreamingLatencyTests/testStreamingLatencyTwoCities
	) 2>&1 | tee "${out}"

	local envf="${RESULTS_DIR}/macos.env"
	parse_latency_results macos "${envf}" "${out}"
	load_env_file "${envf}"
	MACOS_DEVICE="${device}"
	MACOS_TINY_MS="${macos_tiny_ms}"
	MACOS_SMALL_MS="${macos_small_ms}"
	MACOS_MEDIUM_MS="${macos_medium_ms}"
	log "macOS Tiny/Small/Medium: ${MACOS_TINY_MS}/${MACOS_SMALL_MS}/${MACOS_MEDIUM_MS}ms (${MACOS_DEVICE})"
}

run_ios() {
	pick_ios_udid
	log "iOS target UDID: ${IOS_UDID}"
	ensure_swift_xcframework

	local proj_dir="${REPO_ROOT}/examples/ios/StreamingLatency"
	if command -v xcodegen >/dev/null 2>&1; then
		(cd "${proj_dir}" && xcodegen generate)
	fi
	[[ -d "${proj_dir}/StreamingLatency.xcodeproj" ]] \
		|| die "missing ${proj_dir}/StreamingLatency.xcodeproj (install xcodegen or commit the project)"

	local out="${RESULTS_DIR}/ios.log"
	local dest="platform=iOS,id=${IOS_UDID}"
	(
		cd "${proj_dir}"
		xcodebuild test \
			-project StreamingLatency.xcodeproj \
			-scheme StreamingLatency \
			-destination "${dest}" \
			-only-testing:StreamingLatencyTests/StreamingLatencyTests/testStreamingLatencyTwoCities \
			DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM}" \
			CODE_SIGN_STYLE=Automatic \
			-allowProvisioningUpdates \
			2>&1
	) | tee "${out}"

	local xcresult
	xcresult="$(ls -dt "${HOME}/Library/Developer/Xcode/DerivedData"/StreamingLatency-*/Logs/Test/*.xcresult 2>/dev/null | head -1 || true)"
	if [[ -n "${xcresult}" ]]; then
		xcrun xcresulttool get log --legacy --path "${xcresult}" 2>/dev/null \
			| rg "MOONSHINE_LATENCY" >>"${out}" || true
	fi

	local envf="${RESULTS_DIR}/ios.env"
	parse_latency_results ios "${envf}" "${out}"
	load_env_file "${envf}"
	IOS_DEVICE="${device}"
	IOS_TINY_MS="${ios_tiny_ms}"
	IOS_SMALL_MS="${ios_small_ms}"
	IOS_MEDIUM_MS="${ios_medium_ms}"
	log "iOS Tiny/Small/Medium: ${IOS_TINY_MS}/${IOS_SMALL_MS}/${IOS_MEDIUM_MS}ms (${IOS_DEVICE})"
}

update_readme() {
	[[ -n "${MACOS_TINY_MS}" && -n "${ANDROID_TINY_MS}" && -n "${IOS_TINY_MS}" ]] \
		|| die "--update-readme needs macOS, Android, and iOS results for all three models"
	python3 - "${REPO_ROOT}/README.md" \
		"${MACOS_TINY_MS}" "${MACOS_SMALL_MS}" "${MACOS_MEDIUM_MS}" \
		"${ANDROID_TINY_MS}" "${ANDROID_SMALL_MS}" "${ANDROID_MEDIUM_MS}" \
		"${IOS_TINY_MS}" "${IOS_SMALL_MS}" "${IOS_MEDIUM_MS}" <<'PY'
import re, sys
path = sys.argv[1]
macos = {"tiny": sys.argv[2], "small": sys.argv[3], "medium": sys.argv[4]}
android = {"tiny": sys.argv[5], "small": sys.argv[6], "medium": sys.argv[7]}
ios = {"tiny": sys.argv[8], "small": sys.argv[9], "medium": sys.argv[10]}
text = open(path).read()
THRESHOLD = 0.05  # only rewrite a cell if relative change exceeds 5%

header_pat = re.compile(
    r"(\| Model\s+\| WER\s+\| # Parameters\s+\| MacBook Pro\s+\| Linux x86\s+\| R\. Pi 5\s+\|)"
    r"(?:\s*Pixel 10a\s+\|\s*iPad \(A16\)\s+\|)?"
)
sep_pat = re.compile(
    r"(\| -+\s+\| -+\s+\| -+\s+\| -+\s+\| -+\s+\| -+\s+\|)"
    r"(?:\s*-+\s+\|\s*-+\s+\|)?"
)

def add_cols(match):
    base = match.group(1).rstrip()
    if not base.endswith("|"):
        base += " |"
    return base + " Pixel 10a | iPad (A16) |"

def add_sep(match):
    base = match.group(1).rstrip()
    if not base.endswith("|"):
        base += " |"
    return base + " --------- | ---------- |"

text2, n = header_pat.subn(add_cols, text, count=1)
if n != 1:
    sys.exit("could not find README latency table header")
text2, n = sep_pat.subn(add_sep, text2, count=1)
if n != 1:
    sys.exit("could not find README latency table separator")

def ms(v):
    return f"{int(round(float(v)))}ms"

def parse_ms(cell):
    if not cell:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*ms", cell.strip(), re.I)
    return float(m.group(1)) if m else None

def choose_ms(new_v, old_cell):
    """Keep the README value unless the new measurement differs by >5%."""
    new_f = float(new_v)
    old_f = parse_ms(old_cell)
    if old_f is None or old_f == 0:
        return ms(new_f)
    if abs(new_f - old_f) / old_f > THRESHOLD:
        return ms(new_f)
    return old_cell.strip()

changed = []
rows = []
for line in text2.splitlines():
    if not line.startswith("| Moonshine ") and not line.startswith("| Whisper "):
        rows.append(line)
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    while len(cells) < 8:
        cells.append("")
    name = cells[0]
    size = None
    if name.startswith("Moonshine Tiny Streaming"):
        size = "tiny"
    elif name.startswith("Moonshine Small Streaming"):
        size = "small"
    elif name.startswith("Moonshine Medium Streaming"):
        size = "medium"
    if size is not None:
        for idx, label, measured in (
            (3, f"Mac/{size}", macos[size]),
            (6, f"Pixel/{size}", android[size]),
            (7, f"iPad/{size}", ios[size]),
        ):
            old = cells[idx]
            new = choose_ms(measured, old)
            if new != old.strip():
                changed.append(f"{label}: {old or '(empty)'} -> {new}")
            cells[idx] = new
    else:
        if not cells[6]:
            cells[6] = "—"
        if not cells[7]:
            cells[7] = "—"
    rows.append("| " + " | ".join(cells[:8]) + " |")
open(path, "w").write("\n".join(rows) + ("\n" if text.endswith("\n") else ""))
if changed:
    print(f"Updated {path} ({len(changed)} cell(s) >{THRESHOLD:.0%}):")
    for c in changed:
        print(f"  {c}")
else:
    print(f"No README latency updates (all within {THRESHOLD:.0%} of existing values)")
PY
}

ensure_assets

# two_cities.wav is still useful as a local fallback; models download from CDN.
if [[ ! -f "${REPO_ROOT}/test-assets/two_cities.wav" ]]; then
	log "two_cities.wav missing locally; tests will download it"
fi

RUN_MACOS=1
RUN_ANDROID=1
RUN_IOS=1
if [[ "${MACOS_ONLY}" -eq 1 ]]; then
	RUN_ANDROID=0
	RUN_IOS=0
fi
if [[ "${ANDROID_ONLY}" -eq 1 ]]; then
	RUN_MACOS=0
	RUN_IOS=0
fi
if [[ "${IOS_ONLY}" -eq 1 ]]; then
	RUN_MACOS=0
	RUN_ANDROID=0
fi
if [[ "${SKIP_MACOS}" -eq 1 ]]; then
	RUN_MACOS=0
fi

MACOS_TINY_MS=""; MACOS_SMALL_MS=""; MACOS_MEDIUM_MS=""; MACOS_DEVICE=""
ANDROID_TINY_MS=""; ANDROID_SMALL_MS=""; ANDROID_MEDIUM_MS=""; ANDROID_DEVICE=""
IOS_TINY_MS=""; IOS_SMALL_MS=""; IOS_MEDIUM_MS=""; IOS_DEVICE=""

if [[ "${RUN_MACOS}" -eq 1 ]]; then
	run_macos
fi
if [[ "${RUN_ANDROID}" -eq 1 ]]; then
	run_android
fi
if [[ "${RUN_IOS}" -eq 1 ]]; then
	run_ios
fi

log "Results:"
[[ -n "${MACOS_TINY_MS}" ]] && log "  macOS (${MACOS_DEVICE}): tiny=${MACOS_TINY_MS}ms small=${MACOS_SMALL_MS}ms medium=${MACOS_MEDIUM_MS}ms"
[[ -n "${ANDROID_TINY_MS}" ]] && log "  Android (${ANDROID_DEVICE}): tiny=${ANDROID_TINY_MS}ms small=${ANDROID_SMALL_MS}ms medium=${ANDROID_MEDIUM_MS}ms"
[[ -n "${IOS_TINY_MS}" ]] && log "  iOS (${IOS_DEVICE}): tiny=${IOS_TINY_MS}ms small=${IOS_SMALL_MS}ms medium=${IOS_MEDIUM_MS}ms"

if [[ "${UPDATE_README}" -eq 1 ]]; then
	update_readme
fi

{
	echo "macos_tiny_ms=${MACOS_TINY_MS:-}"
	echo "macos_small_ms=${MACOS_SMALL_MS:-}"
	echo "macos_medium_ms=${MACOS_MEDIUM_MS:-}"
	echo "macos_device=${MACOS_DEVICE:-}"
	echo "android_tiny_ms=${ANDROID_TINY_MS:-}"
	echo "android_small_ms=${ANDROID_SMALL_MS:-}"
	echo "android_medium_ms=${ANDROID_MEDIUM_MS:-}"
	echo "android_device=${ANDROID_DEVICE:-}"
	echo "ios_tiny_ms=${IOS_TINY_MS:-}"
	echo "ios_small_ms=${IOS_SMALL_MS:-}"
	echo "ios_medium_ms=${IOS_MEDIUM_MS:-}"
	echo "ios_device=${IOS_DEVICE:-}"
} >"${RESULTS_DIR}/summary.env"
log "wrote ${RESULTS_DIR}/summary.env"
