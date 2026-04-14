#!/usr/bin/env bash
# Verify iOS and Android examples build standalone: either from GitHub Release
# archives (default) or from a temporary copy of examples/android and examples/ios
# (--local-examples).
#
# Usage:
#   ./scripts/test-examples.sh [--repo OWNER/REPO] [--tag vX.Y.Z] [--workdir DIR] [--keep-workdir]
#   ./scripts/test-examples.sh --local-examples [--workdir DIR] [--keep-workdir]
#
# With --local-examples, skips GitHub downloads and copies this repository's
# examples/android and examples/ios into a temp tree (same layout as extracted
# release archives), then runs the same Gradle / xcodebuild checks.
#
# Environment:
#   ANDROID_HOME or ANDROID_SDK_ROOT — required for Android (unless SKIP_ANDROID=1)
#   GITHUB_TOKEN — optional; avoids anonymous rate limits on api.github.com if needed
#   SKIP_ANDROID=1 / SKIP_IOS=1 — skip that platform
#   TEST_EXAMPLES_TAG — same as --tag when --tag is omitted (e.g. v0.0.56)
#   TEST_EXAMPLES_USE_LOCAL=1 — same as --local-examples
#
# Defaults:
#   --repo moonshine-ai/moonshine
#   Archives: ios-examples.tar.gz, android-examples.tar.gz (see README "Getting Started")

set -euo pipefail

REPO="${GITHUB_REPOSITORY:-moonshine-ai/moonshine}"
TAG=""
WORKDIR=""
KEEP_WORKDIR=0
SKIP_ANDROID="${SKIP_ANDROID:-0}"
SKIP_IOS="${SKIP_IOS:-0}"
USE_LOCAL_EXAMPLES=0

usage() {
	cat <<'EOF'
Usage:
  test-examples.sh [--repo OWNER/REPO] [--tag vX.Y.Z] [--workdir DIR] [--keep-workdir]
  test-examples.sh --local-examples [--workdir DIR] [--keep-workdir]

Default mode: downloads ios-examples.tar.gz and android-examples.tar.gz from
GitHub Releases, extracts them, and runs standalone builds:
  Android: every directory containing ./gradlew → ./gradlew assembleDebug
  macOS:   every *.xcodeproj → xcodebuild (iOS Simulator, no code signing)

--local-examples: copy <repo>/examples/android and <repo>/examples/ios into the
work directory (temporary copy; does not modify the originals), then run the
same build steps. Implies repository root is the parent of scripts/.

Options:
  --repo OWNER/REPO   GitHub repository (default: moonshine-ai/moonshine); only for downloads
  --tag vX.Y.Z        Use .../releases/download/TAG/... (default: latest/download); only for downloads
  --local-examples    Use examples/android and examples/ios from this checkout instead of archives
  --workdir DIR       Extract / copy and build here instead of a fresh mktemp directory
  --keep-workdir      Do not delete the work directory on exit (implies useful with --workdir)

Environment:
  ANDROID_HOME        SDK path for Android Gradle
  SKIP_ANDROID=1      Skip Android builds
  SKIP_IOS=1          Skip iOS builds (also implied on non-Darwin)
  TEST_EXAMPLES_USE_LOCAL=1   Same as --local-examples
EOF
}

log() {
	echo "[test-examples] $*"
}

die() {
	echo "[test-examples] ERROR: $*" >&2
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--repo)
		REPO="$2"
		shift 2
		;;
	--tag)
		TAG="$2"
		shift 2
		;;
	--workdir)
		WORKDIR="$2"
		shift 2
		;;
	--keep-workdir)
		KEEP_WORKDIR=1
		shift
		;;
	--local-examples)
		USE_LOCAL_EXAMPLES=1
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		die "unknown option: $1 (try --help)"
		;;
	esac
done

TAG="${TAG:-${TEST_EXAMPLES_TAG:-}}"
if [[ "${TEST_EXAMPLES_USE_LOCAL:-}" == "1" ]]; then
	USE_LOCAL_EXAMPLES=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${USE_LOCAL_EXAMPLES}" -eq 1 && -n "${TAG}" ]]; then
	log "note: --tag / TEST_EXAMPLES_TAG is ignored with --local-examples"
fi

if [[ -z "${WORKDIR}" ]]; then
	WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/moonshine-test-examples.XXXXXX")"
fi
mkdir -p "${WORKDIR}"

cleanup() {
	if [[ "${KEEP_WORKDIR}" -eq 1 ]]; then
		log "keeping workdir: ${WORKDIR}"
		return
	fi
	rm -rf "${WORKDIR}"
}

if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
	trap cleanup EXIT
fi

download_url_for() {
	local filename="$1"
	if [[ -n "${TAG}" ]]; then
		echo "https://github.com/${REPO}/releases/download/${TAG}/${filename}"
	else
		echo "https://github.com/${REPO}/releases/latest/download/${filename}"
	fi
}

download_one() {
	local filename="$1"
	local url
	url="$(download_url_for "${filename}")"
	log "downloading ${filename}"
	log "  URL: ${url}"
	local out="${WORKDIR}/${filename}"
	local curl_opts=(-fSL --retry 3 --connect-timeout 30)
	if [[ -n "${GITHUB_TOKEN:-}" ]]; then
		curl_opts+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
	fi
	if ! curl "${curl_opts[@]}" -o "${out}" "${url}"; then
		die "failed to download ${filename}. Check --repo / --tag and that the release assets exist."
	fi
}

extract_tgz() {
	local filename="$1"
	local dest="$2"
	mkdir -p "${dest}"
	log "extracting ${filename} → ${dest}"
	tar -xzf "${WORKDIR}/${filename}" -C "${dest}"
}

# Copy examples/android and examples/ios from the repository (parent of scripts/)
# into the same layout used after extracting the release tarballs.
copy_local_example_trees() {
	local android_dest="$1"
	local ios_dest="$2"
	local android_src="${REPO_ROOT}/examples/android"
	local ios_src="${REPO_ROOT}/examples/ios"
	[[ -d "${android_src}" ]] || die "missing directory: ${android_src}"
	[[ -d "${ios_src}" ]] || die "missing directory: ${ios_src}"
	rm -rf "${android_dest}" "${ios_dest}"
	mkdir -p "${android_dest}" "${ios_dest}"
	log "copying ${android_src}/ → ${android_dest}/"
	cp -a "${android_src}/." "${android_dest}/"
	log "copying ${ios_src}/ → ${ios_dest}/"
	cp -a "${ios_src}/." "${ios_dest}/"
}

pick_xcode_scheme() {
	local project="$1"
	python3 - "$project" <<'PY'
import json, os, subprocess, sys

proj = sys.argv[1]
out = subprocess.run(
    ["xcodebuild", "-list", "-json", "-project", proj],
    capture_output=True,
    text=True,
    check=True,
).stdout
data = json.loads(out)
schemes = []
if "project" in data and isinstance(data["project"], dict):
    schemes = data["project"].get("schemes") or []
if not schemes:
    sys.exit(0)
base = os.path.splitext(os.path.basename(proj))[0]
for s in schemes:
    if s == base:
        print(s)
        sys.exit(0)
for s in schemes:
    low = s.lower()
    if "test" in low and s != base:
        continue
    print(s)
    sys.exit(0)
print(schemes[0])
PY
}

run_android_builds() {
	local root="$1"
	if [[ "${SKIP_ANDROID}" == "1" ]]; then
		log "SKIP_ANDROID=1 — skipping Android builds"
		return 0
	fi
	local sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
	if [[ -z "${sdk}" && -d "${HOME}/Library/Android/sdk" ]]; then
		sdk="${HOME}/Library/Android/sdk"
	fi
	if [[ -z "${sdk}" ]]; then
		die "ANDROID_HOME (or ANDROID_SDK_ROOT) is not set; required for Gradle."
	fi
	export ANDROID_HOME="${sdk}"

	local found=0
	while IFS= read -r gw; do
		[[ -z "${gw}" ]] && continue
		found=1
		local dir
		dir="$(dirname "${gw}")"
		log "Android: ./gradlew assembleDebug in ${dir}"
		(
			cd "${dir}"
			chmod +x ./gradlew
			./gradlew assembleDebug --no-daemon --warning-mode all
		)
	done < <(find "${root}" -type f -name gradlew 2>/dev/null)

	if [[ "${found}" -eq 0 ]]; then
		die "no gradlew found under ${root} — unexpected android-examples layout"
	fi
}

run_ios_builds() {
	local root="$1"
	if [[ "$(uname -s)" != "Darwin" ]]; then
		log "not macOS — skipping iOS xcodebuild (set SKIP_IOS=1 to silence cross-platform CI)"
		return 0
	fi
	if [[ "${SKIP_IOS}" == "1" ]]; then
		log "SKIP_IOS=1 — skipping iOS builds"
		return 0
	fi
	if ! command -v xcodebuild >/dev/null 2>&1; then
		die "xcodebuild not found in PATH"
	fi

	local found=0
	while IFS= read -r proj; do
		[[ -z "${proj}" ]] && continue
		found=1
		local scheme
		scheme="$(pick_xcode_scheme "${proj}")"
		if [[ -z "${scheme}" ]]; then
			log "warning: no scheme for ${proj} — skipping"
			continue
		fi
		log "iOS: xcodebuild -project \"${proj}\" -scheme \"${scheme}\""
		xcodebuild \
			-project "${proj}" \
			-scheme "${scheme}" \
			-configuration Debug \
			-destination 'generic/platform=iOS Simulator' \
			CODE_SIGNING_ALLOWED=NO \
			CODE_SIGNING_REQUIRED=NO \
			build
	done < <(find "${root}" -type d -name '*.xcodeproj' 2>/dev/null)

	if [[ "${found}" -eq 0 ]]; then
		die "no *.xcodeproj found under ${root} — unexpected ios-examples layout"
	fi
}

main() {
	log "repo=${REPO} tag=${TAG:-<latest>} workdir=${WORKDIR} local_examples=${USE_LOCAL_EXAMPLES}"

	local ios_root="${WORKDIR}/ios-examples-tree"
	local android_root="${WORKDIR}/android-examples-tree"

	if [[ "${USE_LOCAL_EXAMPLES}" -eq 1 ]]; then
		copy_local_example_trees "${android_root}" "${ios_root}"
	else
		download_one "ios-examples.tar.gz"
		download_one "android-examples.tar.gz"
		extract_tgz "ios-examples.tar.gz" "${ios_root}"
		extract_tgz "android-examples.tar.gz" "${android_root}"
	fi

	run_android_builds "${android_root}"
	run_ios_builds "${ios_root}"

	log "all requested example builds succeeded"
}

main
