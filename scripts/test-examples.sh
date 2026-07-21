#!/usr/bin/env bash -ex
# Verify iOS and Android examples build standalone: either from GitHub Release
# archives (default) or from a temporary copy of examples/android and examples/ios
# (--local-examples).
#
# Usage:
#   ./scripts/test-examples.sh [--repo OWNER/REPO] [--tag vX.Y.Z] [--workdir DIR] [--keep-workdir]
#   ./scripts/test-examples.sh --local-examples [--workdir DIR] [--keep-workdir]
#   ./scripts/test-examples.sh --local-library [--workdir DIR]
#
# With --local-examples, skips GitHub downloads and copies this repository's
# examples/android and examples/ios into a temp tree (same layout as extracted
# release archives), then runs the same Gradle / xcodebuild checks.
#
# With --local-library (implies --local-examples), the examples are built against
# THIS checkout's library instead of the published artifacts, so you can verify a
# library change (e.g. a new API or a lowered minSdk) before publishing:
#   - Android: builds + installs the AAR into the local Maven cache (via
#     scripts/build-android.sh local), injects mavenLocal() as the first
#     repository (via a Gradle init script) so it takes precedence, and syncs each
#     example's requested moonshine-voice version to this checkout's coordinates().
#     The example apps' minSdk (like the library's) comes from their Gradle files.
#   - iOS/macOS: copies this checkout's swift/ package into the temp iOS tree and
#     rewrites each example's project.pbxproj to reference that local package
#     (XCLocalSwiftPackageReference) instead of the remote moonshine-swift Git
#     package. Requires swift/Moonshine.xcframework to exist; when it is missing,
#     scripts/build-swift.sh is run to build it first.
#
# Environment:
#   ANDROID_HOME or ANDROID_SDK_ROOT — required for Android (unless SKIP_ANDROID=1)
#   GITHUB_TOKEN — optional; avoids anonymous rate limits on api.github.com if needed
#   SKIP_ANDROID=1 / SKIP_IOS=1 — skip that platform
#   TEST_EXAMPLES_TAG — same as --tag when --tag is omitted (e.g. v0.0.70)
#   TEST_EXAMPLES_USE_LOCAL=1 — same as --local-examples
#
# Defaults:
#   --repo moonshine-ai/moonshine
#   Archives: one asset per example app, named <platform>-<project>.tar.gz
#   (e.g. android-Transcriber.tar.gz, ios-IntentRecognizer.tar.gz). Names are
#   resolved from this repo's examples/android and examples/ios directories.

set -euo pipefail

REPO="${GITHUB_REPOSITORY:-moonshine-ai/moonshine}"
TAG=""
WORKDIR=""
KEEP_WORKDIR=0
SKIP_ANDROID="${SKIP_ANDROID:-0}"
SKIP_IOS="${SKIP_IOS:-0}"
USE_LOCAL_EXAMPLES=0
USE_LOCAL_LIBRARY=0
# Path to a Gradle init script (created at runtime when --local-library is used)
# that injects mavenLocal() as the first dependency-resolution repository.
LOCAL_LIBRARY_INIT_SCRIPT=""

usage() {
	cat <<'EOF'
Usage:
  test-examples.sh [--repo OWNER/REPO] [--tag vX.Y.Z] [--workdir DIR] [--keep-workdir]
  test-examples.sh --local-examples [--workdir DIR] [--keep-workdir]

Default mode: downloads each published example archive for Android and iOS
from GitHub Releases (see header comment for naming), merges them under one
tree per platform, and runs standalone builds:
  Android: every directory containing ./gradlew → ./gradlew assembleDebug
  macOS:   every *.xcodeproj → xcodebuild (iOS Simulator, no code signing)

--local-examples: copy <repo>/examples/android and <repo>/examples/ios into the
work directory (temporary copy; does not modify the originals), then run the
same build steps. Implies repository root is the parent of scripts/.

--local-library: implies --local-examples. Build the Android examples against
this checkout's AAR (installed into the local Maven cache) and the iOS examples
against this checkout's swift/ package (referenced locally in place of the remote
moonshine-swift package), instead of the published artifacts. Use this to verify
library changes (e.g. a new API or a lowered minSdk) before publishing.

Options:
  --repo OWNER/REPO   GitHub repository (default: moonshine-ai/moonshine); only for downloads
  --tag vX.Y.Z        Use .../releases/download/TAG/... (default: latest/download); only for downloads
  --local-examples    Use examples/android and examples/ios from this checkout instead of archives
  --local-library     Build + consume this checkout's Android AAR from the local Maven cache
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
	--local-library)
		USE_LOCAL_LIBRARY=1
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

# List immediate child directories of examples/<platform>/ (used to know which
# release assets to download: <platform>-<dirname>.tar.gz).
list_example_project_names() {
	local platform="$1"
	local src="${REPO_ROOT}/examples/${platform}"
	[[ -d "${src}" ]] || die "missing directory: ${src}"
	find "${src}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | LC_ALL=C sort
}

# Download every <platform>-<project>.tar.gz for that platform and extract into
# dest_root (same layout as the old monolithic *-examples.tar.gz).
download_platform_example_archives() {
	local platform="$1"
	local dest_root="$2"
	local name

	mkdir -p "${dest_root}"
	while IFS= read -r name; do
		[[ -z "${name}" ]] && continue
		local archive="${platform}-${name}.tar.gz"
		download_one "${archive}"
		extract_tgz "${archive}" "${dest_root}"
		rm -f "${WORKDIR}/${archive}"
	done < <(list_example_project_names "${platform}")
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

# Portable in-place edit (BSD/macOS sed and GNU sed differ on -i), applied to
# each given file with the supplied sed expression.
portable_sed_inplace() {
	local expr="$1"
	shift
	local f tmp
	for f in "$@"; do
		tmp="$(mktemp)"
		sed "${expr}" "${f}" >"${tmp}"
		mv "${tmp}" "${f}"
	done
}

# Read the moonshine-voice version this checkout would publish, parsed from the
# root build.gradle.kts coordinates("ai.moonshine", "moonshine-voice", "X.Y.Z").
read_library_version() {
	local version
	version="$(sed -n 's/.*coordinates("ai.moonshine", *"moonshine-voice", *"\([^"]*\)").*/\1/p' "${REPO_ROOT}/build.gradle.kts" | head -n1)"
	[[ -n "${version}" ]] || die "could not parse moonshine-voice version from ${REPO_ROOT}/build.gradle.kts"
	echo "${version}"
}

# Build the AAR from this checkout and install it into the local Maven cache so
# the example builds can resolve it. Delegates to build-android.sh so there is a
# single source of truth for the Gradle invocation.
publish_local_library() {
	log "building + installing local AAR into ~/.m2 (scripts/build-android.sh local)"
	"${SCRIPT_DIR}/build-android.sh" local
}

# Write a Gradle init script that adds mavenLocal() as the first dependency
# resolution repository for every build. Using beforeSettings ensures it is
# consulted before the examples' google()/mavenCentral() entries, so the freshly
# installed local AAR wins over any same-versioned artifact on Maven Central.
write_local_library_init_script() {
	local path="${WORKDIR}/local-library.init.gradle"
	cat >"${path}" <<'EOF'
// Injected by test-examples.sh --local-library. Adds mavenLocal() as the first
// dependency-resolution repository so examples resolve the AAR that
// build-android.sh installed into ~/.m2 rather than the published artifact.
beforeSettings { settings ->
    settings.dependencyResolutionManagement {
        repositories {
            mavenLocal()
        }
    }
}
EOF
	LOCAL_LIBRARY_INIT_SCRIPT="${path}"
	log "wrote local-library init script: ${path}"
}

# Point the copied Android examples at the locally-built library: sync each
# example's requested moonshine-voice version to this checkout's version.
apply_local_library_overrides() {
	local android_root="$1"
	local version
	version="$(read_library_version)"
	log "syncing example moonshine-voice version to ${version}"

	local toml
	while IFS= read -r toml; do
		[[ -z "${toml}" ]] && continue
		portable_sed_inplace \
			"s/^\(moonshineVoice[[:space:]]*=[[:space:]]*\)\"[^\"]*\"/\1\"${version}\"/" \
			"${toml}"
	done < <(find "${android_root}" -type f -name libs.versions.toml 2>/dev/null)
}

# Directory name the local swift/ package is copied to inside the iOS tree.
LOCAL_SWIFT_PACKAGE_DIR="local-moonshine-swift"

# Ensure a locally-built XCFramework exists for the swift package to wrap. The
# example apps consume the MoonshineVoice product, whose binaryTarget points at
# swift/Moonshine.xcframework; build it (via build-swift.sh) if it is missing.
ensure_local_swift_package() {
	local framework="${REPO_ROOT}/swift/Moonshine.xcframework"
	if [[ -d "${framework}" ]]; then
		log "using existing ${framework} (run scripts/build-swift.sh to refresh if core changed)"
		return 0
	fi
	log "swift/Moonshine.xcframework missing — building it via scripts/build-swift.sh"
	"${SCRIPT_DIR}/build-swift.sh"
}

# Copy this checkout's swift/ package into the iOS tree so the examples can
# reference it locally. Excludes the SwiftPM .build cache to keep the copy small;
# the XCFramework and sources are preserved.
copy_local_swift_package() {
	local ios_root="$1"
	local dest="${ios_root}/${LOCAL_SWIFT_PACKAGE_DIR}"
	rm -rf "${dest}"
	mkdir -p "${dest}"
	log "copying ${REPO_ROOT}/swift/ → ${dest}/"
	# rsync is available on macOS (the only platform that runs the iOS builds).
	rsync -a --exclude '.build' "${REPO_ROOT}/swift/." "${dest}/"
}

# Rewrite one project.pbxproj so its remote moonshine-swift package reference
# becomes a local reference at ${relpath}. The XCSwiftPackageProductDependency
# links by productName, so only the package reference object needs to change.
rewrite_pbxproj_to_local_package() {
	local pbxproj="$1"
	local relpath="$2"
	python3 - "$pbxproj" "$relpath" <<'PY'
import sys

pbxproj, relpath = sys.argv[1], sys.argv[2]
with open(pbxproj, "r") as f:
    text = f.read()

marker = ' /* XCRemoteSwiftPackageReference "moonshine-swift" */ = {'
out = []
idx = 0
changed = 0
while True:
    pos = text.find(marker, idx)
    if pos == -1:
        out.append(text[idx:])
        break
    # The 24-char object id immediately precedes the marker on the same line.
    line_start = text.rfind("\n", 0, pos) + 1
    obj_id = text[line_start:pos].strip()
    out.append(text[idx:line_start])
    # Brace-match from the opening '{' at the end of the marker.
    i = pos + len(marker)  # position just after '{'
    depth = 1
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if i < len(text) and text[i] == ";":
        i += 1
    out.append(
        '\t\t{id} /* XCLocalSwiftPackageReference "{name}" */ = {{\n'
        "\t\t\tisa = XCLocalSwiftPackageReference;\n"
        '\t\t\trelativePath = "{rel}";\n'
        "\t\t}};".format(id=obj_id, name=relpath, rel=relpath)
    )
    idx = i
    changed += 1

if changed == 0:
    sys.exit(0)

text = "".join(out)
# Fix up the (cosmetic) packageReferences entry and section header comments.
text = text.replace(
    '/* XCRemoteSwiftPackageReference "moonshine-swift" */',
    '/* XCLocalSwiftPackageReference "{}" */'.format(relpath),
)
text = text.replace(
    "/* Begin XCRemoteSwiftPackageReference section */",
    "/* Begin XCLocalSwiftPackageReference section */",
)
text = text.replace(
    "/* End XCRemoteSwiftPackageReference section */",
    "/* End XCLocalSwiftPackageReference section */",
)
with open(pbxproj, "w") as f:
    f.write(text)
print("rewrote {} package reference(s) in {}".format(changed, pbxproj))
PY
}

# Point every copied iOS example at the local swift package.
apply_local_library_overrides_ios() {
	local ios_root="$1"
	local pkg_abs="${ios_root}/${LOCAL_SWIFT_PACKAGE_DIR}"
	local proj
	while IFS= read -r proj; do
		[[ -z "${proj}" ]] && continue
		# relativePath is resolved against the directory containing the .xcodeproj.
		local proj_dir
		proj_dir="$(cd "$(dirname "${proj}")" && pwd)"
		local relpath
		relpath="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "${pkg_abs}" "${proj_dir}")"
		log "iOS local package: ${proj} -> relativePath ${relpath}"
		rewrite_pbxproj_to_local_package "${proj}/project.pbxproj" "${relpath}"
	done < <(find "${ios_root}" -type d -name '*.xcodeproj' 2>/dev/null)
}

pick_xcode_scheme() {
	local project="$1"
	python3 - "$project" <<'PY'
import json, os, subprocess, sys

proj = sys.argv[1]
proc = subprocess.run(
    ["xcodebuild", "-list", "-json", "-project", proj],
    capture_output=True,
    text=True,
)
if proc.returncode != 0:
    # Surface xcodebuild's own diagnostics instead of hiding them behind a bare
    # exit code. Exit 74 here is usually a Swift Package resolution problem
    # (e.g. a moved tag tripping SPM's trust-on-first-use fingerprint, or a
    # release-asset checksum that no longer matches Package.swift).
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.stderr.write(
        "\n[pick_xcode_scheme] 'xcodebuild -list' failed for {} "
        "(exit {}).\n".format(proj, proc.returncode)
    )
    sys.exit(proc.returncode)
data = json.loads(proc.stdout)
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
			local gradle_args=(assembleDebug --no-daemon --warning-mode all)
			if [[ -n "${LOCAL_LIBRARY_INIT_SCRIPT}" ]]; then
				gradle_args+=(--init-script "${LOCAL_LIBRARY_INIT_SCRIPT}")
			fi
			./gradlew "${gradle_args[@]}"
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
	log "repo=${REPO} tag=${TAG:-<latest>} workdir=${WORKDIR} local_examples=${USE_LOCAL_EXAMPLES} local_library=${USE_LOCAL_LIBRARY}"

	local ios_root="${WORKDIR}/ios-examples-tree"
	local android_root="${WORKDIR}/android-examples-tree"

	if [[ "${USE_LOCAL_LIBRARY}" -eq 1 && "${SKIP_ANDROID}" != "1" ]]; then
		publish_local_library
		write_local_library_init_script
	fi

	if [[ "${USE_LOCAL_EXAMPLES}" -eq 1 ]]; then
		copy_local_example_trees "${android_root}" "${ios_root}"
	else
		download_platform_example_archives "android" "${android_root}"
		download_platform_example_archives "ios" "${ios_root}"
	fi

	if [[ "${USE_LOCAL_LIBRARY}" -eq 1 && "${SKIP_ANDROID}" != "1" ]]; then
		apply_local_library_overrides "${android_root}"
	fi

	if [[ "${USE_LOCAL_LIBRARY}" -eq 1 && "${SKIP_IOS}" != "1" && "$(uname -s)" == "Darwin" ]]; then
		ensure_local_swift_package
		copy_local_swift_package "${ios_root}"
		apply_local_library_overrides_ios "${ios_root}"
	fi

	run_android_builds "${android_root}"
	run_ios_builds "${ios_root}"

	log "all requested example builds succeeded"
}

main
