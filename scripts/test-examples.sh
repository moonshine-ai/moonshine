#!/usr/bin/env bash -ex
# Verify the iOS, Android, portable C++, and web examples build standalone: either
# from GitHub Release archives (default) or from a temporary copy of
# examples/android, examples/ios, examples/c++, and packaged web demos
# (--local-examples).
#
# The C++ example is exercised exactly like the Linux quickstart in README.md:
# fetch the library + a small model + sample audio with download-library.sh,
# compile transcriber.cpp with a single compiler command, and run it. It is
# platform-aware (macOS and Linux; skipped elsewhere) so it validates whatever
# host it runs on. Windows has its own scripts/test-examples.bat.
#
# Web demos are the five self-contained archives from
# scripts/web-example-archive.sh (web-stt.tar.gz, …). Each is extracted (or
# staged from this checkout), served with its bundled serve.mjs, and checked for
# the Cross-Origin Isolation headers plus the shared assets the page needs.
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
#   - C++: hands download-library.sh a moonshine-voice-<platform>.tar.gz packaged
#     from this checkout by scripts/publish-binary.sh, instead of letting it fetch
#     the published archive for a release that does not exist yet. That compiles
#     core unless an earlier stage left a build in core/build to package as-is.
#   - Web: stages the same self-contained trees publish-examples.sh would upload,
#     and points serve.mjs at this checkout via MOONSHINE_REPO_ROOT so ?local=1
#     mounts keep working during the smoke test.
#
# Environment:
#   ANDROID_HOME or ANDROID_SDK_ROOT — required for Android (unless SKIP_ANDROID=1)
#   GITHUB_TOKEN — optional; avoids anonymous rate limits on api.github.com if needed
#   SKIP_ANDROID=1 / SKIP_IOS=1 / SKIP_CPP=1 / SKIP_WEB=1 — skip that platform
#   CXX — C++ compiler for the C++ example (default: g++)
#   TEST_EXAMPLES_TAG — same as --tag when --tag is omitted (e.g. v0.1.1)
#   TEST_EXAMPLES_USE_LOCAL=1 — same as --local-examples
#
# Defaults:
#   --repo moonshine-ai/moonshine
#   Archives: one asset per example app, named <platform>-<project>.tar.gz
#   (e.g. android-Transcriber.tar.gz, ios-TextToSpeech.tar.gz, web-stt.tar.gz).
#   Names are resolved from this repo's examples/<platform> directories (web demos
#   from scripts/web-example-archive.sh list).

set -euo pipefail

REPO="${GITHUB_REPOSITORY:-moonshine-ai/moonshine}"
TAG=""
WORKDIR=""
KEEP_WORKDIR=0
SKIP_ANDROID="${SKIP_ANDROID:-0}"
SKIP_IOS="${SKIP_IOS:-0}"
SKIP_CPP="${SKIP_CPP:-0}"
SKIP_WEB="${SKIP_WEB:-0}"
USE_LOCAL_EXAMPLES=0
USE_LOCAL_LIBRARY=0
# Path to a Gradle init script (created at runtime when --local-library is used)
# that injects mavenLocal() as the first dependency-resolution repository.
LOCAL_LIBRARY_INIT_SCRIPT=""
# Path to the library archive the C++ example is built against with
# --local-library. Empty otherwise, which leaves download-library.sh fetching the
# published archive as a user would.
LOCAL_LIBRARY_ARCHIVE_CPP=""
WEB_ARCHIVE_SCRIPT=""

usage() {
	cat <<'EOF'
Usage:
  test-examples.sh [--repo OWNER/REPO] [--tag vX.Y.Z] [--workdir DIR] [--keep-workdir]
  test-examples.sh --local-examples [--workdir DIR] [--keep-workdir]

Default mode: downloads each published example archive for Android, iOS, and
web from GitHub Releases (see header comment for naming), merges them under one
tree per platform, and runs standalone builds:
  Android: every directory containing ./gradlew → ./gradlew assembleDebug
  macOS:   every *.xcodeproj → xcodebuild (iOS Simulator, no code signing)
  Web:     each web-<demo>.tar.gz → node serve.mjs + HTTP smoke test

--local-examples: copy <repo>/examples/android and <repo>/examples/ios into the
work directory (temporary copy; does not modify the originals), stage the web
demo archives from this checkout, then run the same build / smoke steps.
Implies repository root is the parent of scripts/.

--local-library: implies --local-examples. Build the Android examples against
this checkout's AAR (installed into the local Maven cache) and the iOS examples
against this checkout's swift/ package (referenced locally in place of the remote
moonshine-swift package), instead of the published artifacts. Use this to verify
library changes (e.g. a new API or a lowered minSdk) before publishing.

Options:
  --repo OWNER/REPO   GitHub repository (default: moonshine-ai/moonshine); only for downloads
  --tag vX.Y.Z        Use .../releases/download/TAG/... (default: latest/download); only for downloads
  --local-examples    Use examples from this checkout instead of archives
  --local-library     Build + consume this checkout's Android AAR from the local Maven cache
  --workdir DIR       Extract / copy and build here instead of a fresh mktemp directory
  --keep-workdir      Do not delete the work directory on exit (implies useful with --workdir)

Environment:
  ANDROID_HOME        SDK path for Android Gradle
  SKIP_ANDROID=1      Skip Android builds
  SKIP_IOS=1          Skip iOS builds (also implied on non-Darwin)
  SKIP_CPP=1          Skip the portable C++ example build/run
  SKIP_WEB=1          Skip the web example smoke tests
  CXX                 C++ compiler for the C++ example (default: g++)
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
WEB_ARCHIVE_SCRIPT="${SCRIPT_DIR}/web-example-archive.sh"
[[ -f "${WEB_ARCHIVE_SCRIPT}" ]] || die "missing ${WEB_ARCHIVE_SCRIPT}"

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

# Obtain the portable C++ example (examples/c++) into ${dest}: copy it from this
# checkout with --local-examples, otherwise download and unpack the published
# cpp-examples.tar.gz release asset. Either way ${dest} ends up containing
# transcriber.cpp, download-library.sh, etc. directly.
obtain_cpp_example() {
	local dest="$1"
	rm -rf "${dest}"
	mkdir -p "${dest}"
	if [[ "${USE_LOCAL_EXAMPLES}" -eq 1 ]]; then
		local src="${REPO_ROOT}/examples/c++"
		[[ -d "${src}" ]] || die "missing directory: ${src}"
		log "copying ${src}/ → ${dest}/"
		cp -a "${src}/." "${dest}/"
	else
		download_one "cpp-examples.tar.gz"
		log "extracting cpp-examples.tar.gz → ${dest}"
		# The archive has a top-level c++/ directory; strip it so the sources
		# land directly in ${dest}.
		tar -xzf "${WORKDIR}/cpp-examples.tar.gz" -C "${dest}" --strip-components=1
		rm -f "${WORKDIR}/cpp-examples.tar.gz"
	fi
}

# Package this checkout's library the way the release ships it, so the C++ example
# can be built against it. download-library.sh fetches the archive for the release
# it was cut for, which does not exist yet when a release is rehearsed, and handing
# the example an older published archive would defeat the point of building it
# against the library it ships beside.
#
# publish-binary.sh is the only definition of what that archive contains, so call it
# rather than assembling a lookalike here, and always call it: a
# moonshine-voice-<platform>.tar.gz sitting at the repository root can be months
# old, and silently testing against that would turn this check into theatre. It
# compiles core unless an earlier stage left a build it can package as-is.
ensure_local_library_archive_cpp() {
	local arch
	arch="$(uname -m)"
	case "${arch}" in
	arm64 | aarch64) arch=arm64 ;;
	x86_64 | amd64) arch=x86_64 ;;
	*) die "unsupported architecture for the C++ example: ${arch}" ;;
	esac
	local platform prebuilt
	if [[ "$(uname -s)" == "Darwin" ]]; then
		platform="macos-${arch}"
		prebuilt="${REPO_ROOT}/core/build/moonshine.framework/Versions/A/moonshine"
	else
		platform="linux-${arch}"
		prebuilt="${REPO_ROOT}/core/build/libmoonshine.so"
	fi

	if [[ -f "${prebuilt}" ]]; then
		log "C++: packaging the existing core build in core/build for ${platform}"
		"${SCRIPT_DIR}/publish-binary.sh" skip-build
	else
		log "C++: building and packaging the ${platform} library archive"
		"${SCRIPT_DIR}/publish-binary.sh"
	fi

	local archive="${REPO_ROOT}/moonshine-voice-${platform}.tar.gz"
	[[ -f "${archive}" ]] || die "publish-binary.sh left no archive at ${archive}"
	LOCAL_LIBRARY_ARCHIVE_CPP="${archive}"
}

# Build and run the portable C++ transcriber example, mirroring the README
# quickstart: download-library.sh fetches the prebuilt library plus a small
# model and sample audio, then transcriber.cpp is compiled with a single
# compiler command and run. macOS links the static libmoonshine.a (plus the
# CoreFoundation/Foundation frameworks); Linux links the shared libmoonshine.so
# and bakes in an $ORIGIN rpath so the co-located libonnxruntime.so.1 is found
# without LD_LIBRARY_PATH. Other operating systems (e.g. Windows, which has its
# own test-examples.bat) are skipped.
run_cpp_build() {
	local root="$1"
	if [[ "${SKIP_CPP}" == "1" ]]; then
		log "SKIP_CPP=1 — skipping C++ example build"
		return 0
	fi
	local os
	os="$(uname -s)"
	if [[ "${os}" != "Darwin" && "${os}" != "Linux" ]]; then
		log "C++ example test only runs on macOS and Linux — skipping on ${os}"
		return 0
	fi

	local cxx="${CXX:-g++}"
	command -v "${cxx}" >/dev/null 2>&1 || die "C++ compiler '${cxx}' not found (set CXX or SKIP_CPP=1)"

	# Packaging the local library rebuilds core when no earlier stage has, so only
	# do it once we know the example is actually going to be built here.
	if [[ "${USE_LOCAL_LIBRARY}" -eq 1 ]]; then
		ensure_local_library_archive_cpp
	fi

	# download-library.sh always extracts into a folder named "moonshine-voice"
	# regardless of OS/architecture, so the -I/-L paths are the same everywhere.
	local platform_dir="moonshine-voice"

	(
		cd "${root}"
		log "C++: fetching library, model, and sample audio via download-library.sh"
		chmod +x ./download-library.sh
		MOONSHINE_LIBRARY_ARCHIVE="${LOCAL_LIBRARY_ARCHIVE_CPP}" ./download-library.sh

		log "C++: compiling transcriber.cpp against ${platform_dir}"
		if [[ "${os}" == "Darwin" ]]; then
			"${cxx}" transcriber.cpp \
				-I"${platform_dir}/include" \
				-L"${platform_dir}/lib" \
				-lmoonshine \
				-o transcriber \
				-framework CoreFoundation \
				-framework Foundation
		else
			"${cxx}" transcriber.cpp \
				-I"${platform_dir}/include" \
				-L"${platform_dir}/lib" \
				-lmoonshine \
				-Wl,-rpath,'$ORIGIN/'"${platform_dir}/lib" \
				-o transcriber
		fi

		log "C++: running transcriber"
		./transcriber | tee transcriber-output.txt
		if ! grep -q "Line completed:" transcriber-output.txt; then
			die "C++ transcriber produced no completed lines — see output above"
		fi
		log "C++ example build and run succeeded"
	)
}

# --- Web examples -----------------------------------------------------------

# Stage or download every self-contained web-<demo>.tar.gz into dest_root/<demo>/.
obtain_web_examples() {
	local dest_root="$1"
	local name
	rm -rf "${dest_root}"
	mkdir -p "${dest_root}"

	if [[ "${USE_LOCAL_EXAMPLES}" -eq 1 ]]; then
		while IFS= read -r name; do
			[[ -z "${name}" ]] && continue
			# Pack then extract so we exercise the same archive layout publish
			# uploads, not only the stage helper.
			local archive_path="${WORKDIR}/web-${name}.tar.gz"
			log "packing + extracting web demo ${name}"
			"${WEB_ARCHIVE_SCRIPT}" pack "${name}" "${archive_path}"
			extract_tgz "web-${name}.tar.gz" "${dest_root}"
			rm -f "${archive_path}"
			[[ -f "${dest_root}/${name}/serve.mjs" ]] ||
				die "web-${name}.tar.gz is not self-contained (missing ${name}/serve.mjs)"
			[[ -d "${dest_root}/${name}/assets" ]] ||
				die "web-${name}.tar.gz is not self-contained (missing ${name}/assets)"
			[[ -f "${dest_root}/${name}/${name}/index.html" ]] ||
				die "web-${name}.tar.gz is not self-contained (missing ${name}/${name}/index.html)"
		done < <("${WEB_ARCHIVE_SCRIPT}" list)
		return 0
	fi

	while IFS= read -r name; do
		[[ -z "${name}" ]] && continue
		local archive="web-${name}.tar.gz"
		download_one "${archive}"
		# Archive top-level is <demo>/; extract into dest_root so we get
		# dest_root/<demo>/serve.mjs.
		extract_tgz "${archive}" "${dest_root}"
		rm -f "${WORKDIR}/${archive}"
		[[ -f "${dest_root}/${name}/serve.mjs" ]] ||
			die "${archive} is not self-contained (missing ${name}/serve.mjs)"
		[[ -d "${dest_root}/${name}/assets" ]] ||
			die "${archive} is not self-contained (missing ${name}/assets)"
		[[ -f "${dest_root}/${name}/${name}/index.html" ]] ||
			die "${archive} is not self-contained (missing ${name}/${name}/index.html)"
	done < <("${WEB_ARCHIVE_SCRIPT}" list)
}

# Pick a free TCP port on localhost.
web_free_port() {
	python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

# Assert a URL returns 200 with the Cross-Origin Isolation headers SharedArrayBuffer needs.
web_assert_isolated() {
	local url="$1"
	local tmp headers status
	tmp="$(mktemp)"
	headers="$(mktemp)"
	status="$(curl -sS -o "${tmp}" -D "${headers}" -w '%{http_code}' "${url}" || true)"
	if [[ "${status}" != "200" ]]; then
		rm -f "${tmp}" "${headers}"
		die "GET ${url} → HTTP ${status}"
	fi
	if ! grep -qi '^Cross-Origin-Opener-Policy:[[:space:]]*same-origin' "${headers}"; then
		rm -f "${tmp}" "${headers}"
		die "${url} missing Cross-Origin-Opener-Policy: same-origin"
	fi
	if ! grep -qi '^Cross-Origin-Embedder-Policy:[[:space:]]*require-corp' "${headers}"; then
		rm -f "${tmp}" "${headers}"
		die "${url} missing Cross-Origin-Embedder-Policy: require-corp"
	fi
	if [[ ! -s "${tmp}" ]]; then
		rm -f "${tmp}" "${headers}"
		die "${url} returned an empty body"
	fi
	rm -f "${tmp}" "${headers}"
}

# Serve one staged/extracted demo and confirm the page + shared assets are reachable.
smoke_test_web_demo() {
	local demo_root="$1"
	local demo
	demo="$(basename "${demo_root}")"
	[[ -f "${demo_root}/serve.mjs" ]] || die "no serve.mjs in ${demo_root}"

	local port pid=""
	port="$(web_free_port)"
	log "web ${demo}: starting serve.mjs on :${port}"

	cleanup_web_server() {
		if [[ -n "${pid}" ]]; then
			kill "${pid}" 2>/dev/null || true
			wait "${pid}" 2>/dev/null || true
			pid=""
		fi
	}
	trap cleanup_web_server RETURN

	(
		cd "${demo_root}"
		# With --local-examples, keep the /wasm and /test-assets mounts pointed at
		# this checkout so the smoke test can optionally hit them later.
		if [[ "${USE_LOCAL_EXAMPLES}" -eq 1 ]]; then
			export MOONSHINE_REPO_ROOT="${REPO_ROOT}"
		fi
		exec node serve.mjs "${port}"
	) >/dev/null 2>&1 &
	pid=$!

	local origin="http://127.0.0.1:${port}"
	local deadline=$((SECONDS + 15))
	until curl -fsS "${origin}/" >/dev/null 2>&1; do
		if ! kill -0 "${pid}" 2>/dev/null; then
			die "web ${demo}: serve.mjs exited before becoming ready"
		fi
		if ((SECONDS >= deadline)); then
			die "web ${demo}: serve.mjs did not become ready on :${port}"
		fi
		sleep 0.15
	done

	web_assert_isolated "${origin}/"
	web_assert_isolated "${origin}/${demo}/"
	web_assert_isolated "${origin}/assets/moonshine-ui.js"
	web_assert_isolated "${origin}/assets/moonshine.css"
	web_assert_isolated "${origin}/assets/snippets.js"

	# The demo HTML must pull shared chrome from /assets/, not a CDN copy we
	# forgot to ship.
	curl -fsS "${origin}/${demo}/" | grep -q '/assets/moonshine-ui.js' ||
		die "web ${demo}: index.html does not reference /assets/moonshine-ui.js"

	if [[ "${USE_LOCAL_EXAMPLES}" -eq 1 && -f "${REPO_ROOT}/wasm/dist/index.js" ]]; then
		web_assert_isolated "${origin}/wasm/dist/index.js"
	fi

	log "web ${demo}: smoke test passed"
	cleanup_web_server
	trap - RETURN
}

run_web_tests() {
	local dest_root="$1"
	local demo_root
	obtain_web_examples "${dest_root}"
	while IFS= read -r demo_root; do
		[[ -z "${demo_root}" ]] && continue
		smoke_test_web_demo "${demo_root}"
	done < <(find "${dest_root}" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort)
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

	if [[ "${SKIP_CPP}" != "1" ]]; then
		local cpp_root="${WORKDIR}/cpp-example-tree"
		obtain_cpp_example "${cpp_root}"
		run_cpp_build "${cpp_root}"
	else
		log "SKIP_CPP=1 — skipping C++ example test"
	fi

	if [[ "${SKIP_WEB}" != "1" ]]; then
		run_web_tests "${WORKDIR}/web-examples-tree"
	else
		log "SKIP_WEB=1 — skipping web example tests"
	fi

	log "all requested example builds succeeded"
}

main
