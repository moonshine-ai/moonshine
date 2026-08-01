#! /bin/bash

# Rewrite every version string in the repo from one release version to another.
#
# Usage:
#   scripts/update-version.sh [--check] <old_version> <new_version>
#   scripts/update-version.sh --verify <version>
#
# With --check, run the consistency checks and report what would be rewritten
# without touching any files. With --verify, just assert that the whole repo is
# consistently at one version (used by the release preflight).
#
# Normally you don't call this directly -- scripts/start-candidate.sh runs it
# when a development candidate branch is cut, so the branch carries the version
# it will ship with for the whole development cycle.
#
# The script cross-checks KNOWN_FILES against a repo-wide grep in both
# directions and refuses to run if they disagree, so a new file that embeds the
# version can't silently miss a bump, and a file that quietly stopped tracking
# the version can't rot unnoticed.

set -euo pipefail

# Add text files that contain a version string and need to be updated to this list.
KNOWN_FILES=(
	./README.md
	./core/CMakeLists.txt
	./python/pyproject.toml
	./python/setup.py
	./python/src/moonshine_voice/__init__.py
	./build.gradle.kts
	./wasm/package.json
	./wasm/package-lock.json
	./examples/macos/BasicTranscription/BasicTranscription.xcodeproj/project.pbxproj
	./examples/macos/BasicTranscription/Package.swift
	./examples/macos/MicTranscription/MicTranscription.xcodeproj/project.pbxproj
	./examples/macos/MicTranscription/Package.swift
	./examples/macos/TextToSpeech/Package.swift
	./examples/macos/AgentFlow/Package.swift
	./examples/ios/Transcriber/Transcriber.xcodeproj/project.pbxproj
	./examples/ios/TextToSpeech/TextToSpeech.xcodeproj/project.pbxproj
	./examples/ios/AgentFlow/AgentFlow.xcodeproj/project.pbxproj
	./examples/android/Transcriber/gradle/libs.versions.toml
	./examples/android/TextToSpeech/gradle/libs.versions.toml
	./examples/android/AgentFlow/gradle/libs.versions.toml
	./scripts/build-wasm.sh
	./scripts/publish-swift.sh
	./scripts/publish-binary.sh
	./scripts/publish-binary.bat
	./scripts/build-pip-docker.sh
	./scripts/publish-examples.sh
	./scripts/publish-examples.bat
	./scripts/test-examples.sh
	./examples/c++/download-library.sh
	./examples/c++/README.md
)

main() {
	local mode=rewrite
	case "${1:-}" in
	--check)
		mode=check
		shift
		;;
	--verify)
		mode=verify
		shift
		;;
	esac

	local old_version new_version
	if [ "${mode}" = verify ]; then
		if [ $# -ne 1 ]; then
			echo "Usage: $0 --verify <version>" >&2
			exit 1
		fi
		old_version="$1"
		new_version=""
	else
		if [ $# -ne 2 ]; then
			echo "Usage: $0 [--check] <old_version> <new_version>" >&2
			echo "       $0 --verify <version>" >&2
			exit 1
		fi
		old_version="$1"
		new_version="$2"
	fi

	for v in "${old_version}" ${new_version:+"${new_version}"}; do
		if ! [[ "${v}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
			echo "Version '${v}' is not in X.Y.Z form (no leading 'v')." >&2
			exit 1
		fi
	done
	if [ -n "${new_version}" ] && [ "${old_version}" = "${new_version}" ]; then
		echo "Old and new versions are both '${new_version}'; nothing to do." >&2
		exit 1
	fi

	cd "$(dirname "$(cd "$(dirname "$0")" && pwd)")"

	# Two ways a naive substring match goes wrong, both of which have already
	# silently corrupted this repo (see commit 52b0d1c, which rewrote an Xcode
	# object ID and a sample progress bar):
	#
	#   - unescaped dots are regex wildcards, so 0.0.60 matches "0B0F60"
	#   - even escaped, a bare match hits digits embedded in longer tokens, so
	#     0.0.60 matches inside "60.0/60.0"
	#
	# So the version is escaped, and must be bounded by something that is not a
	# digit or a dot on each side. Discovery uses grep -E, which only needs to
	# find one match per file; the rewrite uses perl, whose zero-width
	# lookaround avoids the overlap bug that bites a sed group when the same
	# line carries two version strings.
	local old_pattern="${old_version//./\\.}"
	local bounded="(^|[^0-9.])${old_pattern}([^0-9.]|$)"

	local actual_files=()
	while IFS= read -r line; do
		actual_files+=("$line")
	done < <(grep -rlIE \
		--exclude-dir=.git \
		--exclude-dir=__pycache__ \
		--exclude-dir=.venv \
		--exclude-dir=build \
		--exclude-dir=.build \
		--exclude-dir=.cxx \
		--exclude-dir=third-party \
		--exclude-dir=artifacts \
		--exclude-dir=lang-specific \
		--exclude-dir=node_modules \
		--exclude-dir=.release-state \
		--exclude=Package.resolved \
		--exclude=uv.lock \
		--exclude=PKG-INFO \
		--exclude=cli-transcriber.sln \
		--exclude=transcriber-test.cpp \
		--exclude=silero-vad-model-data.h \
		--exclude=spanish-unicode-tables.cpp \
		--exclude=russian.cpp \
		--exclude=arabic-ipa.cpp \
		--exclude=moonshine-cpp-test.cpp \
		--exclude=.env \
		--exclude=icon_mic.xml \
		--exclude=en_US-saikat.onnx.json \
		--exclude=portuguese-rules.cpp \
		--exclude=*.bin \
		--exclude=zh_hans.txt \
		--exclude=ipa-postprocess.cpp \
		--exclude=*.tsv \
		--exclude=ko_kr.txt \
		--exclude=vad_mel_tables.cc \
		--exclude=mel_tables.cc \
		--exclude=zipvoice-voices-data.cpp \
		--exclude=plda_vbx.cpp \
		--exclude=community1_cpp_annote_embedded.cpp \
		--exclude=hindi-numbers.cpp \
		--exclude=hindi.cpp \
		--exclude=neural_tts_demo_data.cc \
		--exclude=start-candidate.sh \
		--exclude=finish-release.sh \
		--exclude=build-all-platforms.sh \
		--exclude=release-process.md \
		"${bounded}" .)

	local unknown=()
	local file known found
	# bash 3.2 (the macOS default) treats an empty array expansion as an unbound
	# variable under set -u, and an empty match set is exactly what --verify sees
	# when the repo is at some other version, so guard every expansion.
	for file in ${actual_files[@]+"${actual_files[@]}"}; do
		found=false
		for known in "${KNOWN_FILES[@]}"; do
			if [[ "${file}" == "${known}" ]]; then
				found=true
				break
			fi
		done
		if [[ "${found}" = false ]]; then
			unknown+=("${file}")
		fi
	done
	if [ ${#unknown[@]} -gt 0 ]; then
		echo "These files contain ${old_version} but are not in KNOWN_FILES:" >&2
		printf '  %s\n' "${unknown[@]}" >&2
		echo >&2
		echo "If they should track the release version, add them to the" >&2
		echo "KNOWN_FILES array in $0. If the match is incidental, add an" >&2
		echo "--exclude for them to the grep above." >&2
		exit 1
	fi

	local stale=()
	local actual
	for file in "${KNOWN_FILES[@]}"; do
		found=false
		for actual in ${actual_files[@]+"${actual_files[@]}"}; do
			if [[ "${file}" == "${actual}" ]]; then
				found=true
				break
			fi
		done
		if [[ "${found}" = false ]]; then
			stale+=("${file}")
		fi
	done
	if [ ${#stale[@]} -gt 0 ]; then
		echo "These files are in KNOWN_FILES but do not contain ${old_version}:" >&2
		for file in "${stale[@]}"; do
			if [ -f "${file}" ]; then
				echo "  ${file} (has: $(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "${file}" \
					| sort -u -V | paste -sd' ' - || echo 'no version string'))" >&2
			else
				echo "  ${file} (missing)" >&2
			fi
		done
		echo >&2
		echo "Bring them back in sync with ${old_version} by hand, or remove them" >&2
		echo "from the KNOWN_FILES array in $0 if they no longer carry a version." >&2
		exit 1
	fi

	if [ "${mode}" = verify ]; then
		echo "All ${#KNOWN_FILES[@]} version-bearing files are at ${old_version}."
		return 0
	fi

	if [ "${mode}" = check ]; then
		echo "Checks pass: ${#KNOWN_FILES[@]} files consistently at ${old_version}."
		echo "Would rewrite ${old_version} -> ${new_version} in:"
		printf '  %s\n' "${KNOWN_FILES[@]}"
		return 0
	fi

	echo "Rewriting ${old_version} -> ${new_version} in ${#KNOWN_FILES[@]} files..."
	for file in "${KNOWN_FILES[@]}"; do
		OLD="${old_version}" NEW="${new_version}" perl -pi -e \
			's/(?<![0-9.])\Q$ENV{OLD}\E(?![0-9.])/$ENV{NEW}/g' "${file}"
		echo "  ${file}"
	done

	local missed=()
	for file in "${KNOWN_FILES[@]}"; do
		if grep -qE "${bounded}" "${file}"; then
			missed+=("${file}")
		fi
	done
	if [ ${#missed[@]} -gt 0 ]; then
		echo >&2
		echo "These files still contain ${old_version} after the rewrite:" >&2
		printf '  %s\n' "${missed[@]}" >&2
		exit 1
	fi
	echo "Verified: no ${old_version} left in any version-bearing file."
}

main "$@"
