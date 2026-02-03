#! /bin/bash -ex

OLD_VERSION=$1
NEW_VERSION=$2

# Add text files that contain a version string and need to be updated to this list.
KNOWN_FILES=(
	./README.md
	./core/CMakeLists.txt
	./python/pyproject.toml
	./python/setup.py
	./build.gradle.kts
	./examples/macos/BasicTranscription/BasicTranscription.xcodeproj/project.pbxproj
	./examples/macos/BasicTranscription/Package.swift
	./examples/macos/MicTranscription/MicTranscription.xcodeproj/project.pbxproj
	./examples/macos/MicTranscription/Package.swift
	./examples/ios/Transcriber/Transcriber.xcodeproj/project.pbxproj
	./examples/android/Transcriber/gradle/libs.versions.toml
	./scripts/publish-swift.sh
	./scripts/publish-binary.sh
	./scripts/publish-binary.bat
)

ACTUAL_FILES=()
while IFS= read -r line; do
    ACTUAL_FILES+=("$line")
done < <(grep -rlI \
  --exclude-dir=.git \
  --exclude-dir=__pycache__ \
  --exclude-dir=.venv \
  --exclude-dir=build \
  --exclude-dir=.build \
  --exclude-dir=.cxx \
  --exclude-dir=third-party \
  --exclude=Package.resolved \
  --exclude=uv.lock \
  --exclude=PKG-INFO \
  --exclude=cli-transcriber.sln \
  --exclude=transcriber-test.cpp \
  "$OLD_VERSION" .)
for FILE in "${ACTUAL_FILES[@]}"; do
	echo "Checking file '$FILE'"
    found_in_known_files=false
    for known in "${KNOWN_FILES[@]}"; do
		echo "Checking known file '$known'"
        if [[ "$FILE" == "$known" ]]; then
            found_in_known_files=true
            break
        fi
    done
    if [[ "$found_in_known_files" = false ]]; then
        echo "File '$FILE' is not in the known files list but it contains the old version string"
		echo "If this is intentional, add the file to the KNOWN_FILES array in the update-version.sh script"
        exit 1
    fi
done

for FILE in "${KNOWN_FILES[@]}"; do
    found_in_actual_files=false
    echo "Checking file '$FILE'"
    for actual in "${ACTUAL_FILES[@]}"; do
		echo "Checking actual file '$actual'"
        if [[ "$FILE" == "$actual" ]]; then
            found_in_actual_files=true
            break
        fi
    done
    if [[ "$found_in_actual_files" = false ]]; then
        echo "File '$FILE' is in the known files list but doesn't contain the old version string"
        echo "If this is intentional, remove the file from the KNOWN_FILES array in the update-version.sh script"
        exit 1
    fi
    sed -i '' "s/$OLD_VERSION/$NEW_VERSION/" $FILE
done