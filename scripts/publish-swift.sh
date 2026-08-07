#!/bin/bash -ex

FRAMEWORK_NAME="Moonshine"
VERSION="0.1.1"
REPO="moonshine-ai/moonshine-swift"

# Check that the XCFramework exists
if [ ! -d "swift/$FRAMEWORK_NAME.xcframework" ]; then
	echo "Error: swift/$FRAMEWORK_NAME.xcframework not found"
	echo "Run scripts/build-swift.sh first, then run this script."
	exit 1
fi

TMP_DIR=$(mktemp -d)
gh repo clone $REPO $TMP_DIR
# Mirror this checkout into the cloned package repo. --delete drops renamed or
# removed Swift sources that would otherwise linger (e.g. DialogFlow.swift after
# the rename to AgentFlow.swift) and break consumers with "invalid redeclaration".
# .git / .build are excluded so the clone metadata and any local SPM cache stay put.
rsync -a --delete \
	--exclude '.git/' \
	--exclude '.build/' \
	swift/ "${TMP_DIR}/"
cd $TMP_DIR

XCFRAMEWORK_PATH="$FRAMEWORK_NAME.xcframework"

ZIP_NAME="$FRAMEWORK_NAME.xcframework.zip"

zip -r $ZIP_NAME $FRAMEWORK_NAME.xcframework

echo "Computing checksum..."
CHECKSUM=$(swift package compute-checksum "$ZIP_NAME")
echo "Checksum: $CHECKSUM"

cp Package.swift.remote Package.swift
sed -i '' "s/checksum: \".*\"/checksum: \"$CHECKSUM\"/" Package.swift
sed -i '' "s|\"https://github.com/.*\"|\"https://github.com/$REPO/releases/download/v$VERSION/Moonshine.xcframework.zip\"|" Package.swift

rm -rf Tests/MoonshineVoiceTests/test-assets

# Stage adds, updates, and deletions under the package paths (rsync --delete may
# have removed files that git still tracked on moonshine-swift).
git add -A -- .gitignore Package.swift Package.swift.remote README.md Sources Tests
git commit -m "Release v$VERSION"
git push origin main

# Remove the tag if it already exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    git tag -d "v$VERSION"
    git push --delete origin "v$VERSION" || true
fi

git tag v$VERSION && git push --tags

if ! gh release view v$VERSION >/dev/null 2>&1; then
	gh release create v$VERSION $ZIP_NAME \
		--repo $REPO \
		--title v$VERSION \
		--notes v$VERSION
fi
