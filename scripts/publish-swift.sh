#!/bin/bash -ex

FRAMEWORK_NAME="Moonshine"
VERSION="0.1.1"
REPO="moonshine-ai/moonshine-swift"
TAG="v${VERSION}"

# Check that the XCFramework exists
if [ ! -d "swift/$FRAMEWORK_NAME.xcframework" ]; then
	echo "Error: swift/$FRAMEWORK_NAME.xcframework not found"
	echo "Run scripts/build-swift.sh first, then run this script."
	exit 1
fi

TMP_DIR=$(mktemp -d)
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

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

ZIP_NAME="$FRAMEWORK_NAME.xcframework.zip"
rm -f "${ZIP_NAME}"
zip -r $ZIP_NAME $FRAMEWORK_NAME.xcframework

echo "Computing checksum..."
CHECKSUM=$(swift package compute-checksum "$ZIP_NAME")
echo "Checksum: $CHECKSUM"

cp Package.swift.remote Package.swift
sed -i '' "s/checksum: \".*\"/checksum: \"$CHECKSUM\"/" Package.swift
sed -i '' "s|\"https://github.com/.*\"|\"https://github.com/$REPO/releases/download/$TAG/Moonshine.xcframework.zip\"|" Package.swift

rm -rf Tests/MoonshineVoiceTests/test-assets

# Stage adds, updates, and deletions under the package paths (rsync --delete may
# have removed files that git still tracked on moonshine-swift).
git add -A -- .gitignore Package.swift Package.swift.remote README.md Sources Tests
if git diff --cached --quiet; then
	echo "No package-source changes to commit (Sources already match this checkout)."
else
	git commit -m "Release ${TAG}"
	git push origin main
fi

# Drop any prior tag/release for this version so a resume can republish cleaned
# Sources (SPM resolves Swift sources from the git tag, not the zip alone).
if git rev-parse "${TAG}" >/dev/null 2>&1; then
	git tag -d "${TAG}"
fi
git push --delete origin "${TAG}" 2>/dev/null || true
if gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
	gh release delete "${TAG}" --repo "${REPO}" --yes
fi

git tag "${TAG}"
git push origin "${TAG}"

gh release create "${TAG}" "${ZIP_NAME}" \
	--repo "${REPO}" \
	--title "${TAG}" \
	--notes "${TAG}"
