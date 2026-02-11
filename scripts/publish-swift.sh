#!/bin/bash -ex

FRAMEWORK_NAME="Moonshine"
VERSION="0.0.48"
REPO="moonshine-ai/moonshine-swift"

# Check that the XCFramework exists
if [ ! -d "swift/$FRAMEWORK_NAME.xcframework" ]; then
	echo "Error: swift/$FRAMEWORK_NAME.xcframework not found"
	echo "Run scripts/build-swift.sh first, then run this script."
	exit 1
fi

TMP_DIR=$(mktemp -d)
gh repo clone $REPO $TMP_DIR
cp -R -P swift/* $TMP_DIR/
cp swift/.gitignore $TMP_DIR/
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

git add Package.swift Sources Tests .gitignore
git commit -a -m "Release v$VERSION"
git push origin main

git tag v$VERSION && git push --tags

gh release create v$VERSION $ZIP_NAME \
	--repo $REPO \
	--title v$VERSION \
	--notes v$VERSION