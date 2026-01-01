#!/bin/bash -ex

FRAMEWORK_NAME="Moonshine"
VERSION="0.0.7"
REPO="moonshine-ai/moonshine-v2"

XCFRAMEWORK_PATH="swift/$FRAMEWORK_NAME.xcframework"

# Check that the XCFramework exists
if [ ! -d "$XCFRAMEWORK_PATH" ]; then
	echo "Error: $XCFRAMEWORK_PATH not found"
	echo "Run scripts/build-swift.sh first, then run this script."
	exit 1
fi

TMP_DIR=$(mktemp -d)
cp -R -P $XCFRAMEWORK_PATH $TMP_DIR/
cd $TMP_DIR

ZIP_NAME="$FRAMEWORK_NAME.xcframework.zip"

zip -r $ZIP_NAME $FRAMEWORK_NAME.xcframework

echo "Creating GitHub release v$VERSION..."
gh release create "v$VERSION" \
	"$ZIP_NAME" \
	--repo "$REPO" \
	--title "v$VERSION" \
	--notes "Release v$VERSION of the Moonshine Voice Swift package."

echo "Computing checksum..."
CHECKSUM=$(swift package compute-checksum "$ZIP_NAME")
echo "Checksum: $CHECKSUM"

echo "Done! Next steps:"
echo "  1. Update swift/package.swift with the new checksum '${CHECKSUM}' and url 'https://github.com/$REPO/releases/download/v$VERSION/$ZIP_NAME'"
echo "  2. Commit and push: git add package.swift && git commit -m 'Release v$VERSION' && git push"
echo "  3. Tag the repo: git tag v$VERSION && git push --tags"
