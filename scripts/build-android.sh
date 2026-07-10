#!/bin/bash -ex

# Builds the Android AAR. By default this only assembles the release artifact
# locally, which is handy for testing build configuration changes (e.g. minSdk /
# compileSdk) without pushing anything to Maven Central.
#
# Modes:
#   (none)    build the release AAR only (assembleRelease)
#   local     build and install the AAR into the local Maven cache (~/.m2) via
#             publishToMavenLocal, so examples can consume it with mavenLocal().
#             See scripts/test-examples.sh --local-library.
#   publish   build and publish/release to Maven Central (the old
#             publish-android.sh behavior).
#
# Usage:
#   ./scripts/build-android.sh            # build the AAR only
#   ./scripts/build-android.sh local      # build and install to ~/.m2
#   ./scripts/build-android.sh publish    # build and publish to Maven Central

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

cd ${REPO_ROOT_DIR}

case "$1" in
publish)
    ./gradlew -Pandroid.useAndroidX=true publishAndReleaseToMavenCentral
    echo "Android published to Maven Central"
    ;;
local)
    ./gradlew -Pandroid.useAndroidX=true publishToMavenLocal
    echo "Android AAR installed to local Maven cache (~/.m2)"
    ;;
"")
    ./gradlew -Pandroid.useAndroidX=true assembleRelease
    echo "Android AAR built (pass 'local' to install to ~/.m2, 'publish' to release to Maven Central)"
    ;;
*)
    echo "Unknown argument: $1 (expected one of: <none>, local, publish)" >&2
    exit 1
    ;;
esac
