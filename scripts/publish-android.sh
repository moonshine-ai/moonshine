#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

cd ${REPO_ROOT_DIR}

./gradlew publishAllPublicationsToMavenCentralRepository

echo "Android published"