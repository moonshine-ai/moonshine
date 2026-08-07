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

# SPM records a trust-on-first-use fingerprint per version. Moving/retagging
# vX.Y.Z to a new commit makes later xcodebuild resolves fail with
# "does not match previously recorded value" on this machine (where
# publish-examples runs next). Drop the cached fingerprint + binary artifact.
clear_spm_caches_for_retagged_swift_package() {
	local version="$1"
	local fp_dir="${HOME}/Library/org.swift.swiftpm/security/fingerprints"
	local f
	if [[ -d "${fp_dir}" ]]; then
		for f in "${fp_dir}"/moonshine-swift-*.json; do
			[[ -f "${f}" ]] || continue
			python3 - "${f}" "${version}" <<'PY'
import json, sys
path, version = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
vfs = data.get("versionFingerprints") or {}
if version in vfs:
    del vfs[version]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"cleared SPM fingerprint for {version} in {path}")
PY
		done
	fi
	local artifacts="${HOME}/Library/Caches/org.swift.swiftpm/artifacts"
	if [[ -d "${artifacts}" ]]; then
		find "${artifacts}" -maxdepth 1 -iname "*moonshine*swift*v${version}*" -exec rm -rf {} +
		find "${artifacts}" -maxdepth 1 -iname "*moonshine_swift*v${version}*" -exec rm -rf {} +
	fi
	# Stale checkouts under DerivedData also pin the old tag SHA.
	find "${HOME}/Library/Developer/Xcode/DerivedData" \
		-type d -path '*/SourcePackages/checkouts/moonshine-swift' \
		-prune -exec rm -rf {} + 2>/dev/null || true
}

clear_spm_caches_for_retagged_swift_package "${VERSION}"
