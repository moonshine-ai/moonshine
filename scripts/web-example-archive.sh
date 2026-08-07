#!/usr/bin/env bash
# Stage or pack a self-contained web example archive.
#
# Each archive is meant to extract and run with nothing else from the repo:
#
#   tar xzf web-stt.tar.gz
#   cd web-stt   # or whatever directory you extracted into
#   node serve.mjs
#   # open http://localhost:8080/stt/
#
# Contents:
#   serve.mjs, assets/, <demo>/, mic-check/ (linked from the demos), README.md,
#   and a tiny index.html that redirects to the demo.
#
# Usage:
#   scripts/web-example-archive.sh list
#   scripts/web-example-archive.sh stage <demo> <dest_dir>
#   scripts/web-example-archive.sh pack  <demo> <tar_path>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEB_ROOT="${REPO_ROOT}/examples/web"

# The five demos shipped on GitHub Releases. Not assets/, mic-check/, or the
# landing page — those are bundled into each demo archive as needed.
WEB_DEMOS=(stt tts agent-flow dictation meeting-notes)

die() {
	echo "[web-example-archive] ERROR: $*" >&2
	exit 1
}

list_demos() {
	printf '%s\n' "${WEB_DEMOS[@]}"
}

is_known_demo() {
	local demo="$1"
	local name
	for name in "${WEB_DEMOS[@]}"; do
		[[ "${name}" == "${demo}" ]] && return 0
	done
	return 1
}

# Fill dest_dir with a runnable tree for one demo. dest_dir is replaced.
stage_demo() {
	local demo="$1"
	local dest="$2"
	is_known_demo "${demo}" || die "unknown web demo '${demo}' (want one of: ${WEB_DEMOS[*]})"
	[[ -d "${WEB_ROOT}/${demo}" ]] || die "missing ${WEB_ROOT}/${demo}"
	[[ -d "${WEB_ROOT}/assets" ]] || die "missing ${WEB_ROOT}/assets"
	[[ -f "${WEB_ROOT}/serve.mjs" ]] || die "missing ${WEB_ROOT}/serve.mjs"

	rm -rf "${dest}"
	mkdir -p "${dest}"

	cp -a "${WEB_ROOT}/serve.mjs" "${dest}/serve.mjs"
	cp -a "${WEB_ROOT}/assets" "${dest}/assets"
	cp -a "${WEB_ROOT}/${demo}" "${dest}/${demo}"
	# Several demos link to /mic-check/ for the microphone diagnostic.
	if [[ -d "${WEB_ROOT}/mic-check" ]]; then
		cp -a "${WEB_ROOT}/mic-check" "${dest}/mic-check"
	fi

	cat >"${dest}/index.html" <<EOF
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=/${demo}/" />
    <title>Moonshine Voice — ${demo}</title>
    <link rel="canonical" href="/${demo}/" />
  </head>
  <body>
    <p>Open the <a href="/${demo}/">${demo}</a> example.</p>
  </body>
</html>
EOF

	cat >"${dest}/README.md" <<EOF
# Moonshine Voice — ${demo}

Self-contained web example. Requires [Node.js](https://nodejs.org/) 18+.

\`\`\`bash
node serve.mjs
\`\`\`

Then open http://localhost:8080/${demo}/

The page loads \`@moonshine-ai/moonshine-wasm\` from jsDelivr and downloads
models on first use (cached afterwards). No API key or account is required.

\`serve.mjs\` sets the Cross-Origin Isolation headers browsers need for
SharedArrayBuffer; a plain static file server will not work.
EOF
}

# Build web-<demo>.tar.gz at tar_path. The archive's top-level directory is the
# demo name so extracting yields ./<demo>/serve.mjs rather than scattering files
# into the current directory — matching android-Transcriber.tar.gz → Transcriber/.
pack_demo() {
	local demo="$1"
	local tar_path="$2"
	local stage parent base
	parent="$(dirname "${tar_path}")"
	mkdir -p "${parent}"
	stage="$(mktemp -d "${TMPDIR:-/tmp}/moonshine-web-${demo}.XXXXXX")"
	# shellcheck disable=SC2064
	trap "rm -rf '${stage}'" RETURN
	stage_demo "${demo}" "${stage}/${demo}"
	rm -f "${tar_path}"
	tar -czf "${tar_path}" -C "${stage}" "${demo}"
	base="$(basename "${tar_path}")"
	echo "[web-example-archive] packed ${base} ($(du -h "${tar_path}" | awk '{print $1}'))"
}

usage() {
	sed -n '2,20p' "$0" | sed 's/^# \?//'
}

case "${1:-}" in
list) list_demos ;;
stage)
	[[ $# -eq 3 ]] || die "usage: $0 stage <demo> <dest_dir>"
	stage_demo "$2" "$3"
	;;
pack)
	[[ $# -eq 3 ]] || die "usage: $0 pack <demo> <tar_path>"
	pack_demo "$2" "$3"
	;;
-h | --help | "") usage; exit 0 ;;
*) die "unknown command '${1}' (try --help)" ;;
esac
