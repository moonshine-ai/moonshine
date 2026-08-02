#!/bin/bash
# Captures the Windows guest's screen to a PNG. Run on the Linux build host.
#
# The guest is headless and, until sshd comes up, completely mute -- this is the
# only way to see what an unattended install is doing. A Setup dialog sitting on
# screen means autounattend.xml left one of Setup's questions unanswered.
#
# Usage: ./screenshot.sh [output.png]

set -euo pipefail

VM_NAME="${VM_NAME:-moonshine-win}"
OUT="${1:-/tmp/${VM_NAME}-screen.png}"
RAW="$(mktemp)"
trap 'rm -f "${RAW}"' EXIT

sudo virsh screenshot "${VM_NAME}" "${RAW}" >/dev/null
# virsh writes the file as root.
sudo chmod a+r "${RAW}"

# Which format comes back depends on the guest's video device: libvirt hands
# back PNG for most of them but still PPM for some, so convert only if needed.
if [ "$(head -c 4 "${RAW}" | od -An -tx1 | tr -d ' \n')" = "89504e47" ]; then
    cp "${RAW}" "${OUT}"
else
    pnmtopng "${RAW}" > "${OUT}"
fi

echo "${OUT}"
