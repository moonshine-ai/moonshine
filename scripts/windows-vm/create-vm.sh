#!/bin/bash
# Creates the Windows KVM guest that runs the windows stage of
# build-all-platforms.sh. Run this on the Linux build host (PC1), not the Mac.
#
# The install is hands-off: this builds a small ISO holding autounattend.xml and
# bootstrap.ps1, hands it to Setup alongside the Windows media, and Setup answers
# its own prompts. When it finishes the guest is reachable over SSH with the key
# baked in here, and provision-toolchain.ps1 takes over from there.
#
# Re-running this destroys and recreates the guest, so it is the recovery path
# when the VM gets into a bad state -- there is nothing to preserve on it that
# is not either in git or reinstallable.
#
# Usage:
#   SSH_PUBKEY="ssh-ed25519 AAAA..." ./create-vm.sh
#
# Environment:
#   SSH_PUBKEY    (required) public key allowed to log into the guest
#   VM_PASSWORD   (required) password for the guest's administrator account
#   WIN_ISO       Windows installation media (default ~/isos/win11-ent-eval-x64.iso)
#   VM_NAME       libvirt domain name (default moonshine-win)
#   VM_USER       guest account name; must match WINDOWS_CLOUD_USER (default pete)
#   VM_VCPUS      (default 8)
#   VM_RAM_MB     (default 16384)
#   VM_DISK_GB    (default 200)

set -euo pipefail

VM_NAME="${VM_NAME:-moonshine-win}"
VM_USER="${VM_USER:-pete}"
VM_VCPUS="${VM_VCPUS:-8}"
VM_RAM_MB="${VM_RAM_MB:-16384}"
VM_DISK_GB="${VM_DISK_GB:-200}"
WIN_ISO="${WIN_ISO:-${HOME}/isos/win11-ent-eval-x64.iso}"
COMPUTER_NAME="${COMPUTER_NAME:-MOONSHINE-WIN}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR=/var/lib/libvirt/images

if [ -z "${SSH_PUBKEY:-}" ]; then
    echo "SSH_PUBKEY is required (the key that will log into the guest)." >&2
    exit 1
fi
if [ -z "${VM_PASSWORD:-}" ]; then
    echo "VM_PASSWORD is required (guest administrator password)." >&2
    exit 1
fi
if [ ! -f "${WIN_ISO}" ]; then
    echo "Windows media not found at ${WIN_ISO}." >&2
    exit 1
fi

echo "== Removing any previous ${VM_NAME} =="
if sudo virsh dominfo "${VM_NAME}" >/dev/null 2>&1; then
    sudo virsh destroy "${VM_NAME}" >/dev/null 2>&1 || true
    # --nvram also discards the UEFI variable store, so the rebuilt guest boots
    # from the install media instead of a stale boot entry.
    sudo virsh undefine "${VM_NAME}" --nvram --remove-all-storage >/dev/null 2>&1 || true
fi

echo "== Building the answer-file ISO =="
# libvirt's qemu runs as its own user and cannot read $HOME, so everything the
# guest needs lives under the libvirt image directory.
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT

sed -e "s|@@USERNAME@@|${VM_USER}|g" \
    -e "s|@@PASSWORD@@|${VM_PASSWORD}|g" \
    -e "s|@@COMPUTERNAME@@|${COMPUTER_NAME}|g" \
    "${SCRIPT_DIR}/autounattend.xml.in" > "${STAGE_DIR}/autounattend.xml"
sed -e "s|@@SSH_PUBKEY@@|${SSH_PUBKEY}|g" \
    "${SCRIPT_DIR}/bootstrap.ps1.in" > "${STAGE_DIR}/bootstrap.ps1"

# A malformed answer file is not reported by Setup: it silently reverts to
# prompting, which from the outside is indistinguishable from the file never
# being found, and costs an hour to tell apart. Catch it here instead.
if ! command -v xmllint >/dev/null; then
    echo "xmllint is required to validate the answer file (apt install libxml2-utils)." >&2
    exit 1
fi
xmllint --noout "${STAGE_DIR}/autounattend.xml"

UNATTEND_ISO="${IMAGE_DIR}/${VM_NAME}-unattend.iso"
xorriso -as mkisofs -quiet -J -r -V UNATTEND \
    -o "${STAGE_DIR}/unattend.iso" "${STAGE_DIR}/autounattend.xml" \
    "${STAGE_DIR}/bootstrap.ps1"
sudo install -m 0644 "${STAGE_DIR}/unattend.iso" "${UNATTEND_ISO}"

WIN_ISO_DEST="${IMAGE_DIR}/$(basename "${WIN_ISO}")"
if [ ! -f "${WIN_ISO_DEST}" ]; then
    echo "== Staging the Windows media where libvirt can read it =="
    sudo install -m 0644 "${WIN_ISO}" "${WIN_ISO_DEST}"
fi

echo "== Defining and starting ${VM_NAME} =="
# Plain UEFI without Secure Boot, and a SATA disk with an e1000e NIC rather than
# virtio: Windows Setup has inbox drivers for those, so the install needs no
# driver injection. The tradeoff is some I/O throughput, which barely shows up
# in a build that is dominated by MSVC compile time.
#
# The CPU topology is spelled out because QEMU otherwise presents each vCPU as
# its own socket, and Windows client editions are licensed for at most two --
# so the guest silently ignores most of them and builds at a fraction of the
# speed. One socket with every core under it is what makes them all usable.
sudo virt-install \
    --name "${VM_NAME}" \
    --osinfo win11 \
    --vcpus "${VM_VCPUS}" \
    --cpu "host-passthrough,topology.sockets=1,topology.cores=${VM_VCPUS},topology.threads=1" \
    --memory "${VM_RAM_MB}" \
    --disk "path=${IMAGE_DIR}/${VM_NAME}.qcow2,size=${VM_DISK_GB},format=qcow2,bus=sata,cache=writeback,discard=unmap" \
    --disk "path=${WIN_ISO_DEST},device=cdrom,bus=sata,readonly=on" \
    --disk "path=${UNATTEND_ISO},device=cdrom,bus=sata,readonly=on" \
    --network network=default,model=e1000e \
    --tpm backend.type=emulator,backend.version=2.0,model=tpm-crb \
    --boot "cdrom,hd,loader=/usr/share/OVMF/OVMF_CODE_4M.fd,loader.readonly=yes,loader.type=pflash,loader.secure=no,nvram.template=/usr/share/OVMF/OVMF_VARS_4M.fd" \
    --graphics vnc,listen=127.0.0.1 \
    --video vga \
    --noautoconsole

# The Windows media stops at "Press any key to boot from CD or DVD" and falls
# through to a blank disk if nobody answers, which strands the guest in the UEFI
# shell. Nothing is listening for the keystroke on a headless box, so send it.
echo "== Answering the boot prompt =="
for _ in $(seq 1 20); do
    sudo virsh send-key "${VM_NAME}" --codeset linux KEY_ENTER >/dev/null 2>&1 || true
    sleep 2
done

# Do not bring the guest back on host reboot. build-all-platforms.sh starts it
# for the windows stage and shuts it down afterwards; leaving autostart on would
# burn 8 host cores whenever the Linux box reboots.
sudo virsh autostart --disable "${VM_NAME}" >/dev/null

echo
echo "${VM_NAME} is installing. Expect roughly 15-25 minutes unattended."
echo "Watch it with: ${SCRIPT_DIR}/screenshot.sh"
echo "Autostart is disabled; start it with: sudo virsh start ${VM_NAME}"
