#!/usr/bin/env bash
# Flash the cpp-tiny predict_spelling_tiny.uf2 to a Raspberry Pi Pico 2
# (RP2350) that's been put into BOOTSEL mode.
#
# Usage:
#   cpp-tiny/example-rp2350/scripts/flash.sh                   # default UF2
#   cpp-tiny/example-rp2350/scripts/flash.sh path/to/other.uf2 # override path
#
# Procedure:
#   1. Hold BOOTSEL on the Pico 2 while plugging it into USB.
#   2. The board mounts as /Volumes/RP2350 on macOS.
#   3. Run this script. It waits for that volume (up to MOUNT_TIMEOUT_S
#      seconds), copies the .uf2 over, then waits for the volume to
#      vanish -- which is how macOS surfaces "the Pico rebooted into
#      your firmware and dropped off the bus".
#
# Notes:
#   - We assume /Volumes/RP2350 per the project's standing convention.
#     If your board mounts with a different label (Pico 1 uses
#     /Volumes/RPI-RP2), override with: VOLUME=/Volumes/RPI-RP2 ./flash.sh
#   - macOS sometimes reports "Resource busy" right after `cp` returns,
#     because the Pico is already rebooting and tearing the mount down
#     mid-copy. That's expected and harmless -- the file already landed
#     in flash before the disconnect. We treat any cp non-zero as an
#     error UNLESS the volume disappears within DISMOUNT_GRACE_S.

set -u  # not -e: we want to inspect cp's exit code ourselves.

VOLUME="${VOLUME:-/Volumes/RP2350}"
MOUNT_TIMEOUT_S="${MOUNT_TIMEOUT_S:-30}"
DISMOUNT_GRACE_S="${DISMOUNT_GRACE_S:-10}"

# Resolve the UF2 path. The firmware lands under the example's target dir of the
# top-level build tree: cpp-tiny/build/example-rp2350/predict_spelling_tiny.uf2.
# SCRIPT_DIR = cpp-tiny/example-rp2350/scripts, so ../../build/example-rp2350.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_UF2="$SCRIPT_DIR/../../build/example-rp2350/predict_spelling_tiny.uf2"
UF2="${1:-$DEFAULT_UF2}"

if [[ ! -f "$UF2" ]]; then
  echo "ERROR: .uf2 not found at: $UF2" >&2
  echo "       Build it first with:" >&2
  echo "         cd cpp-tiny && cmake -B build -S . && cmake --build build -j" >&2
  exit 1
fi

UF2_SIZE_HR="$(du -h "$UF2" | awk '{print $1}')"
echo "UF2:    $UF2  ($UF2_SIZE_HR)"
echo "Volume: $VOLUME"

# 1. Wait for BOOTSEL mount.
if [[ ! -d "$VOLUME" ]]; then
  echo -n "Waiting for $VOLUME (hold BOOTSEL on the Pico 2 and plug in USB)..."
  WAITED=0
  while [[ ! -d "$VOLUME" ]]; do
    if (( WAITED >= MOUNT_TIMEOUT_S )); then
      echo ""
      echo "ERROR: $VOLUME did not appear within ${MOUNT_TIMEOUT_S}s." >&2
      echo "       Make sure you're holding BOOTSEL while connecting." >&2
      exit 2
    fi
    sleep 1
    echo -n "."
    WAITED=$((WAITED + 1))
  done
  echo " mounted."
else
  echo "$VOLUME is already mounted; proceeding."
fi

# 2. Copy with a progress bar. cp may return non-zero because the Pico
#    reboots mid-write (sometimes before cp's final fsync returns). We
#    disambiguate "harmless reboot race" from "real failure" by checking
#    whether the volume disappears shortly after.
#
#    Progress: we run cp in the background and poll the destination file
#    size against the source size. The RP2350 throttles USB mass-storage
#    writes (the bootrom flashes each block before ACKing), so the growing
#    dest size is a faithful progress signal. If the size can't be read
#    mid-write on some mount, we fall back to an elapsed-time counter.
draw_bar() {  # $1 = percent (0..100)
  local pct=$1 width=30 filled i bar=""
  filled=$(( pct * width / 100 ))
  for ((i = 0; i < width; i++)); do
    if (( i < filled )); then bar+="#"; else bar+="-"; fi
  done
  printf "\r  [%s] %3d%%" "$bar" "$pct"
}

echo "Copying..."
SRC_BYTES=$(stat -f%z "$UF2" 2>/dev/null || echo 0)
DEST="$VOLUME/$(basename "$UF2")"
START_NS=$(date +%s)
cp "$UF2" "$DEST" 2>/tmp/flash_cp_err.$$ &
CP_PID=$!

LAST_PCT=-1
while kill -0 "$CP_PID" 2>/dev/null; do
  if (( SRC_BYTES > 0 )); then
    DST_BYTES=$(stat -f%z "$DEST" 2>/dev/null || echo 0)
    PCT=$(( DST_BYTES * 100 / SRC_BYTES ))
    (( PCT > 100 )) && PCT=100
    if (( PCT != LAST_PCT )); then draw_bar "$PCT"; LAST_PCT=$PCT; fi
  else
    printf "\r  copying... %ds" "$(( $(date +%s) - START_NS ))"
  fi
  sleep 0.3
done
wait "$CP_PID"
CP_STATUS=$?
# Snap the bar to 100% on success so it doesn't end at, say, 97%.
if (( SRC_BYTES > 0 && CP_STATUS == 0 )); then draw_bar 100; fi
printf "\n"
ELAPSED=$(( $(date +%s) - START_NS ))

if [[ $CP_STATUS -ne 0 ]]; then
  # Give the device a few seconds to drop the volume -- that's the
  # signal that the firmware took over even though cp's last syscall
  # failed.
  echo "cp exited $CP_STATUS after ${ELAPSED}s; checking whether the Pico rebooted..."
  WAITED=0
  while [[ -d "$VOLUME" && $WAITED -lt $DISMOUNT_GRACE_S ]]; do
    sleep 1
    WAITED=$((WAITED + 1))
  done
  if [[ ! -d "$VOLUME" ]]; then
    echo "Volume disappeared -- treating as success (reboot-during-write race)."
    rm -f "/tmp/flash_cp_err.$$"
    exit 0
  fi
  echo "ERROR: cp failed and the volume is still mounted." >&2
  echo "----- cp stderr -----" >&2
  cat "/tmp/flash_cp_err.$$" >&2 || true
  rm -f "/tmp/flash_cp_err.$$"
  exit 3
fi
rm -f "/tmp/flash_cp_err.$$"

# 3. Wait for the volume to disappear, confirming the Pico rebooted
#    into the new firmware. If it's still mounted after the grace
#    period something weird happened (e.g., we copied to a stale
#    mount), but the file did transfer so we don't hard-fail.
echo "Copy done in ${ELAPSED}s. Waiting for $VOLUME to drop (Pico reboot)..."
WAITED=0
while [[ -d "$VOLUME" && $WAITED -lt $DISMOUNT_GRACE_S ]]; do
  sleep 1
  WAITED=$((WAITED + 1))
done
if [[ -d "$VOLUME" ]]; then
  echo "WARNING: $VOLUME is still mounted after ${DISMOUNT_GRACE_S}s. The"
  echo "         .uf2 was written but the Pico may not have rebooted."
  echo "         Try power-cycling the board."
  exit 0
fi
echo "Done. Pico rebooted into the new firmware."

# 4. Hint at where to look for the boot log. We don't auto-attach so we
#    don't fight any serial monitor the user already has open. The
#    recommended workflow is to leave monitor.sh running in another
#    terminal across BOTH flashes -- it auto-reattaches on device drop
#    and is the only way to capture the firmware's boot banner before
#    the RP2350's USB CDC enumeration settles. Attaching AFTER the
#    boot already happened means cat sees an idle, silent device.
echo ""
echo "To read the boot log (recommended: leave running BEFORE you flash):"
echo "  cpp-tiny/example-rp2350/scripts/monitor.sh"
echo ""
echo "Alternative (one-shot, won't see boot if started after this point):"
echo "  ls /dev/cu.usbmodem*"
echo "  screen /dev/cu.usbmodem<TAB> 115200    # Ctrl-A K to quit"
echo ""
echo "  (NOTE: use /dev/cu.* on macOS, never /dev/tty.*. open() on the tty"
echo "   variant blocks forever waiting for DCD that the Pico doesn't drive.)"
