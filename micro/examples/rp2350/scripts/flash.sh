#!/usr/bin/env bash
# Flash a moonshine-micro moonshine_micro_echo*.uf2 to a Raspberry Pi Pico 2
# (RP2350) that's been put into BOOTSEL mode.
#
# Usage:
#   moonshine-micro/examples/rp2350/scripts/flash.sh <variant> [path/to/other.uf2]
#
#   <variant> selects which firmware to flash (required):
#     test      moonshine_micro_echo_test.uf2  (embedded-clip accuracy sweep)
#     echo      moonshine_micro_echo.uf2       (live mic/speaker echo service)
#     wifi      moonshine_micro_echo_wifi.uf2  (voice WiFi setup; pico2_w only)
#
#   An optional second argument overrides the .uf2 path entirely.
#
# Procedure:
#   1. Hold BOOTSEL on the Pico 2 while plugging it into USB.
#   2. The board mounts as /Volumes/RP2350 (or /Volumes/RPI-RP2 on some
#      boards, e.g. the Pico 2 W) on macOS.
#   3. Run this script. It waits for either volume (up to MOUNT_TIMEOUT_S
#      seconds), copies the .uf2 over, then waits for the volume to
#      vanish -- which is how macOS surfaces "the Pico rebooted into
#      your firmware and dropped off the bus".
#
# Notes:
#   - We wait for /Volumes/RP2350 or /Volumes/RPI-RP2, whichever appears
#     first. The RP2350 bootrom usually labels its mass-storage volume
#     RP2350, but some boards (notably the Pico 2 W) enumerate with the
#     older RPI-RP2 label. To force a single explicit path, override with:
#     VOLUME=/Volumes/SomeLabel ./flash.sh <variant>
#   - macOS sometimes reports "Resource busy" right after `cp` returns,
#     because the Pico is already rebooting and tearing the mount down
#     mid-copy. That's expected and harmless -- the file already landed
#     in flash before the disconnect. We treat any cp non-zero as an
#     error UNLESS the volume disappears within DISMOUNT_GRACE_S.

set -u  # not -e: we want to inspect cp's exit code ourselves.

MOUNT_TIMEOUT_S="${MOUNT_TIMEOUT_S:-30}"
DISMOUNT_GRACE_S="${DISMOUNT_GRACE_S:-10}"

# Candidate BOOTSEL mount points, tried in order. A single explicit override
# via VOLUME=/Volumes/Foo wins outright; otherwise we accept either the RP2350
# label or the older RPI-RP2 label (which the Pico 2 W tends to use).
if [[ -n "${VOLUME:-}" ]]; then
  VOLUME_CANDIDATES=("$VOLUME")
else
  VOLUME_CANDIDATES=(/Volumes/RP2350 /Volumes/RPI-RP2)
fi

# Echo the first currently-mounted candidate (empty if none are mounted yet).
find_mounted_volume() {
  local v
  for v in "${VOLUME_CANDIDATES[@]}"; do
    if [[ -d "$v" ]]; then
      printf '%s' "$v"
      return 0
    fi
  done
  return 1
}

# Require a variant argument and map it to the firmware artifact name.
usage() {
  cat >&2 <<'EOF'
Usage:
  moonshine-micro/examples/rp2350/scripts/flash.sh <variant> [path/to/other.uf2]

Where <variant> is one of:
  test      moonshine_micro_echo_test.uf2  (embedded-clip accuracy sweep)
  echo      moonshine_micro_echo.uf2       (live mic/speaker echo service)
  wifi      moonshine_micro_echo_wifi.uf2  (voice WiFi setup; pico2_w only)

Example:
  moonshine-micro/examples/rp2350/scripts/flash.sh echo
EOF
}

VARIANT="${1:-}"
if [[ -z "$VARIANT" ]]; then
  echo "ERROR: missing required <variant> argument." >&2
  echo "" >&2
  usage
  exit 64  # EX_USAGE
fi

case "$VARIANT" in
  test) UF2_NAME="moonshine_micro_echo_test.uf2" ;;
  echo) UF2_NAME="moonshine_micro_echo.uf2" ;;
  wifi) UF2_NAME="moonshine_micro_echo_wifi.uf2" ;;
  *)
    echo "ERROR: unknown variant '$VARIANT' (expected: test, echo, or wifi)." >&2
    echo "" >&2
    usage
    exit 64  # EX_USAGE
    ;;
esac

# Resolve the UF2 path. The firmware lands under the example's target dir of the
# top-level build tree: moonshine-micro/build/examples/rp2350/<UF2_NAME>.
# SCRIPT_DIR = moonshine-micro/examples/rp2350/scripts, so ../../../build/examples/rp2350.
# An optional second argument overrides the resolved path entirely.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_UF2="$SCRIPT_DIR/../../../build/examples/rp2350/$UF2_NAME"
UF2="${2:-$DEFAULT_UF2}"

if [[ ! -f "$UF2" ]]; then
  echo "ERROR: .uf2 not found at: $UF2" >&2
  echo "       Build it first with:" >&2
  echo "         cd moonshine-micro && cmake -B build -S . && cmake --build build -j" >&2
  exit 1
fi

UF2_SIZE_HR="$(du -h "$UF2" | awk '{print $1}')"
echo "UF2:    $UF2  ($UF2_SIZE_HR)"
echo "Volume: ${VOLUME_CANDIDATES[*]}"

# 1. Wait for BOOTSEL mount. Accept whichever candidate appears first, then
#    pin VOLUME to that single path for the rest of the script.
VOLUME="$(find_mounted_volume || true)"
if [[ -z "$VOLUME" ]]; then
  echo -n "Waiting for ${VOLUME_CANDIDATES[*]} (hold BOOTSEL on the Pico 2 and plug in USB)..."
  WAITED=0
  while :; do
    VOLUME="$(find_mounted_volume || true)"
    [[ -n "$VOLUME" ]] && break
    if (( WAITED >= MOUNT_TIMEOUT_S )); then
      echo ""
      echo "ERROR: none of [${VOLUME_CANDIDATES[*]}] appeared within ${MOUNT_TIMEOUT_S}s." >&2
      echo "       Make sure you're holding BOOTSEL while connecting." >&2
      exit 2
    fi
    sleep 1
    echo -n "."
    WAITED=$((WAITED + 1))
  done
  echo " mounted ($VOLUME)."
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
# -X: don't copy macOS extended attributes / resource forks. The BOOTSEL
# volume is FAT and rejects xattrs ("could not copy extended attributes ...
# Operation not permitted"), which made cp exit non-zero AFTER the .uf2 data
# had already streamed across -- a spurious failure on RPI-RP2 mounts.
cp -X "$UF2" "$DEST" 2>/tmp/flash_cp_err.$$ &
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
echo "  moonshine-micro/examples/rp2350/scripts/monitor.sh"
echo ""
echo "Alternative (one-shot, won't see boot if started after this point):"
echo "  ls /dev/cu.usbmodem*"
echo "  screen /dev/cu.usbmodem<TAB> 115200    # Ctrl-A K to quit"
echo ""
echo "  (NOTE: use /dev/cu.* on macOS, never /dev/tty.*. open() on the tty"
echo "   variant blocks forever waiting for DCD that the Pico doesn't drive.)"
