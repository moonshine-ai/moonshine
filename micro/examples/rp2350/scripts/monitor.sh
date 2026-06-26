#!/usr/bin/env bash
# Robust USB CDC monitor for the Pico 2 boot log. Solves four problems
# with using `screen` directly:
#
#   1. `screen` exits immediately if the tty doesn't exist yet -- so if
#      you flash and then type the screen command, you race the Pico's
#      USB enumeration and lose the first batch of output.
#   2. `screen` exits on a brief device drop, which happens any time
#      the firmware is reset (BOOTSEL re-entered, watchdog reboot,
#      power cycle, USB hub glitch, ...).
#   3. By the time you'd notice and re-run screen, the boot banner
#      and per-clip lines have already been printed and discarded.
#   4. `screen` is a full-screen TUI -- you lose the parent shell, can't
#      tee to a logfile easily, and Ctrl-C doesn't exit it (you have to
#      remember Ctrl-A k y).
#
# CRITICAL: on macOS we MUST use /dev/cu.usbmodem*, NOT /dev/tty.usbmodem*.
# The two paths point at the same underlying USB CDC device, but the
# kernel handles open() very differently:
#
#   /dev/tty.usbmodem*  - "modem callin" device. open() blocks until
#                         the device asserts DCD (carrier detect).
#                         Pico TinyUSB CDC never drives the modem
#                         control lines, so this open() hangs FOREVER,
#                         `cat` reads zero bytes, the log file stays
#                         empty, and the script looks completely
#                         broken even though the firmware is happily
#                         printing. This bit us hard once already.
#   /dev/cu.usbmodem*   - "callout" device. open() returns immediately
#                         regardless of modem control state. This is
#                         what `screen`, `minicom`, and every other
#                         macOS serial monitor actually use.
#
# After picking the right device node, we still have to drop the kernel
# line discipline into raw mode (`stty -f $TTY raw -echo ...`) because
# the default cooked mode otherwise mangles non-ASCII bytes (NL/CR
# translation, ICANON line buffering, ECHO, ...) before `cat` reads
# them.
#
# This script:
#   - Polls for /dev/cu.usbmodem* until it appears (or honors an
#     explicit path passed on the command line).
#   - Switches the tty into raw mode with `stty -f` (BSD/macOS form).
#   - Attaches via `cat`, which now passes raw bytes straight through.
#   - Re-attaches automatically on EOF / read errors so a Pico reset
#     doesn't kill the capture session.
#   - Tees everything to a logfile (default: ./pico_monitor.log) so
#     you can re-read the full session after Ctrl-C.
#
# Recommended workflow:
#   Terminal A:  moonshine-micro/examples/rp2350/scripts/monitor.sh  # leave running
#   Terminal B:  moonshine-micro/examples/rp2350/scripts/flash.sh    # flash whenever
#
# That way the monitor is already attached when the Pico finishes
# booting, and the firmware's 30 s "wait for host" loop is overkill
# but harmless (it just falls through as soon as Terminal A's cat
# asserts DTR).
#
# Usage:
#   moonshine-micro/examples/rp2350/scripts/monitor.sh   # auto-detect /dev/cu.usbmodem*
#   moonshine-micro/examples/rp2350/scripts/monitor.sh /dev/cu.usbmodem1101
#   LOG=/tmp/my.log moonshine-micro/examples/rp2350/scripts/monitor.sh  # custom log
#   LOG= moonshine-micro/examples/rp2350/scripts/monitor.sh             # no logfile
#
# (If you pass /dev/tty.usbmodem* explicitly, you will see an empty log
# and the script will appear to hang -- macOS blocks open() on the tty
# device until carrier is asserted. Pass /dev/cu.usbmodem* instead.)

set -u

# Default timeout bumped from 60s to 300s. The previous default was too
# tight against flash.sh, which can spend 40+ seconds copying the UF2
# to /Volumes/RP2350 (slow USB MSC writes on the RP2350 bootloader) and
# then a few more seconds rebooting -- so monitor.sh would error out
# before the new firmware ever enumerated as a CDC device. Set
# WAIT_TIMEOUT_S=N in the environment to override.
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-300}"
LOG="${LOG-pico_monitor.log}"       # use `LOG=` (empty) to disable

TTY="${1:-}"

# Helper: emit a "still waiting..." line every PROGRESS_EVERY_S seconds
# while polling for a device. Without this, monitor.sh looks hung when
# the user is staring at it -- which is exactly the symptom that
# pushed us to debug "the reconnect logic doesn't work" the first
# time around.
PROGRESS_EVERY_S=10

_wait_for_any_usbmodem() {
  local label="$1"
  local waited=0
  while true; do
    # /dev/cu.* (callout), NOT /dev/tty.* (callin). See header comment:
    # open() on /dev/tty.usbmodem* blocks forever on macOS waiting for
    # DCD that the Pico's TinyUSB stack never asserts.
    for candidate in /dev/cu.usbmodem*; do
      if [[ -c "$candidate" ]]; then
        TTY="$candidate"
        return 0
      fi
    done
    if (( waited >= WAIT_TIMEOUT_S * 5 )); then
      return 1
    fi
    if (( waited > 0 && waited % (PROGRESS_EVERY_S * 5) == 0 )); then
      echo "Monitor: $label (waiting $((waited / 5))s / ${WAIT_TIMEOUT_S}s)..."
    fi
    sleep 0.2
    waited=$((waited + 1))
  done
}

_wait_for_specific_tty() {
  local path="$1"
  local label="$2"
  local waited=0
  while [[ ! -c "$path" ]]; do
    if (( waited >= WAIT_TIMEOUT_S * 5 )); then
      return 1
    fi
    if (( waited > 0 && waited % (PROGRESS_EVERY_S * 5) == 0 )); then
      echo "Monitor: $label (waiting $((waited / 5))s / ${WAIT_TIMEOUT_S}s)..."
    fi
    sleep 0.2
    waited=$((waited + 1))
  done
  return 0
}

# 1. Locate / wait for the tty.
if [[ -n "$TTY" ]]; then
  echo "Monitor: waiting for $TTY (up to ${WAIT_TIMEOUT_S}s)..."
  if ! _wait_for_specific_tty "$TTY" "still waiting for $TTY"; then
    echo "ERROR: $TTY did not appear within ${WAIT_TIMEOUT_S}s" >&2
    exit 2
  fi
else
  echo "Monitor: waiting for /dev/cu.usbmodem* (up to ${WAIT_TIMEOUT_S}s)..."
  if ! _wait_for_any_usbmodem "still waiting for /dev/cu.usbmodem*"; then
    echo "ERROR: no /dev/cu.usbmodem* appeared within ${WAIT_TIMEOUT_S}s" >&2
    echo "       Is the Pico plugged in (not in BOOTSEL mode)?" >&2
    exit 2
  fi
fi
echo "Monitor: attached to $TTY${LOG:+  (logging to $LOG)}"
echo "         Ctrl-C to stop."

# 2. Read loop. `cat` exits on EOF whenever the device drops (firmware
#    reset, USB unplug, etc.). We just reopen and keep going.
#
#    The `</dev/null` redirect on cat prevents it from grabbing stdin,
#    so Ctrl-C reaches the shell instead of being eaten by cat.
#    The 2>/dev/null hides the brief "Input/output error" cat prints
#    when the device disappears -- we surface our own line below.
trap 'echo; echo "Monitor: stopping."; exit 0' INT TERM

if [[ -n "$LOG" ]]; then
  : > "$LOG"        # truncate at session start so old logs don't bleed in
fi

FIRST_ATTACH=1
while true; do
  if [[ -c "$TTY" ]]; then
    if (( ! FIRST_ATTACH )); then
      # Always announce reattachment, even when the kernel happens to
      # hand back the same /dev/cu.usbmodem suffix. The original
      # script stayed silent in the same-path case, which looked
      # identical to the "monitor is hung" failure mode and was
      # exactly the symptom that made us think reconnect was broken.
      echo "Monitor: reattached to $TTY. NOTE: any boot output emitted"
      echo "         before this line was sent into the void -- if the"
      echo "         banner is missing, press BOOTSEL+RESET on the Pico"
      echo "         to re-run the firmware from the top."
    fi
    FIRST_ATTACH=0
    # Put the tty into raw mode. Without this, macOS's default cooked /
    # canonical line discipline silently swallows the Pico's CDC bytes
    # and `cat` reads zero. `stty -f` is the BSD-form to act on a
    # specific device path (Linux would use `stty -F`). The flags below
    # disable input processing, echo, special-character handling, and
    # any baud-rate guessing; the CDC virtual UART ignores baud anyway
    # but the call is required to clear all the cooked-mode bits.
    stty -f "$TTY" raw -echo -echoe -echok -echoctl -echoke \
      -ixon -ixoff -ixany 115200 2>/dev/null || true
    if [[ -n "$LOG" ]]; then
      cat "$TTY" </dev/null 2>/dev/null | tee -a "$LOG"
    else
      cat "$TTY" </dev/null 2>/dev/null
    fi
    echo
    echo "Monitor: device dropped, waiting for it to come back..."
  fi
  # Poll for the same path coming back. If the user explicitly passed
  # a tty, stick with it; otherwise re-scan in case the kernel
  # assigned a different /dev/cu.usbmodem suffix on re-enumeration.
  if [[ -n "${1:-}" ]]; then
    if ! _wait_for_specific_tty "$TTY" "still waiting for $TTY to come back"; then
      echo "ERROR: $TTY did not come back within ${WAIT_TIMEOUT_S}s" >&2
      exit 2
    fi
  else
    if ! _wait_for_any_usbmodem "still waiting for /dev/cu.usbmodem* to come back"; then
      echo "ERROR: no /dev/cu.usbmodem* came back within ${WAIT_TIMEOUT_S}s" >&2
      echo "       Is the Pico plugged in (not in BOOTSEL mode)?" >&2
      exit 2
    fi
    # _wait_for_any_usbmodem updates $TTY in place; surface a one-line
    # heads-up if it landed on a different suffix than we had before.
    # (The full "reattached" banner above runs on every iteration; this
    # one is just so the user can see that the kernel re-numbered the
    # device, which on macOS happens whenever stdio_init_all runs twice.)
  fi
done
