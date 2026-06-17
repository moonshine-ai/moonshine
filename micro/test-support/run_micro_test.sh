#!/usr/bin/env bash
# Run a TFLM micro_test.h binary and translate its output into an exit code.
#
# micro_test.h wraps the suite in `while (true) { ... }` (it is designed to run
# forever on an embedded target and stream results over serial). On the host we
# read until the first pass/fail banner, then quit -- closing the pipe sends the
# looping binary a SIGPIPE so it terminates. Exit 0 iff all tests passed.
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: run_micro_test.sh <test-binary> [args...]" >&2
  exit 64
fi

# `sed q` quits at the first banner, closing the pipe and stopping the binary.
out="$("$@" 2>&1 | sed -e '/~~~ALL TESTS PASSED~~~/q' -e '/~~~SOME TESTS FAILED~~~/q')"
printf '%s\n' "$out"
printf '%s\n' "$out" | grep -q "~~~ALL TESTS PASSED~~~"
