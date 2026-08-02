#!/bin/bash
# Types text into the Windows guest's console via the hypervisor. Run on the
# Linux build host.
#
# This exists for the window before sshd is up -- during Windows Setup or when
# the guest has broken badly enough that SSH is gone -- where the only way in is
# to act like a keyboard. Pair it with screenshot.sh to read the result.
#
# Usage:
#   ./send-keys.sh "dir e:\\"          # types the string, no Enter
#   ./send-keys.sh --enter "dir e:\\"  # types it and presses Enter
#   ./send-keys.sh --key KEY_F10 --modifier KEY_LEFTSHIFT   # a single chord

set -euo pipefail

VM_NAME="${VM_NAME:-moonshine-win}"
PRESS_ENTER=0
SINGLE_KEY=""
MODIFIER=""

while [ $# -gt 0 ]; do
    case "$1" in
        --enter) PRESS_ENTER=1; shift ;;
        --key) SINGLE_KEY="$2"; shift 2 ;;
        --modifier) MODIFIER="$2"; shift 2 ;;
        *) break ;;
    esac
done

send() {
    sudo virsh send-key "${VM_NAME}" --codeset linux --holdtime 60 "$@" >/dev/null
}

if [ -n "${SINGLE_KEY}" ]; then
    if [ -n "${MODIFIER}" ]; then
        send "${MODIFIER}" "${SINGLE_KEY}"
    else
        send "${SINGLE_KEY}"
    fi
    exit 0
fi

TEXT="${1:-}"

for (( i = 0; i < ${#TEXT}; i++ )); do
    char="${TEXT:i:1}"
    key=""
    shifted=0
    case "${char}" in
        [a-z]) key="KEY_$(echo "${char}" | tr '[:lower:]' '[:upper:]')" ;;
        [A-Z]) key="KEY_${char}"; shifted=1 ;;
        1) key=KEY_1 ;; 2) key=KEY_2 ;; 3) key=KEY_3 ;; 4) key=KEY_4 ;;
        5) key=KEY_5 ;; 6) key=KEY_6 ;; 7) key=KEY_7 ;; 8) key=KEY_8 ;;
        9) key=KEY_9 ;; 0) key=KEY_0 ;;
        ' ') key=KEY_SPACE ;;
        '.') key=KEY_DOT ;;
        ',') key=KEY_COMMA ;;
        '/') key=KEY_SLASH ;;
        '\') key=KEY_BACKSLASH ;;
        '-') key=KEY_MINUS ;;
        '=') key=KEY_EQUAL ;;
        ';') key=KEY_SEMICOLON ;;
        "'") key=KEY_APOSTROPHE ;;
        '[') key=KEY_LEFTBRACE ;;
        ']') key=KEY_RIGHTBRACE ;;
        '`') key=KEY_GRAVE ;;
        ':') key=KEY_SEMICOLON; shifted=1 ;;
        '"') key=KEY_APOSTROPHE; shifted=1 ;;
        '_') key=KEY_MINUS; shifted=1 ;;
        '+') key=KEY_EQUAL; shifted=1 ;;
        '|') key=KEY_BACKSLASH; shifted=1 ;;
        '?') key=KEY_SLASH; shifted=1 ;;
        '<') key=KEY_COMMA; shifted=1 ;;
        '>') key=KEY_DOT; shifted=1 ;;
        '!') key=KEY_1; shifted=1 ;;
        '@') key=KEY_2; shifted=1 ;;
        '#') key=KEY_3; shifted=1 ;;
        '$') key=KEY_4; shifted=1 ;;
        '%') key=KEY_5; shifted=1 ;;
        '^') key=KEY_6; shifted=1 ;;
        '&') key=KEY_7; shifted=1 ;;
        '*') key=KEY_8; shifted=1 ;;
        '(') key=KEY_9; shifted=1 ;;
        ')') key=KEY_0; shifted=1 ;;
        '{') key=KEY_LEFTBRACE; shifted=1 ;;
        '}') key=KEY_RIGHTBRACE; shifted=1 ;;
        '~') key=KEY_GRAVE; shifted=1 ;;
        *) echo "send-keys: no mapping for '${char}'" >&2; exit 1 ;;
    esac

    if [ "${shifted}" -eq 1 ]; then
        send KEY_LEFTSHIFT "${key}"
    else
        send "${key}"
    fi
done

if [ "${PRESS_ENTER}" -eq 1 ]; then
    send KEY_ENTER
fi
