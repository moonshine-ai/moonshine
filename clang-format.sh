#!/usr/bin/env bash
set -euo pipefail

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}

find . \
  -path './.env' -prune -o \
  -path './cpp/build' -prune -o \
  -type f \( \
    -name '*.c' -o \
    -name '*.cc' -o \
    -name '*.cpp' -o \
    -name '*.h' -o \
    -name '*.hpp' \
  \) -print \
| while IFS= read -r file; do
    echo "clang-format: $file"
    "$CLANG_FORMAT" -i "$file"
  done
