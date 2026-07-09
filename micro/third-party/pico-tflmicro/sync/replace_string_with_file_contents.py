#!/usr/bin/env python

import sys

if len(sys.argv) < 4 or (len(sys.argv) % 2 != 0):
    print(f"Usage: {sys.argv[0]} <templace file> <list file> <replacement pattern> [<file> <pattern> ...]")
    sys.exit(1)

replacement_contents_list = []
replacement_patterns_list = []
for i in range(2, len(sys.argv), 2):
    with open(sys.argv[i]) as f:
        replacement_contents_list.append(f.read())
    replacement_patterns_list.append(sys.argv[i + 1])

with open(sys.argv[1], "rt") as fin:
    for line in fin:
        for i in range(len(replacement_contents_list)):
            replacement_contents = replacement_contents_list[i]
            replacement_pattern = replacement_patterns_list[i]
            line = line.replace(replacement_pattern, replacement_contents)
        sys.stdout.write(line)
