#!/usr/bin/env python

import glob
import os

for folder in glob.glob("tests/*"):
    if not os.path.isdir(folder):
        continue
    name = os.path.basename(folder)

    cmake_path = os.path.join(folder, "CMakeLists.txt")
    with open(cmake_path, 'r') as cmake_file: 
        cmake_lines = cmake_file.read().split("\n")

    has_already_been_updated = False
    new_lines = []
    for cmake_line in cmake_lines:
        if cmake_line.startswith("target_sources"):
            new_lines.append("pico_enable_stdio_usb(" + name + " 1)")
            new_lines.append("pico_enable_stdio_uart(" + name + " 0)")
            new_lines.append("")
        if cmake_line.startswith("pico_enable_stdio"):
            has_already_been_updated = True
            break
        new_lines.append(cmake_line)
    
    if has_already_been_updated:
        continue
    
    with open(cmake_path, 'w') as cmake_file:
        cmake_file.write("\n".join(new_lines)) 
