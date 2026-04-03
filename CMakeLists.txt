cmake_minimum_required(VERSION 3.22.1)

# Top-level entry: C++ core (libmoonshine, moonshine-tts, tools, tests).
project(moonshine_monorepo LANGUAGES CXX)

include(CTest)

add_subdirectory(core)
