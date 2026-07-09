# TensorFlow Lite Micro

An Open Source Machine Learning Framework for Everyone.

## Introduction

This is a version of the [TensorFlow Lite Micro library](https://www.tensorflow.org/lite/microcontrollers)
for the Raspberry Pi Pico microcontroller. It allows you to run machine 
learning models to do things like voice recognition, detect people in images,
recognize gestures from an accelerometer, and other sensor analysis tasks.
This version has scripts to upstream changes from the Google codebase. It also
takes advantage of the RP2040's dual cores for increased speed on some 
operations.

## Getting Started

First you'll need to follow the Pico setup instructions to initialize the
development environment on your machine. Once that is done, make sure that the
`PICO_SDK_PATH` environment variable has been set to the location of the Pico
SDK, either in the shell you're building in, or the CMake configure environment
variable setting of the extension if you're using VS Code.

You should then be able to build the library, tests, and examples. The easiest 
way to build is using VS Code's CMake integration, by loading the project and
choosing the build option at the bottom of the window.

Alternatively you can build the entire project, including tests, by running the
following commands from a terminal once you're in this repo's directory:

```bash
mkdir build
cd build
cmake ..
make
```

## What's Included

There are several example applications included. The simplest one to begin with
is the hello_world project. This demonstrates the fundamentals of deploying an 
ML model on a device, driving the Pico's LED in a learned sine-wave pattern.
Once you have built the project, a UF2 file you can copy to the Pico should be
present at `build/examples/hello_world/hello_world.uf2`.

Another example is the person detector, but since the Pico doesn't come with
image inputs you'll need to write some code to hook up your own sensor. You can
find a fork of TFLM for the Arducam Pico4ML that does this at [arducam.com/pico4ml-an-rp2040-based-platform-for-tiny-machine-learning/](https://www.arducam.com/pico4ml-an-rp2040-based-platform-for-tiny-machine-learning/).

## Contributing

This repository (https://github.com/raspberrypi/pico-tflmicro) is read-only,
because it has been automatically generated from the master TensorFlow 
repository at https://github.com/tensorflow/tensorflow. It's maintained by
@petewarden on a best effort basis, so bugs and PRs may not get addressed. You
can generate an updated version of this generated project by running the command:

```
sync/sync_with_upstream.sh
```

This should create a Pico-compatible project from the latest version of the
TensorFlow repository.

## Learning More

The [TensorFlow website](https://www.tensorflow.org/lite/microcontrollers) has
information on training, tutorials, and other resources.

The [TinyML Book](https://tinymlbook.com) is a guide to using TensorFlow Lite Micro
across a variety of different systems.

[TensorFlowLite Micro: Embedded Machine Learning on TinyML Systems](https://arxiv.org/pdf/2010.08678.pdf)
has more details on the design and implementation of the framework.

## Licensing

The TensorFlow source code is covered by the Apache 2 license described in 
src/tensorflow/LICENSE, components from other libraries have the appropriate
licenses included in their third_party folders.