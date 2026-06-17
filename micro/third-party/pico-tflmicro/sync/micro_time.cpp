/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Raspberry Pi Pico-specific implementation of timing functions.

#include "tensorflow/lite/micro/micro_time.h"

#include "tensorflow/lite/micro/debug_log.h"

// These are headers from the RP2's SDK.
#include "hardware/timer.h"  // NOLINT

namespace tflite {
namespace {
// Pico's time_us_32() returns microseconds.
const uint32_t kClocksPerSecond = 1000000;
}  // namespace

uint32_t ticks_per_second() { return kClocksPerSecond; }

uint32_t GetCurrentTimeTicks() {
  return static_cast<uint32_t>(time_us_32());
}

}  // namespace tflite