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

#ifndef SIGNAL_SRC_CIRCULAR_BUFFER_H_
#define SIGNAL_SRC_CIRCULAR_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above
struct CircularBuffer {
  // Max number of elements, value passed-in to CircularBufferAlloc.
  size_t capacity;
  // Next position to read.
  size_t read;
  // Next position to write.
  size_t write;
  // Flag to indicate emptiness.
  int32_t empty;
  // Auto-generated size variable
  int32_t buffer_size;
  // Array of the circular buffer elements (integers).
  int16_t* buffer;
};

// Returns the size of the memory that the circular buffer needs
// in order to hold `capacity` items.
size_t CircularBufferGetNeededMemory(size_t capacity);

// Initialize an instance of the circular buffer that holds `capacity` items.
// `state` points to a memory allocation of size `state_size`. The size
//  should be greater or equal to the value returned by
//  CircularBufferGetNeededMemory(capacity). Fails if it isn't.
//  On success, returns a pointer to the circular buffer's object.
CircularBuffer* CircularBufferInit(size_t capacity, void* state,
                                   size_t state_size);

// Reset a circular buffer to its initial empty state
void CircularBufferReset(CircularBuffer* cb);

size_t CircularBufferCapacity(const CircularBuffer* cb);

bool CircularBufferFull(const CircularBuffer* cb);

bool CircularBufferEmpty(const CircularBuffer* cb);

// Returns the number of elements ready to read
size_t CircularBufferAvailable(const CircularBuffer* cb);

// Returns the number of elements available to write.
size_t CircularBufferCanWrite(const CircularBuffer* cb);

// Adds a single `value` to the buffer and advances the write pointer.
void CircularBufferAdd(CircularBuffer* cb, int16_t value);

// Writes `n` `values` into the buffer and advances the write pointer.
void CircularBufferWrite(CircularBuffer* cb, const int16_t* values, size_t n);

// Writes `n` zeros into the buffer and advances the write pointer.
void CircularBufferWriteZeros(CircularBuffer* cb, size_t n);

// Returns a pointer to a buffer where elements can be written, and
// advances the write pointer as though they have already been written.
// Fails if `n` elements are not available contiguously at the current
// write position.
int16_t* CircularBufferReserveForWrite(CircularBuffer* cb, size_t n);

// Copies the final region (`count` elements) of the buffer `n` times, to
// the end of the buffer.
void CircularBufferExtend(CircularBuffer* cb, size_t count, int32_t n);

// Reads a single value from the buffer and advances the read pointer
int16_t CircularBufferRemove(CircularBuffer* cb);

// Reads the value at the given `index`, does not modify the read pointer.
int16_t CircularBufferPeek(const CircularBuffer* cb, size_t index);

// Rewinds to restore the previous `n` values read
void CircularBufferRewind(CircularBuffer* cb, size_t n);

// Returns a pointer directly into the circular buffer at the given `index`.
// Caller is responsible for not reading past the end.
const int16_t* CircularBufferPeekDirect(const CircularBuffer* cb, size_t index);

// Returns a pointer into the circular buffer at the current read pointer,
// setting `n` to the number of values available to be read from here.
const int16_t* CircularBufferPeekMax(const CircularBuffer* cb, size_t* n);

// Copies `n` `values` from the buffer and does not advance the read
// pointer and does not update the empty flag.
void CircularBufferGet(CircularBuffer* cb, size_t n, int16_t* values);

// Discards the next `n` values by advancing the read index.
// Valid for n > 0.
void CircularBufferDiscard(CircularBuffer* cb, size_t n);

// Shifts the buffer with `n` values (`n` can be negative) by moving
// the read index.
void CircularBufferShift(CircularBuffer* cb, int n);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_SRC_CIRCULAR_BUFFER_H_
