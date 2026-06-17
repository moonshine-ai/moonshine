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

#include "signal/src/circular_buffer.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT assert

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above
void CircularBufferReset(tflm_signal::CircularBuffer* cb) {
  cb->read = 0;
  cb->write = 0;
  cb->empty = 1;
  cb->buffer = (int16_t*)(cb + 1);
  memset(cb->buffer, 0, sizeof(cb->buffer[0]) * cb->buffer_size);
}

size_t CircularBufferGetNeededMemory(size_t capacity) {
  return sizeof(CircularBuffer) + sizeof(int16_t) * 2 * capacity;
}

CircularBuffer* CircularBufferInit(size_t capacity, void* state,
                                   size_t state_size) {
  ASSERT(CircularBufferGetNeededMemory(capacity) >= state_size);
  CircularBuffer* cb = (CircularBuffer*)state;
  cb->buffer_size = 2 * capacity;
  cb->capacity = capacity;
  CircularBufferReset(cb);
  return cb;
}

size_t CircularBufferCapacity(const tflm_signal::CircularBuffer* cb) {
  return cb->capacity;
}

bool CircularBufferFull(const tflm_signal::CircularBuffer* cb) {
  return cb->read == cb->write && cb->empty == 0;
}

bool CircularBufferEmpty(const tflm_signal::CircularBuffer* cb) {
  return cb->empty == 1;
}

size_t CircularBufferAvailable(const tflm_signal::CircularBuffer* cb) {
  const int32_t diff = cb->write - cb->read;
  if (diff > 0) {
    return diff;
  } else if (diff < 0) {
    return cb->capacity + diff;
  } else if (cb->empty == 1) {
    return 0;
  } else {
    return cb->capacity;
  }
}

size_t CircularBufferCanWrite(const tflm_signal::CircularBuffer* cb) {
  return cb->capacity - CircularBufferAvailable(cb);
}

void CircularBufferAdd(tflm_signal::CircularBuffer* cb, int16_t value) {
  ASSERT(!CircularBufferFull(cb));
  cb->buffer[cb->write] = value;
  cb->buffer[cb->write + cb->capacity] = value;
  if (++cb->write == cb->capacity) {
    cb->write = 0;
  }
  cb->empty = 0;
}

void CircularBufferWrite(tflm_signal::CircularBuffer* cb, const int16_t* values,
                         size_t n) {
  if (n > 0) {
    ASSERT(CircularBufferCanWrite(cb) >= n);
    size_t write = cb->write;
    int16_t* buffer = cb->buffer;
    const size_t capacity = cb->capacity;
    const size_t end = write + n;

    memcpy(buffer + write, values, n * sizeof(int16_t));
    if (end < capacity) {
      memcpy(buffer + capacity + write, values, n * sizeof(int16_t));
      write += n;
    } else {
      const size_t n1 = capacity - write;
      const size_t nbytes1 = n1 * sizeof(int16_t);
      memcpy(buffer + capacity + write, values, nbytes1);
      const size_t n2 = end - capacity;
      if (n2 > 0) {
        const size_t nbytes2 = n2 * sizeof(int16_t);
        memcpy(buffer, values + n1, nbytes2);
      }
      write = n2;
    }
    cb->write = write;
    cb->empty = 0;
  }
}

void CircularBufferWriteZeros(tflm_signal::CircularBuffer* cb, size_t n) {
  if (n > 0) {
    ASSERT(CircularBufferCanWrite(cb) >= n);
    size_t write = cb->write;
    int16_t* buffer = cb->buffer;
    const size_t capacity = cb->capacity;
    const size_t end = write + n;

    memset(buffer + write, 0, n * sizeof(int16_t));
    if (end < capacity) {
      memset(buffer + capacity + write, 0, n * sizeof(int16_t));
      write += n;
    } else {
      const size_t n1 = capacity - write;
      const size_t nbytes1 = n1 * sizeof(int16_t);
      memset(buffer + capacity + write, 0, nbytes1);
      const size_t n2 = end - capacity;
      if (n2 > 0) {
        const size_t nbytes2 = n2 * sizeof(int16_t);
        memset(buffer, 0, nbytes2);
      }
      write = n2;
    }
    cb->write = write;
    cb->empty = 0;
  }
}

int16_t* CircularBufferReserveForWrite(tflm_signal::CircularBuffer* cb,
                                       size_t n) {
  ASSERT(cb->write + n <= cb->capacity);
  int16_t* write_ptr = cb->buffer + cb->write;
  cb->write += n;
  if (cb->write == cb->capacity) {
    cb->write = 0;
  }
  cb->empty = cb->empty && n == 0;
  return write_ptr;
}

void CircularBufferExtend(tflm_signal::CircularBuffer* cb, size_t count,
                          int32_t n) {
  if (n > 0 && count > 0) {
    ASSERT(CircularBufferCanWrite(cb) >= count * n);
    ASSERT(CircularBufferAvailable(cb) >= count);
    const size_t capacity = cb->capacity;
    // start pos of region to copy
    const size_t start =
        (count > cb->write) ? cb->write + capacity - count : cb->write - count;
    const size_t end = start + count;
    int i;
    if (end <= capacity) {
      // the source elements are contiguous
      for (i = 0; i < n; ++i) {
        CircularBufferWrite(cb, cb->buffer + start, count);
      }
    } else {
      // the source elements wrap around the end of the buffer
      for (i = 0; i < n; ++i) {
        const size_t n1 = capacity - start;
        const size_t n2 = count - n1;
        CircularBufferWrite(cb, cb->buffer + start, n1);
        CircularBufferWrite(cb, cb->buffer, n2);
      }
    }
  }
  // Note: no need to update empty flag
}

int16_t CircularBufferRemove(tflm_signal::CircularBuffer* cb) {
  ASSERT(!CircularBufferEmpty(cb));
  const int16_t result = cb->buffer[cb->read];
  if (++cb->read == cb->capacity) {
    cb->read = 0;
  }
  if (cb->read == cb->write) {
    cb->empty = 1;
  }
  return result;
}

int16_t CircularBufferPeek(const tflm_signal::CircularBuffer* cb,
                           size_t index) {
  ASSERT(CircularBufferAvailable(cb) > index);
  size_t target = cb->read + index;
  while (target >= cb->capacity) {
    target -= cb->capacity;
  }
  return cb->buffer[target];
}

void CircularBufferRewind(tflm_signal::CircularBuffer* cb, size_t n) {
  ASSERT(n <= CircularBufferCanWrite(cb));
  if (n > cb->read) {
    // Must add before subtracting because types are unsigned.
    cb->read = (cb->read + cb->capacity) - n;
  } else {
    cb->read -= n;
  }
  if (n > 0) cb->empty = 0;
}

const int16_t* CircularBufferPeekDirect(const tflm_signal::CircularBuffer* cb,
                                        size_t index) {
  ASSERT(CircularBufferAvailable(cb) > index);
  size_t target = cb->read + index;
  while (target >= cb->capacity) {
    target -= cb->capacity;
  }
  return cb->buffer + target;
}

const int16_t* CircularBufferPeekMax(const tflm_signal::CircularBuffer* cb,
                                     size_t* n) {
  if (CircularBufferAvailable(cb) > 0) {
    *n = (cb->write <= cb->read) ? cb->capacity - cb->read
                                 : cb->write - cb->read;
    return cb->buffer + cb->read;
  } else {
    *n = 0;
    return NULL;
  }
}

void CircularBufferGet(tflm_signal::CircularBuffer* cb, size_t n,
                       int16_t* values) {
  ASSERT(CircularBufferAvailable(cb) >= n);
  const int16_t* buffer = cb->buffer;
  const size_t read = cb->read;
  const size_t end = read + n;
  const size_t capacity = cb->capacity;
  if (end <= capacity) {
    memcpy(values, buffer + read, n * sizeof(int16_t));
  } else {
    const size_t n1 = capacity - read;
    const size_t n2 = end - capacity;
    const size_t nbytes1 = n1 * sizeof(int16_t);
    const size_t nbytes2 = n2 * sizeof(int16_t);
    memcpy(values, buffer + read, nbytes1);
    memcpy(values + n1, buffer, nbytes2);
  }
}

void CircularBufferDiscard(tflm_signal::CircularBuffer* cb, size_t n) {
  ASSERT(n > 0);
  ASSERT(CircularBufferAvailable(cb) >= n);
  cb->read += n;
  if (cb->read >= cb->capacity) {
    cb->read -= cb->capacity;
  }
  if (cb->read == cb->write) {
    cb->empty = 1;
  }
}

void CircularBufferShift(tflm_signal::CircularBuffer* cb, int n) {
  if (n < 0) {
    ASSERT(-n <= (int)cb->capacity);
    if ((int)cb->read < -n) {
      // First add then subtract to ensure positivity as types are unsigned.
      cb->read += cb->capacity;
    }
    cb->read += n;
  } else {
    ASSERT(n <= (int)cb->capacity);
    cb->read += n;
    if (cb->read >= cb->capacity) {
      cb->read -= cb->capacity;
    }
  }
}

}  // namespace tflm_signal
}  // namespace tflite
