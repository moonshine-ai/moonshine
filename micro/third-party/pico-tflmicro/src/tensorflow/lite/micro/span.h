/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef TENSORFLOW_LITE_MICRO_SPAN_H_
#define TENSORFLOW_LITE_MICRO_SPAN_H_

#include <array>
#include <cstddef>

namespace tflite {

// A poor man's std::span, we should consider using the Pigweed span instead.
template <typename T>
class Span {
 public:
  constexpr Span(T* data, size_t size) noexcept : data_(data), size_(size) {}

  template <size_t N>
  constexpr Span(T (&data)[N]) noexcept : data_(data), size_(N) {}

  template <size_t N>
  constexpr Span(std::array<T, N>& array) noexcept
      : data_(array.data()), size_(N) {}

  constexpr T& operator[](size_t idx) const noexcept { return *(data_ + idx); }

  constexpr T* data() const noexcept { return data_; }
  constexpr size_t size() const noexcept { return size_; }

 private:
  T* data_;
  size_t size_;
};

template <typename A, typename B>
bool operator==(const Span<A>& a, const Span<B>& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }

  return true;
}

template <typename A, typename B>
bool operator!=(const Span<A>& a, const Span<B>& b) {
  return !(a == b);
}

}  // end namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_SPAN_H_
