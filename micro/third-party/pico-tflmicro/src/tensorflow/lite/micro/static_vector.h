// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_MICRO_STATIC_VECTOR_H_
#define TENSORFLOW_LITE_MICRO_STATIC_VECTOR_H_

#include <array>
#include <cassert>
#include <cstddef>

#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {

template <typename T, std::size_t MaxSize>
class StaticVector {
  // A staticlly-allocated vector. Add to the interface as needed.

 private:
  std::array<T, MaxSize> array_;
  std::size_t size_{0};

 public:
  using iterator = typename decltype(array_)::iterator;
  using const_iterator = typename decltype(array_)::const_iterator;
  using pointer = typename decltype(array_)::pointer;
  using reference = typename decltype(array_)::reference;
  using const_reference = typename decltype(array_)::const_reference;

  StaticVector() {}

  StaticVector(std::initializer_list<T> values) {
    for (const T& v : values) {
      push_back(v);
    }
  }

  static constexpr std::size_t max_size() { return MaxSize; }
  std::size_t size() const { return size_; }
  bool full() const { return size() == max_size(); }
  iterator begin() { return array_.begin(); }
  const_iterator begin() const { return array_.begin(); }
  iterator end() { return begin() + size(); }
  const_iterator end() const { return begin() + size(); }
  pointer data() { return array_.data(); }
  reference operator[](int i) { return array_[i]; }
  const_reference operator[](int i) const { return array_[i]; }
  void clear() { size_ = 0; }

  template <std::size_t N>
  bool operator==(const StaticVector<T, N>& other) const {
    return std::equal(begin(), end(), other.begin(), other.end());
  }

  template <std::size_t N>
  bool operator!=(const StaticVector<T, N>& other) const {
    return !(*this == other);
  }

  void push_back(const T& t) {
    TF_LITE_ASSERT(!full());
    *end() = t;
    ++size_;
  }
};

template <typename T, typename... U>
StaticVector(T, U...) -> StaticVector<T, 1 + sizeof...(U)>;

}  // end namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_STATIC_VECTOR_H_
