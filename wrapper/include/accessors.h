/* Copyright 2025 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
 */

#pragma once

#include <iostream>
#include <type_traits>
#include <vector>

#include "cupynumeric.h"
#include "jlcxx/jlcxx.hpp"
#include "legion.h"

using coord_t = long long;

// To auto-magically generate templated classes and their
// respective member functions you must define a `BuildParameterList`
// which allows `apply_combination` to work. You also need to define a
// struct which defines `apply` as a type for how to create the type properly
// with a given set of template parameters. Then you need a class to wrap the
// template instantiation. This should overload the call, operator(), which is
// passed as a functor to apply_combiation.....what a mess

struct CN_NDArray {
  cupynumeric::NDArray obj;
};

template <typename T, int n_dims>
class NDArrayAccessor {
 public:
  NDArrayAccessor() {}
  ~NDArrayAccessor() {}
  // static
  T read(void* arr, const std::vector<uint64_t>& dims) {
    auto p = Realm::Point<n_dims>(0);
    for (int i = 0; i < n_dims; ++i) {
      p[i] = dims[i];
    }
    auto acc = ((CN_NDArray*)arr)->obj.get_read_accessor<T, n_dims>();
    return acc.read(p);
  }

  // static
  void write(void* arr, const std::vector<uint64_t>& dims, T val) {
    auto p = Realm::Point<n_dims>(0);
    for (int i = 0; i < n_dims; ++i) {
      p[i] = dims[i];
    }
    auto acc = ((CN_NDArray*)arr)->obj.get_write_accessor<T, n_dims>();
    acc.write(p, val);  // DOES THIS HAVE A RETURN??
  }
};

namespace jlcxx {
template <typename T, int n_dims>
struct BuildParameterList<NDArrayAccessor<T, n_dims>> {
  using type = ParameterList<T, std::integral_constant<int_t, n_dims>>;
};
}  // namespace jlcxx

struct ApplyNDArrayAccessor {
  template <typename T, typename n_dims>
  using apply = NDArrayAccessor<T, n_dims::value>;
};

struct WrapNDArrayAccessor {
  template <typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped) {
    using WrappedT = typename TypeWrapperT::type;
    wrapped.template constructor<WrappedT>();
    wrapped.method("read", &WrappedT::read);
    wrapped.method("write", &WrappedT::write);
  }
};
