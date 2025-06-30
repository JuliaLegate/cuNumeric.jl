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

#include <initializer_list>
#include <iostream>
#include <string>  //needed for return type of toString methods
#include <type_traits>

#include "accessors.h"
#include "cupynumeric.h"
#include "cupynumeric/operators.h"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"
#include "legate.h"
#include "legion.h"

// cpp wrapper headers
#include "types.h"
#include "ufi.h"

struct WrapCppOptional {
  template <typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped) {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.template constructor<typename WrappedT::value_type>();
  }
};

cupynumeric::NDArray get_slice(cupynumeric::NDArray arr,
                               std::vector<legate::Slice> slices) {
  switch (slices.size()) {
    case 1: {
      std::initializer_list<legate::Slice> slice_list = {slices[0]};
      return arr[slice_list];
    }
    case 2: {
      std::initializer_list<legate::Slice> slice_list = {slices[0], slices[1]};
      return arr[slice_list];
    }
    default: {
      assert(0 && "dim gteq 3 not supported yet\b");
    }
  };
  assert(0 && "you should not enter here\n");
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
  wrap_unary_ops(mod);
  wrap_binary_ops(mod);
  wrap_unary_reds(mod);

  using jlcxx::ParameterList;
  using jlcxx::Parametric;
  using jlcxx::TypeVar;

  // These are the types/dims used to generate templated functions
  // i.e. only these types/dims can be used from Julia side
  using fp_types = ParameterList<double, float>;
  using int_types = ParameterList<int8_t, int16_t, int32_t, int64_t>;
  using uint_types = ParameterList<uint8_t, uint16_t, uint32_t, uint64_t>;

  using all_types =
      ParameterList<double, float, int8_t, int16_t, int32_t, int64_t, uint8_t,
                    uint16_t, uint32_t, uint64_t>;
  using allowed_dims = ParameterList<std::integral_constant<int_t, 1>,
                                     std::integral_constant<int_t, 2>,
                                     std::integral_constant<int_t, 3>>;

  mod.method("initialize_cunumeric", &cupynumeric::initialize);

  auto ndarry_type = mod.add_type<cupynumeric::NDArray>("NDArray")
                         .constructor<const cupynumeric::NDArray&>();

  mod.add_type<Parametric<TypeVar<1>>>("StdOptional")
      .apply<std::optional<cupynumeric::NDArray>>(WrapCppOptional());

  ndarry_type.method("dim", &cupynumeric::NDArray::dim)
      .method("_size",
              &cupynumeric::NDArray::size)  // hide with underscore cause in
                                            // Julia `size` is same as shape
      .method("shape", &cupynumeric::NDArray::shape)
      .method("type", &cupynumeric::NDArray::type)
      .method("copy", &cupynumeric::NDArray::copy)
      .method("assign",
              (void(cupynumeric::NDArray::*)(const cupynumeric::NDArray&)) &
                  cupynumeric::NDArray::assign)
      .method("_reshape", (cupynumeric::NDArray(cupynumeric::NDArray::*)(
                              std::vector<int64_t>)) &
                              cupynumeric::NDArray::reshape)
      .method("as_type", &cupynumeric::NDArray::as_type)
      .method("unary_op", &cupynumeric::NDArray::unary_op)
      // ew but necessary cause theres an overload in operators.cc
      .method("unary_reduction", static_cast<void (cupynumeric::NDArray::*)(
                                     int32_t, cupynumeric::NDArray)>(
                                     &cupynumeric::NDArray::unary_reduction))
      .method("binary_op", &cupynumeric::NDArray::binary_op)
      .method("get_store", &cupynumeric::NDArray::get_store)
      .method("random", &cupynumeric::NDArray::random)
      .method("fill", &cupynumeric::NDArray::fill)
      .method("add", (cupynumeric::NDArray(cupynumeric::NDArray::*)(
                         const cupynumeric::NDArray&) const) &
                         cupynumeric::NDArray::operator+)
      .method("multiply", (cupynumeric::NDArray(cupynumeric::NDArray::*)(
                              const cupynumeric::NDArray&) const) &
                              cupynumeric::NDArray::operator*)
      .method("add_scalar", (cupynumeric::NDArray(cupynumeric::NDArray::*)(
                                const legate::Scalar&) const) &
                                cupynumeric::NDArray::operator+)
      .method("multiply_scalar", (cupynumeric::NDArray(cupynumeric::NDArray::*)(
                                     const legate::Scalar&) const) &
                                     cupynumeric::NDArray::operator*);

  mod.method("get_slice", &get_slice);

  auto ndarray_accessor =
      mod.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("NDArrayAccessor");
  ndarray_accessor
      .apply_combination<ApplyNDArrayAccessor, all_types, allowed_dims>(
          WrapNDArrayAccessor());

  mod.method("_zeros", &cupynumeric::zeros);  // operators.cc, 152
  mod.method("_full", &cupynumeric::full);    // operators.cc, 162
  mod.method("_dot", &cupynumeric::dot);      // operators.cc, 263
  mod.method("_sum", &cupynumeric::sum);      // operators.cc, 303
  mod.method("_add", &cupynumeric::add);
  mod.method("_multiply", &cupynumeric::multiply);
  mod.method("_random_ndarray", &cupynumeric::random);

  mod.add_type<std::vector<std::shared_ptr<cupynumeric::NDArray>>>(
         "VectorNDArray")
      .method("push_back",
              [](std::vector<std::shared_ptr<cupynumeric::NDArray>>& v,
                 const cupynumeric::NDArray& x) {
                v.push_back(std::make_shared<cupynumeric::NDArray>(x));
              });

  wrap_cuda_methods(mod);
}
