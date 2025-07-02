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
#include "types.h"
#include "ufi.h"

struct WrapCppOptional {
  template <typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped) {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.template constructor<typename WrappedT::value_type>();
  }
};

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

  mod.add_type<CN_NDArray>("CN_NDArray");

  auto ndarray_accessor =
      mod.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("NDArrayAccessor");
  ndarray_accessor
      .apply_combination<ApplyNDArrayAccessor, all_types, allowed_dims>(
          WrapNDArrayAccessor());

  mod.add_type<std::vector<std::shared_ptr<CN_NDArray>>>("VectorNDArray")
      .method("push_back", [](std::vector<std::shared_ptr<CN_NDArray>>& v,
                              const CN_NDArray& x) {
        v.push_back(std::make_shared<CN_NDArray>(x));
      });

  wrap_cuda_methods(mod);
}
