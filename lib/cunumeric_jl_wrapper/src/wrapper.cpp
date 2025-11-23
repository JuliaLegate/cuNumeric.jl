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
#include "mathtypes/half.h"
#include "realm.h"
#include "types.h"
#include "ufi.h"

struct WrapCppOptional {
  template <typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped) {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.template constructor<typename WrappedT::value_type>();
  }
};

legate::LogicalArray* get_store(CN_NDArray* arr) {
  auto res = arr->obj.get_store();
  return new legate::LogicalArray(std::move(res));
}

struct GetPtrFunctor {
  template <legate::Type::Code CODE, int DIM>
  void* operator()(legate::LogicalArray* arr) {
#ifndef HAVE_CUDA
    // Check if FLOAT16 is being used without CUDA support
    if (CODE == legate::Type::Code::FLOAT16) {
      throw std::runtime_error(
          "FLOAT16 type is not supported when building without CUDA");
    }
#endif
    using CppT = typename legate_util::code_to_cxx<CODE>::type;
    auto rf = arr->get_physical_array();
    auto shp = rf.shape<DIM>();
    return rf.data().write_accessor<CppT, DIM>().ptr(Realm::Point<DIM>(shp.lo));
  }
};

inline void* get_ptr(legate::LogicalArray* arr) {
  int dim = arr->dim();
  legate::Type::Code code = arr->type().code();
  return legate::double_dispatch(dim, code, GetPtrFunctor{}, arr);
}

legate::Library get_lib() {
  auto runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  return runtime->get_library();
}

void register_tasks() {
  auto library = get_lib();
  ufi::LoadPTXTask::register_variants(library);
  ufi::RunPTXTask::register_variants(library);
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
                    uint16_t, uint32_t, uint64_t, bool>;
  using allowed_dims = ParameterList<
      std::integral_constant<int_t, 1>, std::integral_constant<int_t, 2>,
      std::integral_constant<int_t, 3>, std::integral_constant<int_t, 4>>;

  mod.method("initialize_cunumeric", &cupynumeric::initialize);

  mod.add_type<CN_NDArray>("CN_NDArray");
  mod.method("_get_store", &get_store);
  mod.method("_get_ptr", &get_ptr);
  mod.method("get_lib", &get_lib);
  mod.method("register_tasks", &register_tasks);

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
