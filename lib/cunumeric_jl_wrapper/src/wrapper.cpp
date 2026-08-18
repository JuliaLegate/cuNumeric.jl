/* Copyright 2026 Northwestern University,
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
 *            Nader Rahhal <naderrahhal2026@u.northwestern.edu>
 */

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>  //needed for return type of toString methods
#include <type_traits>
#include <vector>

#include "accessors.h"
#include "cupynumeric.h"
#include "cupynumeric/operators.h"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"
#include "legate.h"
#include "legion.h"
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

legate::LogicalArray get_store(CN_NDArray* arr) { return arr->obj.get_store(); }

legate::Library get_lib() {
  auto runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  return runtime->get_library();
}

void* nda_store_to_ndarray(legate::LogicalStore st) {
  return static_cast<void*>(new CN_NDArray{cupynumeric::as_array(st)});
}

// Legate.jl wraps ManualTask::add_input/add_output for a whole partition, but
// not the overloads taking a projection. GEEV needs one: its eigenvalue store
// has one fewer dimension than the launch domain. Julia picks the source
// dimensions; this only translates them into a SymbolicPoint.
static legate::SymbolicPoint make_projection(const std::vector<int32_t>& dims) {
  std::vector<legate::SymbolicExpr> exprs;
  exprs.reserve(dims.size());
  for (auto d : dims) {
    exprs.push_back(legate::dimension(static_cast<uint32_t>(d)));
  }
  return legate::SymbolicPoint{std::move(exprs)};
}

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
void register_tasks() {
  auto library = get_lib();
  ufi::LoadPTXTask::register_variants(library);
  ufi::RunPTXTask::register_variants(library);
  ufi::RunPTXBroadcastTask::register_variants(library);
}
#endif

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
  wrap_unary_ops(mod);
  wrap_binary_ops(mod);
  wrap_unary_reds(mod);
  wrap_linalg_ops(mod);
  wrap_bitgenerator_ops(mod);

  using jlcxx::ParameterList;
  using jlcxx::Parametric;
  using jlcxx::TypeVar;
  using legate_util::HalfType;

  mod.map_type<HalfType>("Float16");

  // These are the types/dims used to generate templated functions
  // i.e. only these types/dims can be used from Julia side
  using fp_types = ParameterList<double, float, HalfType>;
  using int_types = ParameterList<int8_t, int16_t, int32_t, int64_t>;
  using uint_types = ParameterList<uint8_t, uint16_t, uint32_t, uint64_t>;

  using all_types =
      ParameterList<double, float, int8_t, int16_t, int32_t, int64_t, uint8_t,
                    uint16_t, uint32_t, uint64_t, bool, std::complex<double>,
                    std::complex<float>, HalfType>;
  using allowed_dims = ParameterList<
      std::integral_constant<int_t, 1>, std::integral_constant<int_t, 2>,
      std::integral_constant<int_t, 3>, std::integral_constant<int_t, 4>>;

  mod.method("initialize_cunumeric", &cupynumeric::initialize);

  mod.add_type<CN_NDArray>("CN_NDArray");
  mod.method("_get_store", &get_store);
  mod.method("get_lib", &get_lib);
  mod.method("nda_store_to_ndarray", &nda_store_to_ndarray);

  // True when the loaded cuSolver provides cusolverDnXgeev. Without it
  // cupynumeric has no GPU eigenvalue kernel for general matrices.
  mod.method("cusolver_has_geev", &cupynumeric_cusolver_has_geev);

  mod.method("add_input_proj",
             [](legate::ManualTask& task,
                std::shared_ptr<legate::LogicalStorePartition> part,
                const std::vector<int32_t>& dims) {
               task.add_input(*part, make_projection(dims));
             });
  mod.method("add_output_proj",
             [](legate::ManualTask& task,
                std::shared_ptr<legate::LogicalStorePartition> part,
                const std::vector<int32_t>& dims) {
               task.add_output(*part, make_projection(dims));
             });

  // Marking a task as throwing also grows its leaf allocation pools by
  // --max-exception-size (4096 bytes by default), which the cupynumeric
  // decomposition tasks rely on: their mapper only declares enough zero-copy
  // memory for a single status flag, while the batched GPU kernels allocate one
  // per matrix. cupynumeric's own Python launchers always set this.
  mod.method("task_throws_exception", [](legate::ManualTask& task, bool value) {
    task.throws_exception(value);
  });
  mod.method("task_throws_exception", [](legate::AutoTask& task, bool value) {
    task.throws_exception(value);
  });

  // Legate.jl Scalar has no vector constructors. BITGENERATOR (and similar)
  // tasks take fixed-array scalars; these helpers pack them from Julia
  // pointers.
  mod.method("add_vector_scalar_i64",
             [](legate::AutoTask& task, const int64_t* p, int32_t n) {
               std::vector<int64_t> v;
               if (n > 0) {
                 v.assign(p, p + n);
               }
               task.add_scalar_arg(legate::Scalar(std::move(v)));
             });
  mod.method("add_vector_scalar_f32",
             [](legate::AutoTask& task, const float* p, int32_t n) {
               std::vector<float> v;
               if (n > 0) {
                 v.assign(p, p + n);
               }
               task.add_scalar_arg(legate::Scalar(std::move(v)));
             });
  mod.method("add_vector_scalar_f64",
             [](legate::AutoTask& task, const double* p, int32_t n) {
               std::vector<double> v;
               if (n > 0) {
                 v.assign(p, p + n);
               }
               task.add_scalar_arg(legate::Scalar(std::move(v)));
             });

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

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  mod.method("register_tasks", &register_tasks);
  wrap_cuda_methods(mod);
#endif
}
