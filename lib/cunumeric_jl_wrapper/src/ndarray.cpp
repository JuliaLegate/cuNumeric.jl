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

#include <cupynumeric.h>
#include <cupynumeric/ndarray.h>
#include <cupynumeric/operators.h>
#include <cupynumeric/runtime.h>
#include <deps/realm/machine.h>
#include <deps/realm/machine_impl.h>
#include <legate.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "ndarray_c_api.h"

extern "C" {

using cupynumeric::full;
using cupynumeric::NDArray;
using cupynumeric::random;
using cupynumeric::zeros;

using legate::Scalar;

struct CN_NDArray {
  NDArray obj;
};

struct CN_Type {
  legate::Type obj;
};

struct CN_Store {
  legate::LogicalStore obj;
};

size_t nda_get_number_of_runtimes() {
  Legion::Machine legion_machine{Legion::Machine::get_machine()};
  return legion_machine.get_address_space_count();
}

CN_NDArray* nda_zeros_array(int32_t dim, const uint64_t* shape, CN_Type type) {
  std::vector<uint64_t> shp(shape, shape + dim);
  NDArray result = zeros(shp, type.obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_full_array(int32_t dim, const uint64_t* shape, CN_Type type,
                           const void* value) {
  std::vector<uint64_t> shp(shape, shape + dim);
  Scalar s(type.obj, value, true);
  NDArray result = full(shp, s);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_random(CN_NDArray* arr, int32_t code) { arr->obj.random(code); }

CN_NDArray* nda_random_array(int32_t dim, const uint64_t* shape) {
  std::vector<uint64_t> shp(shape, shape + dim);
  NDArray result = random(shp);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_reshape_array(CN_NDArray* arr, int32_t dim,
                              const uint64_t* shape) {
  std::vector<int64_t> shp(shape, shape + dim);
  NDArray result = cupynumeric::reshape(arr->obj, shp, "C");
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_from_scalar(CN_Type type, const void* value) {
  Scalar s(type.obj, value, true);
  auto runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  auto scalar_store = runtime->create_scalar_store(s);
  return new CN_NDArray{cupynumeric::as_array(scalar_store)};
  // return new CN_NDArray{NDArray(std::move(scalar_store))};
}

// CN_NDArray* nda_from_scalar_0D(CN_Type type, const void* value) {
//   Scalar s(type.obj, value, true);
//   return new CN_NDArray{
//       legate::Runtime::get_runtime()->create_store(s, legate::Shape{})};
// }

CN_NDArray* nda_astype(CN_NDArray* arr, CN_Type type) {
  NDArray result = arr->obj.as_type(type.obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_fill_array(CN_NDArray* arr, CN_Type type, const void* value) {
  Scalar s(type.obj, value, true);
  arr->obj.fill(s);
}

void nda_multiply(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out) {
  cupynumeric::multiply(rhs1->obj, rhs2->obj, out->obj);
}

void nda_add(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out) {
  cupynumeric::add(rhs1->obj, rhs2->obj, out->obj);
}

// NEW

CN_NDArray* nda_unique(CN_NDArray* arr) {
  NDArray result = cupynumeric::unique(arr->obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_sort(CN_NDArray* arr, int32_t axis, bool stable) {
  const char* kind = stable ? "stable" : "quicksort";
  NDArray result =
      cupynumeric::sort(arr->obj, std::optional<int32_t>{axis}, kind);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_sort_inplace(CN_NDArray* arr, int32_t axis, bool stable) {
  arr->obj.sort(arr->obj, false, std::optional<int32_t>{axis}, stable);
}

CN_NDArray* nda_argsort(CN_NDArray* arr, int32_t axis, bool stable) {
  const char* kind = stable ? "stable" : "quicksort";
  NDArray result =
      cupynumeric::argsort(arr->obj, std::optional<int32_t>{axis}, kind);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_ravel(CN_NDArray* arr) {
  NDArray result = cupynumeric::ravel(arr->obj, "C");
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_trace(CN_NDArray* arr, int32_t offset, int32_t a1, int32_t a2,
                      CN_Type type) {
  NDArray result = cupynumeric::trace(arr->obj, offset, a1, a2, type.obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_eye(int32_t rows, CN_Type type) {
  NDArray result = cupynumeric::eye(rows, rows, 0, type.obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_diag(CN_NDArray* arr, int32_t k) {
  NDArray result = cupynumeric::diag(arr->obj, k);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_transpose(CN_NDArray* arr) {
  NDArray result = cupynumeric::transpose(arr->obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_multiply_scalar(CN_NDArray* rhs1, CN_Type type,
                                const void* value) {
  Scalar s(type.obj, value, true);
  NDArray result = rhs1->obj * s;
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_add_scalar(CN_NDArray* rhs1, CN_Type type, const void* value) {
  Scalar s(type.obj, value, true);
  NDArray result = rhs1->obj + s;
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_dot(CN_NDArray* rhs1, CN_NDArray* rhs2) {
  NDArray result = cupynumeric::dot(rhs1->obj, rhs2->obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_three_dot_arg(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out) {
  out->obj.dot(rhs1->obj, rhs2->obj);
}

CN_NDArray* nda_transpose_axes(CN_NDArray* arr, const int32_t* axes,
                               int32_t n) {
  std::vector<int32_t> axis_vec(axes, axes + n);
  NDArray result = cupynumeric::transpose(arr->obj, std::move(axis_vec));
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_squeeze(CN_NDArray* arr, const int32_t* axes, int32_t n) {
  if (n <= 0) {
    NDArray result = cupynumeric::squeeze(arr->obj, std::nullopt);
    return new CN_NDArray{NDArray(std::move(result))};
  }
  std::vector<int32_t> axis_vec(axes, axes + n);
  NDArray result = cupynumeric::squeeze(
      arr->obj,
      std::optional<std::reference_wrapper<std::vector<int32_t> const>>{
          axis_vec});
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_diagonal(CN_NDArray* arr, int32_t offset, int32_t axis1,
                         int32_t axis2) {
  NDArray result = cupynumeric::diagonal(arr->obj, offset, axis1, axis2, true);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_contract(CN_NDArray* out, const char* lhs_modes, int32_t n_lhs,
                  CN_NDArray* rhs1, const char* rhs1_modes, int32_t n_rhs1,
                  CN_NDArray* rhs2, const char* rhs2_modes, int32_t n_rhs2,
                  const char* extent_keys, const int32_t* extents,
                  int32_t n_extents) {
  std::vector<char> lhs(lhs_modes, lhs_modes + n_lhs);
  std::vector<char> r1(rhs1_modes, rhs1_modes + n_rhs1);
  std::vector<char> r2(rhs2_modes, rhs2_modes + n_rhs2);
  std::map<char, int> mode2extent;
  for (int32_t i = 0; i < n_extents; ++i) {
    mode2extent.emplace(extent_keys[i], extents[i]);
  }
  out->obj.contract(lhs, rhs1->obj, r1, rhs2->obj, r2, mode2extent);
}

CN_NDArray* nda_copy(CN_NDArray* arr) {
  NDArray result = arr->obj.copy();
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_assign(CN_NDArray* arr, CN_NDArray* other) {
  arr->obj.assign(other->obj);
}

void nda_move(CN_NDArray* dst, CN_NDArray* src) {
  dst->obj.operator=(std::move(src->obj));
}

void nda_destroy_array(CN_NDArray* arr) {
  if (arr != NULL) {
    delete arr;
  }
}

int32_t nda_array_dim(const CN_NDArray* arr) { return arr->obj.dim(); }

uint64_t nda_array_size(const CN_NDArray* arr) { return arr->obj.size(); }

int32_t nda_array_type_code(const CN_NDArray* arr) {
  return static_cast<int32_t>(arr->obj.type().code());
}

CN_Type* nda_array_type(const CN_NDArray* arr) {
  return new CN_Type{arr->obj.type()};
}

uint64_t nda_nbytes(CN_NDArray* arr) {
  return static_cast<uint64_t>(arr->obj.type().size()) * nda_array_size(arr);
}

void nda_array_shape(const CN_NDArray* arr, uint64_t* out_shape) {
  const auto& shp = arr->obj.shape();
  for (size_t i = 0; i < shp.size(); ++i) out_shape[i] = shp[i];
}

void nda_binary_op(CN_NDArray* out, CuPyNumericBinaryOpCode op_code,
                   const CN_NDArray* rhs1, const CN_NDArray* rhs2) {
  out->obj.binary_op(op_code, rhs1->obj, rhs2->obj);
}

void nda_binary_reduction(CN_NDArray* out, CuPyNumericBinaryOpCode op_code,
                          const CN_NDArray* rhs1, const CN_NDArray* rhs2) {
  out->obj.binary_reduction(op_code, rhs1->obj, rhs2->obj);
}

CN_NDArray* nda_array_equal(const CN_NDArray* rhs1, const CN_NDArray* rhs2) {
  return new CN_NDArray{cupynumeric::array_equal(rhs1->obj, rhs2->obj)};
}

void nda_unary_op(CN_NDArray* out, CuPyNumericUnaryOpCode op_code,
                  CN_NDArray* input) {
  out->obj.unary_op(op_code, input->obj);
}

void nda_unary_reduction(CN_NDArray* out, CuPyNumericUnaryRedCode op_code,
                         CN_NDArray* input) {
  out->obj.unary_reduction(op_code, input->obj);
}

// COUNT_NONZERO's result is an integer count, not the input dtype.
// Passing res_dtype (not dtype) keeps the source type. dtype=int64 would
// cast the input before counting.
static std::optional<legate::Type> unary_red_res_dtype(
    CuPyNumericUnaryRedCode op_code) {
  switch (op_code) {
    case CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_COUNT_NONZERO:
      return legate::int64();
    default:
      return std::nullopt;
  }
}

static bool is_arg_reduction(CuPyNumericUnaryRedCode op_code) {
  switch (op_code) {
    case CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ARGMAX:
    case CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ARGMIN:
    case CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANARGMAX:
    case CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANARGMIN:
      return true;
    default:
      return false;
  }
}

}  // extern "C"

// Matches cupynumeric Argval<T> { int64_t arg; T arg_value; }.
// Templates cannot live in the extern "C" block.
template <typename T>
struct ArgvalCompat {
  int64_t arg;
  T arg_value;
};

template <typename T>
static void fill_arg_identity(NDArray& acc, const legate::Type& argred_type,
                              bool is_argmax) {
  ArgvalCompat<T> id;
  id.arg = std::numeric_limits<int64_t>::min();
  id.arg_value = is_argmax ? std::numeric_limits<T>::lowest()
                           : std::numeric_limits<T>::max();
  if (argred_type.size() != sizeof(id)) {
    throw std::runtime_error("argred identity layout mismatch");
  }
  acc.fill(Scalar(argred_type, &id, true));
}

static void fill_arg_identity(NDArray& acc, const legate::Type& src_type,
                              const legate::Type& argred_type, bool is_argmax) {
  switch (src_type.code()) {
    case legate::Type::Code::BOOL:
      fill_arg_identity<bool>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::INT8:
      fill_arg_identity<int8_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::INT16:
      fill_arg_identity<int16_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::INT32:
      fill_arg_identity<int32_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::INT64:
      fill_arg_identity<int64_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::UINT8:
      fill_arg_identity<uint8_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::UINT16:
      fill_arg_identity<uint16_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::UINT32:
      fill_arg_identity<uint32_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::UINT64:
      fill_arg_identity<uint64_t>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::FLOAT32:
      fill_arg_identity<float>(acc, argred_type, is_argmax);
      break;
    case legate::Type::Code::FLOAT64:
      fill_arg_identity<double>(acc, argred_type, is_argmax);
      break;
    default:
      throw std::runtime_error(
          "argmax/argmin are not supported for this element type");
  }
}

// Public unary_reduction fills identity via type_dispatch on the *output*
// type. For ARGMAX that output is a struct, which Legate rejects. Fill from
// the source dtype (like Python), then launch SCALAR_UNARY_RED and GETARG.
static CN_NDArray* nda_arg_reduction(CuPyNumericUnaryRedCode op_code,
                                     CN_NDArray* input) {
  const bool is_argmax =
      op_code == CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ARGMAX ||
      op_code == CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANARGMAX;
  auto* runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  const std::vector<uint64_t> scalar_shape{};
  auto argred_type = runtime->get_argred_type(input->obj.type());
  // C++ get_argred_type does not attach redops; Python does this on first
  // use. record_reduction_operator throws if called twice for the same type.
  static std::unordered_set<legate::Type::Code> registered_argred;
  if (registered_argred.insert(input->obj.type().code()).second) {
    auto ids = cupynumeric_register_reduction_ops(
        static_cast<int>(input->obj.type().code()));
    argred_type.record_reduction_operator(
        legate::ReductionOpKind::MAX,
        legate::GlobalRedopID{ids.argmax_redop_id});
    argred_type.record_reduction_operator(
        legate::ReductionOpKind::MIN,
        legate::GlobalRedopID{ids.argmin_redop_id});
  }
  NDArray acc = runtime->create_array(scalar_shape, argred_type);
  fill_arg_identity(acc, input->obj.type(), argred_type, is_argmax);

  auto task =
      runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_SCALAR_UNARY_RED);
  task.add_reduction(acc.get_store(), is_argmax ? legate::ReductionOpKind::MAX
                                                : legate::ReductionOpKind::MIN);
  task.add_input(input->obj.get_store());
  task.add_scalar_arg(Scalar(static_cast<int32_t>(op_code)));
  task.add_scalar_arg(Scalar(input->obj.shape()));
  task.add_scalar_arg(Scalar(false));
  runtime->submit(std::move(task));

  NDArray idx = runtime->create_array(scalar_shape, legate::int64());
  idx.unary_op(
      static_cast<int32_t>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_GETARG),
      acc);
  return new CN_NDArray{NDArray(std::move(idx))};
}

extern "C" {

CN_NDArray* nda_unary_reduction_axes(CuPyNumericUnaryRedCode op_code,
                                     CN_NDArray* input, const int32_t* axes,
                                     int32_t num_axes, bool keepdims) {
  if (is_arg_reduction(op_code)) {
    return nda_arg_reduction(op_code, input);
  }
  std::vector<int32_t> axis_vec(axes, axes + num_axes);
  NDArray result = input->obj._perform_unary_reduction(
      static_cast<int32_t>(op_code), input->obj, axis_vec,
      std::nullopt,                  // dtype
      unary_red_res_dtype(op_code),  // res_dtype
      std::nullopt,                  // out
      keepdims, {},                  // args
      std::nullopt,                  // initial
      std::nullopt                   // where
  );
  return new CN_NDArray{NDArray(std::move(result))};
}

static legate::Slice to_legate_slice(const CN_Slice& slice) {
  std::optional<int64_t> start =
      slice.has_start ? std::optional<int64_t>{slice.start} : std::nullopt;
  std::optional<int64_t> stop =
      slice.has_stop ? std::optional<int64_t>{slice.stop} : std::nullopt;
  return legate::Slice(start, stop);
}

CN_NDArray* nda_get_slice(CN_NDArray* arr, const CN_Slice* slices,
                          int32_t ndim) {
  switch (ndim) {
    case 1: {
      std::initializer_list<legate::Slice> slice_list = {
          to_legate_slice(slices[0])};
      NDArray result = arr->obj[slice_list];
      return new CN_NDArray{NDArray(std::move(result))};
    }
    case 2: {
      std::initializer_list<legate::Slice> slice_list = {
          to_legate_slice(slices[0]), to_legate_slice(slices[1])};
      NDArray result = arr->obj[slice_list];
      return new CN_NDArray{NDArray(std::move(result))};
    }
    case 3: {
      std::initializer_list<legate::Slice> slice_list = {
          to_legate_slice(slices[0]), to_legate_slice(slices[1]),
          to_legate_slice(slices[2])};
      NDArray result = arr->obj[slice_list];
      return new CN_NDArray{NDArray(std::move(result))};
    }
    default:
      return nullptr;
  }
}

CN_NDArray* nda_store_to_ndarray(CN_Store* st) {
  return new CN_NDArray{cupynumeric::as_array(st->obj)};
}

// Mirrors cupynumeric deferred.searchsorted: fill + MIN/MAX reduction.
CN_NDArray* nda_searchsorted(CN_NDArray* a, CN_NDArray* v, bool left) {
  auto* runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  NDArray out = runtime->create_array(v->obj.shape(), legate::int64());
  const int64_t n = static_cast<int64_t>(a->obj.size());
  out.fill(Scalar(left ? n : int64_t{0}));

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_SEARCHSORTED);
  auto p_out =
      task.add_reduction(out.get_store(), left ? legate::ReductionOpKind::MIN
                                               : legate::ReductionOpKind::MAX);
  task.add_input(a->obj.get_store());
  auto p_v = task.add_input(v->obj.get_store());
  task.add_constraint(legate::broadcast(p_v));
  task.add_constraint(legate::broadcast(p_out));
  task.add_constraint(legate::align(p_out, p_v));
  task.add_scalar_arg(Scalar(left));
  task.add_scalar_arg(Scalar(n));
  runtime->submit(std::move(task));
  return new CN_NDArray{NDArray(std::move(out))};
}
}  // extern "C"
