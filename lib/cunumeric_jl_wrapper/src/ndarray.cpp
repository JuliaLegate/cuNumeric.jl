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
#include <optional>
#include <string_view>
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
  return (uint64_t)nda_array_type(arr)->obj.size() * nda_array_size(arr);
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

NDArray get_slice(NDArray arr, std::vector<legate::Slice> slices) {
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

CN_NDArray* nda_get_slice(CN_NDArray* arr, const CN_Slice* slices,
                          int32_t ndim) {
  std::vector<legate::Slice> slice_vec;
  slice_vec.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    std::optional<int64_t> start = slices[i].has_start
                                       ? std::optional<int64_t>{slices[i].start}
                                       : std::nullopt;
    std::optional<int64_t> stop = slices[i].has_stop
                                      ? std::optional<int64_t>{slices[i].stop}
                                      : std::nullopt;
    slice_vec.emplace_back(legate::Slice(start, stop));
  }
  NDArray result = get_slice(arr->obj, slice_vec);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_attach_external(const void* ptr, size_t size, int dim,
                                const uint64_t* shape, CN_Type type) {
  std::vector<uint64_t> shp_vec(shape, shape + dim);
  legate::Shape shp = legate::Shape(shp_vec);

  legate::ExternalAllocation alloc =
      legate::ExternalAllocation::create_sysmem(ptr, size);
  legate::mapping::DimOrdering ordering =
      legate::mapping::DimOrdering::fortran_order();

  auto store = legate::Runtime::get_runtime()->create_store(shp, type.obj,
                                                            alloc, ordering);
  return new CN_NDArray{cupynumeric::as_array(store)};
}
}  // extern "C"
