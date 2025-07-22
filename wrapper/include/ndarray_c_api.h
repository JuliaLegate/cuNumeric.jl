#ifndef NDARRAY_C_API_H
#define NDARRAY_C_API_H

#include <cupynumeric/cupynumeric_c.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int has_start;  // 0 = open, 1 = present
  int64_t start;
  int has_stop;  // 0 = open, 1 = present
  int64_t stop;
} CN_Slice;

// Opaque handle
typedef struct CN_NDArray CN_NDArray;
typedef struct CN_Type CN_Type;
typedef struct CN_Scalar CN_Scalar;

uint64_t nda_query_device_memory();

// zeros(shape, type?)
//   dim        : number of dimensions
//   shape      : pointer to array[length=dim]
//   CN_Type    : Legate type of object
CN_NDArray* nda_zeros_array(int32_t dim, const uint64_t* shape, CN_Type type);

// full(shape, value)
//   dim   : number of dimensions
//   shape : pointer to array[length=dim]
//   value : double‐precision scalar
CN_NDArray* nda_full_array(int32_t dim, const uint64_t* shape, CN_Type type,
                           const void* value);
void nda_random(CN_NDArray* arr, int32_t code);
CN_NDArray* nda_random_array(int32_t dim, const uint64_t* shape);
CN_NDArray* nda_reshape_array(CN_NDArray* arr, int32_t dim,
                              const uint64_t* shape);
CN_NDArray* nda_astype(CN_NDArray* arr, CN_Type type);
void nda_fill_array(CN_NDArray* arr, CN_Type type, const void* value);

void nda_multiply(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out);
void nda_add(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out);
CN_NDArray* nda_multiply_scalar(CN_NDArray* rhs1, CN_Type type,
                                const void* value);
CN_NDArray* nda_add_scalar(CN_NDArray* rhs1, CN_Type type, const void* value);
CN_NDArray* nda_dot(CN_NDArray* rhs1, CN_NDArray* rhs2);
void nda_three_dot_arg(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out);
CN_NDArray* nda_copy(CN_NDArray* arr);
void nda_assign(CN_NDArray* arr, CN_NDArray* other);

void nda_destroy_array(CN_NDArray* arr);

// simple queries
int32_t nda_array_dim(const CN_NDArray* arr);
uint64_t nda_array_size(const CN_NDArray* arr);
int32_t nda_array_type_code(const CN_NDArray* arr);
CN_Type* nda_array_type(const CN_NDArray* arr);
void nda_array_shape(const CN_NDArray* arr, uint64_t* out_shape);
uint64_t nda_nbytes(CN_NDArray* arr);

void nda_binary_op(CN_NDArray* out, CuPyNumericBinaryOpCode op_code,
                   const CN_NDArray* rhs1, const CN_NDArray* rhs2);
void nda_unary_op(CN_NDArray* out, CuPyNumericUnaryOpCode op_code,
                  CN_NDArray* input);
void nda_unary_reduction(CN_NDArray* out, CuPyNumericUnaryRedCode op_code,
                         CN_NDArray* input);
CN_NDArray* nda_get_slice(CN_NDArray* arr, const CN_Slice* slices,
                          int32_t ndim);

#ifdef __cplusplus
}
#endif

#endif  // NDARRAY_C_API_H
