#ifndef CATZILLA_CATZ_MATRIX_H_
#define CATZILLA_CATZ_MATRIX_H_

#include <algorithm>
#include <cassert>
// #include <concepts>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "coord.h"
#include "cuda_utils.h"
#include "macro.h"

// TODO: rename to matrix.h
namespace catz {

// TODO: add conditional compilation. ShapeType should be CoordS
template <typename T, typename ShapeType, typename StrideType>
struct Matrix {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");
  T *data;
  ShapeType shape;
  StrideType stride;

  constexpr __device__ Matrix(T *dt, ShapeType sp, StrideType st)
      : data(dt), shape(sp), stride(st) {}

  constexpr __device__ Matrix(const Matrix &other)
      : data(other.data), shape(other.shape), stride(other.stride) {}

  // move
  constexpr __device__ Matrix(Matrix &&other) noexcept
      : data(other.data), shape(other.shape), stride(other.stride) {
    other.data = nullptr;
  }

  ///////////////////////////////////////////////////////////
  /// dataflow related methods
  ///////////////////////////////////////////////////////////

  // TODO: make it device'd code
  template <typename TileVarType, typename NewShapeType>
  constexpr inline __device__ auto tile(const TileVarType &tile_var,
                                        const NewShapeType &new_shape) {
    auto new_data = data + tile_var.rows * new_shape.rows * stride.rows +
                    tile_var.cols * new_shape.cols * stride.cols;
    return Matrix<T, NewShapeType, StrideType>(new_data, new_shape, stride);
  }

  template <typename TileVarType>
  constexpr inline __device__ auto dist_to(const TileVarType &tile_var) {
    // shape = 8 x 2
    // y in 0-4, x in 0-4
    auto new_data =
        data + tile_var.rows * stride.rows + tile_var.cols * stride.cols;
    auto ret = Matrix<T, ShapeType, StrideType>(new_data, shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ Matrix dist_to(int dist_id) {
    auto row_in_current = IndexDyn(dist_id) / shape.cols;
    auto col_in_current = IndexDyn(dist_id) % shape.cols;
    Matrix ret = Matrix(data + row_in_current * stride.rows +
                            col_in_current * stride.cols,
                        shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ Matrix dist_to_thread() {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    auto lane_id = IndexDyn(threadIdx.y * blockDim.x + threadIdx.x);
    auto row_in_current = lane_id / shape.cols;
    auto col_in_current = lane_id % shape.cols;
    Matrix ret = Matrix(data + row_in_current * stride.rows +
                            col_in_current * stride.cols,
                        shape, stride);
    return std::move(ret);
  }

  ///////////////////////////////////////////////////////////
  /// tensor-core related methods
  ///////////////////////////////////////////////////////////

  template <typename U = T>
  inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      load_fragments(unsigned *loader) {
    // TODO: add loader length check
    auto lane_id = IndexDyn((threadIdx.y * blockDim.x + threadIdx.x) % I32());
    auto lane_rank = lane_id % shape.rows;
    auto lane_group = lane_id / shape.rows;
    const half *data_ptr = data + lane_rank * stride.rows + lane_group * I8();
    if (shape.rows() == 16 && shape.cols() == 8) {
      // LHS X2
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(loader[0]), "=r"(loader[1])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows() == 16 && shape.cols() == 16) {
      // RHS X1
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(loader[0]), "=r"(loader[1]), "=r"(loader[2]), "=r"(loader[3])
          : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows() == 8 && shape.cols() == 8) {
      // LHS X4
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                   : "=r"(loader[0])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows() == 8 && shape.cols() == 16) {
      // RHS X2
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
          : "=r"(loader[0]), "=r"(loader[1])
          : "r"(get_smem_ptr(data_ptr)));
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      load_fragments_c(float *loader) {
    auto lane_id = IndexDyn((threadIdx.y * blockDim.x + threadIdx.x) % I32());
    if (shape.rows() == 16 && shape.cols() == 8) {
      loader[0] =
          data[(lane_id / I4()) * stride.rows + (lane_id % I4()) * I2()];
      loader[1] =
          data[(lane_id / I4()) * stride.rows + (lane_id % I4()) * I2() + I1()];
      loader[2] = data[(lane_id / I4()) * stride.rows +
                       (lane_id % I4()) * I2() + I8() * stride.rows];
      loader[3] = data[(lane_id / I4()) * stride.rows +
                       (lane_id % I4()) * I2() + I8() * stride.rows + I1()];
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      store_fragments_c(float *storer) {
    auto lane_id = IndexDyn((threadIdx.y * blockDim.x + threadIdx.x) % I32());
    if (shape.rows == 16 && shape.cols == 8) {
      data[(lane_id / I4()) * stride.rows + (lane_id % I4()) * I2()] =
          storer[0];
      data[(lane_id / I4()) * stride.rows + (lane_id % I4()) * I2() + I1()] =
          storer[1];
      data[(lane_id / I4()) * stride.rows + (lane_id % I4()) * I2() +
           I8() * stride.rows] = storer[2];
      data[(lane_id / I4()) * stride.rows + (lane_id % I4()) * I2() +
           I8() * stride.rows + I1()] = storer[3];
    }
  }

  ///////////////////////////////////////////////////////////
  /// operators
  ///////////////////////////////////////////////////////////

  template <typename NewShapeType, typename NewStrideType>
  constexpr __device__ void
  operator=(const Matrix<T, NewShapeType, NewStrideType> &other) {
    *data = *other.data;
  }

  constexpr __device__ void operator=(const T other_scalar) {
    *data = other_scalar;
  }

  // half <= float
  template <typename U = T, typename NewShapeType, typename NewStrideType>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator=(const Matrix<float, NewShapeType, NewStrideType> &other) {
    *data = __float2half(*other.data);
  }

  // float <= half
  template <typename U = T, typename NewShapeType, typename NewStrideType>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      operator=(const Matrix<half, NewShapeType, NewStrideType> &other) {
    *data = __half2float(*other.data);
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator=(const float other_scalar) {
    *data = __float2half(other_scalar);
  }

  // operator '<=' is syntax sugar that combines dist-to-threads and '='
  // operator
  constexpr inline __device__ void operator<=(const Matrix &other) {
    // can make 11352
    auto total_threads = IndexDyn(blockDim.x * blockDim.y);
    // int total_threads = 256;
    auto total_elements = shape.rows * shape.cols;
    // int total_elements = 128 * 32;
    auto thread_id = IndexDyn(threadIdx.y * blockDim.x + threadIdx.x);

    auto row_this = thread_id / shape.cols;
    auto col_this = thread_id % shape.cols;
    auto row_other = thread_id / other.shape.cols;
    auto col_other = thread_id % other.shape.cols;

    // TODO: make this config more general
    // #pragma unroll 8
    // NOTE: we need constexpr for avoiding heavy runtime index calc
    // compile-time symbolic index calc, become perf drop
    // for (int i = 0; i < total_elements / total_threads; i++)
    //   (data)[i * total_threads * 32 / 32
    //          + row_this * 32 + col_this]
    //     = ((other.data))[i * total_threads *4096
    //                        /32
    //                      + row_other * 4096 + col_other];
#pragma unroll 16
    for (int i = 0; i < (total_elements / total_threads).value; i++)
      (data)[(i * total_threads * stride.rows / shape.cols +
              row_this * stride.rows + col_this)
                 .value] =
          ((other.data))[(i * total_threads * other.stride.rows /
                              other.shape.cols +
                          row_other * other.stride.rows + col_other)
                             .value];
  }

  //   for (int m = 0; m < 8/* CEIL_DIV(M_TILE, 2) */; m++) {
  //     lhs_shared_mat.tile(Coord(m, kin), Coord(2, 16)).dist_to_thread()
  //       = lhs_mat.tile(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
  //           .tile(Coord(m, kin), Coord(2, 16))
  //           .dist_to_thread();
  //   }
  template <typename U = T, typename NewShapeType, typename NewStrideType>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<=(const Matrix<float, NewShapeType, NewStrideType> &other) {
    int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;

    // OPTION I: normalised loop, with full symbolic index calculation
    // ~ 4180 GFLOPS
#pragma unroll 8
    for (int i = 0; i < total_elements / total_threads; i++)
      (data)[i * total_threads * stride.rows / shape.cols +
             row_this * stride.rows + col_this] =
          ((other.data))[i * total_threads * other.stride.rows /
                             other.shape.cols +
                         row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 1940 GFLOPS
    // for (int i = 0; i < total_elements; i+=total_threads)
    //   (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //   col_this]
    //     = ((other.data))[i * other.stride.rows / other.shape.cols
    //                      + row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 2800 GFLOPS
    // for (int i = 0; i < total_elements; i+=32)
    //   (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //   col_this]
    //     = ((other.data))[i * other.stride.rows / other.shape.cols
    //                      + row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with more-of-literal calculation
    // ~ 2790 GFLOPS
    //
    // for (int i = 0; i < 256; i+=32)
    //   (data)[i * 16 / shape.cols + row_this * 16 + col_this]
    //     = ((other.data))[i * 4096 / other.shape.cols
    //                      + row_other * 4096 + col_other];

    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.rows / shape.cols + row_this *
    //     stride.rows + col_this]
    //       = ((float4*)(other.data))[i * other.stride.rows /
    //       other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
    // } else {
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //     col_this]
    //       = ((other.data))[i * other.stride.rows / other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
    // }
  }

  // TODO: merge it
  // use threads in one wrap to move it,
  // replace m with threadIdx.y
  //
  // for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
  //   out_mat.tile(Coord(blockIdx.x, blockIdx.y), out_sm_tile_shape)
  //       .tile(Coord(m, n), out_reg_tile_shape)
  //     <<= out_shared_mat.tile(Coord(m, n), out_reg_tile_shape);
  // }
  // special operator that allows any same-volume copy, when the volume is
  // dividable by threads volume
  constexpr inline __device__ void operator<<=(const Matrix &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    // int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.rows / shape.cols + row_this * stride.rows + col_this] =
          other.data[i * other.stride.rows / other.shape.cols +
                     row_other * other.stride.rows + col_other];
    }
  }

  template <typename U = T, typename NewShapeType, typename NewStrideType>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<<=(const Matrix<float, NewShapeType, NewStrideType> &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.rows / shape.cols + row_this * stride.rows + col_this] =
          __float2half(other.data[i * other.stride.rows / other.shape.cols +
                                  row_other * other.stride.rows + col_other]);
    }
  }

  ///////////////////////////////////////////////////////////
  /// utils methods
  ///////////////////////////////////////////////////////////

  inline __device__ void fill(T value) {
    auto flat_id = IndexDyn(threadIdx.y * blockDim.x + threadIdx.x);
    auto row_in_current = flat_id / shape.cols;
    auto col_in_current = flat_id % shape.cols;
    auto elements = shape.rows * shape.cols;
    auto threads = IndexDyn(blockDim.x * blockDim.y);
    for (int chunk = 0; chunk < CEIL_DIV(elements.value, threads.value);
         chunk++) {
      data[chunk * threads + flat_id] = value;
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      print() const {
    for (size_t i = 0; i < shape.rows; ++i) {
      for (size_t j = 0; j < shape.cols; ++j) {
        std::cout << data[i * shape.rows + j * shape.cols] << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
};

// #define make_matrix(DT, SP, ST) Matrix(DT, SP, ST)

#define make_matrix(DT, SP) Matrix(DT, SP, Coord((SP).cols, I1()))

} // namespace catz

#endif // CATZILLA_CATZ_MATRIX_H_
