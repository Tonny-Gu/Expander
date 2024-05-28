#pragma once

#include <iostream>
#include "field/M31.hpp"

#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#endif

namespace cuda {

using HostF = gkr::M31_field::M31;
using HostBatchF = gkr::M31_field::VectorizedM31;

template <typename T>
struct Pack3 {
  T v0, v1, v2;

  __device__ __host__ __forceinline__
  constexpr Pack3<T> operator+(const Pack3<T> &rhs) const {
    return Pack3<T>{
      v0 + rhs.v0,
      v1 + rhs.v1,
      v2 + rhs.v2
    };
  }
};

#define CUDA_CHECK(ret) __cuda_check((ret), #ret, __FILE__, __LINE__)

void __cuda_check(cudaError_t err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::abort();
  }
}

static constexpr int64_t div_ceil(int64_t a, int64_t b) {
  return a / b + (a % b > 0);
}

constexpr int32_t block_size = 256;

} // namespace cuda