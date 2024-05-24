#pragma once

#include "cuda/export.hpp"
#include "cuda/common.cuh"
#include "field/M31.hpp"

namespace cuda {

constexpr int32_t mod = 0x7FFFFFFF;

struct CudaM31 {
  
  uint32_t v;

  __device__ __host__ __forceinline__
  static CudaM31 zero() { return CudaM31{0}; }

  __device__ __host__ __forceinline__
  static CudaM31 one() { return CudaM31{1}; }

  __device__ __host__ __forceinline__
  static constexpr CudaM31 mod_reduce(const CudaM31 &rhs) {
    return CudaM31{rhs.v % mod};
  }

  __device__ __host__ __forceinline__
  static constexpr CudaM31 mod_reduce(const int64_t& rhs) {
    uint32_t ret = rhs % mod;
    return CudaM31{ret};
  }

  __device__ __host__ __forceinline__
  static constexpr CudaM31 make(uint32_t v) {
    return CudaM31{v};
  }

  static CudaM31 make(gkr::M31_field::M31 v) {
    return CudaM31{v.x};
  }

  __device__ __host__ __forceinline__
  constexpr CudaM31 operator+(const CudaM31 &rhs) const {
    auto ret = CudaM31{v + rhs.v};
    return mod_reduce(ret);
  }

  __device__ __host__ __forceinline__
  constexpr CudaM31 operator*(const CudaM31 &rhs) const {
    int64_t rhs_i64 = rhs.v;
    int64_t lhs_i64 = v;
    return mod_reduce(lhs_i64 * rhs_i64);
  }

  __device__ __host__ __forceinline__
  constexpr CudaM31 operator-() const {
    uint32_t ret = v == 0 ? 0 : mod - v;
    return CudaM31{ret}; 
  }

  __device__ __host__ __forceinline__
  constexpr CudaM31 operator-(const CudaM31 &rhs) const {
    return *this + (-rhs);
  }

  __device__ __host__ __forceinline__
  constexpr void operator+=(const CudaM31 &rhs) {
    v = (*this + rhs).v;
  }

  __device__ __host__ __forceinline__
  constexpr bool operator==(const CudaM31 &rhs) const {
    return this->v == rhs.v;
  };
};

constexpr CudaM31 inv_2{1ll << 30};

struct CudaBatchM31 {
  static constexpr int32_t batch_size = gkr::M31_field::vectorize_size;
  static constexpr int32_t grid_size = batch_size / block_size;
  // NOTE: should equal to CPU M31 vec size
  static_assert(batch_size % block_size == 0);
  
  CudaM31 elems[batch_size];
};

ASSERT_POD(CudaM31);
ASSERT_POD(CudaBatchM31);
static_assert(sizeof(gkr::M31_field::M31) == sizeof(CudaM31));
static_assert(sizeof(gkr::M31_field::VectorizedM31) == sizeof(CudaBatchM31));

} // namespace cuda