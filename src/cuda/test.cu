// #include <__clang_cuda_builtin_vars.h>
#include <cstdint>

namespace cuda {

namespace m31 {

constexpr int32_t mod = 0x7FFFFFFF;

struct M31 {
  
  uint32_t v;

  __device__ __host__
  constexpr M31 ModReduce(const M31 &rhs) const {
    return M31{rhs.v % mod};
  }

  __device__ __host__
  constexpr M31 ModReduce(const int64_t& rhs) const {
    uint32_t ret = rhs % mod;
    return M31{ret};
  }

  __device__ __host__
  constexpr M31 operator+(const M31 &rhs) const {
    auto ret = M31{v + rhs.v};
    return ModReduce(ret);
  }

  __device__ __host__
  constexpr M31 operator*(const M31 &rhs) const {
    int64_t rhs_i64 = rhs.v;
    int64_t lhs_i64 = v;
    return ModReduce(lhs_i64 * rhs_i64);
  }

  __device__ __host__
  constexpr M31 operator-() const {
    uint32_t ret = v == 0 ? 0 : mod - v;
    return M31{ret}; 
  }

  __device__ __host__
  constexpr M31 operator-(const M31 &rhs) const {
    return *this + (-rhs);
  }

  __device__ __host__
  constexpr bool operator==(const M31 &rhs) const {
    return this->v == rhs.v;
  };
};

constexpr M31 inv_2{1ll << 30};

struct PackedM31 {
  static constexpr int32_t batch_size = 65536;
  static_assert(batch_size % 32 == 0);
  
  M31 elems[batch_size];
};

};

}; 