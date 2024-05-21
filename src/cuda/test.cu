#include "LinearGKR/cuda.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <assert.h>
#include <vector>

namespace cuda {

#define CUDA_CHECK(ret) __cuda_check((ret), #ret, __FILE__, __LINE__)

void __cuda_check(cudaError_t err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::abort();
  }
}

namespace m31 {

constexpr int32_t mod = 0x7FFFFFFF;

struct M31 {
  
  uint32_t v;

  __device__ __host__
  static M31 zero() { return M31{0}; }

  __device__ __host__
  static M31 one() { return M31{1}; }

  __device__ __host__
  static constexpr M31 mod_reduce(const M31 &rhs) {
    return M31{rhs.v % mod};
  }

  __device__ __host__
  static constexpr M31 mod_reduce(const int64_t& rhs) {
    uint32_t ret = rhs % mod;
    return M31{ret};
  }

  __device__ __host__
  static constexpr M31 make(uint32_t v) {
    return M31{v};
  }

  __device__ __host__
  constexpr M31 operator+(const M31 &rhs) const {
    auto ret = M31{v + rhs.v};
    return mod_reduce(ret);
  }

  __device__ __host__
  constexpr M31 operator*(const M31 &rhs) const {
    int64_t rhs_i64 = rhs.v;
    int64_t lhs_i64 = v;
    return mod_reduce(lhs_i64 * rhs_i64);
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
  constexpr void operator+=(const M31 &rhs) {
    v = (*this + rhs).v;
  }

  __device__ __host__
  constexpr bool operator==(const M31 &rhs) const {
    return this->v == rhs.v;
  };
};

constexpr M31 inv_2{1ll << 30};

}; // namespace m31

namespace gkr {

struct BatchF {
  static constexpr int32_t batch_size = 32;
  static constexpr int32_t grid_size = batch_size / block_size;
  // NOTE: should equal to CPU M31 vec size
  static_assert(batch_size % block_size == 0);
  
  F elems[batch_size];
};

template <typename T>
static void __allocate(T*& ptr, int64_t num_elems) {
  CUDA_CHECK(cudaMalloc(&ptr, num_elems * sizeof(T)));
}

template <typename T>
static void __allocate_host(T*& ptr, int64_t num_elems) {
  CUDA_CHECK(cudaMallocHost(&ptr, num_elems * sizeof(T)));
}

static void __deallocate(void *ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

static void __deallocate_host(void *ptr) {
  CUDA_CHECK(cudaFreeHost(ptr));
}

void ScratchPad::init(int64_t max_nb_output, int64_t max_nb_input) {
  this->max_nb_input = max_nb_input;
  this->max_nb_output = max_nb_output;
  __allocate(v_init, max_nb_input);
  __allocate(v_evals, max_nb_input);
  __allocate(hg_evals, max_nb_input);
  __allocate(p, 3);
  __allocate_host(v_evals_host, max_nb_input);
  __allocate_host(hg_evals_host, max_nb_input);
  __allocate_host(p_host, 3);
  __allocate_host(gate_exists_host, max_nb_input);
  // __allocate(eq_evals_at_rx, max_nb_input);
  // __allocate(eq_evals_at_rz1, max_nb_output);
  // __allocate(eq_evals_at_rz2, max_nb_output);
  // __allocate(eq_evals_first_half, max_nb_output);
  // __allocate(eq_evals_second_half, max_nb_output);
  __allocate(gate_exists, max_nb_input);
}

void ScratchPad::deinit() {
  __deallocate(v_init);
  __deallocate(v_evals);
  __deallocate(hg_evals);
  __deallocate(p);
  __deallocate_host(v_evals_host);
  __deallocate_host(hg_evals_host);
  __deallocate_host(p_host);
  __deallocate_host(gate_exists_host);
  // __deallocate(eq_evals_at_rx);
  // __deallocate(eq_evals_at_rz1);
  // __deallocate(eq_evals_at_rz2);
  // __deallocate(eq_evals_first_half);
  // __deallocate(eq_evals_second_half);
  __deallocate(gate_exists);
}

void ScratchPad::check(void *gate_exists_ptr, void *hg_evals_ptr, void *v_evals_ptr, int64_t eval_size) {
  CUDA_CHECK(cudaMemcpy(gate_exists_host, gate_exists,
    sizeof(bool) * max_nb_input, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hg_evals_host, hg_evals,
    sizeof(BatchF) * max_nb_input, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v_evals_host, v_evals,
    sizeof(BatchF) * max_nb_input, cudaMemcpyDeviceToHost));

  auto actual_exists = reinterpret_cast<bool*>(gate_exists_ptr);
  auto actual_hg = reinterpret_cast<BatchF*>(hg_evals_ptr);
  auto actual_v = reinterpret_cast<BatchF*>(v_evals_ptr);
  for (auto i = 0ll; i < eval_size; i++) {
    assert(actual_exists[i] == gate_exists_host[i]);
    for (auto j = 0ll; j < BatchF::batch_size; j++) {
      assert(actual_hg[i].elems[j] == hg_evals_host[i].elems[j]);
      assert(actual_v[i].elems[j] == v_evals_host[i].elems[j]);
    }
  }
  printf("check pass, size=%ld\n", eval_size);
}

static_assert(std::is_trivial_v<F> && std::is_standard_layout_v<F>);
static_assert(std::is_trivial_v<BatchF> && std::is_standard_layout_v<BatchF>);
static_assert(std::is_trivial_v<ScratchPad> && std::is_standard_layout_v<ScratchPad>);

static constexpr int64_t div_ceil(int64_t a, int64_t b) {
  return a / b + (a % b > 0);
}

__global__
static void __poly_eval_at_kernel(BatchF *f_in, BatchF *hg, BatchF *p_out, int32_t eval_size, bool *gate_exists) {

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto p0 = F::zero();
  auto p1 = F::zero();
  auto p2 = F::zero();

  // if (tid >= BatchF::batch_size) {
  //   return;
  // }

  for (auto i = 0; i < eval_size; i++) {
    // NOTE: maybe broadcast is applied to gate_exists
    // TOOD: consider to replace bool with ui64 and load to shared memory
    if (!gate_exists[i * 2] && !gate_exists[i * 2 + 1]) {
      // NOTE: sparsity?
      continue;
    }

    auto f_v_0 = f_in[i * 2].elems[tid];
    auto f_v_1 = f_in[i * 2 + 1].elems[tid];
    auto hg_v_0 = hg[i * 2].elems[tid];
    auto hg_v_1 = hg[i * 2 + 1].elems[tid];

    p0 += f_v_0 * hg_v_0;
    p1 += f_v_1 * hg_v_1;
    p2 += (f_v_0 + f_v_1) * (hg_v_0 + hg_v_1);
  }
  p2 = p1 * F::make(6) + p0 * F::make(3) - p2 * F::make(2);
  p_out[0].elems[tid] = p0;
  p_out[1].elems[tid] = p1;
  p_out[2].elems[tid] = p2;
}

__global__
static void __recv_challenge_kernel(BatchF *f_in, BatchF *f_out, BatchF *hg, F r, int32_t eval_size, bool *gate_exists) {
  
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  // if (tid >= BatchF::batch_size) {
  //   return;
  // }

  for (auto i = 0; i < eval_size; i++) {
    f_out[i].elems[tid] = f_in[2 * i].elems[tid] + (f_in[2 * i + 1].elems[tid] - f_in[2 * i].elems[tid]) * r;
    hg[i].elems[tid] = (!gate_exists[i * 2] && !gate_exists[i * 2 + 1]) ?
      F::zero() : hg[2 * i].elems[tid] + (hg[2 * i + 1].elems[tid] - hg[2 * i].elems[tid]) * r;
  }
}

__global__
static void __update_gate_exists(int32_t eval_size, bool *gate_exists) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= eval_size) {
    return;
  }

  gate_exists[tid] = gate_exists[tid * 2] || gate_exists[tid * 2 + 1];
}

BatchF* SumcheckGKRHelper::poly_eval_at(int32_t eval_size, uint32_t var_idx, uint32_t degree) {
  auto f_in = (var_idx == 0) ? pad->v_init : pad->v_evals;
  __poly_eval_at_kernel<<<BatchF::grid_size, block_size>>>(
    f_in, pad->hg_evals, pad->p, eval_size, pad->gate_exists);
  CUDA_CHECK(cudaMemcpy(pad->p_host, pad->p, sizeof(BatchF) * 3,
    cudaMemcpyDeviceToHost));
  return pad->p_host;
}

void SumcheckGKRHelper::receive_challenge(int32_t eval_size, uint32_t var_idx, const uint32_t r) {
  auto f_in = (var_idx == 0) ? pad->v_init : pad->v_evals;
  __recv_challenge_kernel<<<BatchF::grid_size, block_size>>>
    (f_in, pad->v_evals, pad->hg_evals, F::make(r), eval_size, pad->gate_exists);
  __update_gate_exists<<<div_ceil(eval_size, block_size), block_size>>>(eval_size, pad->gate_exists);
}

void SumcheckGKRHelper::load_f_hg() {
  CUDA_CHECK(cudaMemcpy(pad->v_evals, v_host,
    sizeof(BatchF) * pad->max_nb_input, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pad->hg_evals, hg_host,
    sizeof(BatchF) * pad->max_nb_input, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pad->gate_exists, gate_exists_host,
    sizeof(bool) * pad->max_nb_input, cudaMemcpyHostToDevice));
}

void SumcheckGKRHelper::load_v_init(const void* v_init, int64_t len) {
  CUDA_CHECK(cudaMemcpy(pad->v_init, v_init,
    sizeof(BatchF) * len, cudaMemcpyHostToDevice));
}

void SumcheckGKRHelper::store_f_hg() {
  CUDA_CHECK(cudaMemcpy(v_host, pad->v_evals,
    sizeof(BatchF) * pad->max_nb_input, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hg_host, pad->hg_evals,
    sizeof(BatchF) * pad->max_nb_input, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gate_exists_host, pad->gate_exists,
    sizeof(bool) * pad->max_nb_input, cudaMemcpyDeviceToHost));
}

}; // namespace gkr

}; // namespace cuda