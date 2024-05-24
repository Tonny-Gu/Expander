#pragma once

#include "cuda/export.hpp"
#include "cuda/common.cuh"
#include "cuda/m31.cuh"
#include "cuda/scratchpad.cuh"
#include "field/M31.hpp"

#include <cub/cub.cuh>

namespace cuda {

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

template <int64_t block_dim_y = 16>
__global__
static void __poly_eval_at_kernel(CudaBatchF *f_in, CudaBatchF *hg, CudaBatchF *p_out, bool *gate_exists, int32_t eval_size, int32_t var_idx) {

  auto tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto tid_y = threadIdx.y;
  auto p = Pack3<CudaF>{CudaF::zero(), CudaF::zero(), CudaF::zero()};

  // if (tid >= BatchF::batch_size) {
  //   return;
  // }

  for (auto i = tid_y; i < eval_size; i += block_dim_y) {
    // NOTE: maybe broadcast is applied to gate_exists
    // TOOD: consider to replace bool with ui64 and load to shared memory

    auto left = i << (var_idx + 1);
    auto right = left + (1 << var_idx);
    if (!gate_exists[left] && !gate_exists[right]) {
      // NOTE: sparsity?
      continue;
    }

    auto f_v_0 = f_in[left].elems[tid_x];
    auto f_v_1 = f_in[right].elems[tid_x];
    auto hg_v_0 = hg[left].elems[tid_x];
    auto hg_v_1 = hg[right].elems[tid_x];

    p.v0 += f_v_0 * hg_v_0;
    p.v1 += f_v_1 * hg_v_1;
    p.v2 += (f_v_0 + f_v_1) * (hg_v_0 + hg_v_1);
  }

  if (tid_x < 2 && block_dim_y == 2) {
    p.v0.v = tid_x * 10 + tid_y;
    // printf("(x%d, y%d) p0=%u\n",
    //   tid_x, tid_y, p.v0.v);
  } 

  if constexpr (block_dim_y > 1) {
    using BlockReduce = cub::BlockReduce<Pack3<CudaM31>, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_dim_y>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Pack3<CudaF> p_sum = BlockReduce(temp_storage).Sum(p);

    if (tid_x < 2 && block_dim_y == 2) {
      printf("(x%d, y%d) p0=%u -> %u\n",
        tid_x, tid_y, p.v0.v, p_sum.v0.v);
    } 

    if (tid_y > 0) {
      return;
    }

    // p = p_sum;

    
  }
  
  p_out[0].elems[tid_x] = p.v0;
  p_out[1].elems[tid_x] = p.v1;
  p_out[2].elems[tid_x] = p.v1 * CudaF::make(6) + p.v0 * CudaF::make(3) - p.v2 * CudaF::make(2);
}

__global__
static void __recv_challenge_kernel(CudaBatchF *f_in, CudaBatchF *f_out, CudaBatchF *hg, bool *gate_exists, CudaF r, int32_t eval_size, int32_t var_idx) {
  
  auto tid_x = threadIdx.x + blockIdx.y * blockDim.y;
  auto tid_y = blockIdx.x; // max grid size (2**32-1, 65535, 65535). use dim x for tid_y due to this limitation.

  auto left = tid_y << (var_idx + 1);
  auto right = left + (1 << var_idx);

  f_out[left].elems[tid_x] = f_in[left].elems[tid_x] + (f_in[right].elems[tid_x] - f_in[left].elems[tid_x]) * r;
  hg[left].elems[tid_x] = (!gate_exists[left] && !gate_exists[right]) ?
    CudaF::zero() : hg[left].elems[tid_x] + (hg[right].elems[tid_x] - hg[left].elems[tid_x]) * r;
}

__global__
static void __update_gate_exists_kernel(bool *gate_exists, int32_t eval_size, int32_t var_idx) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= eval_size) {
    return;
  }

  auto left = tid << (var_idx + 1);
  auto right = left + (1 << var_idx);

  gate_exists[left] = gate_exists[left] || gate_exists[right];
}

CudaBatchF* gkr_poly_eval_at(CudaScratchPad *pad, int32_t eval_size, uint32_t var_idx, uint32_t degree) {
  auto f_in = (var_idx == 0) ? pad->v_init : pad->v_evals;
  switch (eval_size) {
  case 1:
  
  case 4:
  case 8:
  case 16:
  // default:
    __poly_eval_at_kernel<1><<<CudaBatchF::grid_size, block_size>>>(
      f_in, pad->hg_evals, pad->p, pad->gate_exists, eval_size, var_idx);
    break;
  
  case 2:
    __poly_eval_at_kernel<2><<<1, dim3(2, 2)>>>(
      f_in, pad->hg_evals, pad->p, pad->gate_exists, eval_size, var_idx);
    break;
  default:
    __poly_eval_at_kernel<32><<<CudaBatchF::batch_size / 32, dim3(32, 32)>>>(
      f_in, pad->hg_evals, pad->p, pad->gate_exists, eval_size, var_idx);
    break;
  }
  
  CUDA_CHECK(cudaMemcpy(pad->p_host, pad->p, sizeof(CudaBatchF) * 3,
    cudaMemcpyDeviceToHost));
  return pad->p_host;
}

void gkr_receive_challenge(CudaScratchPad *pad, int32_t eval_size, uint32_t var_idx, const void *r_ptr) {
  auto r = *reinterpret_cast<const HostF*>(r_ptr);
  auto f_in = (var_idx == 0) ? pad->v_init : pad->v_evals;
  __recv_challenge_kernel<<<dim3(eval_size, CudaBatchF::grid_size), block_size>>>
    (f_in, pad->v_evals, pad->hg_evals, pad->gate_exists, CudaF::make(r), eval_size, var_idx);
  __update_gate_exists_kernel<<<div_ceil(eval_size, block_size), block_size>>>(pad->gate_exists, eval_size, var_idx);
}

} // namespace cuda