#pragma once

#include "cuda/export.hpp"
#include "cuda/common.cuh"
#include "cuda/m31.cuh"
#include "cuda/scratchpad.cuh"
#include "field/M31.hpp"

namespace cuda {

__global__
static void __poly_eval_at_kernel(CudaBatchF *f_in, CudaBatchF *hg, CudaBatchF *p_out, bool *gate_exists, int32_t eval_size, int32_t var_idx) {

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto p0 = CudaF::zero();
  auto p1 = CudaF::zero();
  auto p2 = CudaF::zero();

  // if (tid >= BatchF::batch_size) {
  //   return;
  // }

  for (auto i = 0; i < eval_size; i++) {
    // NOTE: maybe broadcast is applied to gate_exists
    // TOOD: consider to replace bool with ui64 and load to shared memory

    auto left = i << (var_idx + 1);
    auto right = left + (1 << var_idx);
    if (!gate_exists[left] && !gate_exists[right]) {
      // NOTE: sparsity?
      continue;
    }

    auto f_v_0 = f_in[left].elems[tid];
    auto f_v_1 = f_in[right].elems[tid];
    auto hg_v_0 = hg[left].elems[tid];
    auto hg_v_1 = hg[right].elems[tid];

    p0 += f_v_0 * hg_v_0;
    p1 += f_v_1 * hg_v_1;
    p2 += (f_v_0 + f_v_1) * (hg_v_0 + hg_v_1);
  }
  p2 = p1 * CudaF::make(6) + p0 * CudaF::make(3) - p2 * CudaF::make(2);
  p_out[0].elems[tid] = p0;
  p_out[1].elems[tid] = p1;
  p_out[2].elems[tid] = p2;
}

__global__
static void __recv_challenge_kernel(CudaBatchF *f_in, CudaBatchF *f_out, CudaBatchF *hg, bool *gate_exists, CudaF r, int32_t eval_size, int32_t var_idx) {
  
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  // if (tid >= BatchF::batch_size) {
  //   return;
  // }

  for (auto i = 0; i < eval_size; i++) {
    auto left = i << (var_idx + 1);
    auto right = left + (1 << var_idx);
    f_out[left].elems[tid] = f_in[left].elems[tid] + (f_in[right].elems[tid] - f_in[left].elems[tid]) * r;
    hg[left].elems[tid] = (!gate_exists[left] && !gate_exists[right]) ?
      CudaF::zero() : hg[left].elems[tid] + (hg[right].elems[tid] - hg[left].elems[tid]) * r;
  }
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
  __poly_eval_at_kernel<<<CudaBatchF::grid_size, block_size>>>(
    f_in, pad->hg_evals, pad->p, pad->gate_exists, eval_size, var_idx);
  CUDA_CHECK(cudaMemcpy(pad->p_host, pad->p, sizeof(CudaBatchF) * 3,
    cudaMemcpyDeviceToHost));
  return pad->p_host;
}

void gkr_receive_challenge(CudaScratchPad *pad, int32_t eval_size, uint32_t var_idx, const void *r_ptr) {
  auto r = *reinterpret_cast<const HostF*>(r_ptr);
  auto f_in = (var_idx == 0) ? pad->v_init : pad->v_evals;
  __recv_challenge_kernel<<<CudaBatchF::grid_size, block_size>>>
    (f_in, pad->v_evals, pad->hg_evals, pad->gate_exists, CudaF::make(r), eval_size, var_idx);
  __update_gate_exists_kernel<<<div_ceil(eval_size, block_size), block_size>>>(pad->gate_exists, eval_size, var_idx);
}

} // namespace cuda