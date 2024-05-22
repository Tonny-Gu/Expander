#pragma once

#include "cuda/export.hpp"
#include "cuda/common.cuh"
#include "cuda/m31.cuh"
#include "cuda/scratchpad.cuh"
#include "field/M31.hpp"

namespace cuda {

__global__
static void __poly_eval_at_kernel(CudaBatchF *f_in, CudaBatchF *hg, CudaBatchF *p_out, int32_t eval_size, bool *gate_exists) {

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
  p2 = p1 * CudaF::make(6) + p0 * CudaF::make(3) - p2 * CudaF::make(2);
  p_out[0].elems[tid] = p0;
  p_out[1].elems[tid] = p1;
  p_out[2].elems[tid] = p2;
}

__global__
static void __recv_challenge_kernel(CudaBatchF *f_in, CudaBatchF *f_out, CudaBatchF *hg, CudaF r, int32_t eval_size, bool *gate_exists) {
  
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  // if (tid >= BatchF::batch_size) {
  //   return;
  // }

  for (auto i = 0; i < eval_size; i++) {
    f_out[i].elems[tid] = f_in[2 * i].elems[tid] + (f_in[2 * i + 1].elems[tid] - f_in[2 * i].elems[tid]) * r;
    hg[i].elems[tid] = (!gate_exists[i * 2] && !gate_exists[i * 2 + 1]) ?
      CudaF::zero() : hg[2 * i].elems[tid] + (hg[2 * i + 1].elems[tid] - hg[2 * i].elems[tid]) * r;
  }
}

__global__
static void __update_gate_exists_kernel(int32_t eval_size, bool *gate_exists) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= eval_size) {
    return;
  }

  gate_exists[tid] = gate_exists[tid * 2] || gate_exists[tid * 2 + 1];
}

CudaBatchF* gkr_poly_eval_at(CudaScratchPad *pad, int32_t eval_size, uint32_t var_idx, uint32_t degree) {
  auto f_in = (var_idx == 0) ? pad->v_init : pad->v_evals;
  __poly_eval_at_kernel<<<CudaBatchF::grid_size, block_size>>>(
    f_in, pad->hg_evals, pad->p, eval_size, pad->gate_exists);
  CUDA_CHECK(cudaMemcpy(pad->p_host, pad->p, sizeof(CudaBatchF) * 3,
    cudaMemcpyDeviceToHost));
  return pad->p_host;
}

void gkr_receive_challenge(CudaScratchPad *pad, int32_t eval_size, uint32_t var_idx, const void *r_ptr) {
  auto r = *reinterpret_cast<const HostF*>(r_ptr);
  auto f_in = (var_idx == 0) ? pad->v_init : pad->v_evals;
  __recv_challenge_kernel<<<CudaBatchF::grid_size, block_size>>>
    (f_in, pad->v_evals, pad->hg_evals, CudaF::make(r), eval_size, pad->gate_exists);
  __update_gate_exists_kernel<<<div_ceil(eval_size, block_size), block_size>>>(eval_size, pad->gate_exists);
}

} // namespace cuda