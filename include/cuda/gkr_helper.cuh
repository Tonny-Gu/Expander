#pragma once

#include "cuda/export.hpp"
#include "cuda/common.cuh"
#include "cuda/m31.cuh"
#include "cuda/scratchpad.cuh"
#include "cuda/circuit.cuh"
#include "circuit/circuit.hpp"
#include "field/M31.hpp"
#include <cstdint>

namespace cuda {

__global__
static void __poly_eval_at_kernel_phase1(CudaBatchF *f_in, CudaBatchF *hg, Pack3<CudaBatchF*> p_psum, bool *gate_exists, int32_t eval_size, int32_t var_idx) {
  auto tid_x = threadIdx.x + blockIdx.y * blockDim.x;
  auto tid_y = blockIdx.x;

  // if (tid >= BatchF::batch_size) {
  //   return;
  // }

  auto p0 = CudaF::zero();
  auto p1 = CudaF::zero();
  auto p2 = CudaF::zero();

  for (auto i = tid_y; i < eval_size; i += gridDim.x) {
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

    p0 += f_v_0 * hg_v_0;
    p1 += f_v_1 * hg_v_1;
    p2 += (f_v_0 + f_v_1) * (hg_v_0 + hg_v_1);
  }

  p_psum.v0[tid_y].elems[tid_x] = p0;
  p_psum.v1[tid_y].elems[tid_x] = p1;
  p_psum.v2[tid_y].elems[tid_x] = p2;
}

__global__
static void __poly_eval_at_kernel_phase2(Pack3<CudaBatchF*> p_psum, int32_t eval_size, int32_t var_idx) {
  // in-place 2d reduction
  auto tid_x = threadIdx.x + blockIdx.y * blockDim.x;
  auto tid_y = blockIdx.x;

  auto p0 = CudaF::zero();
  auto p1 = CudaF::zero();
  auto p2 = CudaF::zero();

  // #pragma unroll
  for (auto i = tid_y; i < eval_size; i += gridDim.x) {
    auto left = i << (var_idx + 1);
    auto right = left + (1 << var_idx);

    p0 += p_psum.v0[left].elems[tid_x] + p_psum.v0[right].elems[tid_x];
    p1 += p_psum.v1[left].elems[tid_x] + p_psum.v1[right].elems[tid_x];
    p2 += p_psum.v2[left].elems[tid_x] + p_psum.v2[right].elems[tid_x];

    // if (tid_x == 0) {
    //   printf("(blk=%d, eval=%d, i=%d) reduce: %u (%d) + %u (%d) = %u\n",
    //     tid_y, eval_size, var_idx, p_psum.v0[left].elems[tid_x].v, left, p_psum.v0[right].elems[tid_x].v, right, p0.v);
    // }
  }

  auto left = tid_y << (var_idx + 1);
  p_psum.v0[left].elems[tid_x] = p0;
  p_psum.v1[left].elems[tid_x] = p1;
  p_psum.v2[left].elems[tid_x] = p2;

  // if (tid_x == 0) {
  //   printf("(blk=%d, eval=%d, i=%d) write: %u (%d)\n",
  //     tid_y, eval_size, var_idx, p0.v, left);
  // }
  
}

__global__
static void __poly_eval_at_kernel_phase3(Pack3<CudaBatchF*> p_psum, CudaBatchF* p) {
  auto tid_x = threadIdx.x + blockIdx.y * blockDim.x;

  auto p0 = p_psum.v0[0].elems[tid_x];
  auto p1 = p_psum.v1[0].elems[tid_x];
  auto p2 = p_psum.v2[0].elems[tid_x];

  p[0].elems[tid_x] = p0;
  p[1].elems[tid_x] = p1;
  p[2].elems[tid_x] = p1 * CudaF::make(6) + p0 * CudaF::make(3) - p2 * CudaF::make(2);
}

__global__
static void __recv_challenge_kernel(CudaBatchF *f_in, CudaBatchF *f_out, CudaBatchF *hg, bool *gate_exists, CudaF r, int32_t eval_size, int32_t var_idx) {
  
  auto tid_x = threadIdx.x + blockIdx.y * blockDim.x;
  auto tid_y = blockIdx.x; // max grid size (2**32-1, 65535, 65535). use dim x for tid_y due to this limitation.

  auto left = tid_y << (var_idx + 1);
  auto right = left + (1 << var_idx);

  f_out[left].elems[tid_x] = f_in[left].elems[tid_x] + (f_in[right].elems[tid_x] - f_in[left].elems[tid_x]) * r;
  hg[left].elems[tid_x] = (!gate_exists[left] && !gate_exists[right]) ?
    CudaF::zero() : hg[left].elems[tid_x] + (hg[right].elems[tid_x] - hg[left].elems[tid_x]) * r;
  
  // if (tid_x == 0 && tid_y == 0) {
  //   printf("rc: %u\n", f_out[0].elems[0].v);
  // }
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

CudaBatchF* gkr_poly_eval_at(CudaScratchPad *pad, CudaCircuitLayer *layer, int32_t eval_size, uint32_t var_idx, uint32_t degree) {
  auto f_in = (var_idx == 0) ? layer->input_layer_vals.get() : pad->v_evals;
  auto p_psum = Pack3<CudaBatchF*>{pad->p0, pad->p1, pad->p2};
  constexpr int64_t elem_per_thread = 32;
  assert(__builtin_popcount(eval_size) == 1);
  auto grid_dim_x = div_ceil(eval_size, elem_per_thread);

  __poly_eval_at_kernel_phase1<<<dim3(grid_dim_x, CudaBatchF::grid_size), block_size>>>(
    f_in, pad->hg_evals, p_psum, pad->gate_exists, eval_size, var_idx);

  auto i = 0;
  while (grid_dim_x > 1) {
    auto psum_len = grid_dim_x / 2;
    grid_dim_x = div_ceil(psum_len, elem_per_thread); 
    __poly_eval_at_kernel_phase2<<<dim3(grid_dim_x, CudaBatchF::grid_size), block_size>>>
      (p_psum, psum_len, i);
    i += 1;
  }

  __poly_eval_at_kernel_phase3<<<dim3(1, CudaBatchF::grid_size), block_size>>>(p_psum, pad->p);
  
  CUDA_CHECK(cudaMemcpy(pad->p_host, pad->p, sizeof(CudaBatchF) * 3,
    cudaMemcpyDeviceToHost));

  // printf("eval: %u\n", pad->p_host[0].elems[0].v);
  return pad->p_host;
}

void gkr_receive_challenge(CudaScratchPad *pad, CudaCircuitLayer *layer, int32_t eval_size, uint32_t var_idx, const void *r_ptr) {
  auto r = *reinterpret_cast<const HostF*>(r_ptr);
  auto f_in = (var_idx == 0) ? layer->input_layer_vals.get() : pad->v_evals;
  __recv_challenge_kernel<<<dim3(eval_size, CudaBatchF::grid_size), block_size>>>
    (f_in, pad->v_evals, pad->hg_evals, pad->gate_exists, CudaF::make(r), eval_size, var_idx);
  __update_gate_exists_kernel<<<div_ceil(eval_size, block_size), block_size>>>(pad->gate_exists, eval_size, var_idx);
}

template <int64_t nb_input> __global__
static void __prepare_g_x_kernel(CudaGate<nb_input>* gates, int64_t num_gates, CudaBatchF *init_v, CudaBatchF *hg,
    CudaF *eq_evals_at_rz1, bool *gate_exists) {
  auto tid_x = threadIdx.x + blockIdx.y * blockDim.x;
  auto tid_y = blockIdx.x;

  if (tid_y > 0 && gates[tid_y].i_ids[0] == gates[tid_y - 1].i_ids[0]) {
    // backoff to avoid race condition (different blocks write to same hg[x] concurrently). i_ids must be sorted.
    return;
  }

  for (auto i = tid_y; ; i++) {
    auto& gate = gates[i];
    auto& x = gate.i_ids[0];
    auto& z = gate.o_id;
    auto& coef = gate.coef;

    if constexpr (nb_input == 1) {
      hg[x].elems[tid_x] += coef * eq_evals_at_rz1[z];
    } else if constexpr (nb_input == 2) {
      static_assert(nb_input == 2);
      auto& y = gate.i_ids[1];
      hg[x].elems[tid_x] += init_v[y].elems[tid_x] * coef * eq_evals_at_rz1[z];
    }

    gate_exists[x] = true;    

    if (!(i + 1 < num_gates && gates[i].i_ids[0] == gates[i + 1].i_ids[0])) {
      // do things for backoff blocks.
      break;
    }
  }
}

void gkr_prepare_g_x_vals(CudaScratchPad *pad, const void *layer_host, const void *eq_evals_at_rz1, int64_t rz1_len) {
  const auto& layer = *reinterpret_cast<const gkr::CircuitLayer<HostBatchF, HostF>*>(layer_host);
  const auto& layer_gpu = *layer.layer_gpu;

  __clear_device(pad->hg_evals, rz1_len);
  __clear_device(pad->gate_exists, rz1_len);
  __copy_h2d<CudaF>(pad->eq_evals_at_rz1, eq_evals_at_rz1, rz1_len);

  auto mul_size = layer.mul.sparse_evals.size();
  auto add_size = layer.add.sparse_evals.size();
  auto init_v = layer_gpu.input_layer_vals.get();

  __prepare_g_x_kernel<2><<<dim3(mul_size, CudaBatchF::grid_size), block_size>>>(
    layer_gpu.mul_wires_x.get(), mul_size, init_v, pad->hg_evals,
    pad->eq_evals_at_rz1, pad->gate_exists);

  __prepare_g_x_kernel<1><<<dim3(add_size, CudaBatchF::grid_size), block_size>>>(
    layer_gpu.add_wires.get(), add_size, init_v, pad->hg_evals,
    pad->eq_evals_at_rz1, pad->gate_exists);
}

__global__
static void __prepare_h_y_kernel(CudaGate<2>* gates, int64_t num_gates, CudaBatchF *v, CudaBatchF *hg,
    CudaF *eq_evals_at_rz1, CudaF *eq_evals_at_rx, bool *gate_exists) {
  auto tid_x = threadIdx.x + blockIdx.y * blockDim.x;
  auto tid_y = blockIdx.x;

  if (tid_y > 0 && gates[tid_y].i_ids[1] == gates[tid_y - 1].i_ids[1]) {
    // backoff to avoid race condition (different blocks write to same hg[x] concurrently). i_ids must be sorted.
    // printf("gpu b%d: backoff\n", tid_y);
    return;
  }

  for (auto i = tid_y; ; i++) {
    auto& gate = gates[i];
    auto& x = gate.i_ids[0];
    auto& y = gate.i_ids[1];
    auto& z = gate.o_id;
    auto& coef = gate.coef;
      
    hg[y].elems[tid_x] += v[0].elems[tid_x] * coef * eq_evals_at_rz1[z] * eq_evals_at_rx[x];
    gate_exists[y] = true;

    if (tid_x == 0) {
      // printf("gpu b%d: updated %d %u\n", tid_y, y, hg[y].elems[0].v);
    }

    if (!(i + 1 < num_gates && gates[i].i_ids[1] == gates[i + 1].i_ids[1])) {
      // do things for backoff blocks.
      break;
    }
  }
}

void gkr_prepare_h_y_vals(CudaScratchPad *pad, const void *layer_host, const void *eq_evals_at_rx, int64_t rx_len) {
  const auto& layer = *reinterpret_cast<const gkr::CircuitLayer<HostBatchF, HostF>*>(layer_host);
  const auto& layer_gpu = *layer.layer_gpu;

  __clear_device(pad->hg_evals, rx_len);
  __clear_device(pad->gate_exists, rx_len);
  __copy_h2d<CudaF>(pad->eq_evals_at_rx, eq_evals_at_rx, rx_len);

  auto mul_size = layer.mul.sparse_evals.size();

  __prepare_h_y_kernel<<<dim3(mul_size, CudaBatchF::grid_size), block_size>>>(
    layer_gpu.mul_wires_y.get(), mul_size, pad->v_evals, pad->hg_evals,
    pad->eq_evals_at_rz1, pad->eq_evals_at_rx, pad->gate_exists);
}

} // namespace cuda