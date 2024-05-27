#pragma once

#include "cuda/export.hpp"
#include "cuda/common.cuh"
#include "cuda/m31.cuh"
#include "LinearGKR/scratch_pad.hpp"
#include <cstdint>

namespace cuda {

struct CudaScratchPad {
  CudaBatchF *v_init;
  CudaBatchF *v_evals;
  CudaBatchF *hg_evals;
  CudaBatchF *p0; // partial sums
  CudaBatchF *p1;
  CudaBatchF *p2;
  CudaBatchF *p;

  CudaBatchF *v_evals_host;
  CudaBatchF *hg_evals_host;
  CudaBatchF *p_host;
  bool *gate_exists_host;

  CudaF *eq_evals_at_rx;
  CudaF *eq_evals_at_rz1;
  // F *eq_evals_at_rz2;
  // F *eq_evals_first_half;
  // F *eq_evals_second_half;
  
  bool *gate_exists;
  int64_t max_nb_input;
  int64_t max_nb_output;
};

template <typename T>
void __clear_device(T* ptr, int64_t num_elems) {
  CUDA_CHECK(cudaMemset(ptr, 0, num_elems * sizeof(T)));
}

template <typename T>
static void __allocate_device(T*& ptr, int64_t num_elems) {
  CUDA_CHECK(cudaMalloc(&ptr, num_elems * sizeof(T)));
  __clear_device(ptr, num_elems); // NOTE: should explicitly reset memory somewhere else
}

template <typename T>
static void __allocate_host(T*& ptr, int64_t num_elems) {
  CUDA_CHECK(cudaMallocHost(&ptr, num_elems * sizeof(T)));
}

template <typename T>
void __copy_h2d(void *dev_ptr, const void *host_ptr, int64_t num_elems) {
  CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr, sizeof(T) * num_elems, cudaMemcpyHostToDevice));
}

template <typename T>
void __copy_d2h(void *host_ptr, const void *dev_ptr, int64_t num_elems) {
  CUDA_CHECK(cudaMemcpy(host_ptr, dev_ptr, sizeof(T) * num_elems, cudaMemcpyDeviceToHost));
}

static void __deallocate_device(void *ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

static void __deallocate_host(void *ptr) {
  CUDA_CHECK(cudaFreeHost(ptr));
}

void scratchpad_init(CudaScratchPad*& pad, int64_t max_nb_output, int64_t max_nb_input) {
  __host_new(pad);
  pad->max_nb_input = max_nb_input;
  pad->max_nb_output = max_nb_output;
  __allocate_device(pad->v_init, max_nb_input);
  __allocate_device(pad->v_evals, max_nb_input);
  __allocate_device(pad->hg_evals, max_nb_input);
  __allocate_device(pad->p, 3);
  __allocate_device(pad->p0, max_nb_input / 16);
  __allocate_device(pad->p1, max_nb_input / 16);
  __allocate_device(pad->p2, max_nb_input / 16);
  __allocate_host(pad->v_evals_host, max_nb_input);
  __allocate_host(pad->hg_evals_host, max_nb_input);
  __allocate_host(pad->p_host, 3);
  __allocate_host(pad->gate_exists_host, max_nb_input);
  __allocate_device(pad->eq_evals_at_rx, max_nb_input);
  __allocate_device(pad->eq_evals_at_rz1, max_nb_output);
  // __allocate_device(eq_evals_at_rz2, max_nb_output);
  // __allocate_device(eq_evals_first_half, max_nb_output);
  // __allocate_device(eq_evals_second_half, max_nb_output);
  __allocate_device(pad->gate_exists, max_nb_input);
}

void scratchpad_deinit(CudaScratchPad*& pad) {
  __deallocate_device(pad->v_init);
  __deallocate_device(pad->v_evals);
  __deallocate_device(pad->hg_evals);
  __deallocate_device(pad->p);
  __deallocate_host(pad->v_evals_host);
  __deallocate_host(pad->hg_evals_host);
  __deallocate_host(pad->p_host);
  __deallocate_host(pad->gate_exists_host);
  __deallocate_device(pad->eq_evals_at_rx);
  __deallocate_device(pad->eq_evals_at_rz1);
  // __deallocate_device(eq_evals_at_rz2);
  // __deallocate_device(eq_evals_first_half);
  // __deallocate_device(eq_evals_second_half);
  __deallocate_device(pad->gate_exists);
  __host_delete(pad);
}

void scratchpad_check(void *pad_host, int64_t eval_size) {
  auto& pad = *reinterpret_cast<gkr::GKRScratchPad<HostBatchF, HostF>*>(pad_host);
  auto& pad_gpu = *pad.pad_gpu;
  __copy_d2h<bool>(pad_gpu.gate_exists_host, pad_gpu.gate_exists, pad_gpu.max_nb_input);
  __copy_d2h<CudaBatchF>(pad_gpu.hg_evals_host, pad_gpu.hg_evals, pad_gpu.max_nb_input);
  __copy_d2h<CudaBatchF>(pad_gpu.v_evals_host, pad_gpu.v_evals, pad_gpu.max_nb_input);

  auto actual_exists = reinterpret_cast<bool*>(pad.gate_exists);
  auto actual_hg = reinterpret_cast<CudaBatchF*>(pad.hg_evals);
  auto actual_v = reinterpret_cast<CudaBatchF*>(pad.v_evals);
  for (auto i = 0ll; i < eval_size; i++) {
    assert(actual_exists[i] == pad_gpu.gate_exists_host[i]);
    for (auto j = 0ll; j < CudaBatchF::batch_size; j++) {
      assert(actual_hg[i].elems[j] == pad_gpu.hg_evals_host[i].elems[j]);
      // assert(actual_v[i].elems[j] == pad_gpu.v_evals_host[i].elems[j]);
    }
  }
  printf("check pass, size=%ld\n", eval_size);
}

void scratchpad_load(void *pad_host) {
  auto& pad = *reinterpret_cast<gkr::GKRScratchPad<HostBatchF, HostF>*>(pad_host);
  auto& pad_gpu = *pad.pad_gpu;
  // __copy_h2d<CudaBatchF>(pad_gpu.v_evals, pad.v_evals, 1);

  __copy_h2d<CudaBatchF>(pad_gpu.v_evals, pad.v_evals, pad.max_nb_input);
  __copy_h2d<CudaBatchF>(pad_gpu.hg_evals, pad.hg_evals, pad.max_nb_input);
  __copy_h2d<bool>(pad_gpu.gate_exists, pad.gate_exists, pad.max_nb_input);
}

void scratchpad_load_v_init(CudaScratchPad* pad, const void* v_init, int64_t len) {
  CUDA_CHECK(cudaMemcpy(pad->v_init, v_init,
    sizeof(CudaBatchF) * len, cudaMemcpyHostToDevice));
}

void scratchpad_store(void *pad_host) {
  auto& pad = *reinterpret_cast<gkr::GKRScratchPad<HostBatchF, HostF>*>(pad_host);
  auto& pad_gpu = *pad.pad_gpu;
  __copy_d2h<CudaBatchF>(pad.v_evals, pad_gpu.v_evals, 1);

  // __copy_d2h<CudaBatchF>(pad.v_evals, pad_gpu.v_evals, pad.max_nb_input);
  // __copy_d2h<CudaBatchF>(pad.hg_evals, pad_gpu.hg_evals, pad.max_nb_input);
  // __copy_d2h<bool>(pad.gate_exists, pad_gpu.gate_exists, pad.max_nb_input);
}

void scratchpad_test(void *pad_host, const char *msg) {
  // auto& pad = *reinterpret_cast<gkr::GKRScratchPad<HostBatchF, HostF>*>(pad_host);
  // auto& pad_gpu = *pad.pad_gpu;

  // uint32_t tmp;
  // __copy_d2h<uint32_t>(&tmp, &pad_gpu.v_evals[0].elems[0], 1);

  // printf("%s v:%u\n", msg, tmp);
}

ASSERT_POD(CudaScratchPad);

} // namespace cuda