#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "field/M31.hpp"

namespace cuda {

struct CudaBatchM31;
struct CudaM31;

using CudaF = CudaM31;
using CudaBatchF = CudaBatchM31;

struct CudaCircuit;
struct CudaScratchPad;

void scratchpad_init(CudaScratchPad*& pad, int64_t max_nb_output, int64_t max_nb_input);
void scratchpad_deinit(CudaScratchPad*& pad);
void scratchpad_check(void *pad_host, int64_t eval_size);
void scratchpad_load(void *pad_host);
void scratchpad_load_v_init(CudaScratchPad* pad, const void* v_init, int64_t len);
void scratchpad_store(void *pad_host);
void scratchpad_test(void *pad_host, const char *msg);

struct CudaCircuitLayer;
void layer_init(CudaCircuitLayer*& layer);
void layer_load(void* layer_host);

CudaBatchF* gkr_poly_eval_at(CudaScratchPad *pad, CudaCircuitLayer *layer, int32_t eval_size, uint32_t var_idx, uint32_t degree);
void gkr_receive_challenge(CudaScratchPad *pad, CudaCircuitLayer *layer, int32_t eval_size, uint32_t var_idx, const void *r_ptr);
void gkr_prepare_g_x_vals(CudaScratchPad *pad, const void *layer_host, const void *eq_evals_at_rz1, int64_t rz1_len);
void gkr_prepare_h_y_vals(CudaScratchPad *pad, const void *layer_host, const void *eq_evals_at_rx, int64_t rx_len);

template <typename T, typename... Args>
void __host_new(T*& ptr, Args && ...args) {
  // this is a wrapper for clangd
  ptr = new T(std::forward<Args>(args)...);
}

template <typename T>
void __host_delete(T*& ptr) {
  delete ptr;
  ptr = nullptr;
}

#define ASSERT_POD(T) static_assert(std::is_standard_layout_v<T> && std::is_trivial_v<T>);

}