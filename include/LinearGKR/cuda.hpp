#pragma once

#include <cstdint>

namespace cuda {

constexpr int32_t block_size = 32;

namespace m31 {
struct M31;
}

namespace gkr {

using F = m31::M31;

struct BatchF;

struct ScratchPad {
  BatchF *v_init;
  BatchF *v_evals;
  BatchF *hg_evals;
  BatchF *p;

  BatchF *v_evals_host;
  BatchF *hg_evals_host;
  BatchF *p_host;
  bool *gate_exists_host;

  // F *eq_evals_at_rx;
  // F *eq_evals_at_rz1;
  // F *eq_evals_at_rz2;
  // F *eq_evals_first_half;
  // F *eq_evals_second_half;
  
  bool *gate_exists;
  int64_t max_nb_input;
  int64_t max_nb_output;

  void init(int64_t max_nb_output, int64_t max_nb_input);
  void deinit();
  void check(void *gate_exists_ptr, void *hg_evals_ptr, void *v_evals_ptr, int64_t eval_size);
};

struct SumcheckGKRHelper {
  ScratchPad *pad;
  void *v_host;
  void *hg_host;
  void *gate_exists_host;

  void init(ScratchPad& pad, void *v_host, void *hg_host, void *gate_exists) {
    this->pad = &pad;
    this->v_host = v_host;
    this->hg_host = hg_host;
    this->gate_exists_host = gate_exists;
  }

  BatchF* poly_eval_at(int32_t eval_size, uint32_t var_idx, uint32_t degree);
  void receive_challenge(int32_t eval_size, uint32_t var_idx, const uint32_t r);
  void load_f_hg();
  void load_v_init(const void* v_init, int64_t len);
  void store_f_hg();
};

} // namespace gkr

} // namespace cuda