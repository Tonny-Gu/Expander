#pragma once

#include "cuda/export.hpp"
#include "cuda/common.cuh"
#include "cuda/m31.cuh"
#include "cuda/scratchpad.cuh"
#include "field/M31.hpp"
#include "circuit/circuit.hpp"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_new.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

namespace cuda {

template <int64_t nb_input>
struct CudaGate {
  uint32_t o_id;
  uint32_t i_ids[nb_input];
  CudaF coef;
};

struct CudaCircuitLayer {
  uint32_t nb_output_vars;
  uint32_t nb_input_vars;
  thrust::device_ptr<CudaBatchF> input_layer_vals;
  thrust::device_ptr<CudaGate<1>> add_wires;
  thrust::device_ptr<CudaGate<2>> mul_wires;
};

void __layer_tentative_init(gkr::CircuitLayer<HostBatchF, HostF>& layer_host) {
  if (!layer_host.layer_gpu_init) {
    layer_host.layer_gpu_init = true;
    __host_new(layer_host.layer_gpu);
    // printf("allocated: %p\n", &layer_host.layer_gpu);
  }
}

void layer_load_wires(void* layer_host) {
  auto& layer = *reinterpret_cast<gkr::CircuitLayer<HostBatchF, HostF>*>(layer_host);
  __layer_tentative_init(layer);
  auto& layer_gpu = *layer.layer_gpu;

  layer_gpu.nb_input_vars = layer.nb_input_vars;
  layer_gpu.nb_output_vars = layer.nb_output_vars;

  layer_gpu.add_wires = thrust::device_new<CudaGate<1>>(layer.add.sparse_evals.size());
  layer_gpu.mul_wires = thrust::device_new<CudaGate<2>>(layer.mul.sparse_evals.size());

  thrust::host_vector<CudaGate<1>> add_wires_host;
  thrust::host_vector<CudaGate<2>> mul_wires_host;

  for (auto& e : layer.add.sparse_evals) {
    add_wires_host.push_back(CudaGate<1>{e.o_id,
      {e.i_ids[0]}, CudaF::make(e.coef)});
  }

  for (auto& e : layer.mul.sparse_evals) {
    mul_wires_host.push_back(CudaGate<2>{e.o_id,
      {e.i_ids[0], e.i_ids[1]}, CudaF::make(e.coef)});
  }

  thrust::copy(add_wires_host.begin(), add_wires_host.end(), layer_gpu.add_wires);
  thrust::copy(mul_wires_host.begin(), mul_wires_host.end(), layer_gpu.mul_wires);
}

void layer_load_inputs(void* layer_host) {
  auto& layer = *reinterpret_cast<gkr::CircuitLayer<HostBatchF, HostF>*>(layer_host);
  __layer_tentative_init(layer);
  auto& layer_gpu = *layer.layer_gpu;

  layer_gpu.input_layer_vals = thrust::device_new<CudaBatchF>(layer.input_layer_vals.evals.size());
  CUDA_CHECK(cudaMemcpy(layer_gpu.input_layer_vals.get(),
    &layer.input_layer_vals.evals[0], sizeof(CudaBatchF) * layer.input_layer_vals.evals.size(),
    cudaMemcpyHostToDevice));
}

} // namespace cuda