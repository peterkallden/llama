#pragma once

#include "common.cuh"
#include <mutex>

// Serialize cache publication, invalidation, and kernel submission across backend instances.
std::recursive_mutex & ggml_cuda_maxwell_cache_mutex();

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_mul_mat_vec_q_maxwell(
        const void * weights, const block_q8_1 * activations, float * dst,
        ggml_type type, int64_t ncols, int64_t nrows, cudaStream_t stream);
const void * ggml_cuda_maxwell_packed_weights(const ggml_tensor * tensor);
#endif
