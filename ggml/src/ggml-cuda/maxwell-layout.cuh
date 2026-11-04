#pragma once

#include "common.cuh"

// The separate types make it impossible to confuse these tiles with GGML blocks.
struct maxwell_q4_0_tile {
    uint8_t qs[128];
    ggml_half d[8];
};
struct maxwell_q4_k_tile {
    uint8_t qs[128];
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[8];
    uint8_t mins[8];
};
static_assert(sizeof(maxwell_q4_0_tile) == 144, "invalid Maxwell Q4_0 tile");
static_assert(sizeof(maxwell_q4_k_tile) == 148, "invalid Maxwell Q4_K tile");

// Input and output must be distinct device allocations. ncols must be a multiple of 256.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_maxwell_pack(const void * src, void * dst, ggml_type type,
                            int64_t ncols, int64_t nrows, cudaStream_t stream);
void ggml_cuda_maxwell_unpack(const void * src, void * dst, ggml_type type,
                              int64_t ncols, int64_t nrows, cudaStream_t stream);
void ggml_cuda_maxwell_to_f32(const void * src, float * dst, ggml_type type,
                              int64_t ncols, int64_t first_row, int64_t nrows, cudaStream_t stream);
#else
static inline void ggml_cuda_maxwell_pack(
        const void *, void *, ggml_type, int64_t, int64_t, cudaStream_t) {
    GGML_ABORT("Maxwell packing is unavailable on this backend");
}
static inline void ggml_cuda_maxwell_unpack(
        const void *, void *, ggml_type, int64_t, int64_t, cudaStream_t) {
    GGML_ABORT("Maxwell unpacking is unavailable on this backend");
}
static inline void ggml_cuda_maxwell_to_f32(
        const void *, float *, ggml_type, int64_t, int64_t, int64_t, cudaStream_t) {
    GGML_ABORT("Maxwell conversion is unavailable on this backend");
}
#endif
