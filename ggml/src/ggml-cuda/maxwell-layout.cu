#include "maxwell-layout.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Expanding the six-bit coefficients once removes their bit extraction from every Q4_K dot product.
static __device__ void maxwell_unpack_scales(const uint8_t * in, uint8_t * scales, uint8_t * mins) {
    for (int g = 0; g < 8; ++g) {
        if (g < 4) {
            scales[g] = in[g] & 63;
            mins[g] = in[g + 4] & 63;
        } else {
            scales[g] = (in[g + 4] & 15) | ((in[g - 4] >> 6) << 4);
            mins[g] = (in[g + 4] >> 4) | ((in[g] >> 6) << 4);
        }
    }
}

// Reconstructing the original six-bit fields permits lossless interchange with ordinary GGML blocks.
static __device__ void maxwell_pack_scales(const uint8_t * scales, const uint8_t * mins, uint8_t * out) {
    for (int g = 0; g < 4; ++g) {
        out[g] = scales[g] | ((scales[g + 4] >> 4) << 6);
        out[g + 4] = mins[g] | ((mins[g + 4] >> 4) << 6);
        out[g + 8] = (scales[g + 4] & 15) | ((mins[g + 4] & 15) << 4);
    }
}

template<ggml_type type, bool unpack>
static __global__ void maxwell_repack(const void * src, void * dst, int64_t tiles) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tiles) return;
    if constexpr (type == GGML_TYPE_Q4_0) {
        if constexpr (!unpack) {
            const block_q4_0 * in = (const block_q4_0 *) src + 8 * i;
            maxwell_q4_0_tile & out = ((maxwell_q4_0_tile *) dst)[i];
            for (int g = 0; g < 8; ++g) {
                out.d[g] = in[g].d;
                for (int j = 0; j < 16; ++j) out.qs[g * 16 + j] = in[g].qs[j];
            }
        } else {
            const maxwell_q4_0_tile & in = ((const maxwell_q4_0_tile *) src)[i];
            block_q4_0 * out = (block_q4_0 *) dst + 8 * i;
            for (int g = 0; g < 8; ++g) {
                out[g].d = in.d[g];
                for (int j = 0; j < 16; ++j) out[g].qs[j] = in.qs[g * 16 + j];
            }
        }
    } else {
        if constexpr (!unpack) {
            const block_q4_K & in = ((const block_q4_K *) src)[i];
            maxwell_q4_k_tile & out = ((maxwell_q4_k_tile *) dst)[i];
            out.d = in.dm.x; out.dmin = in.dm.y;
            maxwell_unpack_scales(in.scales, out.scales, out.mins);
            for (int j = 0; j < 128; ++j) out.qs[j] = in.qs[j];
        } else {
            const maxwell_q4_k_tile & in = ((const maxwell_q4_k_tile *) src)[i];
            block_q4_K & out = ((block_q4_K *) dst)[i];
            out.dm.x = in.d; out.dm.y = in.dmin;
            maxwell_pack_scales(in.scales, in.mins, out.scales);
            for (int j = 0; j < 128; ++j) out.qs[j] = in.qs[j];
        }
    }
}

// Expand only the requested output rows, keeping prefill's temporary allocation bounded by the caller.
template<ggml_type type>
static __global__ void maxwell_to_f32(const void * src, float * dst, int64_t ncols, int64_t first_row, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t row = first_row + i / ncols;
    const int64_t col = i % ncols;
    const int64_t tile = row * (ncols / 256) + col / 256;
    const int p = col % 256;
    const int g = p / 32;
    const int k = p % 32;
    if constexpr (type == GGML_TYPE_Q4_0) {
        const maxwell_q4_0_tile & w = ((const maxwell_q4_0_tile *) src)[tile];
        const int q = (w.qs[g * 16 + k % 16] >> (4 * (k / 16))) & 15;
        dst[i] = __half2float(w.d[g]) * (q - 8);
    } else {
        const maxwell_q4_k_tile & w = ((const maxwell_q4_k_tile *) src)[tile];
        const int q = (w.qs[(g / 2) * 32 + k] >> (4 * (g % 2))) & 15;
        dst[i] = __half2float(w.d) * w.scales[g] * q - __half2float(w.dmin) * w.mins[g];
    }
}

// Work on complete 256-value tiles so row boundaries and original quantization groups stay intact.
template<bool unpack>
static void maxwell_repack_launch(const void * src, void * dst, ggml_type type, int64_t ncols, int64_t nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols > 0 && ncols % 256 == 0 && nrows > 0 && src != dst);
    const int64_t tiles = ncols / 256 * nrows;
    const int blocks = int((tiles + 127) / 128);
    if (type == GGML_TYPE_Q4_0) maxwell_repack<GGML_TYPE_Q4_0, unpack><<<blocks, 128, 0, stream>>>(src, dst, tiles);
    else {
        GGML_ASSERT(type == GGML_TYPE_Q4_K);
        maxwell_repack<GGML_TYPE_Q4_K, unpack><<<blocks, 128, 0, stream>>>(src, dst, tiles);
    }
}
void ggml_cuda_maxwell_pack(const void * src, void * dst, ggml_type type, int64_t ncols, int64_t nrows, cudaStream_t stream) {
    maxwell_repack_launch<false>(src, dst, type, ncols, nrows, stream);
}
void ggml_cuda_maxwell_unpack(const void * src, void * dst, ggml_type type, int64_t ncols, int64_t nrows, cudaStream_t stream) {
    maxwell_repack_launch<true>(src, dst, type, ncols, nrows, stream);
}
void ggml_cuda_maxwell_to_f32(const void * src, float * dst, ggml_type type, int64_t ncols, int64_t first_row, int64_t nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols > 0 && ncols % 256 == 0 && first_row >= 0 && nrows > 0);
    const int64_t count = ncols * nrows;
    const int blocks = int((count + 255) / 256);
    if (type == GGML_TYPE_Q4_0) maxwell_to_f32<GGML_TYPE_Q4_0><<<blocks, 256, 0, stream>>>(src, dst, ncols, first_row, count);
    else {
        GGML_ASSERT(type == GGML_TYPE_Q4_K);
        maxwell_to_f32<GGML_TYPE_Q4_K><<<blocks, 256, 0, stream>>>(src, dst, ncols, first_row, count);
    }
}
#endif
