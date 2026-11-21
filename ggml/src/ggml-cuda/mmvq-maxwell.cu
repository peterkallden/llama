#include "mmvq-maxwell.cuh"
#include "maxwell-layout.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "vecdotq.cuh"

// Four lanes share each 32-element group, so partial integer sums need only a four-lane reduction.
// All lanes in the warp must participate in the shuffle operations.
static __device__ __forceinline__ int maxwell_sum_group(int value) {
    value += __shfl_xor_sync(0xffffffffu, value, 2, 4);
    value += __shfl_xor_sync(0xffffffffu, value, 1, 4);
    return value;
}

// Maxwell lacks native DP4A, so use scalar integer products and reuse each activation load for two output rows.
// Process 256 weights per warp and apply each 32-weight group's scale and offset before accumulating in FP32.
template<ggml_type type>
static __global__ void mul_mat_vec_q_maxwell(
        const void * __restrict__ weights, const block_q8_1 * __restrict__ activations,
        float * __restrict__ dst, const int64_t ncols, const int64_t nrows) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int64_t row0 = 2 * (int64_t(blockIdx.x) * (blockDim.x / warp_size) + warp);
    if (row0 >= nrows) {
        return;
    }
    const int64_t row1 = row0 + 1 < nrows ? row0 + 1 : row0;
    const int group_in_tile = lane / 4;
    const int lane_in_group = lane % 4;
    float result0 = 0.0f;
    float result1 = 0.0f;

    for (int64_t tile = 0; tile < ncols / QK_K; ++tile) {
        const int64_t group = tile * 8 + group_in_tile;
        const block_q8_1 * x = activations + group;
        const uint32_t x0 = uint32_t(get_int_b4(x->qs, 2 * lane_in_group));
        const uint32_t x1 = uint32_t(get_int_b4(x->qs, 2 * lane_in_group + 1));
        uint32_t w00, w01, w10, w11;
        int shift;
        float d0, d1, min0 = 0.0f, min1 = 0.0f;

        if constexpr (type == GGML_TYPE_Q4_0) {
            const int64_t tiles_per_row = ncols / QK_K;
            const maxwell_q4_0_tile * w0 = (const maxwell_q4_0_tile *) weights + row0 * tiles_per_row + tile;
            const maxwell_q4_0_tile * w1 = (const maxwell_q4_0_tile *) weights + row1 * tiles_per_row + tile;
            const int offset = 2 * (lane_in_group % 2);
            w00 = uint32_t(get_int_b2(w0->qs + 16 * group_in_tile, offset));
            w01 = uint32_t(get_int_b2(w0->qs + 16 * group_in_tile, offset + 1));
            w10 = uint32_t(get_int_b2(w1->qs + 16 * group_in_tile, offset));
            w11 = uint32_t(get_int_b2(w1->qs + 16 * group_in_tile, offset + 1));
            shift = 4 * (lane_in_group / 2);
            d0 = __half2float(w0->d[group_in_tile]);
            d1 = __half2float(w1->d[group_in_tile]);
        } else {
            const int64_t blocks_per_row = ncols / QK_K;
            const maxwell_q4_k_tile * w0 = (const maxwell_q4_k_tile *) weights + row0 * blocks_per_row + tile;
            const maxwell_q4_k_tile * w1 = (const maxwell_q4_k_tile *) weights + row1 * blocks_per_row + tile;
            const int offset = 8 * (group_in_tile / 2) + 2 * lane_in_group;
            w00 = uint32_t(get_int_b4(w0->qs, offset));
            w01 = uint32_t(get_int_b4(w0->qs, offset + 1));
            w10 = uint32_t(get_int_b4(w1->qs, offset));
            w11 = uint32_t(get_int_b4(w1->qs, offset + 1));
            shift = 4 * (group_in_tile % 2);
            d0 = __half2float(w0->d) * w0->scales[group_in_tile];
            d1 = __half2float(w1->d) * w1->scales[group_in_tile];
            min0 = __half2float(w0->dmin) * w0->mins[group_in_tile];
            min1 = __half2float(w1->dmin) * w1->mins[group_in_tile];
        }

        w00 = (w00 >> shift) & 0x0f0f0f0fu;
        w01 = (w01 >> shift) & 0x0f0f0f0fu;
        w10 = (w10 >> shift) & 0x0f0f0f0fu;
        w11 = (w11 >> shift) & 0x0f0f0f0fu;
        int dot0 = 0, dot1 = 0, sum = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q0 = int(int8_t((x0 >> (8 * i)) & 0xffu));
            const int q1 = int(int8_t((x1 >> (8 * i)) & 0xffu));
            dot0 += int((w00 >> (8 * i)) & 0xffu) * q0 + int((w01 >> (8 * i)) & 0xffu) * q1;
            dot1 += int((w10 >> (8 * i)) & 0xffu) * q0 + int((w11 >> (8 * i)) & 0xffu) * q1;
            if constexpr (type == GGML_TYPE_Q4_K) {
                sum += q0 + q1;
            }
        }
        dot0 = maxwell_sum_group(dot0);
        dot1 = maxwell_sum_group(dot1);
        if constexpr (type == GGML_TYPE_Q4_K) {
            sum = maxwell_sum_group(sum);
        }
        if (lane_in_group == 0) {
            const float2 ds = __half22float2(x->ds);
            if constexpr (type == GGML_TYPE_Q4_0) {
                // Q4_0 uses the stored sum of the original F32 activations.
                result0 += d0 * (ds.x * dot0 - 8.0f * ds.y);
                result1 += d1 * (ds.x * dot1 - 8.0f * ds.y);
            } else {
                result0 += ds.x * (d0 * dot0 - min0 * sum);
                result1 += ds.x * (d1 * dot1 - min1 * sum);
            }
        }
    }
    result0 = warp_reduce_sum(result0);
    result1 = warp_reduce_sum(result1);
    if (lane == 0) {
        dst[row0] = result0;
        if (row0 + 1 < nrows) {
            dst[row0 + 1] = result1;
        }
    }
}

// Four warps per block process up to eight rows while reusing activation data across each row pair.
// The caller limits this kernel to the supported formats and shapes.
void ggml_cuda_mul_mat_vec_q_maxwell(
        const void * weights, const block_q8_1 * activations, float * dst,
        ggml_type type, int64_t ncols, int64_t nrows, cudaStream_t stream) {
    constexpr int threads = 128;
    const int64_t blocks = (nrows + 7) / 8;
    if (type == GGML_TYPE_Q4_0) {
        mul_mat_vec_q_maxwell<GGML_TYPE_Q4_0><<<blocks, threads, 0, stream>>>(weights, activations, dst, ncols, nrows);
    } else {
        GGML_ASSERT(type == GGML_TYPE_Q4_K);
        mul_mat_vec_q_maxwell<GGML_TYPE_Q4_K><<<blocks, threads, 0, stream>>>(weights, activations, dst, ncols, nrows);
    }
}

#endif
