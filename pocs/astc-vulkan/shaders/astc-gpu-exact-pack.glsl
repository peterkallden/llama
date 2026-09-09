// Shared physical packing primitives for the audited GPU exact-subset modes.
//
// This file deliberately implements only the binary ISE path used by the
// current subset: QUANT_2 and QUANT_4 weights are powers-of-two alphabets and
// pack as a raw binary stream. Trit/quint ISE and ASTC's more general endpoint
// field interleave remain CPU-audited work; do not use this helper for a new
// mode until its physical descriptor and CPU/Vulkan gates exist.

#ifndef ASTC_GPU_EXACT_PACK_GLSL
#define ASTC_GPU_EXACT_PACK_GLSL

// Mirror the audited CPU descriptors. Keep additions paired with
// astc-vulkan-astc-mode.cpp; CPU tests verify the descriptor and the Vulkan
// smoke verifies these emitted physical payloads.
const uint astc_gpu_mode_d1_luminance_binary_6x6 = 0x104u;
const uint astc_gpu_mode_d1_luminance_binary_5x5 = 0x0e1u;
const uint astc_gpu_mode_d1_luminance_quant4_4x4 = 0x042u;
const uint astc_gpu_mode_d2_luminance_alpha_binary_8x5 = 0x065u;
const uint astc_gpu_mode_d2_luminance_alpha_dual_binary_8x5 = 0x4c1u;
const uint astc_gpu_endpoint_format_luminance = 0u;
const uint astc_gpu_endpoint_format_luminance_alpha = 4u;

uint astc_gpu_bit_reverse8(uint value) {
    value = ((value & 0x55u) << 1u) | ((value >> 1u) & 0x55u);
    value = ((value & 0x33u) << 2u) | ((value >> 2u) & 0x33u);
    return ((value & 0x0Fu) << 4u) | ((value >> 4u) & 0x0Fu);
}

void astc_gpu_payload_bit(inout uvec4 words, uint bit, uint value) {
    const uint word = bit >> 5u;
    const uint shift = bit & 31u;
    if (value != 0u) words[word] |= 1u << shift;
    else words[word] &= ~(1u << shift);
}

void astc_gpu_payload_bits(inout uvec4 words, uint value, uint bits, uint offset) {
    for (uint bit = 0u; bit < bits; ++bit) {
        astc_gpu_payload_bit(words, offset + bit, (value >> bit) & 1u);
    }
}

uint astc_gpu_unorm8(float value) {
    return uint(floor(clamp(value, 0.0, 1.0) * 255.0 + 0.5));
}

// Encodes one value in the binary ISE stream used by the current audited
// modes. Current callers use one or two bits/value, but the bit loop preserves
// the contract if a later audited power-of-two mode needs more.
void astc_gpu_binary_ise_store(inout uint bytes[16], uint index,
                               uint value, uint bits_per_value) {
    const uint offset = index * bits_per_value;
    for (uint bit = 0u; bit < bits_per_value; ++bit) {
        if (((value >> bit) & 1u) != 0u) {
            const uint position = offset + bit;
            bytes[position >> 3u] |= 1u << (position & 7u);
        }
    }
}

// ASTC stores the current subset's raw binary weight stream in reverse-byte,
// bit-reversed physical order. This matches the existing CPU exact-subset
// packer and is intentionally shared by D1 and D2.
void astc_gpu_emit_binary_ise_stream(inout uvec4 payload, in uint bytes[16]) {
    for (uint index = 0u; index < 16u; ++index) {
        astc_gpu_payload_bits(payload, astc_gpu_bit_reverse8(bytes[index]), 8u,
                              (15u - index) * 8u);
    }
}

void astc_gpu_write_normal_header(inout uvec4 payload, uint block_mode,
                                  uint endpoint_format) {
    astc_gpu_payload_bits(payload, block_mode, 11u, 0u);
    astc_gpu_payload_bits(payload, 0u, 2u, 11u); // One partition.
    astc_gpu_payload_bits(payload, endpoint_format, 4u, 13u);
}

void astc_gpu_write_direct_luminance_endpoints(inout uvec4 payload,
                                                uint low, uint high) {
    astc_gpu_payload_bits(payload, low, 8u, 17u);
    astc_gpu_payload_bits(payload, high, 8u, 25u);
}

void astc_gpu_write_direct_luminance_alpha_endpoints(inout uvec4 payload,
                                                      uvec2 low, uvec2 high) {
    astc_gpu_payload_bits(payload, low.x, 8u, 17u);
    astc_gpu_payload_bits(payload, high.x, 8u, 25u);
    astc_gpu_payload_bits(payload, low.y, 8u, 33u);
    astc_gpu_payload_bits(payload, high.y, 8u, 41u);
}

#endif
