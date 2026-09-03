#pragma once

// Offline Block-LDLQ configuration shared by ASTC candidate-selection tools.
//
// Block-LDLQ uses a damped activation Gram/Hessian approximation to regenerate
// the target seen by later ASTC blocks after an earlier legal payload is
// committed. It is appropriate only for offline candidate selection with a
// representative calibration corpus; it is not a Vulkan runtime algorithm and
// does not alter the standard ASTC payload or decoder contract.
//
// The current PoC keeps the matrix/candidate implementation in
// astc-vulkan-latent-smoke.cpp because it uses that tool's private capture and
// artifact types. This small module deliberately owns its stable public knobs
// and terminology first; moving the implementation is deferred until those
// data types have a reusable selection-core boundary.
//
// References:
// - Chee et al., QuIP#: Even Better LLM Quantization with Hadamard Incoherence
//   and Lattice Codebooks, https://arxiv.org/abs/2402.04396
// - van Baalen et al., GPTVQ: The Blessing of Dimensionality for LLM
//   Quantization, https://arxiv.org/abs/2402.15319

enum class astc_vulkan_ldlq_order {
    forward,
    reverse,
    pivot,
};

const char * astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order order);
