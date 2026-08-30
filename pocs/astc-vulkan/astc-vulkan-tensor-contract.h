#pragma once

// Experimental host-side tensor layout contract. This describes how a logical
// row-major weight matrix would map to RGBA ASTC texels; it does not encode
// blocks and is intentionally independent of Vulkan and ggml tensor types.

#include "astc-vulkan-contract.h"

#include <cstdint>

struct ggml_vk_astc_weight_layout {
    uint32_t rows;
    uint32_t columns;
    uint32_t channels_per_texel = 4;

    constexpr uint32_t texel_columns() const {
        return (columns + channels_per_texel - 1) / channels_per_texel;
    }

    constexpr uint64_t texel_count() const {
        return static_cast<uint64_t>(rows) * texel_columns();
    }

    constexpr uint64_t padded_scalar_count() const {
        return texel_count() * channels_per_texel;
    }

    constexpr uint64_t storage_bytes(
            const ggml_vk_astc_format_contract & format) const {
        return ggml_vk_astc_image_storage_bytes(format, texel_columns(), rows);
    }

    constexpr bool is_valid() const {
        return rows != 0 && columns != 0 && channels_per_texel == 4;
    }
};

// Companion metadata is deliberately minimal and private to the PoC. The
// affine transform is applied after ASTC UNORM reconstruction in the shader;
// future packer experiments may extend this without changing ggml semantics.
struct ggml_vk_astc_weight_metadata {
    uint32_t logical_rows;
    uint32_t logical_columns;
    float scale;
    float offset;
};

