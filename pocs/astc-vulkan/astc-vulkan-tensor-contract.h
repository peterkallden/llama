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

// Versioned, private metadata for a future offline ASTC pack artifact. The
// record is intentionally composed of fixed-width scalar fields so a later
// serializer can define an explicit byte order without exposing a GGUF type.
inline constexpr uint32_t ggml_vk_astc_pack_metadata_version = 1;
inline constexpr uint32_t ggml_vk_astc_pack_format_4x4 = 1;
inline constexpr uint32_t ggml_vk_astc_pack_format_6x6 = 2;
inline constexpr uint32_t ggml_vk_astc_pack_format_8x6 = 3;
// Keep existing private artifact IDs stable: artifacts written before 10x6
// used 4 for 8x8. New formats are appended rather than renumbering them.
inline constexpr uint32_t ggml_vk_astc_pack_format_8x8 = 4;
inline constexpr uint32_t ggml_vk_astc_pack_format_10x6 = 5;
inline constexpr uint32_t ggml_vk_astc_pack_mapping_global = 0;
inline constexpr uint32_t ggml_vk_astc_pack_mapping_block_affine = 1;
inline constexpr uint32_t ggml_vk_astc_pack_layout_identity = 0;
inline constexpr uint32_t ggml_vk_astc_pack_layout_grouped = 1;
inline constexpr uint32_t ggml_vk_astc_pack_layout_block_reversed = 2;

struct ggml_vk_astc_pack_metadata {
    uint32_t version;
    uint32_t format_id;
    uint32_t logical_rows;
    uint32_t logical_columns;
    uint32_t texel_columns;
    uint32_t channel_order_packed;
    uint32_t layout_id;
    uint32_t mapping_id;
    uint32_t encoder_quality_percent;
    uint64_t compressed_bytes;
    uint64_t calibration_bytes;

    constexpr bool has_valid_channel_order() const {
        const uint32_t order = channel_order_packed;
        if ((order & ~0xffu) != 0) return false;
        const uint32_t a = (order >> 0) & 3;
        const uint32_t b = (order >> 2) & 3;
        const uint32_t c = (order >> 4) & 3;
        const uint32_t d = (order >> 6) & 3;
        return a != b && a != c && a != d && b != c && b != d && c != d;
    }

    constexpr bool is_valid() const {
        return version == ggml_vk_astc_pack_metadata_version &&
            (format_id == ggml_vk_astc_pack_format_4x4 ||
             format_id == ggml_vk_astc_pack_format_6x6 ||
             format_id == ggml_vk_astc_pack_format_8x6 ||
             format_id == ggml_vk_astc_pack_format_10x6 ||
             format_id == ggml_vk_astc_pack_format_8x8) &&
            logical_rows != 0 && logical_columns != 0 &&
            texel_columns == (logical_columns + 3) / 4 &&
            has_valid_channel_order() &&
            layout_id <= ggml_vk_astc_pack_layout_block_reversed &&
            mapping_id <= ggml_vk_astc_pack_mapping_block_affine &&
            encoder_quality_percent <= 100 && compressed_bytes != 0;
    }
};
