#include "astc-vulkan-embedding-provider.h"

#include "astc-vulkan-embedding-layout.h"
#include "astc-vulkan-embedding-gpu.h"
#include "astc-vulkan-ggml-external-op.h"

#include <astcenc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

#ifndef ASTC_VULKAN_EMBEDDING_SHADER_PATH
#define ASTC_VULKAN_EMBEDDING_SHADER_PATH "astc-embedding-get-rows.comp.spv"
#endif

namespace {

struct metadata {
    uint32_t dimensions = 0;
    uint32_t vocabulary = 0;
    std::string tensor;
    std::string representation;
    std::string footprint;
    std::string payload;
    std::string affine;
};

bool parse_u32(const std::unordered_map<std::string, std::string> & values,
               const char * key, uint32_t & result) {
    const auto it = values.find(key);
    if (it == values.end() || it->second.empty()) return false;
    char * end = nullptr;
    const unsigned long value = std::strtoul(it->second.c_str(), &end, 10);
    if (end == it->second.c_str() || *end != '\0' ||
        value > std::numeric_limits<uint32_t>::max()) return false;
    result = static_cast<uint32_t>(value);
    return true;
}

bool parse_metadata(const std::filesystem::path & path, metadata & result, std::string & error) {
    std::ifstream file(path);
    if (!file) { error = "cannot open embedding metadata: " + path.string(); return false; }
    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(file, line)) {
        const size_t split = line.find('=');
        if (split == std::string::npos || split == 0) continue;
        values.emplace(line.substr(0, split), line.substr(split + 1));
    }
    const auto get = [&](const char * key, std::string & out) {
        const auto it = values.find(key);
        if (it == values.end() || it->second.empty()) return false;
        out = it->second;
        return true;
    };
    uint32_t version = 0;
    if (!parse_u32(values, "version", version) || version != 1 ||
        !get("tensor", result.tensor) || !get("representation", result.representation) ||
        !get("footprint", result.footprint) || !get("payload", result.payload) ||
        !get("affine", result.affine) || !parse_u32(values, "dimensions", result.dimensions) ||
        !parse_u32(values, "vocab", result.vocabulary)) {
        error = "embedding metadata is incomplete or unsupported: " + path.string();
        return false;
    }
    return true;
}

bool parse_metadata_bytes(const std::vector<uint8_t> & bytes, metadata & result, std::string & error) {
    std::istringstream stream(std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(stream, line)) {
        const size_t split = line.find('=');
        if (split != std::string::npos && split != 0) values.emplace(line.substr(0, split), line.substr(split + 1));
    }
    const auto get = [&](const char * key, std::string & out) {
        const auto it = values.find(key); if (it == values.end() || it->second.empty()) return false;
        out = it->second; return true;
    };
    uint32_t version = 0;
    if (!parse_u32(values, "version", version) || version != 1 || !get("tensor", result.tensor) ||
        !get("representation", result.representation) || !get("footprint", result.footprint) ||
        !get("payload", result.payload) || !get("affine", result.affine) ||
        !parse_u32(values, "dimensions", result.dimensions) || !parse_u32(values, "vocab", result.vocabulary)) {
        error = "embedding metadata bytes are incomplete or unsupported"; return false;
    }
    return true;
}

bool read_binary(const std::filesystem::path & path, std::vector<uint8_t> & out, std::string & error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { error = "cannot open embedding resource: " + path.string(); return false; }
    const std::streamsize size = file.tellg();
    if (size < 0) { error = "cannot stat embedding resource: " + path.string(); return false; }
    out.resize(static_cast<size_t>(size));
    file.seekg(0);
    if (size != 0 && !file.read(reinterpret_cast<char *>(out.data()), size)) {
        error = "cannot read embedding resource: " + path.string();
        return false;
    }
    return true;
}

bool read_f32_affine(const std::filesystem::path & path, std::vector<float> & out,
                     uint32_t vocabulary, std::string & error) {
    std::vector<uint8_t> bytes;
    if (!read_binary(path, bytes, error)) return false;
    const size_t expected = static_cast<size_t>(vocabulary) * 2 * sizeof(float);
    if (bytes.size() != expected) {
        error = "embedding affine size mismatch: expected " + std::to_string(expected) +
                ", got " + std::to_string(bytes.size());
        return false;
    }
    out.resize(expected / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    for (float value : out) if (!std::isfinite(value)) {
        error = "embedding affine contains a non-finite value";
        return false;
    }
    for (uint32_t row = 0; row < vocabulary; ++row) {
        if (!(out[2 * row + 1] > 0.0f)) {
            error = "embedding affine contains a non-positive scale";
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> read_spirv(const char * path) {
    std::vector<uint8_t> bytes;
    std::string error;
    if (path == nullptr || !read_binary(path, bytes, error) || bytes.empty() || bytes.size() % sizeof(uint32_t) != 0) return {};
    std::vector<uint32_t> result(bytes.size() / sizeof(uint32_t));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

} // namespace

astc_vulkan_embedding_provider::astc_vulkan_embedding_provider() = default;

astc_vulkan_embedding_provider::~astc_vulkan_embedding_provider() {
    reset();
}

void astc_vulkan_embedding_provider::reset() {
    delete gpu_session_;
    gpu_session_ = nullptr;
    native_node_ = nullptr;
    if (astc_context_ != nullptr) {
        astcenc_context_free(static_cast<astcenc_context *>(astc_context_));
    }
    astc_context_ = nullptr;
    payload_.clear();
    affine_.clear();
    ready_ = false;
    dimensions_ = vocabulary_ = tiles_per_token_ = 0;
    block_width_ = block_height_ = 0;
    last_error_.clear();
    generation_count_ = 0;
}

bool astc_vulkan_embedding_provider::native_bind(ggml_tensor * node, const char * tensor_name) {
    if (!ready_ || node == nullptr || tensor_name == nullptr ||
        std::strcmp(tensor_name, "token_embd.weight") != 0) return false;
    native_node_ = node;
    ggml_vk_astc_external_op::bind_node(node, native_dispatch_callback, this,
                                        native_context_callback, this);
    return true;
}

bool astc_vulkan_embedding_provider::materialize_native(
        const ggml_vk_external_op_dispatch_context * context, std::string & error) {
    if (context == nullptr || context->native_physical_device == 0 || context->native_device == 0 ||
        context->native_queue == 0 || context->native_queue_family == UINT32_MAX) {
        error = "E1 embedding graph context has incomplete Vulkan device handles"; return false;
    }
    const VkDevice device = reinterpret_cast<VkDevice>(context->native_device);
    if (gpu_session_ != nullptr && gpu_session_->device() == device && gpu_session_->ready()) return true;
    const std::vector<uint32_t> spirv = read_spirv(ASTC_VULKAN_EMBEDDING_SHADER_PATH);
    if (spirv.empty()) { error = "cannot load E1 embedding GPU decode SPIR-V"; return false; }
    auto * candidate = new astc_vulkan_embedding_gpu_session();
    if (!candidate->init(reinterpret_cast<VkPhysicalDevice>(context->native_physical_device), device,
                         reinterpret_cast<VkQueue>(context->native_queue), context->native_queue_family,
                         payload_, affine_, vocabulary_, dimensions_, tiles_per_token_, spirv, error)) {
        delete candidate; return false;
    }
    delete gpu_session_;
    gpu_session_ = candidate;
    std::fprintf(stderr, "ASTC embedding GPU atlas ready: %ux%u, %u token columns\n",
                 gpu_session_->atlas_width(), gpu_session_->atlas_height(), gpu_session_->token_columns());
    return true;
}

bool astc_vulkan_embedding_provider::can_record_native(
        const ggml_vk_external_op_dispatch_context * context) {
    if (!ready_ || context == nullptr || context->node != native_node_ ||
        context->native_command_buffer == 0 || context->get_buffer == nullptr ||
        context->node->src[0] == nullptr || context->node->ne[0] != dimensions_ || context->node->ne[1] == 0) return false;
    std::string error;
    if (!materialize_native(context, error)) { last_error_ = error; return false; }
    return true;
}

bool astc_vulkan_embedding_provider::record_native(
        const ggml_vk_external_op_dispatch_context * context) {
    if (!can_record_native(context)) return false;
    ggml_vk_external_op_buffer_view tokens{};
    ggml_vk_external_op_buffer_view output{};
    if (!context->get_buffer(context->backend_context, context->node->src[0], &tokens) ||
        !context->get_buffer(context->backend_context, context->node, &output) ||
        tokens.native_device != context->native_device || output.native_device != context->native_device) return false;
    std::string error;
    const bool ok = gpu_session_->record_external(
        reinterpret_cast<VkCommandBuffer>(context->native_command_buffer),
        reinterpret_cast<VkBuffer>(tokens.native_buffer), tokens.offset, tokens.size,
        reinterpret_cast<VkBuffer>(output.native_buffer), output.offset, output.size,
        static_cast<uint32_t>(context->node->ne[1]), error);
    if (!ok) last_error_ = error;
    return ok;
}

bool astc_vulkan_embedding_provider::native_context_callback(
        const ggml_vk_external_op_dispatch_context * context, void * user_data) {
    return user_data != nullptr && static_cast<astc_vulkan_embedding_provider *>(user_data)->can_record_native(context);
}

bool astc_vulkan_embedding_provider::native_dispatch_callback(
        const ggml_vk_external_op_dispatch_context * context, void * user_data) {
    return user_data != nullptr && static_cast<astc_vulkan_embedding_provider *>(user_data)->record_native(context);
}

bool astc_vulkan_embedding_provider::prepare(const std::string & root, std::string & error) {
    reset();
    namespace fs = std::filesystem;
    const fs::path metadata_path = fs::path(root) / "embedding-10x5.astce";
    metadata meta;
    if (!parse_metadata(metadata_path, meta, error)) { last_error_ = error; return false; }
    if (meta.tensor != "token_embd.weight" || meta.representation != "e1-local" ||
        meta.footprint != "10x5" || meta.dimensions == 0 || meta.vocabulary == 0) {
        error = "embedding annex is not the supported token-local E1 10x5 profile";
        last_error_ = error;
        return false;
    }
    astc_vulkan_embedding_profile profile;
    if (!astc_vulkan_embedding_make_profile(
            astc_vulkan_embedding_representation::kE1Local,
            astc_vulkan_footprint::k10x5, profile, error)) {
        last_error_ = error; return false;
    }
    astc_vulkan_embedding_layout layout;
    layout.profile = profile;
    layout.dimensions = meta.dimensions;
    layout.logical_to_physical.resize(meta.dimensions);
    for (uint32_t i = 0; i < meta.dimensions; ++i) layout.logical_to_physical[i] = i;
    if (!astc_vulkan_embedding_validate_layout(layout, error)) { last_error_ = error; return false; }

    dimensions_ = meta.dimensions;
    vocabulary_ = meta.vocabulary;
    tiles_per_token_ = astc_vulkan_embedding_tile_count(layout);
    const uint64_t expected_payload = static_cast<uint64_t>(vocabulary_) * tiles_per_token_ * 16;
    if (expected_payload > std::numeric_limits<size_t>::max()) {
        error = "embedding payload is too large for this host"; last_error_ = error; return false;
    }
    if (!read_binary(fs::path(root) / meta.payload, payload_, error) ||
        payload_.size() != expected_payload) {
        if (error.empty()) error = "embedding payload size mismatch";
        last_error_ = error; return false;
    }
    if (!read_f32_affine(fs::path(root) / meta.affine, affine_, vocabulary_, error)) {
        last_error_ = error; return false;
    }

    astcenc_config config{};
    const astcenc_error configured = astcenc_config_init(
        ASTCENC_PRF_LDR, 10, 5, 1, ASTCENC_PRE_FAST, 0, &config);
    astcenc_context * context = nullptr;
    const astcenc_error allocated = configured == ASTCENC_SUCCESS
        ? astcenc_context_alloc(&config, 1, &context) : configured;
    if (allocated != ASTCENC_SUCCESS || context == nullptr) {
        error = std::string("embedding ASTC decoder init failed: ") +
                astcenc_get_error_string(allocated);
        last_error_ = error; reset(); return false;
    }
    block_width_ = 10;
    block_height_ = 5;
    astc_context_ = context;
    ready_ = true;
    error.clear();
    return true;
}

bool astc_vulkan_embedding_provider::prepare_from_cache_source(
        const std::shared_ptr<const astc_vulkan_cache_source> & source, std::string & error) {
    reset();
    if (!source) { error = "embedding cache source is null"; return false; }
    astc_vulkan_cache_blob meta_blob, payload_blob, affine_blob;
    if (!source->blob(astc_vulkan_cache_blob_kind::embedding_metadata, meta_blob, error) ||
        !source->blob(astc_vulkan_cache_blob_kind::embedding_payload, payload_blob, error) ||
        !source->blob(astc_vulkan_cache_blob_kind::embedding_affine, affine_blob, error) ||
        meta_blob.size == 0 || payload_blob.size == 0 || affine_blob.size == 0) {
        if (error.empty()) error = "compiled cache has no complete E1 embedding annex";
        return false;
    }
    std::vector<uint8_t> metadata_bytes, affine_bytes;
    if (!source->read_range(astc_vulkan_cache_blob_kind::embedding_metadata, 0, meta_blob.size, metadata_bytes, error) ||
        !source->read_range(astc_vulkan_cache_blob_kind::embedding_payload, 0, payload_blob.size, payload_, error) ||
        !source->read_range(astc_vulkan_cache_blob_kind::embedding_affine, 0, affine_blob.size, affine_bytes, error)) return false;
    metadata meta;
    if (!parse_metadata_bytes(metadata_bytes, meta, error) || meta.tensor != "token_embd.weight" ||
        meta.representation != "e1-local" || meta.footprint != "10x5") return false;
    astc_vulkan_embedding_profile profile;
    if (!astc_vulkan_embedding_make_profile(astc_vulkan_embedding_representation::kE1Local,
                                             astc_vulkan_footprint::k10x5, profile, error)) return false;
    astc_vulkan_embedding_layout layout; layout.profile = profile; layout.dimensions = meta.dimensions;
    layout.logical_to_physical.resize(meta.dimensions);
    for (uint32_t i=0;i<meta.dimensions;++i) layout.logical_to_physical[i]=i;
    if (!astc_vulkan_embedding_validate_layout(layout,error)) return false;
    dimensions_=meta.dimensions; vocabulary_=meta.vocabulary; tiles_per_token_=astc_vulkan_embedding_tile_count(layout);
    const uint64_t expected = uint64_t(vocabulary_) * tiles_per_token_ * 16;
    if (payload_.size() != expected || affine_bytes.size() != size_t(vocabulary_) * 2 * sizeof(float)) {
        error="compiled E1 embedding annex has invalid payload or affine size"; return false;
    }
    affine_.resize(affine_bytes.size()/sizeof(float)); std::memcpy(affine_.data(), affine_bytes.data(), affine_bytes.size());
    for (uint32_t i=0;i<vocabulary_;++i) if (!std::isfinite(affine_[2*i]) || !std::isfinite(affine_[2*i+1]) || !(affine_[2*i+1] > 0)) {
        error="compiled E1 embedding affine is invalid"; return false;
    }
    astcenc_config config{};
    const astcenc_error configured=astcenc_config_init(ASTCENC_PRF_LDR,10,5,1,ASTCENC_PRE_FAST,0,&config);
    astcenc_context * context=nullptr;
    const astcenc_error allocated=configured==ASTCENC_SUCCESS ? astcenc_context_alloc(&config,1,&context) : configured;
    if (allocated != ASTCENC_SUCCESS || !context) { error="compiled E1 ASTC decoder init failed"; reset(); return false; }
    block_width_=10; block_height_=5; astc_context_=context; ready_=true; error.clear(); return true;
}

bool astc_vulkan_embedding_provider::is_ready(const char * tensor_name,
                                              uint32_t dimensions,
                                              uint32_t vocabulary) const {
    return ready_ && tensor_name != nullptr && std::strcmp(tensor_name, "token_embd.weight") == 0 &&
           dimensions == dimensions_ && vocabulary == vocabulary_;
}

bool astc_vulkan_embedding_provider::decode_token(uint32_t token, float * output, uint32_t dimensions) {
    if (!ready_ || output == nullptr || token >= vocabulary_ || dimensions != dimensions_) return false;
    const size_t token_offset = static_cast<size_t>(token) * tiles_per_token_ * 16;
    const uint32_t image_width = tiles_per_token_ * block_width_;
    const size_t pixels = static_cast<size_t>(image_width) * block_height_;
    std::vector<float> decoded(pixels * 4, 0.0f);
    void * decoded_slice = decoded.data();
    astcenc_image image{image_width, block_height_, 1, ASTCENC_TYPE_F32, &decoded_slice};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const astcenc_error status = astcenc_decompress_image(
        static_cast<astcenc_context *>(astc_context_), payload_.data() + token_offset,
        static_cast<size_t>(tiles_per_token_) * 16, &image, &swizzle, 0);
    if (status != ASTCENC_SUCCESS) return false;

    astc_vulkan_embedding_profile profile;
    std::string error;
    if (!astc_vulkan_embedding_make_profile(astc_vulkan_embedding_representation::kE1Local,
                                             astc_vulkan_footprint::k10x5, profile, error)) return false;
    const uint32_t values_per_tile = astc_vulkan_embedding_values_per_tile(profile);
    const auto affine_bias = affine_[static_cast<size_t>(token) * 2 + 0];
    const auto affine_scale = affine_[static_cast<size_t>(token) * 2 + 1];
    for (uint32_t logical = 0; logical < dimensions_; ++logical) {
        const uint32_t tile = logical / values_per_tile;
        const uint32_t in_tile = logical % values_per_tile;
        const uint32_t x = in_tile % block_width_;
        const uint32_t y = in_tile / block_width_;
        const size_t pixel = (static_cast<size_t>(y) * image_width +
                              static_cast<size_t>(tile) * block_width_ + x) * 4;
        output[logical] = affine_bias + affine_scale * (2.0f * decoded[pixel] - 1.0f);
    }
    return true;
}

bool astc_vulkan_embedding_provider::run(const char * tensor_name, const int32_t * token_ids,
                                         uint32_t n_tokens, float * output, uint32_t dimensions) {
    if (!is_ready(tensor_name, dimensions, vocabulary_) || token_ids == nullptr || output == nullptr) return false;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const int32_t token = token_ids[i];
        if (token < 0 || !decode_token(static_cast<uint32_t>(token), output + static_cast<size_t>(i) * dimensions, dimensions)) {
            last_error_ = "embedding token id is out of range or ASTC decode failed";
            return false;
        }
    }
    return true;
}

void astc_vulkan_embedding_provider::generation_begin() {
    ++generation_count_;
}

bool astc_vulkan_embedding_provider::is_ready_callback(void * user_data, const char * tensor_name,
                                                       uint32_t dimensions, uint32_t vocabulary) {
    return user_data != nullptr && static_cast<astc_vulkan_embedding_provider *>(user_data)->is_ready(
        tensor_name, dimensions, vocabulary);
}

bool astc_vulkan_embedding_provider::run_callback(void * user_data, const char * tensor_name,
                                                  const int32_t * token_ids, uint32_t n_tokens,
                                                  float * output, uint32_t dimensions) {
    return user_data != nullptr && static_cast<astc_vulkan_embedding_provider *>(user_data)->run(
        tensor_name, token_ids, n_tokens, output, dimensions);
}

bool astc_vulkan_embedding_provider::native_bind_callback(void * user_data, struct ggml_tensor * node,
                                                          const char * tensor_name) {
    return user_data != nullptr && static_cast<astc_vulkan_embedding_provider *>(user_data)->native_bind(node, tensor_name);
}

void astc_vulkan_embedding_provider::generation_begin_callback(void * user_data) {
    if (user_data != nullptr) static_cast<astc_vulkan_embedding_provider *>(user_data)->generation_begin();
}
