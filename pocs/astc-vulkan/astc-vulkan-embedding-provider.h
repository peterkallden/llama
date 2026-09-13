#pragma once

#include "llama-ext.h"
#include "astc-vulkan-cache.h"
#include "ggml-vulkan-external-op.h"

#include <cstdint>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

// ASTC-owned token embedding provider. It consumes a validated token-local E1
// annex, keeps the normal ggml_get_rows path as a hard fallback, and can
// record a native Vulkan GET_ROWS dispatch after lifecycle/resource checks.
// The explicit research/evidence policy is separate from ASTC legality and
// model correctness.
class astc_vulkan_embedding_provider {
public:
    astc_vulkan_embedding_provider();
    ~astc_vulkan_embedding_provider();

    astc_vulkan_embedding_provider(const astc_vulkan_embedding_provider &) = delete;
    astc_vulkan_embedding_provider & operator=(const astc_vulkan_embedding_provider &) = delete;

    bool prepare(const std::string & root, std::string & error);
    // Compiled ASTCCM containers expose the same immutable annex through the
    // generic cache-source interface, avoiding filesystem materialization.
    bool prepare_from_cache_source(const std::shared_ptr<const astc_vulkan_cache_source> & source,
                                   std::string & error);
    void reset();
    bool ready() const { return ready_; }

    bool is_ready(const char * tensor_name, uint32_t dimensions, uint32_t vocabulary) const;
    bool run(const char * tensor_name, const int32_t * token_ids, uint32_t n_tokens,
             float * output, uint32_t dimensions);

    // The native path owns a token-local sampled ASTC atlas plus affine SSBO,
    // but borrows ggml's token/output buffers and command buffer per graph.
    // It returns false until the active graph device has accepted and
    // materialized the atlas, preserving the normal CPU get_rows fallback.
    bool native_bind(struct ggml_tensor * node, const char * tensor_name);
    void generation_begin();

    static bool is_ready_callback(void * user_data, const char * tensor_name,
                                  uint32_t dimensions, uint32_t vocabulary);
    static bool run_callback(void * user_data, const char * tensor_name,
                             const int32_t * token_ids, uint32_t n_tokens,
                             float * output, uint32_t dimensions);
    static bool native_bind_callback(void * user_data, struct ggml_tensor * node,
                                     const char * tensor_name);
    static void generation_begin_callback(void * user_data);

    uint32_t dimensions() const { return dimensions_; }
    uint32_t vocabulary() const { return vocabulary_; }
    uint32_t tiles_per_token() const { return tiles_per_token_; }
    uint64_t payload_bytes() const { return payload_.size(); }
    const std::vector<uint8_t> & payload() const { return payload_; }
    const std::vector<float> & affine() const { return affine_; }
    const std::string & last_error() const { return last_error_; }

    uint64_t readiness_queries() const { return readiness_queries_.load(); }
    uint64_t readiness_accepts() const { return readiness_accepts_.load(); }
    uint64_t cpu_dispatches() const { return cpu_dispatches_.load(); }
    uint64_t cpu_tokens() const { return cpu_tokens_.load(); }
    uint64_t native_binds() const { return native_binds_.load(); }
    uint64_t native_context_checks() const { return native_context_checks_.load(); }
    uint64_t native_context_accepts() const { return native_context_accepts_.load(); }
    uint64_t native_dispatches() const { return native_dispatches_.load(); }

private:
    bool decode_token(uint32_t token, float * output, uint32_t dimensions);
    bool can_record_native(const ggml_vk_external_op_dispatch_context * context);
    bool record_native(const ggml_vk_external_op_dispatch_context * context);
    bool materialize_native(const ggml_vk_external_op_dispatch_context * context, std::string & error);
    static bool native_context_callback(const ggml_vk_external_op_dispatch_context * context, void * user_data);
    static bool native_dispatch_callback(const ggml_vk_external_op_dispatch_context * context, void * user_data);

    bool ready_ = false;
    uint32_t dimensions_ = 0;
    uint32_t vocabulary_ = 0;
    uint32_t tiles_per_token_ = 0;
    uint32_t block_width_ = 0;
    uint32_t block_height_ = 0;
    std::vector<uint8_t> payload_;
    std::vector<float> affine_; // bias, scale for each vocabulary row
    void * astc_context_ = nullptr; // opaque astcenc_context, kept out of public ABI
    class astc_vulkan_embedding_gpu_session * gpu_session_ = nullptr;
    struct ggml_tensor * native_node_ = nullptr;
    std::string last_error_;
    uint64_t generation_count_ = 0;
    mutable std::atomic<uint64_t> readiness_queries_{0};
    mutable std::atomic<uint64_t> readiness_accepts_{0};
    std::atomic<uint64_t> cpu_dispatches_{0};
    std::atomic<uint64_t> cpu_tokens_{0};
    std::atomic<uint64_t> native_binds_{0};
    std::atomic<uint64_t> native_context_checks_{0};
    std::atomic<uint64_t> native_context_accepts_{0};
    std::atomic<uint64_t> native_dispatches_{0};
};
