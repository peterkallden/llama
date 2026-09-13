#pragma once

#include "astc-vulkan-runtime-overlay.h"
#include "astc-vulkan-scheduler-adapter.h"
#include "astc-vulkan-embedding-provider.h"
#include "ggml-vulkan-external-op.h"

#include "llama-ext.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Prepared D1/D2 cache provider consumed by llama's generic FFN runtime
// bridge. It owns one shared Vulkan device for the overlay and one resident
// image/dispatch binding per eligible tensor. This is intentionally an
// opt-in experimental runtime; a cache miss remains native GGUF execution.
class astc_vulkan_llama_provider {
public:
    ~astc_vulkan_llama_provider();

    struct options {
        // `model_path` is the runtime GGUF for backwards compatibility.
        std::string model_path;
        // Optional exact source GGUF used to build the cache. When omitted,
        // the runtime model is also treated as the cache source.
        std::string source_model_path;
        std::string cache_path = "auto";
        // Optional immutable cache source (for example embedded ASTC sections
        // from a compiled model). The owner must outlive the provider.
        std::shared_ptr<const astc_vulkan_cache_source> cache_source;
        std::string cache_source_fingerprint;
        astc_vulkan_quality_policy policy = astc_vulkan_quality_policy::balanced;
        bool allow_experimental = false;
        bool allow_unverified = false;
        bool require_all_artifacts = false;
    };

    bool prepare(const options & options, std::string & error);
    void reset();

    bool is_ready(uint32_t layer, uint32_t input_columns, uint32_t output_columns) const;
    bool run(uint32_t layer, const float * input, uint32_t n_tokens, uint32_t input_columns,
             float * output, uint32_t output_columns);
    bool ready() const { return ready_; }
    const std::string & last_error() const { return last_error_; }

    static bool is_ready_callback(void * user_data, uint32_t layer,
                                  uint32_t input_columns, uint32_t output_columns);
    static bool run_callback(void * user_data, uint32_t layer,
                             const float * input, uint32_t n_tokens, uint32_t input_columns,
                             float * output, uint32_t output_columns);
    static bool native_bind_callback(void * user_data, struct ggml_tensor * node, uint32_t layer);
    static void native_generation_begin_callback(void * user_data);

    // Name-keyed callbacks are the production seam for all supported matrix
    // roles.  The legacy layer-keyed callbacks above remain for source
    // compatibility with the original FFN-down bridge.
    static bool tensor_is_ready_callback(void * user_data, const char * tensor_name,
                                         uint32_t input_columns, uint32_t output_columns);
    static bool tensor_run_callback(void * user_data, const char * tensor_name,
                                    const float * input, uint32_t n_tokens,
                                    uint32_t input_columns, float * output,
                                    uint32_t output_columns);
    static bool tensor_native_bind_callback(void * user_data, struct ggml_tensor * node,
                                            const char * tensor_name);
    static void tensor_generation_begin_callback(void * user_data);

    // Optional token embedding callbacks.  They are independent from the
    // matrix callbacks above and return false when no validated embedding
    // annex is present, preserving the native GGUF get_rows path.
    static bool embedding_is_ready_callback(void * user_data, const char * tensor_name,
                                            uint32_t dimensions, uint32_t vocabulary);
    static bool embedding_run_callback(void * user_data, const char * tensor_name,
                                       const int32_t * token_ids, uint32_t n_tokens,
                                       float * output, uint32_t dimensions);
    static bool embedding_native_bind_callback(void * user_data, struct ggml_tensor * node,
                                               const char * tensor_name);
    static void embedding_generation_begin_callback(void * user_data);

private:
    struct entry {
        std::string tensor_name;
        uint32_t layer = 0;
        astc_vulkan_scheduler_adapter adapter;
    };

    struct native_binding {
        astc_vulkan_llama_provider * provider = nullptr;
        std::string tensor_name;
        uint32_t layer = 0;
    };

    bool ready_ = false;
    std::string last_error_;
    std::shared_ptr<astc_vulkan_shared_device> shared_device_;
    astc_vulkan_runtime_overlay overlay_;
    // Optional token-local embedding annex.  Absence or failed validation is
    // deliberately non-fatal and leaves ordinary GGUF get_rows in place.
    std::unique_ptr<astc_vulkan_embedding_provider> embedding_provider_;
    std::vector<uint32_t> d1_spirv_;
    std::vector<uint32_t> d2_spirv_;
    // Tensor name is the authoritative runtime key. `layer` remains populated
    // for legacy FFN-down callbacks, but is not sufficient for other matrices.
    std::unordered_map<std::string, std::unique_ptr<entry>> entries_;
    std::vector<std::unique_ptr<native_binding>> native_bindings_;
    // Retained only to materialize the same verified cache artifacts on the
    // actual ggml graph device if prepare() initially observed a different
    // default Vulkan device.
    options prepared_options_;
    // These counters deliberately describe the runtime seam, rather than
    // quality or cache residency. They make it possible to distinguish a
    // native Vulkan dispatch from the temporary CPU custom-op fallback.
    std::atomic<uint64_t> dispatch_calls_{0};
    std::atomic<uint64_t> dispatch_failures_{0};
    std::atomic<uint64_t> dispatch_tokens_{0};
    std::atomic<uint64_t> cpu_fallback_calls_{0};
    std::atomic<uint64_t> native_bind_calls_{0};
    std::atomic<uint64_t> native_context_checks_{0};
    std::atomic<uint64_t> native_context_accepts_{0};
    // A graph device that rejects the artifact format cannot become usable
    // later in the same provider lifetime. Remember it so every bound layer
    // does not repeat a failed materialization attempt.
    uint64_t rejected_graph_device_ = 0;
    std::string rejected_graph_device_error_;
    bool external_op_installed_ = false;

    bool bind_native_node(ggml_tensor * node, uint32_t layer);
    bool bind_native_tensor(ggml_tensor * node, const char * tensor_name);
    bool is_ready_tensor(const char * tensor_name, uint32_t input_columns,
                         uint32_t output_columns) const;
    bool run_tensor(const char * tensor_name, const float * input, uint32_t n_tokens,
                    uint32_t input_columns, float * output, uint32_t output_columns);
    bool materialize_entries(const std::shared_ptr<astc_vulkan_shared_device> & device,
                             std::unordered_map<std::string, std::unique_ptr<entry>> & entries,
                             std::string & error) const;
    bool rebind_to_graph_device(const ggml_vk_external_op_dispatch_context * context,
                                std::string & error);
    bool can_record_native(const std::string & tensor_name,
                           const ggml_vk_external_op_dispatch_context * context);
    bool record_native(const std::string & tensor_name,
                       const ggml_vk_external_op_dispatch_context * context);
    static bool native_dispatch_callback(
            const ggml_vk_external_op_dispatch_context * context, void * user_data);
    static bool native_context_callback(
            const ggml_vk_external_op_dispatch_context * context, void * user_data);
};
