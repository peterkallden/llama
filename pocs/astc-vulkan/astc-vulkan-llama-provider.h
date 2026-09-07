#pragma once

#include "astc-vulkan-runtime-overlay.h"
#include "astc-vulkan-scheduler-adapter.h"
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
        std::string model_path;
        std::string cache_path = "auto";
        astc_vulkan_quality_policy policy = astc_vulkan_quality_policy::balanced;
        bool allow_experimental = false;
        bool allow_unverified = false;
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

private:
    struct entry {
        uint32_t layer = 0;
        astc_vulkan_scheduler_adapter adapter;
    };

    struct native_binding {
        astc_vulkan_llama_provider * provider = nullptr;
        uint32_t layer = 0;
    };

    bool ready_ = false;
    std::string last_error_;
    std::shared_ptr<astc_vulkan_shared_device> shared_device_;
    astc_vulkan_runtime_overlay overlay_;
    std::vector<uint32_t> d1_spirv_;
    std::vector<uint32_t> d2_spirv_;
    std::unordered_map<uint32_t, std::unique_ptr<entry>> entries_;
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
    bool materialize_entries(const std::shared_ptr<astc_vulkan_shared_device> & device,
                             std::unordered_map<uint32_t, std::unique_ptr<entry>> & entries,
                             std::string & error) const;
    bool rebind_to_graph_device(const ggml_vk_external_op_dispatch_context * context,
                                std::string & error);
    bool can_record_native(uint32_t layer,
                           const ggml_vk_external_op_dispatch_context * context);
    bool record_native(uint32_t layer,
                       const ggml_vk_external_op_dispatch_context * context);
    static bool native_dispatch_callback(
            const ggml_vk_external_op_dispatch_context * context, void * user_data);
    static bool native_context_callback(
            const ggml_vk_external_op_dispatch_context * context, void * user_data);
};
