#pragma once

#include "astc-vulkan-runtime-overlay.h"
#include "astc-vulkan-scheduler-adapter.h"

#include "llama-ext.h"

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

private:
    struct entry {
        uint32_t layer = 0;
        astc_vulkan_scheduler_adapter adapter;
    };

    bool ready_ = false;
    std::string last_error_;
    std::shared_ptr<astc_vulkan_shared_device> shared_device_;
    astc_vulkan_runtime_overlay overlay_;
    std::vector<uint32_t> d1_spirv_;
    std::vector<uint32_t> d2_spirv_;
    std::unordered_map<uint32_t, std::unique_ptr<entry>> entries_;
    uint64_t dispatch_calls_ = 0;
    uint64_t dispatch_failures_ = 0;
    uint64_t dispatch_tokens_ = 0;
};
