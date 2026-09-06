#include "astc-vulkan-llama-provider.h"
#include "astc-vulkan-ggml-external-op.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

bool read_spirv(const std::string & path, std::vector<uint32_t> & result) {
    result.clear();
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) return false;
    result.resize(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), size);
    return static_cast<bool>(file);
}

bool parse_ffn_down_layer(const std::string & tensor_name, uint32_t & layer) {
    unsigned parsed = 0;
    char trailing = '\0';
    if (std::sscanf(tensor_name.c_str(), "blk.%u.ffn_down.weight%c", &parsed, &trailing) != 1) {
        return false;
    }
    layer = parsed;
    return true;
}

} // namespace

astc_vulkan_llama_provider::~astc_vulkan_llama_provider() {
    if (ready_) {
        std::fprintf(stderr,
            "ASTC runtime stats: bound_layers=%zu dispatches=%llu dispatch_failures=%llu tokens=%llu\n",
            entries_.size(),
            static_cast<unsigned long long>(dispatch_calls_),
            static_cast<unsigned long long>(dispatch_failures_),
            static_cast<unsigned long long>(dispatch_tokens_));
    }
    reset();
}

void astc_vulkan_llama_provider::reset() {
    if (external_op_installed_) {
        ggml_vk_astc_external_op::uninstall();
        external_op_installed_ = false;
    }
    entries_.clear();
    d1_spirv_.clear();
    d2_spirv_.clear();
    overlay_.reset();
    shared_device_.reset();
    last_error_.clear();
    ready_ = false;
    dispatch_calls_ = 0;
    dispatch_failures_ = 0;
    dispatch_tokens_ = 0;
}

bool astc_vulkan_llama_provider::prepare(const options & options, std::string & error) {
    reset();
    if (options.model_path.empty()) {
        error = "ASTC llama provider requires a runtime model path";
        return false;
    }
    shared_device_ = std::make_shared<astc_vulkan_shared_device>();
    if (!shared_device_->init(error)) {
        reset();
        return false;
    }

    astc_vulkan_runtime_overlay_options overlay_options;
    overlay_options.source_model_path = options.model_path;
    overlay_options.runtime_model_path = options.model_path;
    overlay_options.cache_path = options.cache_path;
    overlay_options.policy.policy = options.policy;
    overlay_options.policy.allow_experimental = options.allow_experimental;
    overlay_options.policy.allow_unverified = options.allow_unverified;
    overlay_options.memory_budget = shared_device_->memory_budget();
    if (!overlay_.prepare(overlay_options, error)) {
        reset();
        return false;
    }

#ifndef ASTC_VULKAN_D1_SHADER_PATH
    error = "ASTC runtime provider was built without the D1 shader path";
    reset();
    return false;
#else
    if (!read_spirv(ASTC_VULKAN_D1_SHADER_PATH, d1_spirv_)) {
        error = "ASTC runtime provider cannot read the D1 shader";
        reset();
        return false;
    }
#endif
#ifndef ASTC_VULKAN_D2_SHADER_PATH
    error = "ASTC runtime provider was built without the D2 shader path";
    reset();
    return false;
#else
    if (!read_spirv(ASTC_VULKAN_D2_SHADER_PATH, d2_spirv_)) {
        error = "ASTC runtime provider cannot read the D2 shader";
        reset();
        return false;
    }
#endif

    for (const auto & plan_entry : overlay_.catalog().plan.entries) {
        if (plan_entry.use_native_fallback) {
            continue;
        }
        uint32_t layer = 0;
        if (!parse_ffn_down_layer(plan_entry.tensor_name, layer)) continue;
        astc_vulkan_page_material material;
        if (!overlay_.resolve_tensor(plan_entry.tensor_name, material, error)) {
            reset();
            return false;
        }
        if (material.resolve.use_native_fallback) continue;
        auto prepared = std::make_unique<entry>();
        prepared->layer = layer;
        prepared->adapter.set_shared_device(shared_device_);
        if (!prepared->adapter.prepare_artifact_from_cache(
                options.model_path, options.cache_path, plan_entry.tensor_name,
                plan_entry.artifact_id, error, options.allow_experimental,
                options.allow_unverified) || !prepared->adapter.ready()) {
            reset();
            return false;
        }
        entries_.emplace(layer, std::move(prepared));
    }
    if (entries_.empty()) {
        error = "ASTC cache contains no resident D1 FFN-down artifacts eligible for runtime";
        reset();
        return false;
    }
    // Install only the generic dispatcher registry.  No graph node is bound
    // yet, so the existing CPU custom-op bridge remains the active path until
    // the native ASTC callback is supplied by a later device-sharing sweep.
    std::string external_error;
    if (ggml_vk_astc_external_op::install(external_error)) {
        external_op_installed_ = true;
    } else {
        std::fprintf(stderr, "ASTC external Vulkan seam unavailable: %s\n",
                     external_error.c_str());
    }
    const auto & summary = overlay_.summary();
    std::fprintf(stderr,
        "ASTC runtime overlay active: planned=%zu resident=%zu native_fallbacks=%zu pages=%zu resident_pages=%zu device_bytes=%llu host_bytes=%llu\n",
        summary.planned_artifacts, summary.resident_artifacts, summary.native_fallbacks,
        summary.pages, summary.resident_pages,
        static_cast<unsigned long long>(summary.resident_device_bytes),
        static_cast<unsigned long long>(summary.resident_host_bytes));
    ready_ = true;
    error.clear();
    return true;
}

bool astc_vulkan_llama_provider::is_ready(
        uint32_t layer, uint32_t input_columns, uint32_t output_columns) const {
    const auto it = entries_.find(layer);
    if (!ready_ || it == entries_.end()) return false;
    const auto & record = it->second->adapter.binding().record;
    return record.width == input_columns && record.height == output_columns;
}

bool astc_vulkan_llama_provider::run(
        uint32_t layer, const float * input, uint32_t n_tokens, uint32_t input_columns,
        float * output, uint32_t output_columns) {
    const auto it = entries_.find(layer);
    if (!is_ready(layer, input_columns, output_columns) || input == nullptr || output == nullptr || n_tokens == 0) {
        ++dispatch_failures_;
        last_error_ = "ASTC runtime provider was called for an unprepared layer";
        return false;
    }
    ++dispatch_calls_;
    dispatch_tokens_ += n_tokens;
    // ggml's CPU custom-op boundary invokes this serially. Keep the vector
    // copies here for the first functional bridge; the in-backend Vulkan path
    // will consume native buffers directly through the same provider policy.
    const size_t input_count = static_cast<size_t>(n_tokens) * input_columns;
    std::vector<float> activations(input, input + input_count);
    std::vector<float> result;
    std::string error;
    const auto & spirv = it->second->adapter.dispatch_kind() ==
        astc_vulkan_scheduler_dispatch_kind::kD2Paired ? d2_spirv_ : d1_spirv_;
    if (!it->second->adapter.run(spirv, activations, result, error) ||
        result.size() != static_cast<size_t>(n_tokens) * output_columns) {
        ++dispatch_failures_;
        last_error_ = error.empty() ? "ASTC runtime dispatch returned invalid output" : error;
        return false;
    }
    std::memcpy(output, result.data(), result.size() * sizeof(float));
    last_error_.clear();
    return true;
}

bool astc_vulkan_llama_provider::is_ready_callback(
        void * user_data, uint32_t layer, uint32_t input_columns, uint32_t output_columns) {
    const auto * provider = static_cast<const astc_vulkan_llama_provider *>(user_data);
    return provider != nullptr && provider->is_ready(layer, input_columns, output_columns);
}

bool astc_vulkan_llama_provider::run_callback(
        void * user_data, uint32_t layer, const float * input, uint32_t n_tokens,
        uint32_t input_columns, float * output, uint32_t output_columns) {
    auto * provider = static_cast<astc_vulkan_llama_provider *>(user_data);
    return provider != nullptr && provider->run(
        layer, input, n_tokens, input_columns, output, output_columns);
}
