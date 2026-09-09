#include "astc-vulkan-llama-provider.h"
#include "astc-vulkan-ggml-external-op.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

bool native_trace_enabled() {
    const char * value = std::getenv("ASTC_VULKAN_NATIVE_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void trace_native_context_reject(const char * reason,
                                 const ggml_vk_external_op_dispatch_context * context,
                                 VkDevice expected_device) {
    if (!native_trace_enabled()) return;
    std::fprintf(stderr,
                 "ASTC native context rejected: %s (ctx-device=%p expected-device=%p cmd=%p node=%p get-buffer=%p)\n",
                 reason,
                 context == nullptr ? nullptr : reinterpret_cast<void *>(context->native_device),
                 reinterpret_cast<void *>(expected_device),
                 context == nullptr ? nullptr : reinterpret_cast<void *>(context->native_command_buffer),
                 context == nullptr ? nullptr : static_cast<void *>(context->node),
                 context == nullptr ? nullptr : reinterpret_cast<void *>(context->get_buffer));
}

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

bool runtime_matrix_candidate(const std::string & tensor_name) {
    if (tensor_name == "output.weight") return true;
    if (tensor_name.rfind("blk.", 0) != 0) return false;
    static constexpr const char * kSuffixes[] = {
        ".ffn_up.weight", ".ffn_gate.weight", ".attn_q.weight",
        ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
    };
    for (const char * suffix : kSuffixes) {
        const size_t suffix_size = std::strlen(suffix);
        if (tensor_name.size() >= suffix_size &&
            tensor_name.compare(tensor_name.size() - suffix_size, suffix_size, suffix) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

astc_vulkan_llama_provider::~astc_vulkan_llama_provider() {
    if (ready_) {
        std::fprintf(stderr,
            "ASTC runtime stats: bound_tensors=%zu native-dispatches=%llu dispatch-failures=%llu "
            "native-tokens=%llu cpu-fallbacks=%llu native-binds=%llu context-checks=%llu "
            "context-accepts=%llu\n",
            entries_.size(),
            static_cast<unsigned long long>(dispatch_calls_.load()),
            static_cast<unsigned long long>(dispatch_failures_.load()),
            static_cast<unsigned long long>(dispatch_tokens_.load()),
            static_cast<unsigned long long>(cpu_fallback_calls_.load()),
            static_cast<unsigned long long>(native_bind_calls_.load()),
            static_cast<unsigned long long>(native_context_checks_.load()),
            static_cast<unsigned long long>(native_context_accepts_.load()));
    }
    reset();
}

void astc_vulkan_llama_provider::reset() {
    ggml_vk_astc_external_op::clear_owner(this);
    if (external_op_installed_) {
        ggml_vk_astc_external_op::uninstall();
        external_op_installed_ = false;
    }
    entries_.clear();
    native_bindings_.clear();
    d1_spirv_.clear();
    d2_spirv_.clear();
    overlay_.reset();
    shared_device_.reset();
    last_error_.clear();
    ready_ = false;
    dispatch_calls_ = 0;
    dispatch_failures_ = 0;
    dispatch_tokens_ = 0;
    cpu_fallback_calls_ = 0;
    native_bind_calls_ = 0;
    native_context_checks_ = 0;
    native_context_accepts_ = 0;
    rejected_graph_device_ = 0;
    rejected_graph_device_error_.clear();
    prepared_options_ = {};
}

bool astc_vulkan_llama_provider::prepare(const options & options, std::string & error) {
    const auto prepare_begin = std::chrono::steady_clock::now();
    reset();
    if (options.model_path.empty()) {
        error = "ASTC llama provider requires a runtime model path";
        return false;
    }
    const std::string source_model_path = options.source_model_path.empty() ?
        options.model_path : options.source_model_path;
    prepared_options_ = options;
    std::string external_error;
    if (ggml_vk_astc_external_op::install(external_error)) {
        external_op_installed_ = true;
    } else {
        std::fprintf(stderr, "ASTC external Vulkan seam unavailable: %s\n",
                     external_error.c_str());
    }

    shared_device_ = std::make_shared<astc_vulkan_shared_device>();
    ggml_vk_external_op_device_context device_context{};
    std::string device_error;
    const bool borrowed_device = external_op_installed_ &&
        ggml_vk_astc_external_op::get_default_device(device_context, device_error) &&
        shared_device_->init_borrowed(
            reinterpret_cast<VkPhysicalDevice>(device_context.native_physical_device),
            reinterpret_cast<VkDevice>(device_context.native_device),
            reinterpret_cast<VkQueue>(device_context.native_queue),
            device_context.native_queue_family, device_error);
    if (!borrowed_device && !shared_device_->init(error)) {
        reset();
        return false;
    }

    astc_vulkan_runtime_overlay_options overlay_options;
    overlay_options.source_model_path = source_model_path;
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
    if (native_trace_enabled()) {
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - prepare_begin).count();
        std::fprintf(stderr, "ASTC prepare: overlay catalog/residency ready after %.3f s\n", elapsed);
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

    if (!materialize_entries(shared_device_, entries_, error)) {
        reset();
        return false;
    }
    if (native_trace_enabled()) {
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - prepare_begin).count();
        std::fprintf(stderr, "ASTC prepare: %zu artifact bindings materialized after %.3f s\n",
                     entries_.size(), elapsed);
    }
    if (entries_.empty()) {
        error = "ASTC cache contains no resident D1/D2 matrix artifacts eligible for runtime";
        reset();
        return false;
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

bool astc_vulkan_llama_provider::materialize_entries(
        const std::shared_ptr<astc_vulkan_shared_device> & device,
        std::unordered_map<std::string, std::unique_ptr<entry>> & entries,
        std::string & error) const {
    if (device == nullptr || !device->ready()) {
        error = "ASTC runtime provider has no Vulkan device for artifact materialization";
        return false;
    }
    std::unordered_map<std::string, std::unique_ptr<entry>> prepared_entries;
    for (const auto & plan_entry : overlay_.catalog().plan.entries) {
        if (plan_entry.use_native_fallback) {
            continue;
        }
        uint32_t layer = 0;
        const bool is_ffn_down = parse_ffn_down_layer(plan_entry.tensor_name, layer);
        if (!is_ffn_down && !runtime_matrix_candidate(plan_entry.tensor_name)) continue;
        astc_vulkan_page_material material;
        if (!overlay_.resolve_tensor(plan_entry.tensor_name, material, error)) {
            return false;
        }
        if (material.resolve.use_native_fallback) continue;
        const auto entry_begin = std::chrono::steady_clock::now();
        if (native_trace_enabled()) {
            std::fprintf(stderr, "ASTC materialize begin: tensor=%s artifact=%s\n",
                         plan_entry.tensor_name.c_str(), plan_entry.artifact_id.c_str());
        }
        // `overlay_` validated the complete GGUF/cache contract exactly once
        // before this loop. Reuse its selected bytes rather than asking every
        // tensor adapter to rehash the same multi-gigabyte GGUF and payload.
        astc_vulkan_scheduler_artifact artifact;
        artifact.record = plan_entry.storage;
        artifact.artifact_id = plan_entry.artifact_id;
        artifact.variant = plan_entry.variant;
        artifact.normalization = plan_entry.normalization;
        artifact.paired_semantic = plan_entry.paired_semantic;
        artifact.evidence = plan_entry.evidence;
        artifact.cache_root = overlay_.catalog().validation.paths.root;
        artifact.payload = std::move(material.payload);
        artifact.layout = std::move(material.paired_layout);
        artifact.pair_map = std::move(material.pair_map);
        artifact.row_scales = std::move(material.row_scales);
        if (artifact.record.representation == astc_vulkan_representation::kPairedD2) {
            artifact.kind = astc_vulkan_scheduler_artifact_kind::kD2;
        } else if (artifact.record.representation == astc_vulkan_representation::kScalar ||
                   artifact.record.representation == astc_vulkan_representation::kGaugeLumaAlpha) {
            artifact.kind = astc_vulkan_scheduler_artifact_kind::kD1;
        } else {
            error = "ASTC model cache contains an unsupported runtime representation";
            return false;
        }
        auto prepared = std::make_unique<entry>();
        prepared->tensor_name = plan_entry.tensor_name;
        prepared->layer = layer;
        prepared->adapter.set_shared_device(device);
        if (!prepared->adapter.prepare_validated_artifact(
                std::move(artifact), error, prepared_options_.allow_experimental,
                prepared_options_.allow_unverified) || !prepared->adapter.ready()) {
            return false;
        }
        prepared_entries.emplace(plan_entry.tensor_name, std::move(prepared));
        if (native_trace_enabled()) {
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - entry_begin).count();
            std::fprintf(stderr, "ASTC materialize done: tensor=%s elapsed=%.3f s\n",
                         plan_entry.tensor_name.c_str(), elapsed);
        }
    }
    entries = std::move(prepared_entries);
    error.clear();
    return true;
}

bool astc_vulkan_llama_provider::is_ready(
        uint32_t layer, uint32_t input_columns, uint32_t output_columns) const {
    if (!ready_) return false;
    const entry * matched = nullptr;
    for (const auto & candidate : entries_) {
        uint32_t legacy_layer = 0;
        if (candidate.second != nullptr &&
            parse_ffn_down_layer(candidate.second->tensor_name, legacy_layer) &&
            legacy_layer == layer) {
            matched = candidate.second.get();
            break;
        }
    }
    if (matched == nullptr) return false;
    const auto & record = matched->adapter.binding().record;
    return record.width == input_columns && record.height == output_columns;
}

bool astc_vulkan_llama_provider::run(
        uint32_t layer, const float * input, uint32_t n_tokens, uint32_t input_columns,
        float * output, uint32_t output_columns) {
    const uint64_t fallback_index = cpu_fallback_calls_.fetch_add(1) + 1;
    if (native_trace_enabled() && fallback_index <= 8) {
        std::fprintf(stderr,
                     "ASTC CPU custom-op fallback: layer=%u tokens=%u input=%u output=%u\n",
                     layer, n_tokens, input_columns, output_columns);
    }
    if (!is_ready(layer, input_columns, output_columns) || input == nullptr || output == nullptr || n_tokens == 0) {
        ++dispatch_failures_;
        last_error_ = "ASTC runtime provider was called for an unprepared layer";
        return false;
    }
    entry * matched = nullptr;
    for (const auto & candidate : entries_) {
        uint32_t legacy_layer = 0;
        if (candidate.second != nullptr &&
            parse_ffn_down_layer(candidate.second->tensor_name, legacy_layer) &&
            legacy_layer == layer) {
            matched = candidate.second.get();
            break;
        }
    }
    if (matched == nullptr) return false;
    ++dispatch_calls_;
    dispatch_tokens_ += n_tokens;
    // ggml's CPU custom-op boundary invokes this serially. Keep the vector
    // copies here for the first functional bridge; the in-backend Vulkan path
    // will consume native buffers directly through the same provider policy.
    const size_t input_count = static_cast<size_t>(n_tokens) * input_columns;
    std::vector<float> activations(input, input + input_count);
    std::vector<float> result;
    std::string error;
    const auto & spirv = matched->adapter.dispatch_kind() ==
        astc_vulkan_scheduler_dispatch_kind::kD2Paired ? d2_spirv_ : d1_spirv_;
    if (!matched->adapter.run(spirv, activations, result, error) ||
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

bool astc_vulkan_llama_provider::native_bind_callback(
        void * user_data, ggml_tensor * node, uint32_t layer) {
    auto * provider = static_cast<astc_vulkan_llama_provider *>(user_data);
    return provider != nullptr && provider->bind_native_node(node, layer);
}

void astc_vulkan_llama_provider::native_generation_begin_callback(void * user_data) {
    auto * provider = static_cast<astc_vulkan_llama_provider *>(user_data);
    if (provider != nullptr) ggml_vk_astc_external_op::clear_owner(provider);
}

bool astc_vulkan_llama_provider::native_context_callback(
        const ggml_vk_external_op_dispatch_context * context, void * user_data) {
    const auto * binding = static_cast<const native_binding *>(user_data);
    if (binding == nullptr || binding->provider == nullptr) return false;
    const bool accepted = binding->provider->can_record_native(binding->tensor_name, context);
    if (accepted) {
        binding->provider->native_context_accepts_.fetch_add(1);
    }
    if (native_trace_enabled()) {
        std::fprintf(stderr, "ASTC native context %s: tensor=%s layer=%u node=%p cmd=%p\n",
                     accepted ? "accepted" : "rejected", binding->tensor_name.c_str(), binding->layer,
                     context == nullptr ? nullptr : static_cast<void *>(context->node),
                     context == nullptr ? nullptr : reinterpret_cast<void *>(context->native_command_buffer));
    }
    return accepted;
}

bool astc_vulkan_llama_provider::native_dispatch_callback(
        const ggml_vk_external_op_dispatch_context * context, void * user_data) {
    const auto * binding = static_cast<const native_binding *>(user_data);
    return binding != nullptr && binding->provider != nullptr &&
           binding->provider->record_native(binding->tensor_name, context);
}

bool astc_vulkan_llama_provider::bind_native_node(ggml_tensor * node, uint32_t layer) {
    if (!ready_ || node == nullptr) return false;
    std::string tensor_name;
    if (node->src[0] != nullptr && node->src[0]->name[0] != '\0') {
        tensor_name = node->src[0]->name;
    }
    auto entry_it = entries_.find(tensor_name);
    if (entry_it == entries_.end()) {
        for (const auto & candidate : entries_) {
            uint32_t legacy_layer = 0;
            if (candidate.second != nullptr &&
                parse_ffn_down_layer(candidate.second->tensor_name, legacy_layer) &&
                legacy_layer == layer) {
                entry_it = entries_.find(candidate.first);
                tensor_name = candidate.first;
                break;
            }
        }
    }
    if (entry_it == entries_.end()) return false;
    native_bind_calls_.fetch_add(1);
    if (native_trace_enabled()) {
        std::fprintf(stderr, "ASTC native bind: tensor=%s layer=%u node=%p\n",
                     tensor_name.c_str(), layer, static_cast<void *>(node));
    }
    native_binding * binding = nullptr;
    for (const auto & candidate : native_bindings_) {
        if (candidate->tensor_name == tensor_name) {
            binding = candidate.get();
            break;
        }
    }
    if (binding == nullptr) {
        auto created = std::make_unique<native_binding>();
        created->provider = this;
        created->tensor_name = tensor_name;
        created->layer = layer;
        binding = created.get();
        native_bindings_.push_back(std::move(created));
    }
    ggml_vk_astc_external_op::bind_node(
        node, native_dispatch_callback, binding, native_context_callback, this);
    return true;
}

bool astc_vulkan_llama_provider::rebind_to_graph_device(
        const ggml_vk_external_op_dispatch_context * context, std::string & error) {
    if (context == nullptr || context->native_physical_device == 0 || context->native_device == 0 ||
        context->native_queue == 0 || context->native_queue_family == UINT32_MAX) {
        error = "ASTC native graph context has incomplete Vulkan device handles";
        return false;
    }
    auto rebound_device = std::make_shared<astc_vulkan_shared_device>();
    if (!rebound_device->init_borrowed(
            reinterpret_cast<VkPhysicalDevice>(context->native_physical_device),
            reinterpret_cast<VkDevice>(context->native_device),
            reinterpret_cast<VkQueue>(context->native_queue),
            context->native_queue_family, error)) {
        return false;
    }
    std::unordered_map<std::string, std::unique_ptr<entry>> rebound_entries;
    if (!materialize_entries(rebound_device, rebound_entries, error)) return false;
    if (rebound_entries.empty()) {
        error = "ASTC graph-device rebind produced no eligible artifacts";
        return false;
    }
    entries_ = std::move(rebound_entries);
    shared_device_ = std::move(rebound_device);
    std::fprintf(stderr, "ASTC native resources rebound to active ggml Vulkan device\n");
    error.clear();
    return true;
}

bool astc_vulkan_llama_provider::can_record_native(
        const std::string & tensor_name, const ggml_vk_external_op_dispatch_context * context) {
    native_context_checks_.fetch_add(1);
    if (!ready_) {
        trace_native_context_reject("provider is not ready", context, VK_NULL_HANDLE);
        return false;
    }
    if (context == nullptr) {
        trace_native_context_reject("missing external context", context, shared_device_ == nullptr ? VK_NULL_HANDLE : shared_device_->device());
        return false;
    }
    if (shared_device_ == nullptr) {
        trace_native_context_reject("missing shared device", context, VK_NULL_HANDLE);
        return false;
    }
    if (context->native_device == 0) {
        trace_native_context_reject("missing Vulkan device", context, shared_device_->device());
        return false;
    }
    if (context->native_command_buffer == 0) {
        trace_native_context_reject("missing command buffer", context, shared_device_->device());
        return false;
    }
    const auto entry_it = entries_.find(tensor_name);
    if (entry_it == entries_.end() || context->node == nullptr) {
        trace_native_context_reject("missing artifact or graph node", context, shared_device_->device());
        return false;
    }
    const auto & record = entry_it->second->adapter.binding().record;
    // The native path binds the ordinary MUL_MAT node, whose activation is
    // normally src[1]. The legacy CPU bridge used src[0]. Accept either only
    // when it has the expected input width and current token extent.
    const auto matches_activation = [&](const ggml_tensor * candidate) {
        return candidate != nullptr && candidate->ne[0] == record.width &&
               candidate->ne[1] == context->node->ne[1];
    };
    if (!matches_activation(context->node->src[0]) &&
        !matches_activation(context->node->src[1])) {
        trace_native_context_reject("native node has no matching activation source", context,
                                    shared_device_->device());
        return false;
    }
    if (reinterpret_cast<VkDevice>(context->native_device) != shared_device_->device()) {
        if (rejected_graph_device_ == context->native_device) {
            trace_native_context_reject(rejected_graph_device_error_.c_str(), context,
                                        shared_device_->device());
            return false;
        }
        std::string error;
        if (!rebind_to_graph_device(context, error)) {
            last_error_ = error.empty() ? "ASTC graph-device rebind failed" : error;
            rejected_graph_device_ = context->native_device;
            rejected_graph_device_error_ = last_error_;
            trace_native_context_reject(last_error_.c_str(), context, shared_device_->device());
            return false;
        }
        rejected_graph_device_ = 0;
        rejected_graph_device_error_.clear();
    }
    if (context->node == nullptr || context->node->src[0] == nullptr) {
        trace_native_context_reject("missing node or activation source", context, shared_device_->device());
        return false;
    }
    if (context->get_buffer == nullptr) {
        trace_native_context_reject("missing buffer-view callback", context, shared_device_->device());
        return false;
    }
    return true;
}

bool astc_vulkan_llama_provider::record_native(
        const std::string & tensor_name, const ggml_vk_external_op_dispatch_context * context) {
    if (!can_record_native(tensor_name, context)) return false;
    const auto it = entries_.find(tensor_name);
    if (it == entries_.end()) return false;
    const auto & record = it->second->adapter.binding().record;
    const ggml_tensor * activation_tensor = nullptr;
    const auto matches_activation = [&](const ggml_tensor * candidate) {
        return candidate != nullptr && candidate->ne[0] == record.width &&
               candidate->ne[1] == context->node->ne[1];
    };
    if (matches_activation(context->node->src[1])) {
        activation_tensor = context->node->src[1];
    } else if (matches_activation(context->node->src[0])) {
        activation_tensor = context->node->src[0];
    }
    if (activation_tensor == nullptr) return false;
    ggml_vk_external_op_buffer_view activation{};
    ggml_vk_external_op_buffer_view output{};
    if (!context->get_buffer(context->backend_context, activation_tensor, &activation) ||
        !context->get_buffer(context->backend_context, context->node, &output) ||
        activation.native_device != context->native_device ||
        output.native_device != context->native_device) {
        return false;
    }
    const auto & adapter = it->second->adapter;
    const auto & spirv = adapter.dispatch_kind() ==
        astc_vulkan_scheduler_dispatch_kind::kD2Paired ? d2_spirv_ : d1_spirv_;
    const uint32_t samples = static_cast<uint32_t>(context->node->ne[1]);
    const uint32_t band_height = static_cast<uint32_t>(adapter.binding().record.height);
    std::string error;
    const bool ok = it->second->adapter.record_native(
        spirv, reinterpret_cast<VkDevice>(context->native_device),
        reinterpret_cast<VkCommandBuffer>(context->native_command_buffer),
        reinterpret_cast<VkBuffer>(activation.native_buffer), activation.offset, activation.size,
        reinterpret_cast<VkBuffer>(output.native_buffer), output.offset, output.size,
        samples, 0, band_height, error);
    if (!ok) {
        ++dispatch_failures_;
        last_error_ = error.empty() ? "ASTC native graph recording failed" : error;
    } else {
        dispatch_calls_.fetch_add(1);
        dispatch_tokens_.fetch_add(samples);
        if (native_trace_enabled()) {
            std::fprintf(stderr, "ASTC native dispatch: tensor=%s tokens=%u rows=%u\n",
                         tensor_name.c_str(), samples, band_height);
        }
        last_error_.clear();
    }
    return ok;
}
