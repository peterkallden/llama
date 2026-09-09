#include "astc-vulkan-tensor-catalog.h"

#include <cctype>
#include <vector>

namespace {

std::vector<std::string> split_name(const std::string & name) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= name.size()) {
        const size_t end = name.find('.', start);
        parts.push_back(name.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return parts;
}

std::string replace_underscores(std::string value) {
    for (char & ch : value) if (ch == '_') ch = '/';
    return value;
}

std::string role_for_component(const std::string & component) {
    if (component == "ffn_down") return "ffn.down";
    if (component == "ffn_up") return "ffn.up";
    if (component == "ffn_gate") return "ffn.gate";
    if (component == "attn_q") return "attention.query";
    if (component == "attn_k") return "attention.key";
    if (component == "attn_v") return "attention.value";
    if (component == "attn_output") return "attention.output";
    if (component == "attn_norm") return "normalization.attention";
    if (component == "ffn_norm") return "normalization.ffn";
    if (component == "post_attention_norm") return "normalization.post_attention";
    if (component == "token_embd") return "embedding.token";
    if (component == "output_norm") return "normalization.output";
    if (component == "output") return "output";
    return component.empty() ? "unknown" : replace_underscores(component);
}

} // namespace

std::string astc_vulkan_tensor_semantic_role(const std::string & name) {
    const std::vector<std::string> parts = split_name(name);
    if (parts.empty()) return "unknown";
    const size_t component_index = parts.size() >= 2 && parts.back() == "weight" ?
        parts.size() - 2 : parts.size() - 1;
    return role_for_component(parts[component_index]);
}

std::string astc_vulkan_tensor_canonical_path(const std::string & name) {
    const std::vector<std::string> parts = split_name(name);
    if (parts.empty()) return "unknown";
    size_t index = 0;
    std::string result;
    if (parts.size() >= 2 && parts[0] == "blk" && !parts[1].empty()) {
        bool numeric = true;
        for (const char ch : parts[1]) {
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                numeric = false;
                break;
            }
        }
        if (numeric) {
            result = "layers/" + parts[1];
            index = 2;
        }
    }
    if (result.empty()) result = "global";
    for (; index < parts.size(); ++index) {
        if (parts[index].empty()) continue;
        result += "/" + replace_underscores(parts[index]);
    }
    return result;
}

astc_vulkan_tensor_role_capability astc_vulkan_tensor_role_capability_for_name(
        const std::string & name) {
    astc_vulkan_tensor_role_capability result;
    result.semantic_role = astc_vulkan_tensor_semantic_role(name);
    if (result.semantic_role == "ffn.down") {
        result.matrix_candidate = true;
        result.native_binding_ready = true;
        result.source_builder_ready = true;
        return result;
    }

    // These roles are deliberately prepared for tensor-name binding and
    // scheduler/catalog discovery, but their source builders remain planned.
    result.matrix_candidate =
        result.semantic_role == "ffn.up" ||
        result.semantic_role == "ffn.gate" ||
        result.semantic_role == "attention.query" ||
        result.semantic_role == "attention.key" ||
        result.semantic_role == "attention.value" ||
        result.semantic_role == "attention.output" ||
        result.semantic_role == "output";
    result.native_binding_ready = result.matrix_candidate;
    return result;
}
