#include "agent/tooling/contracts/tool-result-contracts.h"

#include "memory/memory-types.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

json common_tool_memory_hit_to_json(const common_memory_hit & hit) {
    return {
        {"memory", common_tool_memory_record_to_json(hit.memory)},
        {"score", hit.final_score},
        {"provenance", hit.provenance},
    };
}

} // namespace

json common_tool_resource_descriptor_to_json(
        const agent_resource_descriptor & descriptor) {
    json result = {
        {"uri", descriptor.uri},
        {"name", descriptor.name},
        {"description", descriptor.description},
        {"mime_type", descriptor.mime_type},
        {"size_bytes", descriptor.size_bytes},
        {"scope", common_runtime_resource_scope_name(descriptor.scope)},
        {"metadata", {
            {"purpose", descriptor.metadata.purpose},
            {"content_summary", descriptor.metadata.content_summary},
            {"usage_hint", descriptor.metadata.usage_hint},
            {"limitations", descriptor.metadata.limitations},
            {"keywords", descriptor.metadata.keywords},
            {"entities", descriptor.metadata.entities},
            {"processing_cache_key", descriptor.metadata.processing_cache_key},
            {"declared_language", descriptor.metadata.declared_language},
            {"resolved_language", descriptor.metadata.resolved_language},
            {"language_confidence", descriptor.metadata.language_confidence},
            {"language_source", descriptor.metadata.language_source},
        }},
    };
    if (!descriptor.lineage.parent_uri.empty()) {
        result["lineage"] = {
            {"parent_uri", descriptor.lineage.parent_uri},
            {"chunk_index", descriptor.lineage.chunk_index},
            {"chunk_count", descriptor.lineage.chunk_count},
            {"byte_offset", descriptor.lineage.byte_offset},
            {"byte_length", descriptor.lineage.byte_length},
            {"overlap_bytes", descriptor.lineage.overlap_bytes},
            {"derivation", descriptor.lineage.derivation},
        };
    }
    return result;
}

json common_tool_resource_read_result_to_json(
        const common_tool_resource_read_result & result) {
    json payload = {
        {"resource", common_tool_resource_descriptor_to_json(result.resource)},
        {"representation", result.representation},
        {"content", result.content},
    };
    if (!result.content_encoding.empty()) {
        payload["content_encoding"] = result.content_encoding;
    }
    return payload;
}

json common_tool_resource_inspect_result_to_json(
        const common_tool_resource_inspect_result & result) {
    return {
        {"resource", common_tool_resource_descriptor_to_json(result.resource)},
        {"available_representations", result.available_representations},
    };
}

json common_tool_web_search_result_to_json(
        const common_tool_web_search_result & result) {
    json payload = {
        {"results", result.results},
        {"provider", result.provider},
    };
    if (result.truncated.has_value()) {
        payload["truncated"] = *result.truncated;
    }
    if (result.total_results.has_value()) {
        payload["total_results"] = *result.total_results;
    }
    return payload;
}

json common_tool_web_fetch_result_to_json(
        const common_tool_web_fetch_result & result) {
    return {
        {"url", result.url},
        {"final_url", result.final_url},
        {"status", result.status},
        {"content_type", result.content_type},
        {"title", result.title},
        {"text", result.text},
        {"truncated", result.truncated},
    };
}

json common_tool_web_fetch_inline_result_to_json(
        const common_tool_web_fetch_inline_result & result) {
    return {
        {"url", result.url},
        {"final_url", result.final_url},
        {"status", result.status},
        {"content_type", result.content_type},
        {"title", result.title},
        {"text_excerpt", result.text_excerpt},
        {"text_length", result.text_length},
        {"truncated", result.truncated},
    };
}

json common_tool_memory_record_to_json(
        const common_memory_record & memory) {
    return {
        {"id", memory.id},
        {"kind", common_memory_kind_name(memory.kind)},
        {"content", memory.content},
        {"summary", memory.summary},
        {"scope", common_memory_scope_name(memory.scope)},
        {"importance", memory.importance},
        {"confidence", memory.confidence},
        {"created_at", memory.created_at},
    };
}

json common_tool_memory_search_result_to_json(
        const common_tool_memory_search_payload & result) {
    json values = json::array();
    for (const auto & hit : result.hits) {
        values.push_back(common_tool_memory_hit_to_json(hit));
    }
    return {
        {"results", std::move(values)},
    };
}

json common_tool_memory_get_result_to_json(
        const common_memory_record & memory) {
    return {
        {"memory", common_tool_memory_record_to_json(memory)},
    };
}

json common_tool_memory_remember_result_to_json(
        const common_tool_memory_remember_payload & result) {
    json payload = {
        {"ok", result.ok},
        {"decision", result.decision},
        {"reason", result.reason},
        {"kind", common_memory_kind_name(result.kind)},
        {"scope", common_memory_scope_name(result.scope)},
        {"content", result.content},
        {"related_count", result.related_count},
    };
    if (!result.related.empty()) {
        json related = json::array();
        for (const auto & hit : result.related) {
            related.push_back({
                {"id", hit.id},
                {"kind", common_memory_kind_name(hit.kind)},
                {"score", hit.score},
                {"content", hit.content},
            });
        }
        payload["related"] = std::move(related);
    }
    if (result.id.has_value()) {
        payload["id"] = *result.id;
    }
    if (result.error.has_value()) {
        payload["error"] = *result.error;
    }
    return payload;
}

json common_tool_plan_get_result_to_json(
        const common_tool_plan_get_payload & result) {
    json steps = json::array();
    for (const auto & step : result.steps) {
        json entry = {
            {"id", step.id},
            {"title", step.title},
            {"objective", step.objective},
            {"status", step.status},
            {"selected_tool", step.selected_tool},
        };
        steps.push_back(std::move(entry));
    }

    json payload = {
        {"plan_id", result.plan_id},
        {"version", result.version},
        {"goal", result.goal},
        {"active_step", result.active_step},
        {"next_action", result.next_action},
        {"steps", std::move(steps)},
    };
    if (result.history_count.has_value()) {
        payload["history_count"] = *result.history_count;
    }
    return payload;
}

json common_tool_chat_failure_payload_to_json(
        const std::string & code,
        const std::string & message,
        bool retryable,
        common_tool_failure_class failure_class) {
    return {
        {"ok", false},
        {"error", {
            {"code", code},
            {"message", message},
            {"retryable", retryable},
            {"class", common_tool_failure_class_name(failure_class)},
        }},
    };
}

json common_tool_chat_success_payload_to_json(
        const std::string & output_json_or_text) {
    const auto value = json::parse(output_json_or_text, nullptr, false);
    return value.is_discarded()
        ? json({{"ok", true}, {"result_text", output_json_or_text}})
        : json({{"ok", true}, {"result", value}});
}
