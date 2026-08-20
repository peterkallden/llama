#include "agent-tool-result-json-contracts.h"

using json = nlohmann::ordered_json;

json render_agent_tool_resource_ref_json(const common_runtime_resource_ref & resource) {
    json result = {
        {"uri", resource.uri},
        {"scope", common_runtime_resource_scope_name(resource.scope)},
    };
    if (!resource.name.empty()) {
        result["name"] = resource.name;
    }
    if (!resource.description.empty()) {
        result["description"] = resource.description;
    }
    if (!resource.mime_type.empty()) {
        result["mime_type"] = resource.mime_type;
    }
    if (resource.size_bytes > 0) {
        result["size_bytes"] = resource.size_bytes;
    }
    if (!resource.lineage.parent_uri.empty()) {
        result["lineage"] = {
            {"parent_uri", resource.lineage.parent_uri},
            {"chunk_index", resource.lineage.chunk_index},
            {"chunk_count", resource.lineage.chunk_count},
            {"byte_offset", resource.lineage.byte_offset},
            {"byte_length", resource.lineage.byte_length},
            {"overlap_bytes", resource.lineage.overlap_bytes},
            {"derivation", resource.lineage.derivation},
        };
    }
    json metadata = json::object();
    if (!resource.metadata.purpose.empty()) {
        metadata["purpose"] = resource.metadata.purpose;
    }
    if (!resource.metadata.content_summary.empty()) {
        metadata["content_summary"] = resource.metadata.content_summary;
    }
    if (!resource.metadata.usage_hint.empty()) {
        metadata["usage_hint"] = resource.metadata.usage_hint;
    }
    if (!resource.metadata.limitations.empty()) {
        metadata["limitations"] = resource.metadata.limitations;
    }
    if (!resource.metadata.keywords.empty()) {
        metadata["keywords"] = resource.metadata.keywords;
    }
    if (!resource.metadata.entities.empty()) {
        metadata["entities"] = resource.metadata.entities;
    }
    if (!metadata.empty()) {
        result["metadata"] = std::move(metadata);
    }
    return result;
}

void attach_agent_tool_resource_refs_json(
        const std::vector<common_runtime_resource_ref> & resources,
        json & payload) {
    if (resources.empty()) {
        return;
    }

    json rendered = json::array();
    for (const auto & resource : resources) {
        rendered.push_back(render_agent_tool_resource_ref_json(resource));
    }
    payload["resources"] = std::move(rendered);
}

json make_agent_tool_failure_payload_json(
        const std::string & code,
        const std::string & message,
        bool retryable,
        common_tool_failure_class failure_class,
        const std::vector<common_runtime_resource_ref> & resources) {
    json payload = {
        {"ok", false},
        {"error", {
            {"code", code},
            {"message", message},
            {"retryable", retryable},
            {"class", common_tool_failure_class_name(failure_class)},
        }},
    };
    attach_agent_tool_resource_refs_json(resources, payload);
    return payload;
}

json make_agent_tool_success_payload_json(
        const std::string & output_json_or_text,
        const std::string & summary,
        const std::vector<common_runtime_resource_ref> & resources) {
    const auto value = json::parse(output_json_or_text, nullptr, false);
    json payload = value.is_discarded()
        ? json({{"ok", true}, {"result_text", output_json_or_text}})
        : json({{"ok", true}, {"result", value}});
    if (!summary.empty()) {
        payload["summary"] = summary;
    }
    attach_agent_tool_resource_refs_json(resources, payload);
    return payload;
}

json make_agent_tool_structured_success_payload_json(
        const std::string & structured_content_json,
        const std::string & summary,
        const std::vector<common_runtime_resource_ref> & resources) {
    const auto value = json::parse(structured_content_json, nullptr, false);
    json payload = value.is_discarded()
        ? json({{"ok", true}, {"result_text", structured_content_json}})
        : json({{"ok", true}, {"result", value}});
    if (!summary.empty()) {
        payload["summary"] = summary;
    }
    attach_agent_tool_resource_refs_json(resources, payload);
    return payload;
}

json make_agent_tool_text_success_payload_json(
        const std::string & text,
        const std::string & summary,
        const std::vector<common_runtime_resource_ref> & resources) {
    json payload = {
        {"ok", true},
        {"result_text", text},
    };
    if (!summary.empty()) {
        payload["summary"] = summary;
    }
    attach_agent_tool_resource_refs_json(resources, payload);
    return payload;
}

agent_tool_result make_agent_tool_failure_result(
        const agent_tool_call & call,
        const std::string & failure_code,
        common_tool_failure_class failure_class,
        bool retryable,
        const std::string & safe_summary,
        const std::string & raw_diagnostic,
        std::vector<common_runtime_resource_ref> resources) {
    agent_tool_result result;
    result.ok = false;
    result.tool_call_id = call.id;
    result.tool_name = call.name;
    result.failure_code = failure_code;
    result.failure_class = failure_class;
    result.retryable = retryable;
    result.safe_summary = safe_summary;
    result.raw_diagnostic = raw_diagnostic;
    result.content_summary = result.safe_summary;
    result.resource_refs = std::move(resources);
    result.content_json = make_agent_tool_failure_payload_json(
        result.failure_code.empty() ? "tool_call_rejected" : result.failure_code,
        result.safe_summary.empty() ? "The tool call was rejected by its contract or executor." : result.safe_summary,
        result.retryable,
        result.failure_class,
        result.resource_refs).dump();
    return result;
}

agent_tool_result make_agent_tool_json_success_result(
        const agent_tool_call & call,
        const std::string & output_json_or_text,
        const std::string & summary,
        std::vector<common_runtime_resource_ref> resources) {
    agent_tool_result result;
    result.ok = true;
    result.tool_call_id = call.id;
    result.tool_name = call.name;
    result.content_summary = summary;
    result.resource_refs = std::move(resources);
    result.content_json = make_agent_tool_success_payload_json(
        output_json_or_text,
        result.content_summary,
        result.resource_refs).dump();
    return result;
}

agent_tool_result make_agent_tool_structured_success_result(
        const agent_tool_call & call,
        const std::string & structured_content_json,
        const std::string & summary,
        std::vector<common_runtime_resource_ref> resources) {
    agent_tool_result result;
    result.ok = true;
    result.tool_call_id = call.id;
    result.tool_name = call.name;
    result.content_summary = summary;
    result.resource_refs = std::move(resources);
    result.content_json = make_agent_tool_structured_success_payload_json(
        structured_content_json,
        result.content_summary,
        result.resource_refs).dump();
    return result;
}

agent_tool_result make_agent_tool_text_success_result(
        const agent_tool_call & call,
        const std::string & text,
        const std::string & summary,
        std::vector<common_runtime_resource_ref> resources) {
    agent_tool_result result;
    result.ok = true;
    result.tool_call_id = call.id;
    result.tool_name = call.name;
    result.content_summary = summary;
    result.resource_refs = std::move(resources);
    result.content_json = make_agent_tool_text_success_payload_json(
        text,
        result.content_summary,
        result.resource_refs).dump();
    return result;
}
