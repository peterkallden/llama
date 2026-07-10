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
