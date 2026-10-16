#include "agent-mcp-protocol.h"

#include <utility>

using json = nlohmann::ordered_json;

const char * common_mcp_default_protocol_version() {
    return "2025-06-18";
}

bool common_mcp_is_supported_protocol_version(const std::string & version) {
    return version == "2024-11-05" || version == "2025-06-18" || version == "2025-11-25";
}

std::string common_mcp_negotiate_protocol_version(const std::string & requested) {
    if (requested.empty()) return common_mcp_default_protocol_version();
    return common_mcp_is_supported_protocol_version(requested) ? requested : std::string();
}

bool parse_mcp_stdio_message(std::string & buffer, json & message, std::string & error) {
    message = json();
    error.clear();
    const size_t end = buffer.find('\n');
    if ((end == std::string::npos ? buffer.size() : end) > common_mcp_max_stdio_message_bytes) {
        error = "MCP stdio message exceeds size limit";
        return true;
    }
    if (end == std::string::npos) {
        return false;
    }
    const auto parsed = json::parse(buffer.begin(), buffer.begin() + end, nullptr, false);
    buffer.erase(0, end + 1);
    if (parsed.is_discarded() || !parsed.is_object()) {
        error = "MCP stdio message contained invalid JSON-RPC payload";
        return true;
    }
    message = parsed;
    return true;
}

bool write_mcp_stdio_message(FILE * stream, const json & message, std::string & error) {
    error.clear();
    const std::string line = message.dump() + '\n';
    if (std::fwrite(line.data(), 1, line.size(), stream) != line.size()) {
        error = "failed to write MCP JSON-RPC message";
        return false;
    }
    if (std::fflush(stream) != 0) {
        error = "failed to flush MCP JSON-RPC message";
        return false;
    }
    return true;
}

namespace {

bool parse_resource_ref_from_json(
        const json & item,
        common_runtime_resource_ref & resource) {
    if (!item.is_object()) {
        return false;
    }
    const std::string uri = item.value("uri", "");
    if (uri.empty()) {
        return false;
    }

    resource = {};
    resource.uri = uri;
    resource.name = item.value("name", item.value("title", std::string()));
    resource.description = item.value("description", "");
    resource.mime_type = item.value("mimeType", item.value("mime_type", std::string()));
    resource.size_bytes = item.value("sizeBytes", item.value("size_bytes", size_t(0)));
    resource.scope = common_runtime_resource_scope::turn;
    if (item.contains("lineage") && item["lineage"].is_object()) {
        const auto & lineage = item["lineage"];
        resource.lineage.parent_uri = lineage.value("parent_uri", "");
        resource.lineage.chunk_index = lineage.value("chunk_index", size_t(0));
        resource.lineage.chunk_count = lineage.value("chunk_count", size_t(0));
        resource.lineage.byte_offset = lineage.value("byte_offset", size_t(0));
        resource.lineage.byte_length = lineage.value("byte_length", size_t(0));
        resource.lineage.overlap_bytes = lineage.value("overlap_bytes", size_t(0));
        resource.lineage.derivation = lineage.value("derivation", "");
    }
    if (item.contains("metadata") && item["metadata"].is_object()) {
        const auto & metadata = item["metadata"];
        resource.metadata.purpose = metadata.value("purpose", "");
        resource.metadata.content_summary = metadata.value("content_summary", "");
        resource.metadata.usage_hint = metadata.value("usage_hint", "");
        resource.metadata.limitations = metadata.value("limitations", "");
        if (metadata.contains("keywords") && metadata["keywords"].is_array()) {
            for (const auto & keyword : metadata["keywords"]) {
                if (keyword.is_string()) resource.metadata.keywords.push_back(keyword.get<std::string>());
            }
        }
        if (metadata.contains("entities") && metadata["entities"].is_array()) {
            for (const auto & entity : metadata["entities"]) {
                if (entity.is_string()) resource.metadata.entities.push_back(entity.get<std::string>());
            }
        }
    }
    return true;
}

std::string join_text_content(const json & content) {
    if (!content.is_array()) {
        return {};
    }

    std::string joined;
    for (const auto & item : content) {
        if (!item.is_object()) {
            continue;
        }
        if (item.value("type", "") != "text") {
            continue;
        }
        const std::string text = item.value("text", "");
        if (text.empty()) {
            continue;
        }
        if (!joined.empty()) {
            joined += "\n";
        }
        joined += text;
    }
    return joined;
}

std::vector<common_runtime_resource_ref> extract_resource_links(const json & content) {
    std::vector<common_runtime_resource_ref> resources;
    if (!content.is_array()) {
        return resources;
    }

    for (const auto & item : content) {
        if (!item.is_object()) {
            continue;
        }
        if (item.value("type", "") != "resource_link") {
            continue;
        }
        const std::string uri = item.value("uri", "");
        if (uri.empty()) {
            continue;
        }

        common_runtime_resource_ref resource;
        if (parse_resource_ref_from_json(item, resource)) {
            resources.push_back(std::move(resource));
        }
    }

    return resources;
}

common_tool_failure_class parse_mcp_failure_class(const std::string & value) {
    if (value == "validation") return common_tool_failure_class::validation;
    if (value == "policy")     return common_tool_failure_class::policy;
    if (value == "not_found")  return common_tool_failure_class::not_found;
    if (value == "timeout")    return common_tool_failure_class::timeout;
    if (value == "network")    return common_tool_failure_class::network;
    if (value == "limit")      return common_tool_failure_class::limit;
    return common_tool_failure_class::execution;
}

} // namespace

json make_mcp_jsonrpc_request(
        int id,
        const std::string & method,
        const json & params) {
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method},
        {"params", params},
    };
}

json make_mcp_jsonrpc_notification(
        const std::string & method,
        const json & params) {
    return {
        {"jsonrpc", "2.0"},
        {"method", method},
        {"params", params},
    };
}

json make_mcp_initialize_params() {
    json capabilities = json::object();
    capabilities["extensions"]["io.modelcontextprotocol/ui"] = {
        {"mimeTypes", json::array({"text/html;profile=mcp-app"})},
    };
    return {
        {"protocolVersion", common_mcp_default_protocol_version()},
        {"capabilities", std::move(capabilities)},
        {"clientInfo", {
            {"name", "llama-agent"},
            {"version", "0.1"},
        }},
    };
}

json make_mcp_tools_call_params(
        const std::string & name,
        const json & arguments) {
    return {
        {"name", name},
        {"arguments", arguments},
    };
}

json make_mcp_resources_read_params(
        const std::string & uri) {
    return {
        {"uri", uri},
    };
}

bool parse_mcp_tool_definition(
        const std::string & provider_id,
        const json & item,
        mcp_agent_tool_definition & definition,
        std::string & error) {
    definition = {};
    error.clear();

    if (!item.is_object()) {
        error = "MCP tool definition must be an object";
        return false;
    }

    definition.provider_id = provider_id;
    definition.name = item.value("name", "");
    definition.description = item.value("description", "");
    if (item.contains("inputSchema")) {
        definition.input_schema_json = item["inputSchema"].dump();
    }
    if (item.contains("outputSchema") && item["outputSchema"].is_object()) {
        definition.result_schema_json = item["outputSchema"].dump();
        definition.model_result_schema_json = definition.result_schema_json;
    }
    if (item.contains("_meta") && item["_meta"].is_object() &&
            item["_meta"].contains("ui") && item["_meta"]["ui"].is_object()) {
        auto ui = item["_meta"]["ui"];
        definition.ui_metadata_json = ui.dump();
        definition.ui_resource_uri = ui.value("resourceUri", "");
        if (ui.contains("visibility") && ui["visibility"].is_array()) {
            for (const auto & visibility : ui["visibility"]) {
                if (visibility.is_string()) {
                    const auto value = visibility.get<std::string>();
                    if (value == "model" || value == "app") definition.ui_visibility.push_back(value);
                }
            }
        } else {
            // MCP Apps visibility defaults to both the agent and the app.
            definition.ui_visibility = {"model", "app"};
            ui["visibility"] = definition.ui_visibility;
        }
        definition.ui_metadata_json = ui.dump();
    } else {
        // Ordinary MCP tools have no app surface, but remain available to the
        // model. The MCP Apps default (model + app) applies only when a UI
        // resource is actually declared.
        definition.ui_visibility = {"model"};
    }

    if (item.contains("annotations") && item["annotations"].is_object()) {
        const auto & annotations = item["annotations"];
        if (annotations.contains("readOnlyHint") && annotations["readOnlyHint"].is_boolean()) {
            definition.read_only_declared = true;
            definition.read_only_hint = annotations["readOnlyHint"].get<bool>();
            definition.read_only = definition.read_only_hint;
        }
    }
    if (item.contains("hostPolicy") && item["hostPolicy"].is_object()) {
        const auto & policy = item["hostPolicy"];
        definition.requires_confirmation = policy.value("requiresConfirmation", definition.requires_confirmation);
        definition.uses_network = policy.value("usesNetwork", definition.uses_network);
        definition.writes_memory = policy.value("writesMemory", definition.writes_memory);
        definition.writes_plan = policy.value("writesPlan", definition.writes_plan);
    }

    if (definition.name.empty()) {
        error = "MCP tool definition is missing a name";
        return false;
    }

    return true;
}

bool parse_mcp_tool_call_result(
        const json & rpc_result,
        mcp_agent_tool_call_result & result,
        std::string & error) {
    result = {};
    error.clear();

    if (!rpc_result.is_object()) {
        error = "MCP tools/call response did not contain a result object";
        return false;
    }

    result.ok = !rpc_result.value("isError", false);
    const auto content = rpc_result.value("content", json::array());
    result.text_content = join_text_content(content);
    result.resource_refs = extract_resource_links(content);
    if (rpc_result.contains("structuredContent")) {
        result.structured_content_json = rpc_result["structuredContent"].dump();
    }
    if (!result.ok) {
        result.failure_code = "mcp.tool_error";
        result.failure_class = common_tool_failure_class::execution;
        result.retryable = false;
        result.safe_summary = result.text_content.empty()
            ? "The MCP server reported a tool error."
            : result.text_content;

        if (rpc_result.contains("errorInfo") && rpc_result["errorInfo"].is_object()) {
            const auto & error_info = rpc_result["errorInfo"];
            result.failure_code = error_info.value("code", result.failure_code);
            result.failure_class = parse_mcp_failure_class(
                error_info.value("class", std::string()));
            result.retryable = error_info.value("retryable", result.retryable);
            result.safe_summary = error_info.value("safeSummary", result.safe_summary);
        }
        result.raw_diagnostic = result.safe_summary;
    }

    return true;
}

bool parse_mcp_resources_list_result(
        const json & rpc_result,
        std::vector<mcp_agent_resource_definition> & resources,
        std::string & error) {
    resources.clear();
    error.clear();

    if (!rpc_result.is_object() || !rpc_result.contains("resources") || !rpc_result["resources"].is_array()) {
        error = "MCP resources/list response did not contain a resources array";
        return false;
    }

    for (const auto & item : rpc_result["resources"]) {
        mcp_agent_resource_definition definition;
        if (!parse_resource_ref_from_json(item, definition.resource)) {
            error = "MCP resource entry is invalid";
            return false;
        }
        resources.push_back(std::move(definition));
    }

    return true;
}

bool parse_mcp_resource_read_result(
        const json & rpc_result,
        mcp_agent_resource_read_result & result,
        std::string & error) {
    result = {};
    error.clear();

    if (!rpc_result.is_object() || !rpc_result.contains("contents") || !rpc_result["contents"].is_array() ||
            rpc_result["contents"].empty() || !rpc_result["contents"][0].is_object()) {
        error = "MCP resources/read response did not contain a contents array";
        return false;
    }

    const auto & first = rpc_result["contents"][0];
    if (!parse_resource_ref_from_json(first, result.resource)) {
        error = "MCP resource read content is invalid";
        return false;
    }
    if (first.contains("text") && first["text"].is_string()) {
        result.text_content = first["text"].get<std::string>();
    } else if (first.contains("blob") && first["blob"].is_string()) {
        result.blob_content_base64 = first["blob"].get<std::string>();
    } else {
        error = "MCP resource read content did not contain text or blob";
        return false;
    }
    if (first.contains("_meta") && first["_meta"].is_object()) {
        result.ui_metadata_json = first["_meta"].dump();
    }
    return true;
}
