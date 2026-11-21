#include "agent-mcp-resource-materializer.h"
#include "base64.hpp"
#include "hash/hash.h"
#include <algorithm>
#include <map>
#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::ordered_json;
constexpr size_t max_total_bytes = 10U * 1024U * 1024U;
constexpr size_t max_links = 16;
constexpr size_t page_bytes = 8192;
constexpr size_t max_page_calls = 64;

bool stopped(const agent_tool_context & context, std::string & error) {
    if (!context.execution_control.should_stop()) return false;
    error = "MCP resource acquisition was cancelled or exceeded its deadline";
    return true;
}

void rewrite_exact(json & value, const std::map<std::string, std::string> & replacements) {
    if (value.is_string()) {
        const auto found = replacements.find(value.get<std::string>());
        if (found != replacements.end()) value = found->second;
    } else if (value.is_array() || value.is_object()) {
        for (auto & item : value) rewrite_exact(item, replacements);
    }
}

std::string safe_source_ref(std::string uri) {
    // URLs may contain credentials or transient query parameters. Preserve
    // opaque non-URL identities, but never persist URL userinfo or queries.
    const auto scheme = uri.find("://");
    if (scheme != std::string::npos && (uri.rfind("http://", 0) == 0 || uri.rfind("https://", 0) == 0)) {
        const auto end = uri.find_first_of("/?#", scheme + 3);
        const auto at = uri.find('@', scheme + 3);
        if (at != std::string::npos && (end == std::string::npos || at < end)) uri.erase(scheme + 3, at - scheme - 2);
        const auto query = uri.find_first_of("?#");
        if (query != std::string::npos) uri.resize(query);
    }
    if (uri.size() > 512) uri.resize(512);
    return uri;
}

bool read_paged(agent_mcp_tool_client & client, const agent_tool_context & context,
        const mcp_agent_tool_definition & reader, const common_runtime_resource_ref & ref,
        size_t limit, size_t & page_calls, std::string & bytes, std::string & error) {
    std::string expected_hash;
    size_t expected_size = 0;
    size_t offset = 0;
    while (page_calls < max_page_calls) {
        if (stopped(context, error)) return false;
        ++page_calls;
        mcp_agent_tool_call_result page;
        const json args{{"uri", ref.uri}, {"representation", "bytes"}, {"offset", offset}, {"max_bytes", page_bytes}};
        if (!client.call_tool(context, reader, args.dump(), page, error) || !page.ok) {
            error = "MCP resource page could not be read"; return false;
        }
        const auto payload = json::parse(page.structured_content_json.empty() ? page.text_content : page.structured_content_json, nullptr, false);
        if (!payload.is_object() || !payload.contains("resource") || !payload["resource"].is_object() ||
                !payload.contains("range") || !payload["range"].is_object() ||
                !payload.contains("content") || !payload["content"].is_string() ||
                payload.value("content_encoding", std::string()) != "base64") {
            error = "MCP resource page has an unsupported or incomplete contract"; return false;
        }
        const auto & descriptor = payload["resource"];
        const auto & range = payload["range"];
        if (!descriptor.contains("uri") || !descriptor["uri"].is_string() ||
                descriptor["uri"].get<std::string>() != ref.uri ||
                !descriptor.contains("sha256") || !descriptor["sha256"].is_string() ||
                !descriptor.contains("size_bytes") || !descriptor["size_bytes"].is_number_unsigned() ||
                !range.contains("offset") || !range["offset"].is_number_unsigned() ||
                !range.contains("bytes_read") || !range["bytes_read"].is_number_unsigned() ||
                !range.contains("resource_complete") || !range["resource_complete"].is_boolean()) {
            error = "MCP resource page lacks stable identity or continuation metadata"; return false;
        }
        const auto hash = descriptor["sha256"].get<std::string>();
        const auto size = descriptor["size_bytes"].get<size_t>();
        if (hash.empty() || size > limit || range["offset"].get<size_t>() != offset ||
                (!expected_hash.empty() && (hash != expected_hash || size != expected_size))) {
            error = "MCP resource changed during paging or exceeded the acquisition limit"; return false;
        }
        if (expected_hash.empty()) { expected_hash = hash; expected_size = size; }
        std::string page_bytes_value;
        if (payload["content"].get_ref<const std::string &>().size() > ((page_bytes + 2) / 3) * 4) {
            error = "MCP resource page exceeds the page limit"; return false;
        }
        try { page_bytes_value = base64::decode(payload["content"].get<std::string>()); }
        catch (...) { error = "MCP resource page has invalid base64"; return false; }
        if (page_bytes_value.size() != range["bytes_read"].get<size_t>() ||
                page_bytes_value.size() > page_bytes || offset > size || page_bytes_value.size() > size - offset) {
            error = "MCP resource page has inconsistent byte counts"; return false;
        }
        bytes += page_bytes_value;
        offset += page_bytes_value.size();
        if (range["resource_complete"].get<bool>()) {
            if (offset != size || hash_sha256_hex(bytes.data(), bytes.size()) != expected_hash ||
                    (ref.size_bytes != 0 && ref.size_bytes != size)) {
                error = "MCP resource content does not match its reported identity"; return false;
            }
            return true;
        }
        if (page_bytes_value.empty() || !range.contains("next_offset") || !range["next_offset"].is_number_unsigned() ||
                range["next_offset"].get<size_t>() != offset) {
            error = "MCP resource continuation did not advance"; return false;
        }
    }
    error = "MCP resource exceeded its page budget";
    return false;
}
}

bool agent_materialize_mcp_resources(agent_mcp_tool_client & client,
        agent_resource_store & store, const agent_tool_context & context,
        const std::string & provider_id, const std::string & source_tool,
        mcp_agent_tool_call_result & result, std::string & error) {
    if (!result.ok || result.resource_refs.empty()) { error.clear(); return true; }
    if (context.scope.namespace_id.empty() || context.scope.session_id.empty() || result.resource_refs.size() > max_links) {
        error = "MCP resource links exceed host limits or lack session scope"; return false;
    }
    std::vector<mcp_agent_tool_definition> definitions;
    if (!client.list_tools(context, definitions, error)) { error = "MCP resource reader could not be discovered"; return false; }
    const auto reader = std::find_if(definitions.begin(), definitions.end(), [](const auto & item) {
        return item.name == "resource.read" && item.read_only && !item.requires_confirmation && !item.writes_memory && !item.writes_plan;
    });
    const bool reader_allowed = context.allowed_exposed_tool_names.empty() ||
        std::find(context.allowed_exposed_tool_names.begin(), context.allowed_exposed_tool_names.end(), "remote_resource.read") != context.allowed_exposed_tool_names.end();
    std::map<std::string, std::string> replacements;
    std::vector<common_runtime_resource_ref> local_refs;
    size_t total = 0;
    size_t page_calls = 0;
    for (const auto & ref : result.resource_refs) {
        if (ref.uri.empty() || ref.uri.size() > 512 || stopped(context, error)) return false;
        const auto existing = replacements.find(ref.uri);
        if (existing != replacements.end()) {
            const auto previous = std::find_if(local_refs.begin(), local_refs.end(), [&](const auto & value) { return value.uri == existing->second; });
            if (previous != local_refs.end()) local_refs.push_back(*previous);
            continue;
        }
        if (ref.size_bytes > max_total_bytes - total) { error = "MCP resources exceed the host acquisition limit"; return false; }
        std::string bytes;
        bool complete = false;
        if (reader != definitions.end() && reader_allowed) {
            complete = read_paged(client, context, *reader, ref, max_total_bytes - total, page_calls, bytes, error);
        } else {
            mcp_agent_resource_read_result read;
            if (client.read_resource(ref.uri, read, error) && read.resource.uri == ref.uri) {
                try { bytes = read.blob_content_base64.empty() ? read.text_content : base64::decode(read.blob_content_base64); }
                catch (...) { error = "MCP resource contains invalid base64"; return false; }
                // A zero size is ambiguous in the MCP link contract: require a
                // nonzero length from either descriptor before claiming completeness.
                const size_t declared = read.resource.size_bytes ? read.resource.size_bytes : ref.size_bytes;
                complete = declared > 0 && bytes.size() == declared && bytes.size() <= max_total_bytes - total &&
                    (!ref.size_bytes || ref.size_bytes == declared);
            }
            if (!complete) error = "MCP resource completeness could not be verified";
        }
        if (!complete || stopped(context, error)) return false;
        total += bytes.size();
        agent_resource_put_request put;
        put.name = ref.name.empty() ? "mcp-resource" : ref.name;
        put.description = ref.description;
        put.mime_type = ref.mime_type.empty() ? "application/octet-stream" : ref.mime_type;
        put.bytes = std::move(bytes); put.bytes_are_authoritative = true;
        put.scope = common_runtime_resource_scope::session;
        put.namespace_id = context.scope.namespace_id; put.session_id = context.scope.session_id;
        put.project_id = context.scope.project_id; put.turn_id = context.scope.turn_id;
        put.source_provider = provider_id; put.source_tool = source_tool;
        put.metadata = ref.metadata; put.metadata.source_provider = provider_id;
        put.metadata.source_operation = source_tool;
        put.metadata.source_request_json = json{{"resource_uri", safe_source_ref(ref.uri)}}.dump();
        put.metadata.content_hash = "sha256:" + hash_sha256_hex(put.bytes.data(), put.bytes.size());
        put.lineage.parent_uri = safe_source_ref(ref.uri); put.lineage.chunk_count = 1;
        put.lineage.byte_length = put.bytes.size(); put.lineage.derivation = "mcp-resource-materialization:v1";
        agent_resource_descriptor local;
        if (!store.put_bytes(put, local, error)) { error = "MCP resource could not be saved locally"; return false; }
        replacements.emplace(ref.uri, local.uri);
        local_refs.push_back(local);
    }
    if (!result.structured_content_json.empty()) {
        auto structured = json::parse(result.structured_content_json, nullptr, false);
        if (structured.is_discarded()) { error = "MCP structured result could not be updated safely"; return false; }
        rewrite_exact(structured, replacements);
        result.structured_content_json = structured.dump();
    }
    result.resource_refs = std::move(local_refs);
    error.clear();
    return true;
}
