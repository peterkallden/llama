#include "agent-openapi-provider.h"
#include "agent-openapi-result-projection.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <set>

namespace {

using json = nlohmann::ordered_json;

// An OpenAPI document can declare a very wide response object.  The full
// schema remains authoritative for host validation and result handling, but
// serializing it into every model tool catalog needlessly turns a simple
// operation choice into a provider-specific schema-reading task.  All
// OpenAPI calls are wrapped by the common tool result contract: `result`
// carries the bounded response and `resources` carries host-owned materialized
// representations when available.
constexpr const char * openapi_model_result_schema = R"({
    "type":"object",
    "properties":{
        "result":{"description":"Bounded provider response."},
        "resources":{"type":"array","description":"Host-owned materialized result resources.","items":{"type":"object"}}
    }
})";

std::string paging_description(const agent_openapi_operation & operation) {
    const auto & paging = operation.paging;
    if (paging.kind == agent_openapi_paging_kind::none) return {};

    std::string description = " pagination: host-classified ";
    description += agent_openapi_paging_kind_name(paging.kind);
    description += " operation; bound each collection request";
    if (!paging.page_size_parameter.empty()) {
        description += " with " + paging.page_size_parameter;
    }
    if (!paging.cursor_parameter.empty()) {
        description += "; continue with " + paging.cursor_parameter +
            " only when the response supplies a continuation value";
    } else if (!paging.page_parameter.empty()) {
        description += "; advance " + paging.page_parameter + " explicitly when another page is needed";
    }
    if (!paging.projection_parameter.empty()) {
        description += "; use " + paging.projection_parameter +
            " to request only fields needed for the task";
    }
    description += ".";
    return description;
}

} // namespace

class agent_openapi_tool_provider::client : public agent_mcp_tool_client {
public:
    client(const agent_openapi_catalog & catalog, agent_openapi_executor executor,
            agent_openapi_result_materializer materializer)
        : catalog(catalog), executor(std::move(executor)), materializer(std::move(materializer)) {}

    bool list_tools(const agent_tool_context &, std::vector<mcp_agent_tool_definition> & tools,
                    std::string & error) override {
        tools.clear();
        for (const auto & operation : catalog.operations) {
            mcp_agent_tool_definition definition;
            definition.provider_id = catalog.provider_id;
            definition.name = operation.operation_id;
            definition.description = operation.description.empty() ? operation.summary : operation.description;
            definition.description += paging_description(operation);
            for (const auto & relation : catalog.relations) {
                if (relation.collection_operation_id == operation.operation_id) {
                    definition.description += " workflow: " + catalog.prefix + "." +
                        relation.collection_operation_id + " -> choose/bind " +
                        relation.item_parameter + " -> " + catalog.prefix + "." +
                        relation.item_operation_id + ".";
                } else if (relation.item_operation_id == operation.operation_id) {
                    definition.description += " workflow input: " + catalog.prefix + "." +
                        relation.collection_operation_id + " produces " +
                        relation.item_parameter + ".";
                }
            }
            definition.input_schema_json = operation.input_schema_json;
            std::string model_schema_error;
            definition.model_input_schema_json =
                common_project_model_input_schema_required_parameters(
                    operation.input_schema_json,
                    operation.host_required_parameters,
                    model_schema_error);
            if (!model_schema_error.empty()) {
                error = "OpenAPI model-facing schema projection failed for " +
                    operation.operation_id + ": " + model_schema_error;
                return false;
            }
            definition.result_schema_json = operation.result_schema_json;
            definition.model_result_schema_json = openapi_model_result_schema;
            definition.read_only = operation.read_only;
            definition.requires_confirmation = operation.requires_confirmation;
            definition.uses_network = true;
            tools.push_back(std::move(definition));
        }
        return true;
    }

    bool call_tool(const agent_tool_context & context, const mcp_agent_tool_definition & tool,
                   const std::string & arguments_json, mcp_agent_tool_call_result & result,
                   std::string & error) override {
        return call_operation(context, tool, arguments_json, 0, result, error);
    }

    bool call_agent_tool(const agent_tool_context & context,
            const mcp_agent_tool_definition & tool, const agent_tool_call & call,
            const std::string & normalized_arguments_json,
            mcp_agent_tool_call_result & result, std::string & error) override {
        return call_operation(context, tool, normalized_arguments_json,
            call.workflow_page_limit, result, error);
    }

private:
    bool call_operation(const agent_tool_context & context,
            const mcp_agent_tool_definition & tool, const std::string & arguments_json,
            uint8_t requested_page_limit, mcp_agent_tool_call_result & result,
            std::string & error) {
        const auto it = std::find_if(catalog.operations.begin(), catalog.operations.end(),
            [&](const agent_openapi_operation & operation) { return operation.operation_id == tool.name; });
        if (it == catalog.operations.end()) {
            error = "OpenAPI operation is not in the filtered catalog";
            return false;
        }
        agent_openapi_execution_result execution;
        auto arguments = json::parse(arguments_json, nullptr, false);
        if (!executor || !arguments.is_object() ||
                !executor(context, *it, arguments_json, execution, error)) return false;
        if (execution.ok && requested_page_limit > 1 &&
                it->paging.kind == agent_openapi_paging_kind::cursor &&
                !it->paging.cursor_parameter.empty()) {
            execution = collect_cursor_pages(context, *it, arguments,
                std::min<uint8_t>(requested_page_limit, 5), std::move(execution));
        }
        if (materializer && !materializer(context, *it, arguments_json, execution, error)) return false;
        execution.structured_content_json = agent_openapi_project_model_result(
            execution.structured_content_json, catalog.prefix == "openalex");
        if (requested_page_limit > 1 && !execution.resource_refs.empty()) {
            const auto body = json::parse(execution.structured_content_json, nullptr, false);
            if (body.is_object() && body.contains("pagination") &&
                    body["pagination"].is_object() && body.contains("results") &&
                    body["results"].is_array()) {
                json candidates = json::array();
                const size_t preview_count = std::min<size_t>(10, body["results"].size());
                for (size_t index = 0; index < preview_count; ++index) {
                    const auto & item = body["results"][index];
                    json candidate = json::object();
                    for (const char * key : {"id", "display_name", "title", "publication_year", "doi",
                            "full_text_url", "full_text_url_source"}) {
                        if (item.is_object() && item.contains(key)) candidate[key] = item[key];
                    }
                    if (item.is_object() && item.contains("abstract_text") &&
                            item["abstract_text"].is_string()) {
                        const auto & abstract = item["abstract_text"].get_ref<const std::string &>();
                        size_t preview_bytes = std::min<size_t>(1200, abstract.size());
                        if (abstract.size() > preview_bytes) {
                            while (preview_bytes > 0 &&
                                    (static_cast<unsigned char>(abstract[preview_bytes]) & 0xc0) == 0x80) {
                                --preview_bytes;
                            }
                        }
                        candidate["abstract_text"] = abstract.substr(0, preview_bytes);
                        if (abstract.size() > preview_bytes) candidate["abstract_truncated"] = true;
                    }
                    candidates.push_back(std::move(candidate));
                }
                execution.structured_content_json = json{
                    {"pagination", body["pagination"]},
                    {"result_count", body["results"].size()},
                    {"candidates", std::move(candidates)},
                    {"resource_ref", execution.resource_refs.front().uri},
                }.dump();
            }
        }
        result.ok = execution.ok;
        result.structured_content_json = execution.structured_content_json;
        result.text_content = execution.text_content;
        result.resource_refs = execution.resource_refs;
        result.dataset_refs = execution.dataset_refs;
        result.failure_code = execution.failure_code;
        result.failure_class = execution.failure_class;
        result.retryable = execution.retryable;
        result.safe_summary = execution.safe_summary;
        result.raw_diagnostic = execution.raw_diagnostic;
        result.http = execution.http;
        if (!execution.resource_refs.empty()) result.http.resource_ref = execution.resource_refs.front().uri;
        return true;
    }

    agent_openapi_execution_result collect_cursor_pages(
            const agent_tool_context & context,
            const agent_openapi_operation & operation,
            json arguments,
            uint8_t page_limit,
            agent_openapi_execution_result first) {
        auto merged = json::parse(first.structured_content_json, nullptr, false);
        if (!merged.is_object() || !merged.contains("results") || !merged["results"].is_array()) {
            first.structured_content_json = partial_json(std::move(merged), 1, page_limit,
                "cursor_response_missing_results");
            return first;
        }
        size_t item_count = merged["results"].size();
        uint8_t pages = 1;
        std::set<std::string> seen_cursors;
        json page_provenance = json::array();
        page_provenance.push_back({{"page", 1},
            {"cursor", arguments.contains(operation.paging.cursor_parameter)
                ? arguments[operation.paging.cursor_parameter] : json(nullptr)},
            {"http_status", first.http.status}, {"response_bytes", first.http.response_bytes}});
        bool complete = false;
        std::string stop_reason;
        for (;;) {
            const auto cursor = response_cursor(merged);
            if (!cursor) { stop_reason = "cursor_unavailable"; break; }
            if (cursor->empty()) { complete = true; break; }
            if (pages >= page_limit) { stop_reason = "page_limit"; break; }
            if (!seen_cursors.insert(*cursor).second) { stop_reason = "repeated_cursor"; break; }
            arguments[operation.paging.cursor_parameter] = *cursor;
            agent_openapi_execution_result page;
            std::string page_error;
            if (!executor(context, operation, arguments.dump(), page, page_error) || !page.ok) {
                stop_reason = "page_request_failed";
                first.http = page.http;
                page_provenance.push_back({{"page", static_cast<unsigned>(pages + 1)},
                    {"cursor", *cursor}, {"http_status", page.http.status},
                    {"failure_code", page.failure_code}, {"retryable", page.retryable}});
                const auto failure = json{{"http_status", page.http.status},
                    {"failure_code", page.failure_code}, {"retryable", page.retryable},
                    {"summary", page.safe_summary}};
                merged["pagination"] = {{"pages_fetched", pages}, {"page_limit", page_limit},
                    {"complete", false}, {"stop_reason", stop_reason},
                    {"page_provenance", page_provenance}, {"page_error", failure}};
                first.structured_content_json = merged.dump();
                first.raw_diagnostic = page.raw_diagnostic.empty() ? page_error : page.raw_diagnostic;
                // Keep already fetched evidence usable, but explicitly mark it
                // partial and retain the failed HTTP status in the result.
                first.ok = true;
                return first;
            }
            auto page_json = json::parse(page.structured_content_json, nullptr, false);
            if (!page_json.is_object() || !page_json.contains("results") ||
                    !page_json["results"].is_array()) {
                stop_reason = "cursor_response_missing_results";
                break;
            }
            bool result_limit_reached = false;
            for (auto & item : page_json["results"]) {
                if (item_count >= 10000) { result_limit_reached = true; break; }
                merged["results"].push_back(std::move(item));
                ++item_count;
            }
            if (page_json.contains("meta")) merged["meta"] = page_json["meta"];
            first.http = page.http;
            ++pages;
            page_provenance.push_back({{"page", pages}, {"cursor", *cursor},
                {"http_status", page.http.status}, {"response_bytes", page.http.response_bytes}});
            if (result_limit_reached) { stop_reason = "result_limit"; break; }
        }
        if (stop_reason.empty()) stop_reason = complete ? "exhausted" : "cursor_unavailable";
        merged["pagination"] = {{"pages_fetched", pages}, {"page_limit", page_limit},
            {"complete", complete}, {"stop_reason", stop_reason},
            {"page_provenance", page_provenance}};
        first.structured_content_json = merged.dump();
        first.ok = true;
        return first;
    }

    static std::optional<std::string> response_cursor(const json & value) {
        const json * cursor = nullptr;
        if (value.contains("meta") && value["meta"].is_object() &&
                value["meta"].contains("next_cursor")) cursor = &value["meta"]["next_cursor"];
        else if (value.contains("next_cursor")) cursor = &value["next_cursor"];
        if (cursor == nullptr) return std::nullopt;
        if (cursor->is_null()) return std::string{};
        if (!cursor->is_string()) return std::nullopt;
        return cursor->get<std::string>();
    }

    static std::string partial_json(json value, uint8_t pages, uint8_t limit,
            const std::string & reason) {
        if (!value.is_object()) value = json::object();
        value["pagination"] = {{"pages_fetched", pages}, {"page_limit", limit},
            {"complete", false}, {"stop_reason", reason}};
        return value.dump();
    }

    const agent_openapi_catalog & catalog;
    agent_openapi_executor executor;
    agent_openapi_result_materializer materializer;
};

agent_openapi_tool_provider::agent_openapi_tool_provider(
        agent_openapi_catalog catalog, agent_openapi_executor executor,
        agent_openapi_result_materializer materializer)
    : catalog(std::move(catalog))
    , executor(std::move(executor))
    , materializer(std::move(materializer))
    , client_impl(std::make_unique<client>(this->catalog, this->executor, this->materializer))
    , delegate(std::make_unique<mcp_agent_tool_provider>(
        this->catalog.provider_id,
        *client_impl,
        this->catalog.prefix + ".",
        [this](const common_plan_state & plan, std::string & error) {
            for (const auto & relation : this->catalog.relations) {
                const std::string item_name = this->catalog.prefix + "." + relation.item_operation_id;
                const std::string collection_name = this->catalog.prefix + "." + relation.collection_operation_id;
                for (size_t index = 0; index < plan.steps.size(); ++index) {
                    const auto & step = plan.steps[index];
                    if (!step.tool_call || step.tool_call->name != item_name ||
                            step.tool_call->arguments_json.find("\"$") == std::string::npos) continue;
                    bool producer_before_item = false;
                    for (size_t previous = 0; previous < index; ++previous) {
                        if (plan.steps[previous].tool_call &&
                                plan.steps[previous].tool_call->name == collection_name) {
                            producer_before_item = true;
                            break;
                        }
                    }
                    if (!producer_before_item) {
                        error = "workflow.required_producer_missing: " + item_name +
                            " requires " + collection_name + " before dynamic " +
                            relation.item_parameter + " binding";
                        return false;
                    }
                }
            }
            error.clear();
            return true;
        },
        "openapi")) {}

agent_openapi_tool_provider::~agent_openapi_tool_provider() = default;

std::unique_ptr<agent_tool_view> agent_openapi_tool_provider::resolve_tools(
        const agent_tool_context & context, std::string & error) {
    return delegate->resolve_tools(context, error);
}
