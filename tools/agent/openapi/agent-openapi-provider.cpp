#include "agent-openapi-provider.h"
#include "../tooling/agent-result-view.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <set>

namespace {

using json = nlohmann::ordered_json;

// Plan observations and JSON-pointer bindings use the provider result body as
// their root. Keep the model schema small, but describe host-owned fields that
// are guaranteed by a declared collection relation. The common `{ok, result}`
// execution envelope is added later by the tool runtime.
std::string openapi_model_result_schema(
        const agent_openapi_catalog & catalog,
        const agent_openapi_operation & operation) {
    json schema = {
        {"type", "object"},
        {"description", "Bounded response body returned by this provider operation."},
        {"additionalProperties", true},
    };
    if (!operation.result_view_json.empty()) {
        const auto view = json::parse(operation.result_view_json);
        json fields = json::object();
        for (const auto & field : view["fields"].items()) fields[field.key()] = json::object();
        if (view.contains("derived_fields")) {
            for (const auto & field : view["derived_fields"].items()) {
                fields[field.key()] = {{"type", "string"}};
                if (field.value()["transform"] == "first_nonempty_url") fields[field.key() + "_source"] = {{"type", "string"}};
                if (field.value()["transform"] == "inverted_index_to_text") fields[field.key() + "_truncated"] = {{"type", "boolean"}};
            }
        }
        schema["properties"] = json::object();
        if (view["kind"] == "object") schema["properties"] = std::move(fields);
        else {
            // Describe the output at the configured pointer, including nested envelopes.
            auto * properties = &schema["properties"];
            auto pointer = view["items_pointer"].get<std::string>();
            pointer = view.value("output_pointer", pointer.empty() ? std::string("/items") : pointer);
            json::json_pointer parsed(pointer);
            (void) parsed;
            size_t begin = 1;
            while (begin <= pointer.size()) {
                const size_t end = pointer.find('/', begin);
                auto key = pointer.substr(begin, end == std::string::npos ? end : end - begin);
                for (size_t i = 0; i + 1 < key.size(); ++i) if (key[i] == '~') { key.replace(i, 2, key[i + 1] == '1' ? "/" : "~"); }
                if (end == std::string::npos) {
                    (*properties)[key] = {{"type", "array"}, {"items", {{"type", "object"}, {"properties", fields}}}};
                    break;
                }
                (*properties)[key] = {{"type", "object"}, {"properties", json::object()}};
                properties = &(*properties)[key]["properties"];
                begin = end + 1;
            }
        }
        schema["properties"]["resource_ref"] = {{"type", "string"}};
        schema["properties"]["pagination"] = {{"type", "object"}};
        schema["properties"]["result_count"] = {{"type", "integer"}};
        schema["properties"]["preview_omitted_items"] = {{"type", "integer"}};
        schema["properties"]["view_truncated"] = {{"type", "boolean"}};
        schema["properties"]["view_invalid_fields"] = {{"type", "integer"}};
    }
    const auto relation = std::find_if(catalog.relations.begin(), catalog.relations.end(),
        [&](const auto & item) { return item.collection_operation_id == operation.operation_id; });
    if (relation != catalog.relations.end()) {
        schema["properties"]["host_item_parameter"] = {{"type", "string"}};
        schema["properties"]["host_item_candidates"] = {
                {"type", "array"},
                {"description", "Host-validated item identities for a subsequent lookup."},
                {"items", {
                    {"type", "object"},
                    {"properties", {{relation->item_parameter, {
                        {"type", "string"},
                        {"description", "Identity accepted by the related item lookup."},
                    }}}},
                }},
            };
        schema["properties"]["host_item_candidates_truncated"] = {{"type", "boolean"}};
    }
    return schema.dump();
}

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

void annotate_pagination(json & response, const json & facts) {
    auto & pagination = response["pagination"];
    if (!pagination.is_object()) pagination = json::object();
    for (const auto & fact : facts.items()) pagination[fact.key()] = fact.value();
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
            definition.additional_families = catalog.additional_families;
            definition.name = operation.operation_id;
            definition.description = operation.description.empty() ? operation.summary : operation.description;
            definition.description += paging_description(operation);
            if (!operation.host_required_any_parameters.empty()) {
                definition.description += " At least one of these parameters is required: ";
                for (size_t i = 0; i < operation.host_required_any_parameters.size(); ++i) {
                    if (i) definition.description += ", ";
                    definition.description += operation.host_required_any_parameters[i];
                }
                definition.description += ".";
            }
            for (const auto & relation : catalog.relations) {
                if (relation.collection_operation_id == operation.operation_id) {
                    definition.collection_item_parameter = relation.item_parameter;
                    definition.collection_items_pointer = "/host_item_candidates";
                    definition.description += " workflow: " + catalog.prefix + "." +
                        relation.collection_operation_id + " -> choose/bind " +
                        relation.item_parameter + " -> " + catalog.prefix + "." +
                        relation.item_operation_id +
                        ". Use host_item_candidates for validated lookup identities.";
                } else if (relation.item_operation_id == operation.operation_id) {
                    definition.item_lookup_parameter = relation.item_parameter;
                    definition.description += " workflow input: " + catalog.prefix + "." +
                        relation.collection_operation_id + " produces " +
                        relation.item_parameter + ".";
                }
            }
            definition.input_schema_json = operation.input_schema_json;
            std::string model_schema_error;
            definition.model_input_schema_json =
                common_project_model_input_schema_parameter_requirements(
                    operation.input_schema_json,
                    operation.host_required_parameters,
                    operation.host_required_any_parameters,
                    model_schema_error);
            if (!model_schema_error.empty()) {
                error = "OpenAPI model-facing schema projection failed for " +
                    operation.operation_id + ": " + model_schema_error;
                return false;
            }
            definition.model_result_schema_json = openapi_model_result_schema(catalog, operation);
            // Dataflow bindings consume the projected body, while dataset
            // classification continues to use the upstream response schema.
            definition.result_schema_json = operation.result_view_json.empty()
                ? operation.result_schema_json : definition.model_result_schema_json;
            definition.read_only = operation.read_only;
            // OpenAPI operations are model tools by default. Empty visibility
            // remains an explicit provider choice to hide a tool.
            definition.ui_visibility = {"model"};
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
        if (!executor || !arguments.is_object()) return false;
        if (requested_page_limit > 1 && it->read_only && it->paging.kind == agent_openapi_paging_kind::cursor &&
                !it->paging.initial_cursor.empty() && !arguments.contains(it->paging.cursor_parameter) &&
                (it->paging.page_parameter.empty() || !arguments.contains(it->paging.page_parameter))) {
            arguments[it->paging.cursor_parameter] = it->paging.initial_cursor;
        }
        const auto effective_arguments_json = arguments.dump();
        if (!executor(context, *it, effective_arguments_json, execution, error)) return false;
        if (execution.ok && requested_page_limit > 1 && it->read_only &&
                it->paging.kind == agent_openapi_paging_kind::cursor &&
                !it->paging.cursor_parameter.empty()) {
            execution = collect_cursor_pages(context, *it, arguments,
                std::min<uint8_t>(requested_page_limit, 5), std::move(execution));
        } else if (execution.ok && requested_page_limit > 1) {
            execution.structured_content_json = partial_json(
                json::parse(execution.structured_content_json, nullptr, false), 1,
                std::min<uint8_t>(requested_page_limit, 5), "unsupported_pagination_contract");
        }
        // Materialization replaces the body with a resource descriptor. Preserve
        // the response for a bounded discovery preview before that happens.
        const auto source_content_json = execution.structured_content_json;
        if (execution.ok && materializer && !materializer(context, *it, effective_arguments_json, execution, error)) return false;
        if (execution.ok && !it->result_view_json.empty()) {
            if (execution.resource_refs.empty() && execution.dataset_refs.empty()) {
                execution.structured_content_json = "{}";
            } else if (!execution.resource_refs.empty()) {
                auto metadata = json::parse(execution.structured_content_json, nullptr, false);
                if (!metadata.is_object()) metadata = json::object();
                metadata["resource_ref"] = execution.resource_refs.front().uri;
                execution.structured_content_json = metadata.dump();
            }
        }
        if (execution.ok && !it->result_view_json.empty() && requested_page_limit > 1) {
            const auto source = json::parse(source_content_json, nullptr, false);
            auto metadata = json::parse(execution.structured_content_json, nullptr, false);
            if (source.is_object() && source.contains("pagination") && source["pagination"].is_object()) {
                if (!metadata.is_object()) metadata = json::object();
                metadata["pagination"] = source["pagination"];
                execution.structured_content_json = metadata.dump();
            }
        }
        if (execution.ok && !it->result_view_json.empty() &&
                !agent_result_view_project(it->result_view_json, source_content_json, execution.structured_content_json, error)) {
            execution.ok = false;
            execution.failure_code = "openapi.result_view_invalid_response";
            execution.safe_summary = "Provider response did not match the configured result view.";
            execution.raw_diagnostic = error;
            error.clear();
        }
        if (!it->result_view_json.empty()) {
            execution.text_content = execution.ok ? "Bounded provider result view." : execution.safe_summary;
        }
        if (requested_page_limit > 1 && !execution.resource_refs.empty() && it->result_view_json.empty()) {
            auto preview = json::parse(execution.structured_content_json, nullptr, false);
            const json::json_pointer path(it->paging.items_pointer);
            if (preview.is_object() && preview.contains("pagination") &&
                    preview.contains(path) && preview[path].is_array()) {
                const auto count = preview[path].size();
                json selected = json::array();
                size_t preview_bytes = 0;
                for (const auto & item : preview[path]) {
                    const size_t bytes = item.dump().size();
                    if (selected.size() >= 10 || bytes > 16384 - preview_bytes) break;
                    selected.push_back(item);
                    preview_bytes += bytes;
                }
                const auto omitted = count - selected.size();
                preview[path] = std::move(selected);
                annotate_pagination(preview, {{"preview_omitted_items", omitted},
                    {"resource_ref", execution.resource_refs.front().uri}});
                execution.structured_content_json = preview.dump();
            }
        }
        // Project identity candidates from the original provider response,
        // before a materializer replaces it with a resource descriptor. The
        // catalog relation, rather than a provider name, defines the lookup
        // parameter. The original body remains in the host-owned resource.
        const auto relation = std::find_if(catalog.relations.begin(), catalog.relations.end(),
                [&](const auto & item) { return item.collection_operation_id == it->operation_id; });
        if (execution.ok && relation != catalog.relations.end()) {
            std::vector<agent_openapi_item_reference> references;
            std::string reference_error;
            agent_openapi_result_projection_limits limits;
            if (!make_agent_openapi_item_references(catalog, it->operation_id, source_content_json,
                    limits, references, reference_error)) {
                execution.ok = false;
                execution.failure_code = "openapi.collection_projection_unavailable";
                execution.failure_class = common_tool_failure_class::execution;
                execution.safe_summary = "The provider collection did not match its declared response layout.";
                execution.raw_diagnostic = reference_error;
            } else {
                auto preview = json::parse(execution.structured_content_json, nullptr, false);
                if (!preview.is_object()) preview = json::object();
                preview["host_item_parameter"] = relation->item_parameter;
                preview["host_item_candidates"] = json::array();
                const auto body = json::parse(source_content_json, nullptr, false);
                for (const auto & reference : references) {
                    if (preview["host_item_candidates"].size() >= 5) break;
                    json candidate = {{reference.item_parameter, reference.item_value}};
                    try {
                        const json::json_pointer rows(it->paging.items_pointer);
                        if (body.contains(rows) && body.at(rows).is_array() && reference.row_index < body.at(rows).size()) {
                            const auto & row = body.at(rows).at(reference.row_index);
                            for (const char * key : {"display_name", "title"}) {
                                if (row.is_object() && row.contains(key) && row[key].is_string() && row[key].get<std::string>().size() <= 512)
                                    candidate[key] = row[key];
                            }
                        }
                    } catch (const json::exception &) { /* Identity remains valid without a display title. */ }
                    preview["host_item_candidates"].push_back(std::move(candidate));
                }
                preview["host_item_candidates_truncated"] = references.size() > 5;
                execution.structured_content_json = preview.dump();
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
        const json::json_pointer items_path(operation.paging.items_pointer);
        if (!merged.is_object() || !merged.contains(items_path) || !merged[items_path].is_array()) {
            first.structured_content_json = partial_json(std::move(merged), 1, page_limit,
                "cursor_response_missing_results");
            return first;
        }
        auto & items = merged[items_path];
        // Bound the combined response as well as each HTTP response. This is
        // independent of provider naming and of the page-size parameter.
        constexpr size_t max_items = 10000;
        constexpr size_t max_merged_bytes = 8 * 1024 * 1024;
        if (items.size() > max_items) {
            items.erase(items.begin() + max_items, items.end());
            first.structured_content_json = partial_json(std::move(merged), 1, page_limit, "result_limit");
            return first;
        }
        size_t merged_bytes = merged.dump().size();
        if (merged_bytes > max_merged_bytes) {
            // Preserve the bounded first HTTP response, but never grow it.
            first.structured_content_json = partial_json(std::move(merged), 1, page_limit, "byte_limit");
            return first;
        }
        auto pending_cursor = response_cursor(merged, operation.paging);
        size_t item_count = items.size();
        uint8_t pages = 1;
        std::set<std::string> seen_cursors;
        if (arguments.contains(operation.paging.cursor_parameter) &&
                arguments[operation.paging.cursor_parameter].is_string()) {
            seen_cursors.insert(arguments[operation.paging.cursor_parameter].get<std::string>());
        }
        json page_provenance = json::array();
        page_provenance.push_back({{"page", 1},
            {"cursor", arguments.contains(operation.paging.cursor_parameter)
                ? arguments[operation.paging.cursor_parameter] : json(nullptr)},
            {"http_status", first.http.status}, {"response_bytes", first.http.response_bytes}});
        bool complete = false;
        std::string stop_reason;
        for (;;) {
            const auto cursor = pending_cursor;
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
                    {"retry_after_seconds", page.http.retry_after_seconds},
                    {"summary", page.safe_summary}};
                annotate_pagination(merged, {{"pages_fetched", pages}, {"page_limit", page_limit},
                    {"complete", false}, {"stop_reason", stop_reason},
                    {"page_provenance", page_provenance}, {"page_error", failure}});
                first.structured_content_json = merged.dump();
                first.raw_diagnostic = page.raw_diagnostic.empty() ? page_error : page.raw_diagnostic;
                // Keep already fetched evidence usable, but explicitly mark it
                // partial and retain the failed HTTP status in the result.
                first.ok = true;
                return first;
            }
            auto page_json = json::parse(page.structured_content_json, nullptr, false);
            if (!page_json.is_object() || !page_json.contains(items_path) ||
                    !page_json[items_path].is_array()) {
                stop_reason = "cursor_response_missing_results";
                break;
            }
            // Keep the last response separate: a top-level continuation must
            // advance too, rather than reading the first page's cursor again.
            pending_cursor = response_cursor(page_json, operation.paging);
            std::string limit_reason;
            for (auto & item : page_json[items_path]) {
                if (item_count >= max_items) { limit_reason = "result_limit"; break; }
                const auto item_bytes = item.dump().size() + 1;
                if (item_bytes > max_merged_bytes - merged_bytes) { limit_reason = "byte_limit"; break; }
                merged[items_path].push_back(std::move(item));
                merged_bytes += item_bytes;
                ++item_count;
            }
            // Update continuation fields without overwriting a provider's
            // metadata object, which may itself contain the collected array.
            if (!operation.paging.next_cursor_pointer.empty()) {
                const json::json_pointer cursor_path(operation.paging.next_cursor_pointer);
                if (page_json.contains(cursor_path)) merged[cursor_path] = page_json[cursor_path];
            } else if (page_json.contains("meta") && page_json["meta"].is_object() &&
                    page_json["meta"].contains("next_cursor")) {
                merged["meta"]["next_cursor"] = page_json["meta"]["next_cursor"];
            } else if (page_json.contains("next_cursor")) {
                merged["next_cursor"] = page_json["next_cursor"];
            }
            first.http = page.http;
            ++pages;
            page_provenance.push_back({{"page", pages}, {"cursor", *cursor},
                {"http_status", page.http.status}, {"response_bytes", page.http.response_bytes}});
            if (!limit_reason.empty()) { stop_reason = limit_reason; break; }
        }
        if (stop_reason.empty()) stop_reason = complete ? "exhausted" : "cursor_unavailable";
        annotate_pagination(merged, {{"pages_fetched", pages}, {"page_limit", page_limit},
            {"complete", complete}, {"stop_reason", stop_reason},
            {"page_provenance", page_provenance}});
        first.structured_content_json = merged.dump();
        first.ok = true;
        return first;
    }

    static std::optional<std::string> response_cursor(const json & value,
            const agent_openapi_paging_contract & paging) {
        const json * cursor = nullptr;
        if (!paging.next_cursor_pointer.empty()) {
            const json::json_pointer path(paging.next_cursor_pointer);
            if (value.contains(path)) cursor = &value[path];
        } else if (value.contains("meta") && value["meta"].is_object() &&
                value["meta"].contains("next_cursor")) cursor = &value["meta"]["next_cursor"];
        else if (value.contains("next_cursor")) cursor = &value["next_cursor"];
        if (cursor == nullptr) return std::nullopt;
        if (cursor->is_null()) return std::string{};
        if (!cursor->is_string()) return std::nullopt;
        return cursor->get<std::string>();
    }

    static std::string partial_json(json value, uint8_t pages, uint8_t limit,
            const std::string & reason) {
        if (!value.is_object()) value = json{{"response", value.is_discarded() ? json(nullptr) : value}};
        annotate_pagination(value, {{"pages_fetched", pages}, {"page_limit", limit},
            {"complete", false}, {"stop_reason", reason}});
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
