#include "../tools/agent/openapi/agent-openapi-provider.h"
#include "../tools/agent/openapi/agent-openapi-result-projection.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::ordered_json;

static json read_example_json(const std::string & relative_path) {
    std::ifstream file(std::string(LLAMA_AGENT_SOURCE_DIR) + "/" + relative_path);
    assert(file);
    std::stringstream text;
    text << file.rdbuf();
    return json::parse(text.str());
}

static void test_checked_in_eurostat_provider_examples() {
    const auto host = read_example_json("docs/examples/agent-host-config-eurostat.json");
    const auto & providers = host.at("tools").at("providers");
    assert(providers.size() == 2);
    std::string error;
    for (const auto & provider : providers) {
        agent_host_openapi_provider_config config;
        config.id = provider.at("id").get<std::string>();
        config.base_url = provider.at("base_url").get<std::string>();
        config.prefix = provider.at("prefix").get<std::string>();
        config.access = "read_only";
        config.exposure = "auto";
        const auto spec = read_example_json("docs/examples/" +
            provider.at("spec_path").get<std::string>());
        agent_openapi_catalog catalog;
        assert(build_agent_openapi_catalog(spec, config, catalog, error));
        assert(catalog.operations.size() == 1);
        assert(catalog.operations.front().read_only);
        if (config.prefix == "eurostat") {
            const auto & operation = catalog.operations.front();
            assert(operation.operation_id == "getData");
            assert(operation.query_parameter_map.empty());
            assert(operation.response_media_types == std::vector<std::string>{"application/json"});
            const auto input = json::parse(operation.input_schema_json);
            assert(input["properties"]["geo"]["type"] == "array");
            assert(input["properties"]["time"]["type"] == "array");
            assert(!input["properties"].contains("dimensions"));
        } else {
            assert(config.prefix == "eurostat_sdmx");
            assert(catalog.operations.front().operation_id == "getStructure");
            assert(catalog.operations.front().path_parameters.size() == 4);
            assert(std::find(catalog.operations.front().query_parameters.begin(),
                catalog.operations.front().query_parameters.end(), "lang") ==
                catalog.operations.front().query_parameters.end());
            assert((catalog.operations.front().response_media_types == std::vector<std::string>{
                "application/vnd.sdmx.structure+xml", "application/xml"}));
            assert(json::parse(catalog.operations.front().result_schema_json).value("type", "") == "string");
        }
    }
}

static void test_response_media_type_contract_is_safely_filtered() {
    const auto spec = json::parse(R"({
        "openapi":"3.0.0","paths":{"/resource":{"get":{"operationId":"getResource",
          "responses":{"200":{"content":{"text/xml":{"schema":{"type":"string"}},
            "application/json":{"schema":{"type":"object"}},
            "text/plain\\r\\nX-Evil: yes":{"schema":{"type":"string"}}}}}}}}
    })");
    agent_host_openapi_provider_config config;
    config.id = "media-contract";
    config.base_url = "https://example.test";
    agent_openapi_catalog catalog;
    std::string error;
    assert(build_agent_openapi_catalog(spec, config, catalog, error));
    assert(catalog.operations.size() == 1);
    assert((catalog.operations.front().response_media_types == std::vector<std::string>{
        "application/json", "text/xml"}));
}

static void test_generic_workflow_pagination() {
    auto spec = json::parse(R"({
        "openapi":"3.0.0","paths":{"/records":{"get":{
            "operationId":"listRecords",
            "parameters":[{"name":"continuation","in":"query","schema":{"type":"string"}}],
            "x-agent-pagination":{"cursor_parameter":"continuation",
                "items_pointer":"/payload/entries","next_cursor_pointer":"/links/token"},
            "responses":{"200":{"description":"Records","content":{"application/json":{"schema":{
                "type":"object","properties":{
                    "payload":{"type":"object","properties":{"entries":{"type":"array","items":{"type":"object"}}}},
                    "links":{"type":"object","properties":{"token":{"type":"string","nullable":true}}}
                }
            }}}}}
        }}}}
    )");
    agent_host_openapi_provider_config config;
    config.id = "library";
    config.base_url = "https://library.example";
    agent_openapi_catalog catalog;
    std::string error;
    assert(build_agent_openapi_catalog(spec, config, catalog, error));
    assert(catalog.operations.size() == 1);
    const auto operation = catalog.operations.front();
    assert(operation.paging.cursor_parameter == "continuation");
    assert(operation.paging.items_pointer == "/payload/entries");
    assert(operation.paging.next_cursor_pointer == "/links/token");

    agent_tool_context context;
    context.request_id = "generic-pages";
    context.allow_network = true;
    context.max_calls = 20;
    size_t calls = 0;
    agent_openapi_tool_provider provider(catalog,
        [&](const agent_tool_context &, const agent_openapi_operation &,
                const std::string & arguments, agent_openapi_execution_result & page, std::string &) {
            ++calls;
            const auto args = json::parse(arguments);
            if (calls > 1) assert(args.at("continuation") == "page-" + std::to_string(calls));
            page.ok = true;
            page.structured_content_json = json{
                {"payload", {{"entries", json::array({{{"arbitrary_field", calls}}})}}},
                {"links", {{"token", calls < 3 ? json("page-" + std::to_string(calls + 1)) : json(nullptr)}}},
            }.dump();
            return true;
        },
        [](const agent_tool_context &, const agent_openapi_operation &, const std::string &,
                agent_openapi_execution_result & page, std::string &) {
            const auto raw = json::parse(page.structured_content_json);
            assert(raw["payload"]["entries"].size() == 3);
            common_runtime_resource_ref resource;
            resource.uri = "resource://turn/library-pages";
            page.resource_refs.push_back(resource);
            return true;
        });
    auto view = provider.resolve_tools(context, error);
    assert(view);
    const auto result = view->call({"generic", "library.listRecords", "{}", 5}, error);
    assert(result.ok && calls == 3);
    const auto body = json::parse(result.content_json).at("result");
    assert(body["pagination"]["complete"] == true);
    assert(body["payload"]["entries"][2]["arbitrary_field"] == 3);
    assert(body["links"]["token"].is_null());
    assert(body["pagination"]["preview_omitted_items"] == 0);
    assert(!body.contains("candidates"));

    // Conventional layouts are inferred from the schema without an extension.
    auto & declared = spec["paths"]["/records"]["get"];
    declared.erase("x-agent-pagination");
    declared["parameters"][0]["name"] = "page_token";
    declared["responses"]["200"]["content"]["application/json"]["schema"] = json::parse(
        R"({"type":"object","properties":{"items":{"type":"array","items":{"type":"object"}},"nextCursor":{"type":"string","nullable":true}}})");
    agent_openapi_catalog inferred;
    assert(build_agent_openapi_catalog(spec, config, inferred, error));
    assert(inferred.operations.front().paging.cursor_parameter == "page_token");
    assert(inferred.operations.front().paging.items_pointer == "/items");
    assert(inferred.operations.front().paging.next_cursor_pointer == "/nextCursor");

    calls = 0;
    agent_openapi_tool_provider top_level(inferred,
        [&](const agent_tool_context &, const agent_openapi_operation &,
                const std::string &, agent_openapi_execution_result & page, std::string &) {
            ++calls;
            page.ok = true;
            page.structured_content_json = json{
                {"items", json::array({{{"value", calls}}})},
                {"nextCursor", calls < 3 ? json(std::to_string(calls)) : json(nullptr)},
            }.dump();
            return true;
        });
    auto top_view = top_level.resolve_tools(context, error);
    assert(top_view);
    const auto top = top_view->call({"top", "library.listRecords", "{}", 5}, error);
    assert(top.ok && calls == 3);
    const auto top_body = json::parse(top.content_json).at("result");
    assert(top_body["items"].size() == 3 && top_body["pagination"]["complete"] == true);

    // The initial request cursor belongs to the seen set too.
    calls = 0;
    agent_openapi_tool_provider repeated(inferred,
        [&](const agent_tool_context &, const agent_openapi_operation &,
                const std::string &, agent_openapi_execution_result & page, std::string &) {
            ++calls;
            page.ok = true;
            page.structured_content_json = R"({"items":[],"nextCursor":"same"})";
            return true;
        });
    auto repeated_view = repeated.resolve_tools(context, error);
    const auto repeat = repeated_view->call({"repeat", "library.listRecords", R"({"page_token":"same"})", 5}, error);
    assert(repeat.ok && calls == 1);
    assert(json::parse(repeat.content_json)["result"]["pagination"]["stop_reason"] == "repeated_cursor");

    // An unsupported paging method must not masquerade as complete collection.
    inferred.operations.front().paging.kind = agent_openapi_paging_kind::page_number;
    agent_openapi_tool_provider unsupported(inferred,
        [](const agent_tool_context &, const agent_openapi_operation &, const std::string &,
                agent_openapi_execution_result & page, std::string &) {
            page.ok = true;
            page.structured_content_json = R"({"items":[]})";
            return true;
        });
    auto unsupported_view = unsupported.resolve_tools(context, error);
    const auto partial = unsupported_view->call({"unsupported", "library.listRecords", "{}", 5}, error);
    assert(partial.ok);
    assert(json::parse(partial.content_json)["result"]["pagination"]["stop_reason"] ==
        "unsupported_pagination_contract");

    for (const auto * invalid : {
            R"({"items_pointer":"/bad~2"})",
            R"({"cursor_parameter":"undeclared"})",
            R"({"items_pointer":"/items","next_cursor_pointer":"/items/token"})",
            R"({"items_pointer":"/pagination"})",
            R"({"unknown":"/value"})"}) {
        declared["x-agent-pagination"] = json::parse(invalid);
        error.clear();
        assert(!build_agent_openapi_catalog(spec, config, inferred, error));
        assert(!error.empty());
    }
}

static void test_openapi_query_parameter_serialization_contract() {
    const auto spec = json::parse(R"({
        "openapi":"3.0.0","paths":{"/data/{code}":{"get":{
            "operationId":"getData",
            "x-agent-query-parameter-map":"dimensions",
            "parameters":[
                {"name":"code","in":"path","required":true,"schema":{"type":"string"}},
                {"name":"geo","in":"query","style":"form","explode":true,
                    "schema":{"type":"array","items":{"type":"string"}}},
                {"name":"time","in":"query","style":"form","explode":false,
                    "schema":{"type":"array","items":{"type":"string"}}}
            ],
            "responses":{"200":{"description":"data","content":{"application/json":{"schema":{"type":"object"}}}}}
        }}}}
    )");
    agent_host_openapi_provider_config config;
    config.id = "eurostat";
    config.base_url = "https://example.test";
    agent_openapi_catalog catalog;
    std::string error;
    assert(build_agent_openapi_catalog(spec, config, catalog, error));
    assert(catalog.operations.size() == 1);
    const auto & operation = catalog.operations.front();
    assert(operation.query_parameter_map == "dimensions");
    assert(operation.query_parameter_styles.at("geo") == "form");
    assert(operation.query_parameter_explode.at("geo"));
    const auto schema = json::parse(operation.input_schema_json);
    assert(schema["properties"]["geo"]["type"] == "array");
    assert(schema["properties"]["dimensions"]["type"] == "object");
    agent_openapi_tool_provider provider(catalog,
        [](const agent_tool_context &, const agent_openapi_operation &,
                const std::string &, agent_openapi_execution_result & result, std::string &) {
            result.ok = true;
            result.structured_content_json = R"({"status":"ok"})";
            return true;
        });
    agent_tool_context context;
    context.request_id = "eurostat-schema";
    context.allow_network = true;
    context.max_calls = 4;
    const auto view = provider.resolve_tools(context, error);
    assert(view != nullptr);
    const auto call = view->call({"eurostat-schema-call", "eurostat.getData",
        R"({"code":"DEMO","geo":["SE","NO"],"time":["2020","2021"],"dimensions":{"unit":["P_KM2","P_M2"],"age":"TOTAL"}})"}, error);
    assert(call.ok);
}

int main() {
    test_checked_in_eurostat_provider_examples();
    test_response_media_type_contract_is_safely_filtered();
    test_openapi_query_parameter_serialization_contract();
    test_generic_workflow_pagination();
    agent_openapi_catalog catalog;
    catalog.provider_id = "openalex";
    catalog.prefix = "openalex";
    catalog.additional_families = {"research"};
    catalog.base_url = "https://api.openalex.org";
    agent_openapi_operation operation;
    operation.operation_id = "listWorks";
    operation.method = "GET";
    operation.path = "/works";
    operation.summary = "List works";
    operation.description = operation.summary;
    operation.input_schema_json = R"({"type":"object","additionalProperties":false,"properties":{"cursor":{"type":"string"},"per_page":{"type":"integer"}}})";
    operation.result_schema_json = R"({"type":"object"})";
    operation.read_only = true;
    operation.paging.kind = agent_openapi_paging_kind::cursor;
    operation.paging.cursor_parameter = "cursor";
    operation.paging.page_size_parameter = "per_page";
    catalog.operations.push_back(operation);

    size_t calls = 0;
    size_t materializations = 0;
    json long_abstract_positions = json::array();
    for (int index = 2; index < 900; ++index) long_abstract_positions.push_back(index);
    const auto second_page = json{
        {"meta", {{"next_cursor", nullptr}}},
        {"results", json::array({{{"title", "Second"},
            {"content_url", "https://content.example/full.txt"},
            {"abstract_inverted_index", {{"é", long_abstract_positions}}}}})},
    }.dump();
    agent_openapi_executor executor = [&](const agent_tool_context &,
            const agent_openapi_operation &, const std::string & arguments,
            agent_openapi_execution_result & result, std::string &) {
        const auto args = json::parse(arguments);
        ++calls;
        result.ok = true;
        result.http.provider = "openalex";
        result.http.operation = "listWorks";
        result.http.status = 200;
        if (!args.contains("cursor")) {
            result.structured_content_json = R"({"meta":{"next_cursor":"cursor-2"},"results":[{"title":"First","abstract_inverted_index":{"study":[1],"A":[0]}}]})";
        } else {
            assert(args.at("cursor") == "cursor-2");
            result.structured_content_json = second_page;
        }
        return true;
    };
    agent_openapi_result_materializer materializer = [&](const agent_tool_context &,
            const agent_openapi_operation &, const std::string &,
            agent_openapi_execution_result & result, std::string &) {
        ++materializations;
        const auto raw = json::parse(result.structured_content_json);
        assert(raw.at("results").size() == (calls == 2 ? 2 : 1));
        assert(raw.at("results")[0].contains("abstract_inverted_index"));
        common_runtime_resource_ref resource;
        resource.uri = "resource://turn/openalex-results";
        resource.name = "OpenAlex results";
        resource.description = "Materialized OpenAlex response";
        resource.mime_type = "application/json";
        resource.size_bytes = 512;
        resource.scope = common_runtime_resource_scope::turn;
        result.resource_refs.push_back(std::move(resource));
        return true;
    };
    agent_openapi_tool_provider provider(std::move(catalog), executor, materializer);
    agent_tool_context context;
    context.request_id = "test";
    context.allow_network = true;
    context.max_calls = 10;
    std::string error;
    auto view = provider.resolve_tools(context, error);
    assert(view && error.empty());
    assert(view->exposes_tool("openalex.listWorks"));
    assert(std::any_of(view->chat_tools().begin(), view->chat_tools().end(),
        [](const common_chat_tool & tool) { return tool.name == "openalex.listWorks"; }));
    assert(view->additional_tool_families("openalex.listWorks") == std::vector<std::string>{"research"});
    assert(view->additional_tool_families("openalex.unknown").empty());
    auto result = view->call({"call-1", "openalex.listWorks", "{}", 5}, error);
    assert(result.ok && error.empty());
    assert(calls == 2 && materializations == 1);
    const auto output = json::parse(result.content_json);
    const auto & body = output.at("result");
    assert(body.at("result_count") == 2);
    assert(body.at("resource_ref") == "resource://turn/openalex-results");
    assert(body.at("pagination").at("pages_fetched") == 2);
    assert(body.at("pagination").at("complete") == true);
    assert(body.at("candidates")[0].at("abstract_text") == "A study");
    assert(body.at("candidates")[1].at("full_text_url") == "https://content.example/full.txt");
    assert(body.at("candidates")[1].at("full_text_url_source") == "content_url");
    assert(body.at("candidates")[1].at("abstract_truncated") == true);
    assert(body.at("candidates")[1].at("abstract_text").get<std::string>().size() <= 1200);

    const auto preferred_pdf = json::parse(agent_openapi_project_model_result(
        R"({"best_oa_location":{"pdf_url":"https://files.example/paper.pdf"},"content_url":"https://files.example/paper.txt","landing_page_url":"https://publisher.example/paper"})",
        true));
    assert(preferred_pdf.at("full_text_url") == "https://files.example/paper.pdf");
    const auto landing_only = json::parse(agent_openapi_project_model_result(
        R"({"landing_page_url":"https://publisher.example/paper"})", true));
    assert(!landing_only.contains("full_text_url"));

    // Without a workflow directive an OpenAPI collection remains one request.
    calls = 0;
    materializations = 0;
    result = view->call({"call-2", "openalex.listWorks", "{}"}, error);
    assert(result.ok && calls == 1 && materializations == 1);
    const auto single = json::parse(result.content_json).at("result");
    assert(!single.contains("pagination"));

    // Existing connector status metadata survives the MCP provider boundary.
    agent_openapi_catalog error_catalog;
    error_catalog.provider_id = "openalex";
    error_catalog.base_url = "https://api.openalex.org";
    error_catalog.prefix = "openalex";
    error_catalog.operations.push_back(operation);
    agent_openapi_tool_provider error_provider(
        std::move(error_catalog),
        [](const agent_tool_context &, const agent_openapi_operation &,
                const std::string &, agent_openapi_execution_result & failed,
                std::string &) {
            failed.ok = false;
            failed.http.provider = "openalex";
            failed.http.status = 429;
            failed.http.retry_after_seconds = 30;
            failed.failure_code = "openapi.http_status";
            failed.failure_class = common_tool_failure_class::network;
            failed.retryable = true;
            failed.safe_summary = "Rate limit reached.";
            return true;
        });
    auto error_view = error_provider.resolve_tools(context, error);
    assert(error_view);
    result = error_view->call({"call-3", "openalex.listWorks", "{}"}, error);
    assert(!result.ok && result.http.status == 429 && result.http.retry_after_seconds == 30);
    assert(result.retryable && result.failure_code == "openapi.http_status");

    // Cursor traversal never exceeds the host-provided page budget.
    size_t capped_calls = 0;
    agent_openapi_catalog capped_catalog;
    capped_catalog.provider_id = "openalex";
    capped_catalog.prefix = "openalex";
    capped_catalog.operations.push_back(operation);
    agent_openapi_tool_provider capped_provider(std::move(capped_catalog),
        [&](const agent_tool_context &, const agent_openapi_operation &,
                const std::string & args_json, agent_openapi_execution_result & page,
                std::string &) {
            const auto args = json::parse(args_json);
            ++capped_calls;
            page.ok = true;
            page.structured_content_json = json{
                {"meta", {{"next_cursor", "cursor-" + std::to_string(capped_calls + 1)}}},
                {"results", json::array({{{"title", "Page"}}})},
            }.dump();
            if (args.contains("cursor")) assert(args.at("cursor").is_string());
            return true;
        });
    auto capped_view = capped_provider.resolve_tools(context, error);
    assert(capped_view);
    result = capped_view->call({"call-4", "openalex.listWorks", "{}", 50}, error);
    assert(result.ok && capped_calls == 5);
    const auto capped = json::parse(result.content_json).at("result");
    assert(capped.at("pagination").at("complete") == false);
    assert(capped.at("pagination").at("stop_reason") == "page_limit");

    // A later page failure leaves earlier results usable but explicitly partial,
    // and retains HTTP/rate-limit diagnostics instead of discarding them.
    size_t failed_page_calls = 0;
    agent_openapi_catalog partial_catalog;
    partial_catalog.provider_id = "openalex";
    partial_catalog.prefix = "openalex";
    partial_catalog.operations.push_back(operation);
    agent_openapi_tool_provider partial_provider(std::move(partial_catalog),
        [&](const agent_tool_context &, const agent_openapi_operation &,
                const std::string & args_json, agent_openapi_execution_result & page,
                std::string &) {
            ++failed_page_calls;
            if (!json::parse(args_json).contains("cursor")) {
                page.ok = true;
                page.structured_content_json = R"({"meta":{"next_cursor":"cursor-next"},"results":[{"title":"First"}]})";
                page.http.status = 200;
            } else {
                page.ok = false;
                page.failure_code = "openapi.http_status";
                page.retryable = true;
                page.safe_summary = "Rate limit reached.";
                page.http.status = 429;
                page.http.retry_after_seconds = 30;
            }
            return true;
        });
    auto partial_view = partial_provider.resolve_tools(context, error);
    assert(partial_view);
    result = partial_view->call({"call-5", "openalex.listWorks", "{}", 5}, error);
    assert(result.ok && failed_page_calls == 2 && result.http.status == 429 &&
        result.http.retry_after_seconds == 30);
    const auto partial = json::parse(result.content_json).at("result");
    assert(partial.at("results").size() == 1);
    assert(partial.at("pagination").at("complete") == false);
    assert(partial.at("pagination").at("stop_reason") == "page_request_failed");
    assert(partial.at("pagination").at("page_error").at("failure_code") ==
        "openapi.http_status");
}
