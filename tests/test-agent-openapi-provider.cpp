#include "../tools/agent/openapi/agent-openapi-provider.h"
#include "../tools/agent/tooling/agent-result-view.h"

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

    auto reference_catalog = catalog;
    reference_catalog.relations.push_back({"listRecords", "readRecord", "/records/{record_key}", "record_key"});
    std::vector<agent_openapi_item_reference> references;
    assert(make_agent_openapi_item_references(reference_catalog, "listRecords",
        R"({"payload":{"entries":[{"record_key":"observed"}]},"results":[{"record_key":"decoy"}]})",
        agent_openapi_result_projection_limits{}, references, error));
    assert(references.size() == 1 && references.front().item_value == "observed" &&
        references.front().item_parameter == "record_key");
    assert(!make_agent_openapi_item_references(reference_catalog, "listRecords",
        R"({"results":[{"record_key":"decoy"}]})",
        agent_openapi_result_projection_limits{}, references, error));

    agent_tool_context context;
    context.request_id = "generic-pages";
    context.allow_network = true;
    context.max_calls = 20;
    size_t relation_calls = 0;
    agent_openapi_tool_provider relation_provider(reference_catalog,
        [&](const agent_tool_context &, const agent_openapi_operation &,
                const std::string &, agent_openapi_execution_result & page, std::string &) {
            page.ok = true;
            page.structured_content_json = ++relation_calls == 1
                ? R"({"payload":{"entries":[{"record_key":"R1","title":"Record one"}]}})"
                : R"({"results":[{"record_key":"decoy"}]})";
            return true;
        });
    auto relation_view = relation_provider.resolve_tools(context, error);
    assert(relation_view && error.empty());
    const auto related_tool = std::find_if(relation_view->chat_tools().begin(), relation_view->chat_tools().end(),
        [](const common_chat_tool & tool) { return tool.name == "library.listRecords"; });
    assert(related_tool != relation_view->chat_tools().end());
    const auto related_schema = json::parse(related_tool->result_schema);
    assert(related_schema["properties"]["host_item_candidates"]["items"]["properties"].contains("record_key"));
    auto related_result = relation_view->call({"related-1", "library.listRecords", "{}"}, error);
    assert(related_result.ok && json::parse(related_result.content_json)["result"]["host_item_candidates"][0]["record_key"] == "R1");
    assert(json::parse(related_result.content_json)["result"]["host_item_parameter"] == "record_key");
    related_result = relation_view->call({"related-2", "library.listRecords", "{}"}, error);
    assert(!related_result.ok && related_result.failure_code == "openapi.collection_projection_unavailable");

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

    // First-cursor syntax belongs to the API contract, not a provider-name
    // rule or a model-generated guess. Ordinary reads and explicit positions
    // retain their existing semantics.
    auto initial_spec = spec;
    initial_spec["paths"]["/records"]["get"]["x-agent-pagination"]["initial_cursor"] = "begin-token";
    initial_spec["paths"]["/records"]["get"]["parameters"].push_back(
        {{"name", "page"}, {"in", "query"}, {"schema", {{"type", "integer"}}}});
    agent_openapi_catalog initialized;
    assert(build_agent_openapi_catalog(initial_spec, config, initialized, error));
    assert(initialized.operations.front().paging.initial_cursor == "begin-token");
    std::vector<json> received;
    agent_openapi_tool_provider initialized_provider(initialized,
        [&](const agent_tool_context &, const agent_openapi_operation &, const std::string & input,
                agent_openapi_execution_result & page, std::string &) {
            received.push_back(json::parse(input));
            const auto cursor = received.back().value("continuation", std::string{});
            page.ok = true;
            page.structured_content_json = json{
                {"payload", {{"entries", json::array({{{"value", received.size()}}})}}},
                {"links", {{"token", cursor == "begin-token" ? json("second-page") : json(nullptr)}}}}.dump();
            return true;
        });
    auto initialized_view = initialized_provider.resolve_tools(context, error);
    const auto initialized_result = initialized_view->call({"initialized", "library.listRecords", "{}", 2}, error);
    assert(initialized_result.ok && received.size() == 2 && received[0]["continuation"] == "begin-token" &&
        received[1]["continuation"] == "second-page");
    const auto initialized_body = json::parse(initialized_result.content_json)["result"];
    assert(initialized_body["pagination"]["complete"] == true &&
        initialized_body["pagination"]["page_provenance"][0]["cursor"] == "begin-token");
    assert(initialized_view->call({"resume", "library.listRecords", R"({"continuation":"resume-token"})", 2}, error).ok);
    assert(received.size() == 3 && received.back()["continuation"] == "resume-token");
    assert(initialized_view->call({"single", "library.listRecords", "{}"}, error).ok);
    assert(received.size() == 4 && !received.back().contains("continuation"));
    assert(initialized_view->call({"page", "library.listRecords", R"({"page":3})", 2}, error).ok);
    assert(received.size() == 5 && received.back()["page"] == 3 && !received.back().contains("continuation"));
    auto missing_cursor_spec = initial_spec;
    missing_cursor_spec["paths"]["/records"]["get"]["parameters"] = json::array();
    missing_cursor_spec["paths"]["/records"]["get"]["x-agent-pagination"].erase("cursor_parameter");
    agent_openapi_catalog missing_cursor_catalog;
    assert(!build_agent_openapi_catalog(missing_cursor_spec, config, missing_cursor_catalog, error));
    assert(error.find("initial_cursor requires") != std::string::npos);
    error.clear();

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
            R"({"initial_cursor":false})",
            R"({"initial_cursor":""})",
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

static void test_result_view_contract() {
    std::string error, output = R"({"resource_ref":"resource://source","pagination":{"complete":true}})";
    const std::string policy = R"({"kind":"collection","items_pointer":"/payload/rows","fields":{"id":"/id","name":"/details/name"},"max_items":1,"max_bytes":4096,"materialize":"resource"})";
    assert(agent_result_view_validate(policy, error));
    const std::string raw = R"({"payload":{"rows":[{"id":1,"details":{"name":"First","secret":"retained"}},{"id":2}]},"pagination":{"complete":true}})";
    assert(agent_result_view_project(policy, raw, output, error));
    const auto value = json::parse(output);
    assert(value["payload"]["rows"].size() == 1);
    assert(value["payload"]["rows"][0]["name"] == "First");
    assert(value["preview_omitted_items"] == 1 && value["view_truncated"] == true);
    assert(value["pagination"]["complete"] == true && value["resource_ref"] == "resource://source");
    assert(!value["payload"]["rows"][0].contains("details"));
    assert(json::parse(raw)["payload"]["rows"][0]["details"]["secret"] == "retained");
    assert(agent_result_view_validate(R"({"kind":"object","fields":{},"resource_scope":"session"})", error));
    assert(!agent_result_view_validate(R"({"kind":"object","fields":{},"resource_scope":"project"})", error));
    assert(!agent_result_view_validate(R"({"kind":"object","fields":{"id":"not-a-pointer"}})", error));
    assert(!agent_result_view_validate(R"({"kind":"object","fields":{},"derived_fields":{"x":{"transform":"eval","from":"/x"}}})", error));
    assert(!agent_result_view_validate(R"({"kind":"object","fields":{"resource_ref":"/id"}})", error));
    const std::string root_policy = R"({"kind":"collection","items_pointer":"","fields":{"id":"/id"}})";
    output = "{}";
    assert(agent_result_view_project(root_policy, R"([{"id":1}])", output, error));
    assert(json::parse(output)["items"][0]["id"] == 1);
    const std::string renamed_policy = R"({"kind":"collection","items_pointer":"/rows","output_pointer":"/items","fields":{"id":"/id"}})";
    output = "{}";
    assert(agent_result_view_project(renamed_policy, R"({"rows":[{"id":2}]})", output, error));
    assert(json::parse(output)["items"][0]["id"] == 2);
    output = "{}";
    assert(!agent_result_view_project(policy, R"({"payload":{}})", output, error));
    agent_openapi_result_projection dataset;
    assert(classify_agent_openapi_result_json(raw, {}, dataset, error, "{}", "/payload/rows"));
    assert(dataset.kind == agent_openapi_result_projection_kind::json_resource);
    assert(classify_agent_openapi_result_json(R"({"payload":{"rows":[{"id":1,"name":"a"}]}})", {}, dataset, error, "{}", "/payload/rows"));
    assert(dataset.kind == agent_openapi_result_projection_kind::dataset);
}

void test_agent_openalex_provider_view();

int main() {
    test_checked_in_eurostat_provider_examples();
    test_response_media_type_contract_is_safely_filtered();
    test_openapi_query_parameter_serialization_contract();
    test_generic_workflow_pagination();
    test_result_view_contract();
    test_agent_openalex_provider_view();
}
