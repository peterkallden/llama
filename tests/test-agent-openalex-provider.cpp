#include "../tools/agent/openapi/agent-openapi-provider.h"
#include "../tools/agent/tooling/agent-result-view.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

void test_agent_openalex_provider_view() {
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
    operation.result_view_json = R"({"kind":"collection","items_pointer":"/results","fields":{"id":"/id","title":"/title","display_name":"/display_name"},"derived_fields":{"abstract_text":{"transform":"inverted_index_to_text","from":"/abstract_inverted_index"},"full_text_url":{"transform":"first_nonempty_url","from":["/best_oa_location/pdf_url","/content_urls/pdf","/content_url"]}},"materialize":"resource"})";
    catalog.operations.push_back(operation);
    catalog.relations.push_back({"listWorks", "getWork", "/works/{id}", "id"});

    size_t calls = 0;
    size_t materializations = 0;
    json long_abstract_positions = json::array();
    for (int index = 2; index < 900; ++index) long_abstract_positions.push_back(index);
    const auto second_page = json{
        {"meta", {{"next_cursor", nullptr}}},
        {"results", json::array({{{"id", "W2"}, {"title", "Second"},
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
            result.structured_content_json = R"({"meta":{"next_cursor":"cursor-2"},"results":[{"id":"W1","title":"First","abstract_inverted_index":{"study":[1],"A":[0]}}]})";
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
        result.structured_content_json = R"({"materialized":true})";
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
    const auto collection = std::find_if(view->chat_tools().begin(), view->chat_tools().end(),
        [](const common_chat_tool & tool) { return tool.name == "openalex.listWorks"; });
    assert(collection != view->chat_tools().end());
    const auto model_result_schema = json::parse(collection->result_schema, nullptr, false);
    assert(model_result_schema.is_object() && model_result_schema.value("type", std::string()) == "object");
    assert(!model_result_schema.value("properties", json::object()).contains("result"));
    assert(model_result_schema.value("additionalProperties", false));
    assert(model_result_schema["properties"]["host_item_candidates"]["items"]["properties"].contains("id"));
    assert(model_result_schema["properties"]["results"]["items"]["properties"].contains("abstract_text"));
    assert(collection != view->chat_tools().end() && collection->collection_item_parameter == "id" &&
        collection->collection_items_pointer == "/host_item_candidates");
    assert(view->additional_tool_families("openalex.unknown").empty());
    auto result = view->call({"call-1", "openalex.listWorks", "{}", 5}, error);
    assert(result.ok && error.empty());
    assert(calls == 2 && materializations == 1);
    const auto output = json::parse(result.content_json);
    const auto & body = output.at("result");
    assert(body.at("result_count") == 2);
    assert(body.at("preview_omitted_items") == 0);
    assert(body.at("resource_ref") == "resource://turn/openalex-results");
    assert(body.at("pagination").at("pages_fetched") == 2);
    assert(body.at("pagination").at("complete") == true);
    assert(body.at("results")[0].at("abstract_text") == "A study");
    assert(body.at("host_item_candidates").size() == 2 && body.at("host_item_candidates")[0].at("id") == "W1" &&
        body.at("host_item_candidates")[1].at("title") == "Second");
    assert(body.at("results")[1].at("full_text_url") == "https://content.example/full.txt");
    assert(body.at("results")[1].at("full_text_url_source") == "/content_url");
    assert(body.at("results")[1].at("abstract_text_truncated") == true);
    assert(body.at("results")[1].at("abstract_text").get<std::string>().size() <= 1200);

    std::string projection_error;
    std::string projection = "{}";
    const std::string object_policy = R"({"kind":"object","fields":{"id":"/id","display_name":"/display_name"},"derived_fields":{"abstract_text":{"transform":"inverted_index_to_text","from":"/abstract_inverted_index"},"full_text_url":{"transform":"first_nonempty_url","from":["/best_oa_location/pdf_url","/content_urls/pdf","/content_url"]}},"materialize":"resource"})";
    assert(agent_result_view_project(object_policy, R"({"best_oa_location":{"pdf_url":"https://files.example/paper.pdf"},"content_url":"https://files.example/paper.txt"})", projection, projection_error));
    assert(json::parse(projection).at("full_text_url") == "https://files.example/paper.pdf");
    projection = "{}";
    assert(agent_result_view_project(object_policy, R"({"landing_page_url":"https://publisher.example/paper"})", projection, projection_error));
    assert(!json::parse(projection).contains("full_text_url"));

    // A materialized single work still exposes bounded host-derived fields
    // required for the full-text workflow, while retaining the raw resource.
    agent_openapi_catalog work_catalog;
    work_catalog.provider_id = "openalex";
    work_catalog.prefix = "openalex";
    work_catalog.base_url = "https://api.openalex.org";
    auto work_operation = operation;
    work_operation.operation_id = "getWork";
    work_operation.result_view_json = object_policy;
    work_catalog.operations.push_back(work_operation);
    agent_openapi_tool_provider work_provider(std::move(work_catalog),
        [](const agent_tool_context &, const agent_openapi_operation &, const std::string &,
                agent_openapi_execution_result & work, std::string &) {
            work.ok = true;
            work.structured_content_json = R"({"id":"W1","display_name":"Work","best_oa_location":{"pdf_url":"https://example.org/work.pdf"},"abstract_inverted_index":{"A":[0],"study":[1]},"large_payload":"not exposed"})";
            return true;
        },
        [](const agent_tool_context &, const agent_openapi_operation &, const std::string &,
                agent_openapi_execution_result & work, std::string &) {
            common_runtime_resource_ref resource;
            resource.uri = "resource://turn/work";
            work.resource_refs.push_back(std::move(resource));
            work.structured_content_json = R"({"resource_ref":"resource://turn/work"})";
            return true;
        });
    auto work_view = work_provider.resolve_tools(context, error);
    assert(work_view && error.empty());
    const auto work_result = work_view->call({"work", "openalex.getWork", "{}"}, error);
    assert(work_result.ok && error.empty());
    const auto work = json::parse(work_result.content_json).at("result");
    assert(work.at("resource_ref") == "resource://turn/work");
    assert(work.at("full_text_url") == "https://example.org/work.pdf");
    assert(work.at("full_text_url_source") == "/best_oa_location/pdf_url");
    assert(work.at("abstract_text") == "A study");
    assert(!work.contains("large_payload"));

    // Without a workflow directive an OpenAPI collection remains one request.
    calls = 0;
    materializations = 0;
    result = view->call({"call-2", "openalex.listWorks", "{}"}, error);
    assert(result.ok && calls == 1 && materializations == 1);
    const auto single = json::parse(result.content_json).at("result");
    assert(!single.contains("pagination"));
    assert(single.at("results")[0].at("title") == "First");

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
