#include "tools/agent/cli/agent-cli-host-adapter.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/resource/agent-resource-store.h"

#include "memory/memory-in-memory.h"
#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/openapi/dynamic/agent-openapi-dynamic-admission.h"
#include <nlohmann/json.hpp>

#include <fstream>
#include <algorithm>
#include <filesystem>
#include <iostream>

class host_data_store final : public common_agent_data_store {
public:
    bool put_row(const std::string & dataset, const std::string & row_id,
            const std::string & row_json, std::string & error) override {
        last_dataset = dataset;
        rows.emplace_back(row_id, row_json);
        error.clear();
        return true;
    }
    bool put_dataset_descriptor(const common_agent_dataset_descriptor & value,
            std::string & error) override {
        descriptor = value;
        error.clear();
        return true;
    }
    bool get_dataset_descriptor(const std::string & uri,
            common_agent_dataset_descriptor & value, std::string & error) override {
        if (descriptor.ref.uri != uri) { error = "unknown dataset"; return false; }
        value = descriptor;
        error.clear();
        return true;
    }
    bool execute(const std::string & operation, const std::string & request,
            std::string & result, std::string & error) override {
        if (operation != "data.query") { error = "unexpected data operation"; return false; }
        const auto query = nlohmann::json::parse(request, nullptr, false);
        if (query.is_discarded() || query.value("dataset", std::string()) != descriptor.ref.uri) {
            error = "data.query did not receive the materialized dataset";
            return false;
        }
        result = nlohmann::json({{"columns", {"id", "amount"}}, {"rows", rows.size()},
            {"row_count", rows.size()}}).dump();
        error.clear();
        return true;
    }

    common_agent_dataset_descriptor descriptor;
    std::string last_dataset;
    std::vector<std::pair<std::string, std::string>> rows;
};

int main() {
    const auto spec_path = std::filesystem::temp_directory_path() / "agent-openapi-host-smoke.json";
    std::ofstream spec(spec_path);
    spec << R"({"openapi":"3.0.0","info":{"title":"sales","version":"1"},"paths":{"/sales":{"get":{"operationId":"listSales","responses":{"200":{"content":{"application/json":{"schema":{"type":"array"}}}}}}},"/complex":{"get":{"operationId":"complex","responses":{"200":{"content":{"application/json":{"schema":{"type":"object"}}}}}}}}})";
    spec.close();

    common_memory_in_memory_store memory;
    std::string error;
    if (!memory.open("", error)) { std::cerr << "memory setup failed: " << error << "\n"; return 1; }
    host_data_store data;
    agent_catalogued_resource_store resource_store(
        std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    agent_host_tool_selection_request request;
    request.data_store = &data;
    request.resource_store_config.blob_backend = "in-memory";
    request.resource_store_config.metadata_backend = "in-memory";
    request.tool_context.profile_id = "openapi-host-smoke";
    request.tool_context.allow_network = true;
    request.tool_context.scope.namespace_id = "local";
    request.tool_context.scope.session_id = "session-1";
    request.tool_context.scope.project_id = "project-1";
    request.tool_context.scope.turn_id = "turn-1";
    common_tool_profile profile;
    profile.id = "openapi-host-smoke";
    profile.members = {{"data.query", 1, true, "{}"}, {"openapi.connect", 1, true, "{}"}};
    profile.allow_network = true;
    request.tool_profiles.emplace(profile.id, profile);
    request.dynamic_openapi_registry = std::make_shared<agent_openapi_dynamic_registry>();
    agent_host_openapi_provider_config openapi;
    openapi.id = "sales-api";
    openapi.enabled = true;
    openapi.required = true;
    openapi.spec_path = spec_path.string();
    openapi.base_url = "https://sales.example.invalid";
    openapi.prefix = "sales";
    openapi.access = "read_only";
    openapi.exposure = "auto";
    openapi.allow_private_network = true;
    openapi.connect_timeout_ms = 1000;
    openapi.request_timeout_ms = 2000;
    openapi.max_result_bytes = 1024 * 1024;
    agent_host_config cli_host_config;
    cli_host_config.tool_profile = profile.id;
    cli_host_config.tool_profiles = request.tool_profiles;
    cli_host_config.openapi_providers.push_back(openapi);
    request.openapi_providers.push_back(std::move(openapi));
    request.openapi_executor_overrides.emplace("sales-api", [](
            const agent_tool_context &, const agent_openapi_operation & operation,
            const std::string & arguments_json, agent_openapi_execution_result & result,
            std::string & executor_error) {
        if (arguments_json != "{}") {
            executor_error = "unexpected OpenAPI arguments";
            return false;
        }
        result.ok = true;
        result.http_status = 200;
        result.mime_type = "application/json";
        if (operation.operation_id == "listSales") {
            result.structured_content_json = R"([{"id":1,"amount":12.5},{"id":2,"amount":8}])";
        } else if (operation.operation_id == "complex") {
            result.structured_content_json = R"({"items":[{"id":1}]})";
        } else {
            executor_error = "unexpected OpenAPI operation";
            return false;
        }
        result.text_content = result.structured_content_json;
        executor_error.clear();
        return true;
    });
    agent_openapi_catalog catalog_check;
    nlohmann::json spec_check;
    std::ifstream spec_check_file(spec_path);
    spec_check_file >> spec_check;
    spec_check_file.close();
    if (!build_agent_openapi_catalog(spec_check, request.openapi_providers.front(), catalog_check, error)) {
        std::cerr << "catalog build failed: " << error << "\n";
        return 1;
    }
    if (catalog_check.operations.size() != 2) {
        std::cerr << "unexpected catalog operation count: " << catalog_check.operations.size() << "\n";
        return 1;
    }
    common_agent_cli_tool_selection selection;
    common_memory_query query;
    query.scope = common_memory_scope::session;
    query.session_id = "session-1";
    if (!resolve_agent_host_tool_selection(memory, nullptr, &resource_store, nullptr,
        profile.id, request, query, nullptr, selection, error)) {
        std::cerr << "host selection failed: " << error << "\n";
        return 1;
    }
    if (!selection.tool_view) {
        std::cerr << "host selection returned no tool view\n";
        return 1;
    }
    if (selection.tool_view->chat_tools().empty()) {
        std::cerr << "host selection returned no tools\n";
        return 1;
    }
    const auto connect_capability = selection.tooling.capability_tools.find("openapi.connect");
    if (connect_capability == selection.tooling.capability_tools.end() ||
            std::find(connect_capability->second.begin(), connect_capability->second.end(),
                "openapi.connect") == connect_capability->second.end()) {
        std::cerr << "openapi.connect was not bound to its workflow capability\n";
        return 1;
    }
    std::vector<std::string> expected_host_tool_names;
    for (const auto & operation : catalog_check.operations) {
        expected_host_tool_names.push_back(
            agent_openapi_exposed_tool_name(catalog_check, operation));
    }
    std::vector<std::string> actual_host_tool_names;
    for (const auto & tool : selection.tool_view->chat_tools()) {
        actual_host_tool_names.push_back(tool.name);
    }
    if (!std::all_of(expected_host_tool_names.begin(), expected_host_tool_names.end(),
            [&actual_host_tool_names](const std::string & expected) {
                return std::find(actual_host_tool_names.begin(), actual_host_tool_names.end(),
                    expected) != actual_host_tool_names.end();
            })) {
        std::cerr << "expected=";
        for (const auto & name : expected_host_tool_names) std::cerr << " " << name;
        std::cerr << " actual=";
        for (const auto & name : actual_host_tool_names) std::cerr << " " << name;
        std::cerr << "\n";
        std::cerr << "expected OpenAPI tools were not exposed\n";
        return 1;
    }

    // Exercise dynamic admission through the production-resolved host tool
    // view, then ensure a fresh session turn exposes the usual typed provider.
    const nlohmann::json dynamic_spec = {
        {"openapi", "3.0.3"},
        {"info", {{"title", "Dynamic Catalog"}, {"version", "1"}}},
        {"servers", {{{"url", "https://dynamic.example.invalid/v1"}}}},
        {"paths", {{"/things", {{"get", {{"operationId", "listThings"},
            {"summary", "List things"}, {"responses", {{"200", {{"description", "ok"}}}}}}}}}}},
    };
    agent_resource_put_request spec_resource_request;
    spec_resource_request.name = "dynamic-openapi.json";
    spec_resource_request.description = "Fetched OpenAPI document";
    spec_resource_request.mime_type = "application/json";
    // Match the host resource payload produced by web.fetch: the opaque ref
    // points to a bounded JSON envelope whose text field contains the spec.
    spec_resource_request.text = nlohmann::json({
        {"url", "https://spec.example.invalid/openapi.json"},
        {"final_url", "https://spec.example.invalid/openapi.json"},
        {"status", 200},
        {"content_type", "application/json"},
        {"text", dynamic_spec.dump()},
        {"truncated", false},
    }).dump();
    spec_resource_request.scope = common_runtime_resource_scope::turn;
    spec_resource_request.namespace_id = "local";
    spec_resource_request.session_id = "session-1";
    spec_resource_request.project_id = "project-1";
    spec_resource_request.turn_id = "turn-1";
    agent_resource_descriptor spec_resource;
    if (!resource_store.put_text(spec_resource_request, spec_resource, error)) {
        std::cerr << "failed to materialize dynamic OpenAPI test resource: " << error << "\n";
        return 1;
    }
    const auto connect_result = selection.tool_view->call({
        "connect", "openapi.connect",
        nlohmann::json({{"spec_resource", spec_resource.uri}}).dump()}, error);
    if (!connect_result.ok || connect_result.content_json.find("available_next_turn") == std::string::npos) {
        std::cerr << "dynamic OpenAPI admission failed: " << error << " "
                  << connect_result.raw_diagnostic << "\n";
        return 1;
    }
    const auto dynamic_registration = request.dynamic_openapi_registry->snapshot().front();
    request.openapi_executor_overrides.emplace(dynamic_registration.config.id, [](
            const agent_tool_context &, const agent_openapi_operation & operation,
            const std::string & arguments_json, agent_openapi_execution_result & result,
            std::string & executor_error) {
        if (operation.operation_id != "listThings" || arguments_json != "{}") {
            executor_error = "unexpected dynamic OpenAPI operation or arguments";
            return false;
        }
        result.ok = true;
        result.http_status = 200;
        result.mime_type = "application/json";
        result.structured_content_json = R"([{"id":"thing-1","name":"sample"}])";
        result.text_content = result.structured_content_json;
        executor_error.clear();
        return true;
    });
    request.tool_context.turn_id = "turn-2";
    request.tool_context.scope.turn_id = "turn-2";
    common_agent_cli_tool_selection next_turn_selection;
    if (!resolve_agent_host_tool_selection(memory, nullptr, &resource_store, nullptr,
            profile.id, request, query, nullptr, next_turn_selection, error)) {
        std::cerr << "next-turn dynamic provider resolution failed: " << error << "\n";
        return 1;
    }
    const auto dynamic_prefix = dynamic_registration.config.prefix;
    const auto expected_dynamic_tool = dynamic_prefix + ".listThings";
    if (!next_turn_selection.tool_view || !next_turn_selection.tool_view->exposes_tool(expected_dynamic_tool)) {
        std::cerr << "session dynamic OpenAPI tool was not exposed on the next turn: "
                  << expected_dynamic_tool << "\n";
        return 1;
    }
    const auto read_capability = next_turn_selection.tooling.capability_tools.find("openapi.read");
    if (read_capability == next_turn_selection.tooling.capability_tools.end() ||
            std::find(read_capability->second.begin(), read_capability->second.end(),
                expected_dynamic_tool) == read_capability->second.end()) {
        std::cerr << "dynamic read-only operation was not bound to openapi.read capability\n";
        return 1;
    }
    const auto dynamic_result = next_turn_selection.tool_view->call({
        "dynamic-call", expected_dynamic_tool, "{}"}, error);
    if (!dynamic_result.ok || dynamic_result.resource_refs.size() != 1 ||
            dynamic_result.content_json.find("thing-1") == std::string::npos) {
        std::cerr << "dynamic OpenAPI operation did not materialize its result: " << error << "\n";
        return 1;
    }
    agent_resource_descriptor dynamic_result_resource;
    const auto next_turn_authority = make_agent_resource_read_authority(
        next_turn_selection.tooling.resource_runtime, std::time(nullptr));
    if (!resource_store.stat(dynamic_result.resource_refs.front().uri,
            next_turn_authority, dynamic_result_resource, error) ||
            dynamic_result_resource.scope != common_runtime_resource_scope::session) {
        std::cerr << "dynamic OpenAPI result was not retained at session scope: " << error << "\n";
        return 1;
    }

    // Exercise the production config path as well as the direct host request
    // above.  The CLI must carry providers loaded from host config into the
    // same selection seam; otherwise the model sees tools=0 even though the
    // lower-level OpenAPI host smoke passes.
    args cli_options;
    apply_agent_host_config_to_args(cli_host_config, cli_options);
    if (cli_options.openapi_providers.size() != 1 || cli_options.tool_profile != profile.id) {
        std::cerr << "host config was not copied into CLI options\n";
        return 1;
    }
    common_agent_cli_tool_selection cli_selection;
    if (!resolve_agent_cli_tool_selection(memory, nullptr, &resource_store, nullptr,
            cli_options, query, false, cli_selection, error) ||
            !cli_selection.tool_view) {
        std::cerr << "CLI host-config selection failed: " << error << "\n";
        return 1;
    }
    std::vector<std::string> expected_cli_tool_names;
    for (const auto & operation : catalog_check.operations) {
        expected_cli_tool_names.push_back(
            agent_openapi_exposed_tool_name(catalog_check, operation));
    }
    std::vector<std::string> actual_cli_tool_names;
    for (const auto & tool : cli_selection.tool_view->chat_tools()) {
        actual_cli_tool_names.push_back(tool.name);
    }
    if (!std::all_of(expected_cli_tool_names.begin(), expected_cli_tool_names.end(),
            [&actual_cli_tool_names](const std::string & expected) {
                return std::find(actual_cli_tool_names.begin(), actual_cli_tool_names.end(),
                    expected) != actual_cli_tool_names.end();
            })) {
        std::cerr << "CLI host config did not expose OpenAPI tools\n";
        return 1;
    }
    const auto list_operation_it = std::find_if(catalog_check.operations.begin(),
        catalog_check.operations.end(), [](const agent_openapi_operation & operation) {
            const auto schema = nlohmann::json::parse(operation.result_schema_json,
                nullptr, false);
            return schema.is_object() && schema.value("type", "") == "array";
        });
    if (list_operation_it == catalog_check.operations.end()) {
        std::cerr << "catalog did not expose a collection operation\n";
        return 1;
    }
    const auto & list_operation = *list_operation_it;
    const auto complex_operation_it = std::find_if(catalog_check.operations.begin(),
        catalog_check.operations.end(), [&list_operation](const agent_openapi_operation & operation) {
            return &operation != &list_operation;
        });
    if (complex_operation_it == catalog_check.operations.end()) {
        std::cerr << "catalog did not expose a second operation\n";
        return 1;
    }
    const auto & complex_operation = *complex_operation_it;
    const auto list_tool_name = agent_openapi_exposed_tool_name(catalog_check, list_operation);
    auto list = selection.tool_view->call({"list", list_tool_name, "{}"}, error);
    if (!list.ok || list.dataset_refs.size() != 1 || list.resource_refs.size() != 1) {
        std::cerr << "collection materialization failed: " << error
                  << " failure=" << list.failure_code << "\n";
        return 1;
    }
    if (list.dataset_refs.front().source_resource_uri != list.resource_refs.front().uri ||
            list.dataset_refs.front().source_provider != request.openapi_providers.front().id ||
            list.dataset_refs.front().source_operation != list_operation.operation_id ||
            list.dataset_refs.front().source_request_json != "{}" ||
            list.dataset_refs.front().retrieved_at <= 0 ||
            list.dataset_refs.front().content_hash.empty() ||
            list.content_json.find("materialized") == std::string::npos) {
        std::cerr << "dataset provenance or compact result missing\n";
        return 1;
    }
    agent_resource_descriptor source_descriptor;
    auto authority = make_agent_resource_read_authority(
        selection.tooling.resource_runtime, std::time(nullptr));
    if (!resource_store.stat(
                list.resource_refs.front().uri, authority, source_descriptor, error) ||
            source_descriptor.scope != common_runtime_resource_scope::turn ||
            source_descriptor.turn_id != "turn-1" ||
            source_descriptor.session_id != "session-1") {
        std::cerr << "source resource scope validation failed: " << error << "\n";
        return 1;
    }
    auto query_result = selection.tool_view->call({"query", "data.query",
        std::string("{\"dataset\":\"") + list.dataset_refs.front().uri + "\"}"}, error);
    if (!query_result.ok || data.last_dataset != list.dataset_refs.front().uri || data.rows.size() != 2) {
        std::cerr << "dataset dataflow failed: " << error << "\n";
        return 1;
    }
    auto wrong_scope_request = request;
    wrong_scope_request.tool_context.turn_id = "turn-3";
    wrong_scope_request.tool_context.scope.turn_id = "turn-3";
    common_agent_cli_tool_selection wrong_scope_selection;
    if (!resolve_agent_host_tool_selection(memory, nullptr, &resource_store, nullptr,
            profile.id, wrong_scope_request, query, nullptr,
            wrong_scope_selection, error) || !wrong_scope_selection.tool_view) {
        std::cerr << "wrong-scope selection failed to resolve: " << error << "\n";
        return 1;
    }
    const auto wrong_scope_result = wrong_scope_selection.tool_view->call({
        "wrong-scope", "data.query",
        std::string("{\"dataset\":\"") + list.dataset_refs.front().uri + "\"}"}, error);
    // OpenAPI imports are host-registered input datasets, not turn-derived
    // datasets. The data adapter intentionally allows such backend-owned refs
    // across turns; only explicitly derived dataset URIs are turn-scoped.
    if (!wrong_scope_result.ok) {
        std::cerr << "host-registered OpenAPI dataset was rejected in the next turn: "
                  << wrong_scope_result.failure_code << " " << wrong_scope_result.raw_diagnostic << "\n";
        return 1;
    }

    const auto complex_tool_name = agent_openapi_exposed_tool_name(catalog_check, complex_operation);
    auto complex = selection.tool_view->call({"complex", complex_tool_name, "{}"}, error);
    if (!complex.ok || !complex.dataset_refs.empty() || complex.resource_refs.size() != 1 ||
            complex.content_json.find("items") == std::string::npos) {
        std::cerr << "complex JSON projection failed: " << error << "\n";
        return 1;
    }

    std::filesystem::remove(spec_path);
    std::cout << "agent-cli-openapi-host-smoke: ok\n";
}
