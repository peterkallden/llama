#include "tools/agent/cli/agent-cli-host-adapter.h"
#include "tools/agent/resource/agent-resource-store.h"

#include "memory/memory-in-memory.h"
#include "tools/agent/openapi/agent-openapi-catalog.h"
#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <thread>

class http_server_guard final {
public:
    explicit http_server_guard(httplib::Server & server)
        : server_(server), thread_([this] { server_.listen_after_bind(); }) {
        // The client can otherwise win the race with the listener startup on
        // Windows, turning a connection failure into an unrelated teardown.
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    ~http_server_guard() {
        stop();
    }

    void stop() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    http_server_guard(const http_server_guard &) = delete;
    http_server_guard & operator=(const http_server_guard &) = delete;

private:
    httplib::Server & server_;
    std::thread thread_;
};

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
    httplib::Server server;
    server.Get("/sales", [](const httplib::Request &, httplib::Response & response) {
        response.set_content(R"([{"id":1,"amount":12.5},{"id":2,"amount":8}])", "application/json");
    });
    server.Get("/complex", [](const httplib::Request &, httplib::Response & response) {
        response.set_content(R"({"items":[{"id":1}]})", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) { std::cerr << "could not bind HTTP test server\n"; return 1; }
    http_server_guard server_guard(server);

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
    profile.members = {{"data.query", 1, true, "{}"}};
    request.tool_profiles.emplace(profile.id, profile);
    agent_host_openapi_provider_config openapi;
    openapi.id = "sales-api";
    openapi.enabled = true;
    openapi.required = true;
    openapi.spec_path = spec_path.string();
    openapi.base_url = "http://127.0.0.1:" + std::to_string(port);
    openapi.prefix = "sales";
    openapi.access = "read_only";
    openapi.exposure = "auto";
    openapi.allow_private_network = true;
    openapi.connect_timeout_ms = 1000;
    openapi.request_timeout_ms = 2000;
    openapi.max_result_bytes = 1024 * 1024;
    request.openapi_providers.push_back(std::move(openapi));
    agent_openapi_catalog catalog_check;
    nlohmann::json spec_check;
    std::ifstream spec_check_file(spec_path);
    spec_check_file >> spec_check;
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
    bool has_list = false;
    bool has_complex = false;
    for (const auto & tool : selection.tool_view->chat_tools()) {
        has_list = has_list || tool.name == "sales.listSales";
        has_complex = has_complex || tool.name == "sales.complex";
    }
    if (!has_list || !has_complex) {
        std::cerr << "expected OpenAPI tools were not exposed\n";
        return 1;
    }
    auto list = selection.tool_view->call({"list", "sales.listSales", "{}"}, error);
    if (!list.ok || list.dataset_refs.size() != 1 || list.resource_refs.size() != 1) {
        std::cerr << "collection materialization failed: " << error << "\n";
        return 1;
    }
    if (list.dataset_refs.front().source_resource_uri != list.resource_refs.front().uri ||
            list.dataset_refs.front().source_provider != "sales-api" ||
            list.dataset_refs.front().source_operation != "listSales" ||
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
    wrong_scope_request.tool_context.turn_id = "turn-2";
    wrong_scope_request.tool_context.scope.turn_id = "turn-2";
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
    if (wrong_scope_result.ok || wrong_scope_result.failure_code != "tool.dataset.out_of_scope") {
        std::cerr << "dataset was usable outside its turn scope\n";
        return 1;
    }

    auto complex = selection.tool_view->call({"complex", "sales.complex", "{}"}, error);
    if (!complex.ok || !complex.dataset_refs.empty() || complex.resource_refs.size() != 1 ||
            complex.content_json.find("items") == std::string::npos) {
        std::cerr << "complex JSON projection failed: " << error << "\n";
        return 1;
    }

    // Stop the HTTP thread while the host/tooling objects still exist.  This
    // makes shutdown ordering explicit and avoids a Windows-only teardown
    // failure after an otherwise successful smoke run.
    server_guard.stop();
    std::filesystem::remove(spec_path);
    std::cout << "agent-cli-openapi-host-smoke: ok\n";
}
