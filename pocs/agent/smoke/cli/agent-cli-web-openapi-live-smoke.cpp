#include "tools/agent/cli/agent-cli-host-adapter.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/openapi/dynamic/agent-openapi-dynamic-admission.h"

#include "memory/memory-in-memory.h"
#include "agent/data-store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

constexpr const char * k_default_spec_url =
    "https://raw.githubusercontent.com/peterkallden/llama/kallden/agent-model-adaptation/"
    "docs/examples/openalex-works-openapi.json";
constexpr const char * k_default_openalex_host_config =
    "docs/examples/agent-host-config-openalex.json";

class live_smoke_data_store final : public common_agent_data_store {
public:
    bool list_dataset_descriptors(
            std::vector<common_agent_dataset_descriptor> & descriptors,
            std::string & error) override {
        descriptors.clear();
        error.clear();
        return true;
    }

    bool execute(const std::string &, const std::string &, std::string & result,
            std::string & error) override {
        result.clear();
        error = "dataset operations are outside this live smoke";
        return false;
    }
};

bool resolve_selection(
        common_memory_store & memory,
        live_smoke_data_store & data,
        agent_resource_store & resource_store,
        agent_host_tool_selection_request & request,
        common_agent_cli_tool_selection & selection,
        std::string & error) {
    common_memory_query query;
    request.data_store = &data;
    return resolve_agent_host_tool_selection(
        memory, nullptr, &resource_store, nullptr, request.tool_context.profile_id,
        request, query, nullptr, selection, error);
}

bool configure_request(agent_host_tool_selection_request & request) {
    request.resource_store_config.blob_backend = "in-memory";
    request.resource_store_config.metadata_backend = "in-memory";
    request.tool_context.profile_id = "web-openapi-live";
    request.tool_context.allow_network = true;
    request.tool_context.default_timeout_ms = 60000;
    request.tool_context.scope.namespace_id = "local";
    request.tool_context.scope.session_id = "web-openapi-live-session";
    request.tool_context.scope.project_id = "web-openapi-live-project";
    request.tool_context.scope.turn_id = "turn-1";

    common_tool_profile profile;
    profile.id = "web-openapi-live";
    profile.members = {
        {"web.fetch", 1, true, "{}"},
        {"resource.read", 1, true, "{}"},
        {"openapi.connect", 1, true, "{}"},
    };
    profile.allow_network = true;
    request.tool_profiles.emplace(profile.id, std::move(profile));
    request.dynamic_openapi_registry = std::make_shared<agent_openapi_dynamic_registry>();
    return true;
}

bool run_resource_fetch_smoke(
        common_memory_store & memory,
        live_smoke_data_store & data,
        agent_resource_store & resource_store,
        agent_host_tool_selection_request request) {
    std::string error;
    common_agent_cli_tool_selection selection;
    if (!resolve_selection(memory, data, resource_store, request, selection, error)) {
        std::cerr << "resource smoke tool resolution failed: " << error << "\n";
        return false;
    }
    const auto fetched = selection.tool_view->call({
        "web-resource-live", "web.fetch",
        R"({"url":"https://www.iana.org/domains/example","max_bytes":65536})"}, error);
    if (!fetched.ok || fetched.resource_refs.size() != 1) {
        std::cerr << "live web.fetch did not return a host resource: " << error << " "
                  << fetched.raw_diagnostic << "\n";
        return false;
    }
    const auto read = selection.tool_view->call({
        "web-resource-read", "resource.read",
        nlohmann::json({{"uri", fetched.resource_refs.front().uri},
            {"representation", "text"}, {"max_bytes", 8192}}).dump()}, error);
    if (!read.ok || read.content_json.empty() ||
            read.content_json.find("IANA") == std::string::npos) {
        std::cerr << "live fetched resource could not be read back: " << error << " "
                  << read.content_json.substr(0, 600) << "\n";
        return false;
    }
    std::cout << "web_resource_fetch_and_read=passed\n";
    return true;
}

bool run_openapi_bootstrap_smoke(
        common_memory_store & memory,
        live_smoke_data_store & data,
        agent_resource_store & resource_store,
        agent_host_tool_selection_request request,
        const std::string & spec_url) {
    std::string error;
    common_agent_cli_tool_selection first_turn;
    if (!resolve_selection(memory, data, resource_store, request, first_turn, error)) {
        std::cerr << "OpenAPI smoke initial tool resolution failed: " << error << "\n";
        return false;
    }
    const auto fetched = first_turn.tool_view->call({
        "openapi-spec-live", "web.fetch",
        nlohmann::json({{"url", spec_url}, {"max_bytes", 500000}}).dump()}, error);
    if (!fetched.ok || fetched.resource_refs.size() != 1) {
        std::cerr << "live OpenAPI spec fetch failed: " << error << " "
                  << fetched.raw_diagnostic << "\n";
        return false;
    }
    const auto resource_read = first_turn.tool_view->call({
        "openapi-spec-read", "resource.read",
        nlohmann::json({{"uri", fetched.resource_refs.front().uri},
            {"representation", "text"}, {"max_bytes", 32768}}).dump()}, error);
    if (!resource_read.ok || resource_read.content_json.find("OpenAlex Works smoke contract") ==
            std::string::npos) {
        std::cerr << "fetched OpenAPI resource did not contain the expected contract: "
                  << error << " " << resource_read.content_json.substr(0, 600) << "\n";
        return false;
    }

    const auto connected = first_turn.tool_view->call({
        "openapi-connect-live", "openapi.connect",
        nlohmann::json({{"spec_resource", fetched.resource_refs.front().uri}}).dump()}, error);
    if (!connected.ok || connected.content_json.find("available_next_turn") == std::string::npos) {
        std::cerr << "live OpenAPI spec admission failed: " << error << " "
                  << connected.raw_diagnostic << "\n";
        return false;
    }

    request.tool_context.turn_id = "turn-2";
    request.tool_context.scope.turn_id = "turn-2";
    common_agent_cli_tool_selection second_turn;
    if (!resolve_selection(memory, data, resource_store, request, second_turn, error)) {
        std::cerr << "OpenAPI smoke next-turn provider resolution failed: " << error << "\n";
        return false;
    }
    const auto registrations = request.dynamic_openapi_registry->snapshot();
    if (registrations.size() != 1) {
        std::cerr << "OpenAPI smoke expected one dynamic provider, got "
                  << registrations.size() << "\n";
        return false;
    }
    const std::string tool_name = registrations.front().catalog.prefix + ".listWorks";
    const auto read_capability = second_turn.tooling.capability_tools.find("openapi.read");
    if (!second_turn.tool_view->exposes_tool(tool_name) ||
            read_capability == second_turn.tooling.capability_tools.end() ||
            std::find(read_capability->second.begin(), read_capability->second.end(), tool_name) ==
                read_capability->second.end()) {
        std::cerr << "OpenAPI operation was not exposed and bound to openapi.read\n";
        return false;
    }
    const auto result = second_turn.tool_view->call({
        "openalex-api-live", tool_name,
        R"({"search":"machine learning","per_page":1,"select":"id,display_name"})"}, error);
    if (!result.ok || result.resource_refs.size() != 1) {
        std::cerr << "live OpenAlex operation failed"
                  << " failure_code=" << result.failure_code
                  << " failure_class=" << common_tool_failure_class_name(result.failure_class)
                  << " retryable=" << (result.retryable ? "true" : "false")
                  << " safe_summary=" << result.safe_summary
                  << " error=" << error
                  << " raw_diagnostic=" << result.raw_diagnostic
                  << " response=" << result.content_json.substr(0, 800) << "\n";
        return false;
    }
    auto response = nlohmann::json::parse(result.content_json, nullptr, false);
    if (response.is_object() && response.contains("result")) response = response["result"];
    if (!response.is_object() || !response.contains("results") ||
            !response["results"].is_array() || response["results"].empty() ||
            !response["results"][0].contains("id")) {
        std::cerr << "OpenAlex response shape was unexpected: "
                  << result.content_json.substr(0, 800) << "\n";
        return false;
    }
    std::cout << "web_fetch_openapi_connect_live_api=passed\n";
    return true;
}

bool run_configured_openalex_smoke(
        common_memory_store & memory,
        live_smoke_data_store & data,
        agent_resource_store & resource_store,
        const std::string & config_path) {
    std::string error;
    agent_host_config config;
    if (!load_agent_host_config(config_path, config, error)) {
        std::cerr << "configured OpenAlex smoke could not load host config: " << error << "\n";
        return false;
    }
    if (config.openapi_providers.size() != 1 || config.openapi_providers.front().id != "openalex") {
        std::cerr << "configured OpenAlex smoke expected one openalex provider\n";
        return false;
    }

    agent_host_tool_selection_request request;
    request.resource_store_config.blob_backend = "in-memory";
    request.resource_store_config.metadata_backend = "in-memory";
    request.tool_context.profile_id = config.tool_profile;
    request.tool_context.allow_network = true;
    request.tool_context.default_timeout_ms = 60000;
    request.tool_context.scope.namespace_id = "local";
    request.tool_context.scope.session_id = "configured-openalex-session";
    request.tool_context.scope.project_id = "configured-openalex-project";
    request.tool_context.scope.turn_id = "turn-1";
    request.openapi_providers = config.openapi_providers;
    request.tool_profiles = config.tool_profiles;
    request.tool_capabilities = config.tool_capabilities;
    request.tool_family_descriptions = config.tool_family_descriptions;

    common_agent_cli_tool_selection selection;
    if (!resolve_selection(memory, data, resource_store, request, selection, error) || !selection.tool_view) {
        std::cerr << "configured OpenAlex provider resolution failed: " << error << "\n";
        return false;
    }
    const auto read_capability = selection.tooling.capability_tools.find("openapi.read");
    const bool has_openalex_context = std::find(selection.tooling.available_context.begin(),
        selection.tooling.available_context.end(), "context.openapi.available") !=
        selection.tooling.available_context.end();
    if (!selection.tool_view->exposes_tool("openalex.listWorks") ||
            !selection.tool_view->exposes_tool("openalex.getWork") ||
            read_capability == selection.tooling.capability_tools.end() ||
            std::find(read_capability->second.begin(), read_capability->second.end(),
                "openalex.listWorks") == read_capability->second.end() || !has_openalex_context) {
        std::cerr << "configured OpenAlex provider was not exposed as an automatic read-only tool view\n";
        return false;
    }
    const auto result = selection.tool_view->call({
        "configured-openalex-list", "openalex.listWorks",
        R"({"search":"machine learning","per_page":1,"select":"id,display_name"})"}, error);
    if (!result.ok || result.resource_refs.size() != 1) {
        std::cerr << "configured OpenAlex operation failed"
                  << " failure_code=" << result.failure_code
                  << " failure_class=" << common_tool_failure_class_name(result.failure_class)
                  << " retryable=" << (result.retryable ? "true" : "false")
                  << " safe_summary=" << result.safe_summary
                  << " error=" << error
                  << " raw_diagnostic=" << result.raw_diagnostic << "\n";
        return false;
    }
    std::cout << "configured_openalex_provider_and_api=passed\n";
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string mode;
    std::string spec_url = k_default_spec_url;
    std::string config_path = k_default_openalex_host_config;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (argument == "--spec-url" && i + 1 < argc) spec_url = argv[++i];
        else if (argument == "--config" && i + 1 < argc) config_path = argv[++i];
        else {
            std::cerr << "usage: llama-agent-web-openapi-live-smoke --mode resource|openapi|configured [--spec-url URL] [--config PATH]\n";
            return 2;
        }
    }
    if (mode != "resource" && mode != "openapi" && mode != "configured") {
        std::cerr << "choose --mode resource, openapi or configured\n";
        return 2;
    }

    common_memory_in_memory_store memory;
    std::string error;
    if (!memory.open("", error)) {
        std::cerr << "memory setup failed: " << error << "\n";
        return 1;
    }
    live_smoke_data_store data;
    agent_catalogued_resource_store resource_store(
        std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    bool passed = false;
    if (mode == "configured") {
        passed = run_configured_openalex_smoke(memory, data, resource_store, config_path);
    } else {
        agent_host_tool_selection_request request;
        configure_request(request);
        passed = mode == "resource"
            ? run_resource_fetch_smoke(memory, data, resource_store, std::move(request))
            : run_openapi_bootstrap_smoke(memory, data, resource_store, std::move(request), spec_url);
    }
    return passed ? 0 : 1;
}
