#include "tools/agent/cli/agent-cli-host-adapter.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/openapi/dynamic/agent-openapi-dynamic-admission.h"

#include "memory/memory-in-memory.h"
#include "agent/data-store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

namespace {

constexpr const char * k_default_spec_url =
    "https://raw.githubusercontent.com/peterkallden/llama/kallden/agent-model-adaptation/"
    "docs/examples/openalex-works-openapi.json";
constexpr const char * k_default_openalex_host_config =
    "docs/examples/agent-host-config-openalex.json";

class live_smoke_data_store final : public common_agent_data_store {
public:
    bool put_row(const std::string & dataset, const std::string & row_id,
            const std::string & row_json, std::string & error) override {
        rows[dataset][row_id] = row_json;
        error.clear();
        ++rows_written;
        return true;
    }

    bool put_rows(const std::string & dataset,
            const std::vector<std::pair<std::string, std::string>> & batch,
            std::string & error) override {
        ++batches_written;
        for (const auto & row : batch) rows[dataset][row.first] = row.second;
        rows_written += batch.size();
        error.clear();
        return true;
    }

    bool put_dataset_descriptor(const common_agent_dataset_descriptor & descriptor,
            std::string & error) override {
        descriptors[descriptor.ref.uri] = descriptor;
        error.clear();
        return true;
    }

    bool get_dataset_descriptor(const std::string & uri,
            common_agent_dataset_descriptor & descriptor, std::string & error) override {
        const auto found = descriptors.find(uri);
        if (found == descriptors.end()) {
            error = "dataset descriptor not found";
            return false;
        }
        descriptor = found->second;
        error.clear();
        return true;
    }

    bool list_dataset_descriptors(
            std::vector<common_agent_dataset_descriptor> & descriptors,
            std::string & error) override {
        descriptors.clear();
        for (const auto & item : this->descriptors) descriptors.push_back(item.second);
        error.clear();
        return true;
    }

    bool execute(const std::string & operation, const std::string & request_json, std::string & result,
            std::string & error) override {
        if (operation != "data.query") {
            result.clear();
            error = "only bounded data.query is supported by this live smoke store";
            return false;
        }
        const auto request = nlohmann::json::parse(request_json, nullptr, false);
        const auto dataset = request.is_object() ? request.value("dataset", std::string{}) : std::string{};
        const auto descriptor = descriptors.find(dataset);
        const auto dataset_rows = rows.find(dataset);
        if (descriptor == descriptors.end() || dataset_rows == rows.end()) {
            result.clear();
            error = "dataset is unavailable in the live smoke store";
            return false;
        }
        const size_t limit = std::min<size_t>(request.value("max_result_rows", request.value("limit", size_t(20))), 100);
        nlohmann::json columns = nlohmann::json::array();
        for (const auto & column : descriptor->second.columns) columns.push_back(column.name);
        nlohmann::json output_rows = nlohmann::json::array();
        for (const auto & row : dataset_rows->second) {
            if (output_rows.size() >= limit) break;
            output_rows.push_back(nlohmann::json::parse(row.second));
        }
        result = nlohmann::json({{"dataset", dataset}, {"columns", columns},
            {"rows", output_rows}, {"row_count", output_rows.size()},
            {"result_truncated", dataset_rows->second.size() > output_rows.size()}}).dump();
        error.clear();
        return true;
    }

    std::map<std::string, common_agent_dataset_descriptor> descriptors;
    std::map<std::string, std::map<std::string, std::string>> rows;
    size_t rows_written = 0;
    size_t batches_written = 0;
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
    request.tool_context.max_calls = 16;
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
        {"dataset.inspect", 1, true, "{}"},
        {"dataset.sample", 1, true, "{}"},
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

    const auto eurostat = selection.tool_view->call({
        "web-eurostat-jsonstat-live", "web.fetch",
        R"({"url":"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/DEMO_R_D3DENS?lang=EN&geo=SE&time=2020","accept":"application/json","max_bytes":1048576})"}, error);
    if (!eurostat.ok || eurostat.resource_refs.size() != 1 ||
            eurostat.content_json.find("application/json") == std::string::npos) {
        std::cerr << "web.fetch could not acquire Eurostat JSON-stat: " << error << " "
                  << "ok=" << (eurostat.ok ? "true" : "false")
                  << " resource_refs=" << eurostat.resource_refs.size()
                  << " json_mime=" << (eurostat.content_json.find("application/json") != std::string::npos ? "true" : "false") << " "
                  << eurostat.content_json.substr(0, 600) << "\n";
        return false;
    }
    const auto eurostat_read = selection.tool_view->call({
        "web-eurostat-jsonstat-read", "resource.read",
        nlohmann::json({{"uri", eurostat.resource_refs.front().uri},
            {"representation", "text"}, {"max_bytes", 32768}}).dump()}, error);
    const auto eurostat_read_json = nlohmann::json::parse(eurostat_read.content_json, nullptr, false);
    const std::string eurostat_body = eurostat_read_json.is_object() &&
            eurostat_read_json.contains("result") && eurostat_read_json["result"].is_object() &&
            eurostat_read_json["result"].contains("content") && eurostat_read_json["result"]["content"].is_string()
        ? eurostat_read_json["result"]["content"].get<std::string>()
        : std::string{};
    if (!eurostat_read.ok || eurostat_body.find(R"("version":"2.0")") == std::string::npos ||
            eurostat_body.find(R"("dimension")") == std::string::npos) {
        std::cerr << "web.fetch Eurostat resource did not preserve JSON-stat: " << error << " "
                  << eurostat_read.content_json.substr(0, 600) << "\n";
        return false;
    }
    const auto eurostat_catalog = selection.tool_view->call({
        "web-eurostat-catalogue-live", "web.fetch",
        R"({"url":"https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en","accept":"text/plain","max_bytes":4194304})"}, error);
    if (!eurostat_catalog.ok || eurostat_catalog.resource_refs.size() != 1 ||
            eurostat_catalog.content_json.find("text/plain") == std::string::npos) {
        std::cerr << "web.fetch could not acquire Eurostat catalogue TSV: " << error << " "
                  << eurostat_catalog.safe_summary << " " << eurostat_catalog.raw_diagnostic << "\n";
        return false;
    }
    const auto catalogue_read = selection.tool_view->call({
        "web-eurostat-catalogue-read", "resource.read",
        nlohmann::json({{"uri", eurostat_catalog.resource_refs.front().uri},
            {"representation", "text"}, {"max_bytes", 4096}}).dump()}, error);
    const auto catalogue_read_json = nlohmann::json::parse(catalogue_read.content_json, nullptr, false);
    const std::string catalogue_body = catalogue_read_json.is_object() &&
            catalogue_read_json.contains("result") && catalogue_read_json["result"].is_object() &&
            catalogue_read_json["result"].contains("content") && catalogue_read_json["result"]["content"].is_string()
        ? catalogue_read_json["result"]["content"].get<std::string>()
        : std::string{};
    if (!catalogue_read.ok || catalogue_body.find("\"title\"\t\"code\"\t\"type\"") == std::string::npos) {
        std::cerr << "Eurostat catalogue resource did not return the expected bounded TSV: " << error << " "
                  << catalogue_read.content_json.substr(0, 600) << "\n";
        return false;
    }

    const std::string catalogue_resource = eurostat_catalog.resource_refs.front().uri;
    const size_t eurostat_catalogue_resource_size = eurostat_catalog.resource_refs.front().size_bytes;
    const size_t rows_before_catalogue = data.rows_written;
    const size_t batches_before_catalogue = data.batches_written;
    const auto first_inspect_start = std::chrono::steady_clock::now();
    const auto first_inspect = selection.tool_view->call({
        "dataset-inspect-eurostat-catalogue", "dataset.inspect",
        nlohmann::json({{"resource", catalogue_resource}}).dump()}, error);
    const auto first_inspect_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - first_inspect_start).count();
    auto first_inspect_json = nlohmann::json::parse(first_inspect.content_json, nullptr, false);
    if (first_inspect_json.is_object() && first_inspect_json.contains("result")) {
        first_inspect_json = first_inspect_json["result"];
    }
    if (!first_inspect.ok || !first_inspect_json.is_object() ||
            !first_inspect_json.contains("dataset") || !first_inspect_json["dataset"].is_string() ||
            !first_inspect_json.contains("rows") || !first_inspect_json["rows"].is_number_integer() ||
            first_inspect_json["rows"].get<size_t>() < 1000 || data.rows_written == rows_before_catalogue ||
            data.batches_written == batches_before_catalogue) {
        std::cerr << "production dataset.inspect did not materialize the fetched Eurostat catalogue: "
                  << first_inspect.safe_summary << " " << first_inspect.raw_diagnostic << " "
                  << first_inspect.content_json.substr(0, 800) << "\n";
        return false;
    }
    const auto imported_dataset = first_inspect_json["dataset"].get<std::string>();
    const size_t catalogue_row_count = first_inspect_json["rows"].get<size_t>();
    const size_t initial_rows_written = data.rows_written;
    const size_t initial_batches_written = data.batches_written;

    const auto second_inspect_start = std::chrono::steady_clock::now();
    const auto second_inspect = selection.tool_view->call({
        "dataset-inspect-eurostat-catalogue-repeat", "dataset.inspect",
        nlohmann::json({{"resource", catalogue_resource}}).dump()}, error);
    const auto second_inspect_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - second_inspect_start).count();
    auto second_inspect_json = nlohmann::json::parse(second_inspect.content_json, nullptr, false);
    if (second_inspect_json.is_object() && second_inspect_json.contains("result")) {
        second_inspect_json = second_inspect_json["result"];
    }
    if (!second_inspect.ok || !second_inspect_json.is_object() ||
            second_inspect_json.value("dataset", std::string{}) != imported_dataset ||
            data.rows_written != initial_rows_written || data.batches_written != initial_batches_written) {
        std::cerr << "repeat dataset.inspect failed to reuse the complete catalogue materialization: "
                  << second_inspect.safe_summary << " " << second_inspect.raw_diagnostic << " "
                  << second_inspect.content_json.substr(0, 800) << "\n";
        return false;
    }

    const auto sample = selection.tool_view->call({
        "dataset-sample-eurostat-catalogue", "dataset.sample",
        nlohmann::json({{"resource", catalogue_resource}, {"rows", 3}}).dump()}, error);
    auto sample_json = nlohmann::json::parse(sample.content_json, nullptr, false);
    if (sample_json.is_object() && sample_json.contains("result")) sample_json = sample_json["result"];
    if (!sample.ok || !sample_json.is_object() || !sample_json.contains("rows") ||
            !sample_json["rows"].is_array() || sample_json["rows"].empty() ||
            !sample_json["rows"][0].contains("code")) {
        std::cerr << "dataset.sample could not query the shared Eurostat materialization: "
                  << sample.safe_summary << " " << sample.raw_diagnostic << " "
                  << sample.content_json.substr(0, 800) << "\n";
        return false;
    }
    std::cout << "web_resource_fetch_and_read=passed\n";
    std::cout << "web_eurostat_jsonstat_fetch_and_read=passed\n";
    std::cout << "web_eurostat_catalogue_tsv_fetch_and_read=passed\n";
    std::cout << "eurostat_catalogue_resource_bytes=" << eurostat_catalogue_resource_size << "\n";
    std::cout << "eurostat_catalogue_rows=" << catalogue_row_count << "\n";
    std::cout << "eurostat_catalogue_first_import_rows=" << initial_rows_written - rows_before_catalogue << "\n";
    std::cout << "eurostat_catalogue_first_import_batches=" << initial_batches_written - batches_before_catalogue << "\n";
    std::cout << "eurostat_catalogue_first_inspect_us=" << first_inspect_us << "\n";
    std::cout << "eurostat_catalogue_repeat_inspect_us=" << second_inspect_us << "\n";
    std::cout << "eurostat_catalogue_repeat_import_rows=0\n";
    std::cout << "eurostat_catalogue_sample=passed\n";
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
    const std::vector<std::string> expected_openalex_tools = {
        "openalex.listWorks", "openalex.getWork",
        "openalex.listAuthors", "openalex.getAuthor",
        "openalex.listInstitutions", "openalex.getInstitution",
        "openalex.listTopics", "openalex.getTopic",
    };
    const auto unexpected_openalex_tool = std::find_if(
        selection.tooling.tools.begin(), selection.tooling.tools.end(),
        [&](const common_chat_tool & tool) {
            return tool.name.rfind("openalex.", 0) == 0 &&
                std::find(expected_openalex_tools.begin(), expected_openalex_tools.end(), tool.name) ==
                    expected_openalex_tools.end();
        });
    if (std::any_of(expected_openalex_tools.begin(), expected_openalex_tools.end(),
                [&](const std::string & name) { return !selection.tool_view->exposes_tool(name); }) ||
            unexpected_openalex_tool != selection.tooling.tools.end() ||
            read_capability == selection.tooling.capability_tools.end() ||
            std::find(read_capability->second.begin(), read_capability->second.end(),
                "openalex.listTopics") == read_capability->second.end() || !has_openalex_context) {
        std::cerr << "configured OpenAlex provider did not expose exactly the intended read-only research surface\n";
        return false;
    }
    const auto works_tool = std::find_if(selection.tooling.tools.begin(), selection.tooling.tools.end(),
        [](const common_chat_tool & tool) { return tool.name == "openalex.listWorks"; });
    const auto works_schema = works_tool == selection.tooling.tools.end()
        ? nlohmann::json::object()
        : nlohmann::json::parse(works_tool->parameters, nullptr, false);
    const auto properties = works_schema.is_object()
        ? works_schema.value("properties", nlohmann::json::object())
        : nlohmann::json::object();
    const std::string search_description = properties.value("search", nlohmann::json::object())
        .value("description", "");
    const std::string filter_description = properties.value("filter", nlohmann::json::object())
        .value("description", "");
    if (works_tool == selection.tooling.tools.end() ||
            search_description.find("titles") == std::string::npos ||
            filter_description.find("topics.id:T") == std::string::npos ||
            filter_description.find("publication_year:2024") == std::string::npos) {
        std::cerr << "configured OpenAlex model-facing parameter schemas omitted title or topic filter guidance\n"
                  << "  search: " << search_description << "\n"
                  << "  filter: " << filter_description << "\n";
        return false;
    }
    const auto result = selection.tool_view->call({
        "configured-openalex-list", "openalex.listWorks",
        R"({"search":"machine learning","per_page":1,"select":"id,display_name,publication_year,primary_topic,topics"})"}, error);
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

bool run_configured_openalex_full_text_smoke(
        common_memory_store & memory,
        live_smoke_data_store & data,
        agent_resource_store & resource_store,
        const std::string & config_path) {
    std::string error;
    agent_host_config config;
    if (!load_agent_host_config(config_path, config, error)) {
        std::cerr << "configured OpenAlex full-text smoke could not load host config: " << error << "\n";
        return false;
    }

    agent_host_tool_selection_request request;
    request.resource_store_config.blob_backend = "in-memory";
    request.resource_store_config.metadata_backend = "in-memory";
    request.tool_context.profile_id = config.tool_profile;
    request.tool_context.allow_network = true;
    request.tool_context.default_timeout_ms = 60000;
    request.tool_context.scope.namespace_id = "local";
    request.tool_context.scope.session_id = "configured-openalex-full-text-session";
    request.tool_context.scope.project_id = "configured-openalex-full-text-project";
    request.tool_context.scope.turn_id = "turn-1";
    request.openapi_providers = config.openapi_providers;
    request.tool_profiles = config.tool_profiles;
    request.tool_capabilities = config.tool_capabilities;
    request.tool_family_descriptions = config.tool_family_descriptions;

    common_agent_cli_tool_selection selection;
    if (!resolve_selection(memory, data, resource_store, request, selection, error) || !selection.tool_view) {
        std::cerr << "configured OpenAlex full-text tool resolution failed: " << error << "\n";
        return false;
    }
    for (const auto & name : {"openalex.getWork", "web.fetch", "resource.inspect", "resource.read"}) {
        if (!selection.tool_view->exposes_tool(name)) {
            std::cerr << "configured OpenAlex full-text smoke omitted tool " << name << "\n";
            return false;
        }
    }

    // This OpenAlex record currently declares an openly reachable PDF. The
    // smoke intentionally derives the URL from the returned work record and
    // never hard-codes it into web.fetch.
    const auto record_result = selection.tool_view->call({
        "configured-openalex-full-text-work", "openalex.getWork", R"({"id":"W3038568908"})"}, error);
    if (!record_result.ok || record_result.resource_refs.size() != 1) {
        std::cerr << "configured OpenAlex full-text work lookup failed: "
                  << record_result.safe_summary << " " << record_result.raw_diagnostic << "\n";
        return false;
    }
    auto record = nlohmann::json::parse(record_result.content_json, nullptr, false);
    if (record.is_object() && record.contains("result")) record = record["result"];
    const auto location = record.value("best_oa_location", nlohmann::json::object());
    const auto pdf_url = location.value("pdf_url", std::string{});
    if (!record.is_object() || !record.contains("abstract_inverted_index") ||
            !record.contains("abstract_text") || pdf_url.empty()) {
        std::cerr << "configured OpenAlex full-text work did not expose a readable abstract and open PDF location: "
                  << record_result.content_json.substr(0, 1000) << "\n";
        return false;
    }

    const auto fetched = selection.tool_view->call({
        "configured-openalex-full-text-fetch", "web.fetch",
        nlohmann::json({{"url", pdf_url}, {"max_bytes", 8 * 1024 * 1024}}).dump()}, error);
    if (!fetched.ok || fetched.resource_refs.size() != 1) {
        std::cerr << "OpenAlex-returned full-text URL could not be fetched: "
                  << fetched.safe_summary << " " << fetched.raw_diagnostic << "\n";
        return false;
    }
    const auto inspected = selection.tool_view->call({
        "configured-openalex-full-text-inspect", "resource.inspect",
        nlohmann::json({{"uri", fetched.resource_refs.front().uri}}).dump()}, error);
    if (!inspected.ok || inspected.content_json.find("application/pdf") == std::string::npos) {
        std::cerr << "OpenAlex full text was not preserved as a PDF host resource: "
                  << inspected.content_json.substr(0, 1000) << "\n";
        return false;
    }
    const auto text = selection.tool_view->call({
        "configured-openalex-full-text-read", "resource.read",
        nlohmann::json({{"uri", fetched.resource_refs.front().uri},
            {"representation", "text"}, {"max_bytes", 8192}}).dump()}, error);
    if (text.ok && !text.content_json.empty()) {
        std::cout << "configured_openalex_full_text_representation=passed\n";
    } else if (text.failure_code == "tool.resource_read.representation_unavailable") {
        // This proves the safe boundary for a PDF without an extractable text
        // layer. The abstract is still available above; callers must report
        // this limitation rather than infer unobserved sections.
        std::cout << "configured_openalex_full_text_representation=unavailable\n";
    } else {
        std::cerr << "OpenAlex PDF text materialization failed unexpectedly: "
                  << text.safe_summary << " " << text.raw_diagnostic << "\n";
        return false;
    }
    std::cout << "configured_openalex_abstract_to_full_text_resource=passed\n";
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
            std::cerr << "usage: llama-agent-web-openapi-live-smoke --mode resource|openapi|configured|openalex-fulltext [--spec-url URL] [--config PATH]\n";
            return 2;
        }
    }
    if (mode != "resource" && mode != "openapi" && mode != "configured" && mode != "openalex-fulltext") {
        std::cerr << "choose --mode resource, openapi, configured or openalex-fulltext\n";
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
    } else if (mode == "openalex-fulltext") {
        passed = run_configured_openalex_full_text_smoke(memory, data, resource_store, config_path);
    } else {
        agent_host_tool_selection_request request;
        configure_request(request);
        passed = mode == "resource"
            ? run_resource_fetch_smoke(memory, data, resource_store, std::move(request))
            : run_openapi_bootstrap_smoke(memory, data, resource_store, std::move(request), spec_url);
    }
    return passed ? 0 : 1;
}
