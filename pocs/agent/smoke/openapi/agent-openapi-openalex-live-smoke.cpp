#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/openapi/agent-openapi-http.h"
#include "tools/agent/openapi/agent-openapi-provider.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::string spec_path_from_args(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--spec") return argv[i + 1];
    }
    const char * configured = std::getenv("OPENALEX_OPENAPI_SPEC");
    return configured == nullptr ? std::string() : configured;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string spec_path = spec_path_from_args(argc, argv);
    if (spec_path.empty()) {
        std::cerr << "usage: llama-agent-openapi-openalex-live-smoke --spec PATH\n";
        return 2;
    }

    nlohmann::json document;
    std::ifstream input(spec_path);
    if (!input || !(input >> document)) {
        std::cerr << "could not read OpenAlex smoke spec: " << spec_path << "\n";
        return 1;
    }

    agent_host_openapi_provider_config config;
    config.id = "openalex";
    config.base_url = "https://api.openalex.org";
    config.prefix = "openalex";
    config.access = "read_only";
    config.exposure = "auto";
    config.auth.type = "none";
    config.connect_timeout_ms = 5000;
    config.request_timeout_ms = 30000;
    config.max_result_bytes = 2 * 1024 * 1024;

    agent_openapi_catalog catalog;
    std::string error;
    if (!build_agent_openapi_catalog(document, config, catalog, error)) {
        std::cerr << "OpenAlex catalog failed: " << error << "\n";
        return 1;
    }
    const auto operation = std::find_if(catalog.operations.begin(), catalog.operations.end(),
        [](const agent_openapi_operation & candidate) {
            return candidate.operation_id == "listWorks";
        });
    if (operation == catalog.operations.end() ||
            std::find(operation->query_parameters.begin(), operation->query_parameters.end(), "search") ==
                operation->query_parameters.end() ||
            std::find(operation->query_parameters.begin(), operation->query_parameters.end(), "per_page") ==
                operation->query_parameters.end() ||
            std::find(operation->query_parameters.begin(), operation->query_parameters.end(), "select") ==
                operation->query_parameters.end()) {
        std::cerr << "OpenAlex catalog did not expose referenced query parameters\n";
        return 1;
    }

    agent_openapi_tool_provider provider(
        std::move(catalog), make_agent_openapi_http_executor(config));
    agent_tool_context context;
    context.allow_network = true;
    context.default_timeout_ms = config.request_timeout_ms;
    context.default_max_result_bytes = config.max_result_bytes;
    auto view = provider.resolve_tools(context, error);
    if (!view) {
        std::cerr << "OpenAlex provider view failed: " << error << "\n";
        return 1;
    }
    const auto result = view->call({
        "openalex-live-1",
        "openalex.listWorks",
        R"({"search":"machine learning","per_page":1,"select":"id,display_name"})"
    }, error);
    if (!result.ok) {
        std::cerr << "OpenAlex live request failed: " << error << "\n";
        return 1;
    }
    const auto envelope = nlohmann::json::parse(result.content_json, nullptr, false);
    const auto response = envelope.is_object() && envelope.contains("result")
        ? envelope["result"] : envelope;
    if (response.is_discarded() || !response.is_object() ||
            !response.contains("meta") || !response["meta"].is_object() ||
            !response.contains("results") || !response["results"].is_array() ||
            response["results"].empty() || !response["results"][0].is_object() ||
            !response["results"][0].contains("id") ||
            !response["results"][0].contains("display_name")) {
        std::cerr << "OpenAlex response shape was not the expected list envelope: "
                  << result.content_json.substr(0, 1000) << "\n";
        return 1;
    }
    std::cout << "openalex_list_works=passed\n";
    std::cout << "openalex_first_id=" << response["results"][0]["id"].get<std::string>() << "\n";
    return 0;
}
