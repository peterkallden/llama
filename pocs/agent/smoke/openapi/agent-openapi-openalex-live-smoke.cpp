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

nlohmann::json response_body(const std::string & content_json) {
    auto envelope = nlohmann::json::parse(content_json, nullptr, false);
    if (envelope.is_object() && envelope.contains("result")) return envelope["result"];
    return envelope;
}

bool call_list(
        agent_tool_view & view,
        const std::string & tool_name,
        const std::string & arguments,
        nlohmann::json & response,
        std::string & error) {
    const auto result = view.call({"openalex-live", tool_name, arguments}, error);
    if (!result.ok) {
        std::cerr << tool_name << " failed: " << error << "\n";
        return false;
    }
    response = response_body(result.content_json);
    if (!response.is_object() || !response.contains("meta") ||
            !response["meta"].is_object() || !response.contains("results") ||
            !response["results"].is_array()) {
        std::cerr << tool_name << " returned an unexpected list envelope: "
                  << result.content_json.substr(0, 1000) << "\n";
        return false;
    }
    return true;
}

std::string short_openalex_id(const nlohmann::json & records) {
    if (!records.is_array() || records.empty() || !records.front().is_object() ||
            !records.front().contains("id") || !records.front()["id"].is_string()) return {};
    const std::string full_id = records.front()["id"].get<std::string>();
    const size_t slash = full_id.find_last_of('/');
    return slash == std::string::npos ? full_id : full_id.substr(slash + 1);
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
    config.exposure = "include";
    for (const auto * operation_id : {"listWorks", "getWork", "listAuthors", "getAuthor",
            "listInstitutions", "getInstitution", "listTopics", "getTopic"}) {
        config.operations.emplace(operation_id, agent_host_openapi_operation_policy{});
    }
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
            std::find(operation->query_parameters.begin(), operation->query_parameters.end(), "filter") ==
                operation->query_parameters.end() ||
            std::find(operation->query_parameters.begin(), operation->query_parameters.end(), "per_page") ==
                operation->query_parameters.end() ||
            std::find(operation->query_parameters.begin(), operation->query_parameters.end(), "select") ==
                operation->query_parameters.end()) {
        std::cerr << "OpenAlex catalog did not expose search, filter and bounded projection parameters\n";
        return 1;
    }
    for (const auto * operation_id : {"listWorks", "getWork", "listAuthors", "getAuthor",
            "listInstitutions", "getInstitution", "listTopics", "getTopic"}) {
        if (std::none_of(catalog.operations.begin(), catalog.operations.end(),
                [&](const agent_openapi_operation & candidate) {
                    return candidate.operation_id == operation_id;
                })) {
            std::cerr << "OpenAlex catalog did not expose allowlisted operation " << operation_id << "\n";
            return 1;
        }
    }

    agent_openapi_tool_provider provider(
        std::move(catalog), make_agent_openapi_http_executor(config));
    agent_tool_context context;
    context.allow_network = true;
    context.max_calls = 8;
    context.default_timeout_ms = config.request_timeout_ms;
    context.default_max_result_bytes = config.max_result_bytes;
    auto view = provider.resolve_tools(context, error);
    if (!view) {
        std::cerr << "OpenAlex provider view failed: " << error << "\n";
        return 1;
    }
    nlohmann::json works;
    if (!call_list(*view, "openalex.listWorks",
            R"({"search":"machine learning","per_page":1,"select":"id,display_name,publication_year,primary_topic,topics"})",
            works, error) || works["results"].empty() ||
            !works["results"][0].contains("display_name")) return 1;
    std::cout << "openalex_title_text_search=passed\n";

    nlohmann::json authors;
    nlohmann::json institutions;
    nlohmann::json topics;
    if (!call_list(*view, "openalex.listAuthors",
            R"({"search":"Yann LeCun","per_page":1,"select":"id,display_name"})",
            authors, error) ||
        !call_list(*view, "openalex.listInstitutions",
            R"({"search":"Stanford University","per_page":1,"select":"id,display_name"})",
            institutions, error) ||
        !call_list(*view, "openalex.listTopics",
            R"({"search":"machine learning","per_page":1,"select":"id,display_name"})",
            topics, error)) return 1;
    const std::string author_id = short_openalex_id(authors["results"]);
    const std::string institution_id = short_openalex_id(institutions["results"]);
    const std::string topic_id = short_openalex_id(topics["results"]);
    if (author_id.empty() || institution_id.empty() || topic_id.empty()) {
        std::cerr << "OpenAlex entity lookup did not return usable IDs\n";
        return 1;
    }
    std::cout << "openalex_author_institution_topic_resolution=passed\n";

    for (const auto & filter : {
            "author.id:" + author_id + ",publication_year:2024",
            "institutions.id:" + institution_id + ",publication_year:2024",
            "topics.id:" + topic_id + ",publication_year:2024"}) {
        const std::string arguments = nlohmann::json({
            {"filter", filter}, {"per_page", 1},
            {"select", "id,display_name,publication_year,primary_topic,topics"}
        }).dump();
        nlohmann::json filtered_works;
        if (!call_list(*view, "openalex.listWorks", arguments, filtered_works, error)) return 1;
    }
    std::cout << "openalex_author_institution_topic_year_filters=passed\n";
    std::cout << "openalex_first_title=" << works["results"][0]["display_name"].get<std::string>() << "\n";
    return 0;
}
