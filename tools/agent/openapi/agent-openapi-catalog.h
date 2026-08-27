#pragma once

#include "../host/agent-host-openapi-provider-config.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

enum class agent_openapi_access {
    read,
    write,
    destructive,
};

struct agent_openapi_operation {
    std::string operation_id;
    std::string method;
    std::string path;
    std::string summary;
    std::string description;
    std::string input_schema_json = R"({"type":"object"})";
    std::vector<std::string> path_parameters;
    std::vector<std::string> query_parameters;
    agent_openapi_access access = agent_openapi_access::write;
    bool read_only = false;
    bool requires_confirmation = false;
    std::string result_schema_json = R"({"type":"object"})";
};

struct agent_openapi_relation {
    std::string collection_operation_id;
    std::string item_operation_id;
    std::string resource_path;
    std::string item_parameter;
};

struct agent_openapi_catalog {
    std::string provider_id;
    std::string base_url;
    std::string prefix;
    std::vector<agent_openapi_operation> operations;
    std::vector<agent_openapi_relation> relations;
};

// Parse an OpenAPI 3 document and apply the host-owned exposure policy. This
// function deliberately does not resolve references or perform HTTP calls.
bool build_agent_openapi_catalog(
    const nlohmann::json & document,
    const agent_host_openapi_provider_config & config,
    agent_openapi_catalog & catalog,
    std::string & error);

std::string agent_openapi_exposed_tool_name(
    const agent_openapi_catalog & catalog,
    const agent_openapi_operation & operation);

std::string agent_openapi_access_name(agent_openapi_access access);
