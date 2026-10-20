#pragma once

#include "agent-host-provider-auth-config.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct agent_host_investigation_source_config {
    std::string id;
    std::string dialect = "elasticsearch";
    std::string base_url;
    std::vector<std::string> allowed_indices;
    std::vector<std::string> allowed_namespaces;
    std::vector<std::string> allowed_projects;
    std::map<std::string, std::string> fields;
    agent_host_provider_auth_config auth;
    bool allow_private_network = false;
    uint32_t connect_timeout_ms = 5000;
    uint32_t request_timeout_ms = 30000;
    size_t max_page_size = 500;
    size_t max_result_bytes = 4 * 1024 * 1024;
};

struct agent_host_investigation_config {
    bool enabled = false;
    std::vector<agent_host_investigation_source_config> sources;
    size_t max_events_per_query = 5000;
    size_t max_nodes_per_expansion = 100;
    size_t max_edges_per_expansion = 200;
    size_t max_expansions_per_turn = 12;
    size_t max_materialized_rows = 10000;
};
