#include "tools/agent/host/agent-host-config.h"

#include <iostream>

int main() {
    using json = nlohmann::ordered_json;
    std::string error;
    agent_host_config defaults;
    if (!parse_agent_host_config_json(json::object(), defaults, error) || defaults.investigation.enabled) {
        std::cerr << "investigation must remain disabled by default: " << error << '\n';
        return 1;
    }
    agent_host_config default_roundtrip;
    if (!parse_agent_host_config_json(agent_host_config_to_json(defaults), default_roundtrip, error) ||
            default_roundtrip.investigation.enabled) return 2;

    json configured = json::object();
    configured["stores"]["data"] = {{"backend", "cozo"}, {"path", "var/investigation.cozo"}};
    configured["investigation"] = {
        {"enabled", true},
        {"workflow_package", "docs/examples/agent-investigation-workflows-v1.json"},
        {"max_events_per_query", 2500},
        {"sources", json::array({{{"id", "security-logs"}, {"dialect", "elasticsearch"},
            {"base_url", "https://elastic.example.invalid:9200"},
            {"allowed_indices", json::array({"logs-security-2026.10"})},
            {"allowed_namespaces", json::array({"tenant-a"})},
            {"fields", {{"event_time", "@timestamp"}, {"host_id", "host.id"},
                {"user_id", "user.id"}}},
            {"relation_profiles", json::array({{{"id", "auth-observed-v1"},
                {"event_type", "authentication"}, {"from_kind", "user"}, {"to_kind", "host"},
                {"relation_kind", "authenticated_to"}}})},
            {"auth", {{"type", "bearer"}, {"token_env", "ELASTIC_TOKEN"}}},
            {"max_page_size", 500}}})},
    };
#ifdef LLAMA_MEMORY_USE_COZO
    if (!parse_agent_host_config_json(configured, defaults, error) || !defaults.investigation.enabled ||
            defaults.investigation.sources.size() != 1 ||
            defaults.investigation.sources[0].fields.at("host_id") != "host.id" ||
            defaults.investigation.sources[0].relation_profiles.size() != 1 ||
            defaults.investigation.sources[0].relation_profiles[0].id != "auth-observed-v1" ||
            defaults.investigation.sources[0].auth.token_env != "ELASTIC_TOKEN") {
        std::cerr << "investigation source config did not parse: " << error << '\n';
        return 3;
    }
    agent_host_config configured_roundtrip;
    if (!parse_agent_host_config_json(agent_host_config_to_json(defaults), configured_roundtrip, error) ||
            configured_roundtrip.investigation.sources.size() != 1 ||
            configured_roundtrip.investigation.sources[0].allowed_indices[0] != "logs-security-2026.10" ||
            configured_roundtrip.investigation.sources[0].allowed_namespaces[0] != "tenant-a" ||
            configured_roundtrip.investigation.sources[0].relation_profiles.size() != 1 ||
            configured_roundtrip.investigation.sources[0].relation_profiles[0].event_type != "authentication" ||
            configured_roundtrip.investigation.max_events_per_query != 2500 ||
            configured_roundtrip.investigation.workflow_package != "docs/examples/agent-investigation-workflows-v1.json") return 4;
    auto missing_workflow_package = configured;
    missing_workflow_package["investigation"].erase("workflow_package");
    if (parse_agent_host_config_json(missing_workflow_package, configured_roundtrip, error) ||
            error.find("requires investigation.workflow_package") == std::string::npos) return 5;
    configured["investigation"]["sources"][0]["allowed_indices"][0] = "*";
    if (parse_agent_host_config_json(configured, configured_roundtrip, error) ||
            error.find("without wildcards") == std::string::npos) return 6;
#else
    if (parse_agent_host_config_json(configured, defaults, error) ||
            error.find("requires a build with LLAMA_MEMORY_COZO") == std::string::npos) return 7;
#endif
    return 0;
}
