#include "tools/agent/host/agent-host-config.h"

#include <iostream>

int main() {
    using json = nlohmann::ordered_json;
    std::string error;
    agent_host_config config;
    if (!parse_agent_host_config_json(json::object(), config, error) || config.runtime_context_size != 0) {
        std::cerr << "omitted context should select automatic sizing: " << error << '\n'; return 1;
    }
    agent_host_config roundtrip;
    if (!parse_agent_host_config_json(agent_host_config_to_json(config), roundtrip, error) ||
            roundtrip.runtime_context_size != 0 || roundtrip.context_budgets.resource_chunk_max_bytes != 0 ||
            agent_host_config_to_json(config)["runtime"]["context_budgets"].contains("resource_chunk_max_bytes") ||
            !roundtrip.context_budgets.configured_prompt_values.empty()) return 2;
    json explicit_config = {{"runtime", {{"context_size", 8192},
        {"context_budgets", {{"memory_chars", 4096}}}}}};
    if (!parse_agent_host_config_json(explicit_config, config, error) || config.runtime_context_size != 8192 ||
            !parse_agent_host_config_json(agent_host_config_to_json(config), roundtrip, error) ||
            roundtrip.runtime_context_size != 8192 ||
            roundtrip.context_budgets.configured_prompt_values.count("memory_chars") != 1 ||
            roundtrip.context_budgets.configured_prompt_values.size() != 1) return 3;
    explicit_config["runtime"]["context_size"] = -1;
    if (parse_agent_host_config_json(explicit_config, config, error)) return 4;
    explicit_config["runtime"]["context_size"] = 8192;
    explicit_config["runtime"]["context_budgets"]["resource_chunk_max_bytes"] = 4096;
    if (!parse_agent_host_config_json(explicit_config, config, error) ||
            !parse_agent_host_config_json(agent_host_config_to_json(config), roundtrip, error) ||
            roundtrip.context_budgets.resource_chunk_max_bytes != 4096 ||
            roundtrip.context_budgets.configured_prompt_values.count("resource_chunk_max_bytes") != 1) return 5;
    explicit_config["runtime"]["context_budgets"]["resource_chunk_max_bytes"] = 0;
    if (parse_agent_host_config_json(explicit_config, config, error) && validate_agent_host_config(config, error)) return 6;
    config = agent_host_config{};
    roundtrip = agent_host_config{};
    json source_config = json::object();
    source_config["sandbox"]["development_targets"]["contracts"] = {
        {"build_program", "cmake"}, {"test_program", "ctest"}, {"source_paths", {"common/agent", "tools/agent"}}};
    if (!parse_agent_host_config_json(source_config, config, error) ||
            !parse_agent_host_config_json(agent_host_config_to_json(config), roundtrip, error) ||
            roundtrip.sandbox.development_targets.at("contracts").source_paths.size() != 2) { std::cerr << "source metadata roundtrip: " << error << "\n"; return 7; }
    source_config["sandbox"]["development_targets"]["contracts"]["source_paths"] = {"../outside"};
    if (parse_agent_host_config_json(source_config, config, error)) return 8;
    agent_host_config authority_config;
    agent_host_mcp_inbound_token_config token;
    token.id="web"; token.project_id="default"; token.token_env="TEST_WEB_TOKEN"; token.tool_profile="all-configured";
    token.allow_project_selection=true;
    authority_config.inbound_mcp_tokens.push_back(token);
    authority_config.inbound_mcp_jwt_allow_project_selection=true;
    if (!parse_agent_host_config_json(agent_host_config_to_json(authority_config), roundtrip, error) ||
        !roundtrip.inbound_mcp_tokens.front().allow_project_selection ||
        !roundtrip.inbound_mcp_jwt_allow_project_selection) { std::cerr << error; return 9; }
}
