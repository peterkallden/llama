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
            roundtrip.runtime_context_size != 0 || !roundtrip.context_budgets.configured_prompt_values.empty()) return 2;
    json explicit_config = {{"runtime", {{"context_size", 8192},
        {"context_budgets", {{"memory_chars", 4096}}}}}};
    if (!parse_agent_host_config_json(explicit_config, config, error) || config.runtime_context_size != 8192 ||
            !parse_agent_host_config_json(agent_host_config_to_json(config), roundtrip, error) ||
            roundtrip.runtime_context_size != 8192 ||
            roundtrip.context_budgets.configured_prompt_values.count("memory_chars") != 1 ||
            roundtrip.context_budgets.configured_prompt_values.size() != 1) return 3;
    explicit_config["runtime"]["context_size"] = -1;
    if (parse_agent_host_config_json(explicit_config, config, error)) return 4;
}
