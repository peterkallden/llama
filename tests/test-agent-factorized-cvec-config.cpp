#include "tools/agent/host/agent-host-config.h"

#include <iostream>
#include <string>

int main() {
    using json = nlohmann::ordered_json;
    std::string error;
    agent_host_config config;

    if (!parse_agent_host_config_json(json::object(), config, error) ||
            config.adaptation_flydelta_factorized_cvec_batch) {
        std::cerr << "factorized cvec must default off: " << error << '\n';
        return 1;
    }
    const auto serialized = agent_host_config_to_json(config);
    if (!serialized["runtime"]["adaptation"]["flydelta"].contains("factorized_cvec_batch") ||
            serialized["runtime"]["adaptation"]["flydelta"]["factorized_cvec_batch"] != false) {
        std::cerr << "factorized cvec default was not serialized as false\n";
        return 1;
    }

    json enabled = {
        {"model", {{"backend", "server-context"}}},
        {"runtime", {{"adaptation", {{"flydelta", {
            {"enabled", true},
            {"worker_count", 1},
            {"queue_path", "var/flydelta/queue"},
            {"batch_mode", "auto"},
            {"batch_parallelism", 2},
            {"factorized_cvec_batch", true},
        }}}}}},
        {"limits", {{"worker_count", 2}}},
    };
    if (!parse_agent_host_config_json(enabled, config, error) ||
            !config.adaptation_flydelta_factorized_cvec_batch) {
        std::cerr << "valid factorized cvec config rejected: " << error << '\n';
        return 1;
    }
    const auto roundtrip = agent_host_config_to_json(config);
    agent_host_config roundtrip_config;
    if (!parse_agent_host_config_json(roundtrip, roundtrip_config, error) ||
            !roundtrip_config.adaptation_flydelta_factorized_cvec_batch) {
        std::cerr << "factorized cvec config did not round-trip: " << error << '\n';
        return 1;
    }

    for (const json & invalid : {
            json{{"model", {{"backend", "server-context"}}}, {"runtime", {{"adaptation", {{"flydelta", {
                {"enabled", true}, {"queue_path", "var/flydelta/queue"},
                {"batch_mode", "disabled"}, {"batch_parallelism", 2},
                {"factorized_cvec_batch", true},
            }}}}}}},
            json{{"model", {{"backend", "server-context"}}}, {"runtime", {{"adaptation", {{"flydelta", {
                {"enabled", true}, {"queue_path", "var/flydelta/queue"},
                {"batch_mode", "auto"}, {"batch_parallelism", 1},
                {"factorized_cvec_batch", true},
            }}}}}}},
            json{{"model", {{"backend", "cli"}}}, {"runtime", {{"adaptation", {{"flydelta", {
                {"enabled", true}, {"queue_path", "var/flydelta/queue"},
                {"batch_mode", "auto"}, {"batch_parallelism", 2},
                {"factorized_cvec_batch", true},
            }}}}}}},
    }) {
        if (parse_agent_host_config_json(invalid, config, error)) {
            std::cerr << "invalid factorized cvec config was accepted\n";
            return 1;
        }
    }

    std::cout << "agent_factorized_cvec_config=passed\n";
    return 0;
}
