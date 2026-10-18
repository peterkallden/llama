#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/daemon/agent-daemon-adapter.h"

#include <iostream>
#include <string>

int main() {
    using json = nlohmann::ordered_json;
    std::string error;
    agent_host_config config;

    if (!parse_agent_host_config_json(json::object(), config, error) ||
            config.adaptation_flydelta_disable_factorized_cvec_batch ||
            config.adaptation_flydelta_outcome_guidance.mode !=
                common_flydelta_outcome_guidance_mode::shadow) {
        std::cerr << "factorized cvec must not be disabled by default: " << error << '\n';
        return 1;
    }
    const auto serialized = agent_host_config_to_json(config);
    if (!serialized["runtime"]["adaptation"]["flydelta"].contains("disable_factorized_cvec_batch") ||
            serialized["runtime"]["adaptation"]["flydelta"]["disable_factorized_cvec_batch"] != false ||
            serialized["runtime"]["adaptation"]["flydelta"].contains("factorized_cvec_batch") ||
            serialized["runtime"]["adaptation"]["flydelta"]["outcome_guidance"]["mode"] != "shadow") {
        std::cerr << "factorized cvec opt-out default was not serialized canonically\n";
        return 1;
    }

    json compatible = {
        {"model", {{"backend", "server-context"}}},
        {"models", {
            {"schema_version", 1},
            {"directory", "models"},
            {"bases", {{"agent", {{"kind", "generation"}, {"backend", "server-context"},
                {"path", "model.gguf"}, {"load", "resident"}}}}},
            {"profiles", {{"agent-default", {{"base", "agent"}, {"context_size", 4096},
                {"n_parallel", 2}, {"n_sequences", 2}, {"load", "resident"}}}}},
            {"routing", {{"default_profile", "agent-default"}}},
        }},
        {"runtime", {{"adaptation", {{"flydelta", {
            {"enabled", true},
            {"worker_count", 1},
            {"queue_path", "var/flydelta/queue"},
            {"batch_mode", "auto"},
            {"batch_parallelism", 2},
        }}}}}},
        {"limits", {{"worker_count", 2}}},
    };
    if (!parse_agent_host_config_json(compatible, config, error) ||
            config.adaptation_flydelta_disable_factorized_cvec_batch) {
        std::cerr << "default compatible factorized cvec config rejected: " << error << '\n';
        return 1;
    }
    compatible["runtime"]["adaptation"]["flydelta"]["outcome_guidance"] = {
        {"mode", "experimental"}, {"seed", uint64_t{71}},
        {"harm_penalty", 6.0}, {"min_backoff_groups", 9}, {"max_prior_strength", 3.0}
    };
    if (!parse_agent_host_config_json(compatible, config, error) ||
            config.adaptation_flydelta_outcome_guidance.mode !=
                common_flydelta_outcome_guidance_mode::experimental ||
            config.adaptation_flydelta_outcome_guidance.seed != 71 ||
            config.adaptation_flydelta_outcome_guidance.harm_penalty != 6.0 ||
            config.adaptation_flydelta_outcome_guidance.min_backoff_groups != 9 ||
            config.adaptation_flydelta_outcome_guidance.max_prior_strength != 3.0) {
        std::cerr << "outcome guidance config fields were not parsed: " << error << '\n';
        return 1;
    }
    daemon_options guidance_options;
    apply_agent_host_config_to_daemon_options(config, guidance_options);
    if (guidance_options.adaptation_flydelta_outcome_guidance.mode !=
            common_flydelta_outcome_guidance_mode::experimental ||
            guidance_options.adaptation_flydelta_outcome_guidance.seed != 71) {
        std::cerr << "outcome guidance config did not reach daemon options\n";
        return 1;
    }
    daemon_options enabled_options;
    apply_agent_host_config_to_daemon_options(config, enabled_options);
    if (!enabled_options.adaptation_flydelta_factorized_cvec_batch) {
        std::cerr << "factorized cvec was not enabled for a compatible default config\n";
        return 1;
    }

    const auto roundtrip = agent_host_config_to_json(config);
    agent_host_config roundtrip_config;
    if (!parse_agent_host_config_json(roundtrip, roundtrip_config, error) ||
            roundtrip_config.adaptation_flydelta_disable_factorized_cvec_batch) {
        std::cerr << "factorized cvec config did not round-trip: " << error << '\n';
        return 1;
    }

    auto disabled = compatible;
    disabled["runtime"]["adaptation"]["flydelta"]["disable_factorized_cvec_batch"] = true;
    if (!parse_agent_host_config_json(disabled, config, error) ||
            !config.adaptation_flydelta_disable_factorized_cvec_batch) {
        std::cerr << "explicit factorized cvec disable was not parsed: " << error << '\n';
        return 1;
    }
    daemon_options disabled_options;
    apply_agent_host_config_to_daemon_options(config, disabled_options);
    if (disabled_options.adaptation_flydelta_factorized_cvec_batch) {
        std::cerr << "explicit factorized cvec disable was ignored\n";
        return 1;
    }

    auto unsupported = compatible;
    unsupported["model"]["backend"] = "cli";
    if (!parse_agent_host_config_json(unsupported, config, error)) {
        std::cerr << "non-server-context config unexpectedly rejected: " << error << '\n';
        return 1;
    }
    daemon_options unsupported_options;
    apply_agent_host_config_to_daemon_options(config, unsupported_options);
    if (unsupported_options.adaptation_flydelta_factorized_cvec_batch) {
        std::cerr << "factorized cvec should fall back for a non-server-context backend\n";
        return 1;
    }

    auto legacy_enabled = compatible;
    legacy_enabled["runtime"]["adaptation"]["flydelta"]["factorized_cvec_batch"] = true;
    if (!parse_agent_host_config_json(legacy_enabled, config, error) ||
            config.adaptation_flydelta_disable_factorized_cvec_batch) {
        std::cerr << "legacy enabled config was not migrated: " << error << '\n';
        return 1;
    }
    auto legacy_disabled = compatible;
    legacy_disabled["runtime"]["adaptation"]["flydelta"]["factorized_cvec_batch"] = false;
    if (!parse_agent_host_config_json(legacy_disabled, config, error) ||
            !config.adaptation_flydelta_disable_factorized_cvec_batch) {
        std::cerr << "legacy disabled config was not migrated: " << error << '\n';
        return 1;
    }
    auto conflicting = compatible;
    conflicting["runtime"]["adaptation"]["flydelta"]["factorized_cvec_batch"] = true;
    conflicting["runtime"]["adaptation"]["flydelta"]["disable_factorized_cvec_batch"] = true;
    if (parse_agent_host_config_json(conflicting, config, error)) {
        std::cerr << "ambiguous legacy and opt-out settings were accepted\n";
        return 1;
    }

    std::cout << "agent_factorized_cvec_config=passed\n";
    return 0;
}
