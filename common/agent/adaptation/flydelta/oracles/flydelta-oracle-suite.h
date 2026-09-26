#pragma once

#include "agent/adaptation/flydelta/oracles/flydelta-oracle-contracts.h"

#include <functional>
#include <string>
#include <vector>

struct common_flydelta_oracle_probe {
    std::string probe_id;
    common_flydelta_oracle_probe_kind kind = common_flydelta_oracle_probe_kind::target;
    common_flydelta_oracle_request request;
    common_flydelta_oracle_verdict expected_verdict = common_flydelta_oracle_verdict::unknown;
};

struct common_flydelta_oracle_probe_observation {
    std::string probe_id;
    common_flydelta_oracle_probe_kind kind = common_flydelta_oracle_probe_kind::target;
    common_flydelta_oracle_verdict expected_verdict = common_flydelta_oracle_verdict::unknown;
    common_flydelta_oracle_result baseline;
    common_flydelta_oracle_result candidate;
};

// The runner is the only model/host boundary. It can invoke an agent/server
// context for both arms; Oracle code never creates a direct single-turn path.
using common_flydelta_oracle_probe_runner = std::function<bool(
        const common_flydelta_oracle_probe & probe,
        bool candidate,
        std::string & observed,
        std::string & error)>;

struct common_flydelta_oracle_suite_request {
    std::vector<common_flydelta_oracle_probe> probes;
    common_flydelta_oracle_evaluator_chain evaluators;
    common_flydelta_oracle_probe_runner runner;
};

struct common_flydelta_oracle_suite_result {
    std::vector<common_flydelta_oracle_probe_observation> probes;
    float baseline_success_rate = 0.0f;
    float candidate_success_rate = 0.0f;
    float intervention_gain = 0.0f;
    float false_intervention_rate = 0.0f;
    float control_retention = 0.0f;
    float transfer_gain = 0.0f;
    bool semantically_helped = false;
    bool safe_to_continue = false;
};

bool common_flydelta_oracle_verdict_matches(
        common_flydelta_oracle_verdict expected,
        const common_flydelta_oracle_result & actual);

bool common_flydelta_run_oracle_suite(
        const common_flydelta_oracle_suite_request & request,
        common_flydelta_oracle_suite_result & result,
        std::string & error);
