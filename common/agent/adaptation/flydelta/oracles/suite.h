#pragma once

#include "agent/adaptation/flydelta/oracles/contracts.h"

#include <functional>
#include <cstddef>
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

// Durable, redacted Oracle-suite materialization.  This is deliberately
// separate from common_flydelta_evaluation_report: the lifecycle report keeps
// only oracle_suite_report_ref and revision metadata, while this artifact
// carries the bounded per-probe host-verification observations.  It contains
// no prompts, model output, captures or activation authority.
struct common_flydelta_oracle_suite_fixture_observation {
    std::string probe_id;
    std::string suite_kind;
    std::string fixture_ref;
    std::string verifier_revision;
    std::string outcome;
    bool baseline_known = false;
    bool baseline_passed = false;
    bool candidate_known = false;
    bool candidate_passed = false;
};

struct common_flydelta_oracle_suite_report {
    int schema_version = 1;
    std::string id;
    std::string source_kind = "host_counterfactual_evaluation";
    std::string candidate_id;
    std::string evaluation_revision;
    std::string model_profile_id;
    std::string oracle_ref;
    std::string oracle_revision;
    std::string policy_revision;
    std::vector<common_flydelta_oracle_suite_fixture_observation> observations;
    float baseline_success_rate = 0.0f;
    float candidate_success_rate = 0.0f;
    float intervention_gain = 0.0f;
    float false_intervention_rate = 0.0f;
    float control_retention = 0.0f;
    float transfer_gain = 0.0f;
    bool semantically_helped = false;
    bool safe_to_continue = false;
};

bool common_flydelta_oracle_suite_report_validate(
        const common_flydelta_oracle_suite_report & report,
        std::string & error);

std::string common_flydelta_oracle_suite_report_to_json(
        const common_flydelta_oracle_suite_report & report);

bool common_flydelta_oracle_suite_report_from_json(
        const std::string & text,
        common_flydelta_oracle_suite_report & report,
        std::string & error);

bool common_flydelta_oracle_verdict_matches(
        common_flydelta_oracle_verdict expected,
        const common_flydelta_oracle_result & actual);

bool common_flydelta_run_oracle_suite(
        const common_flydelta_oracle_suite_request & request,
        common_flydelta_oracle_suite_result & result,
        std::string & error);
