#include "agent/adaptation/flydelta/oracles/flydelta-oracle-suite.h"

#include <algorithm>

namespace {

bool expected(const common_flydelta_oracle_probe_observation & observation, bool candidate) {
    const auto & actual = candidate ? observation.candidate : observation.baseline;
    return common_flydelta_oracle_verdict_matches(observation.expected_verdict, actual);
}

bool is_control(const common_flydelta_oracle_probe_observation & observation) {
    return observation.kind == common_flydelta_oracle_probe_kind::control ||
        observation.kind == common_flydelta_oracle_probe_kind::competing;
}

} // namespace

bool common_flydelta_oracle_verdict_matches(
        common_flydelta_oracle_verdict expected,
        const common_flydelta_oracle_result & actual) {
    return expected != common_flydelta_oracle_verdict::unknown && actual.known &&
        actual.verdict == expected;
}

bool common_flydelta_run_oracle_suite(
        const common_flydelta_oracle_suite_request & request,
        common_flydelta_oracle_suite_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (request.probes.empty()) {
        error = "oracle suite requires at least one probe";
        return false;
    }
    if (!request.runner) {
        error = "oracle suite requires an agent/server-context probe runner";
        return false;
    }

    size_t baseline_scored = 0;
    size_t candidate_scored = 0;
    size_t baseline_success = 0;
    size_t candidate_success = 0;
    size_t controls = 0;
    size_t retained_controls = 0;
    size_t transfer_scored = 0;
    size_t transfer_baseline_success = 0;
    size_t transfer_candidate_success = 0;
    bool no_harm = true;

    result.probes.reserve(request.probes.size());
    for (const auto & probe : request.probes) {
        common_flydelta_oracle_probe_observation observation;
        observation.probe_id = probe.probe_id;
        observation.kind = probe.kind;
        observation.expected_verdict = probe.expected_verdict;
        std::string baseline_text;
        std::string candidate_text;
        if (!request.runner(probe, false, baseline_text, error) ||
                !request.runner(probe, true, candidate_text, error)) {
            if (error.empty()) error = "oracle suite probe runner failed";
            return false;
        }
        if (!common_flydelta_oracle_evaluate(
                request.evaluators, probe.request, baseline_text, observation.baseline, error) ||
                !common_flydelta_oracle_evaluate(
                    request.evaluators, probe.request, candidate_text, observation.candidate, error)) {
            return false;
        }
        if (probe.expected_verdict != common_flydelta_oracle_verdict::unknown) {
            ++baseline_scored;
            ++candidate_scored;
            baseline_success += expected(observation, false) ? 1U : 0U;
            candidate_success += expected(observation, true) ? 1U : 0U;
        }
        if (is_control(observation)) {
            ++controls;
            if (observation.baseline.known && observation.candidate.known &&
                    observation.baseline.verdict == observation.candidate.verdict) {
                ++retained_controls;
            } else {
                no_harm = false;
            }
        }
        if (observation.kind == common_flydelta_oracle_probe_kind::transfer &&
                probe.expected_verdict != common_flydelta_oracle_verdict::unknown) {
            ++transfer_scored;
            transfer_baseline_success += expected(observation, false) ? 1U : 0U;
            transfer_candidate_success += expected(observation, true) ? 1U : 0U;
        }
        if (observation.candidate.known &&
                observation.expected_verdict != common_flydelta_oracle_verdict::unknown &&
                !expected(observation, true)) {
            no_harm = false;
        }
        result.probes.push_back(std::move(observation));
    }

    if (baseline_scored != 0) {
        result.baseline_success_rate = static_cast<float>(baseline_success) /
            static_cast<float>(baseline_scored);
        result.candidate_success_rate = static_cast<float>(candidate_success) /
            static_cast<float>(candidate_scored);
    }
    result.intervention_gain = result.candidate_success_rate - result.baseline_success_rate;
    result.false_intervention_rate = controls == 0
        ? 0.0f : 1.0f - static_cast<float>(retained_controls) / static_cast<float>(controls);
    result.control_retention = controls == 0
        ? 1.0f : static_cast<float>(retained_controls) / static_cast<float>(controls);
    const int transfer_delta = static_cast<int>(transfer_candidate_success) -
        static_cast<int>(transfer_baseline_success);
    result.transfer_gain = transfer_scored == 0
        ? 0.0f : static_cast<float>(transfer_delta) / static_cast<float>(transfer_scored);
    result.semantically_helped = result.intervention_gain > 0.0f && no_harm;
    result.safe_to_continue = no_harm && candidate_scored != 0 &&
        result.candidate_success_rate >= result.baseline_success_rate;
    return true;
}
