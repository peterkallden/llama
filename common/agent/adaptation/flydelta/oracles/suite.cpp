#include "agent/adaptation/flydelta/oracles/suite.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

using json = nlohmann::ordered_json;

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

bool common_flydelta_oracle_suite_report_validate(
        const common_flydelta_oracle_suite_report & report,
        std::string & error) {
    error.clear();
    const auto bounded = [](const std::string & value, size_t maximum = 512U) {
        return !value.empty() && value.size() <= maximum;
    };
    const auto metric = [](float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
    };
    if (report.schema_version != 1 || !bounded(report.id) ||
            !bounded(report.source_kind) || !bounded(report.candidate_id) ||
            !bounded(report.evaluation_revision) || !bounded(report.model_profile_id) ||
            !bounded(report.oracle_ref) || !bounded(report.oracle_revision) ||
            !bounded(report.policy_revision) || report.observations.empty() ||
            report.observations.size() > 4096U || !metric(report.baseline_success_rate) ||
            !metric(report.candidate_success_rate) ||
            !std::isfinite(report.intervention_gain) ||
            report.intervention_gain < -1.0f || report.intervention_gain > 1.0f ||
            !metric(report.false_intervention_rate) || !metric(report.control_retention) ||
            !std::isfinite(report.transfer_gain) || report.transfer_gain < -1.0f ||
            report.transfer_gain > 1.0f) {
        error = "FlyDelta Oracle-suite report identity, bounds or metrics are invalid";
        return false;
    }
    std::set<std::string> probe_ids;
    std::set<std::string> fixture_refs;
    for (const auto & observation : report.observations) {
        if (!bounded(observation.probe_id) || !bounded(observation.suite_kind) ||
                !bounded(observation.fixture_ref) ||
                !bounded(observation.verifier_revision) ||
                !bounded(observation.outcome) ||
                (observation.suite_kind != "intended" &&
                 observation.suite_kind != "holdout" &&
                 observation.suite_kind != "retention" &&
                 observation.suite_kind != "agent_regression" &&
                 observation.suite_kind != "paraphrase" &&
                 observation.suite_kind != "transfer" &&
                 observation.suite_kind != "control" &&
                 observation.suite_kind != "competing") ||
                (observation.outcome != "unknown" &&
                 observation.outcome != "helped" &&
                 observation.outcome != "neutral" &&
                 observation.outcome != "harmed") ||
                !probe_ids.insert(observation.probe_id).second ||
                !fixture_refs.insert(observation.fixture_ref).second) {
            error = "FlyDelta Oracle-suite observation is invalid";
            return false;
        }
    }
    return true;
}

std::string common_flydelta_oracle_suite_report_to_json(
        const common_flydelta_oracle_suite_report & report) {
    json observations = json::array();
    for (const auto & observation : report.observations) {
        observations.push_back({
            {"probe_id", observation.probe_id},
            {"suite_kind", observation.suite_kind},
            {"fixture_ref", observation.fixture_ref},
            {"verifier_revision", observation.verifier_revision},
            {"outcome", observation.outcome},
            {"baseline_known", observation.baseline_known},
            {"baseline_passed", observation.baseline_passed},
            {"candidate_known", observation.candidate_known},
            {"candidate_passed", observation.candidate_passed},
        });
    }
    return json{
        {"kind", "flydelta_oracle_suite_report"},
        {"schema_version", report.schema_version},
        {"id", report.id},
        {"source_kind", report.source_kind},
        {"candidate_id", report.candidate_id},
        {"evaluation_revision", report.evaluation_revision},
        {"model_profile_id", report.model_profile_id},
        {"oracle_ref", report.oracle_ref},
        {"oracle_revision", report.oracle_revision},
        {"policy_revision", report.policy_revision},
        {"observations", observations},
        {"metrics", {
            {"baseline_success_rate", report.baseline_success_rate},
            {"candidate_success_rate", report.candidate_success_rate},
            {"intervention_gain", report.intervention_gain},
            {"false_intervention_rate", report.false_intervention_rate},
            {"control_retention", report.control_retention},
            {"transfer_gain", report.transfer_gain},
            {"semantically_helped", report.semantically_helped},
            {"safe_to_continue", report.safe_to_continue},
        }},
    }.dump();
}

bool common_flydelta_oracle_suite_report_from_json(
        const std::string & text,
        common_flydelta_oracle_suite_report & report,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (value.value("kind", "") != "flydelta_oracle_suite_report") {
            error = "FlyDelta Oracle-suite report kind is invalid";
            return false;
        }
        report = {};
        report.schema_version = value.value("schema_version", 0);
        report.id = value.value("id", "");
        report.source_kind = value.value("source_kind", "");
        report.candidate_id = value.value("candidate_id", "");
        report.evaluation_revision = value.value("evaluation_revision", "");
        report.model_profile_id = value.value("model_profile_id", "");
        report.oracle_ref = value.value("oracle_ref", "");
        report.oracle_revision = value.value("oracle_revision", "");
        report.policy_revision = value.value("policy_revision", "");
        const auto metrics = value.value("metrics", json::object());
        report.baseline_success_rate = metrics.value("baseline_success_rate", 0.0f);
        report.candidate_success_rate = metrics.value("candidate_success_rate", 0.0f);
        report.intervention_gain = metrics.value("intervention_gain", 0.0f);
        report.false_intervention_rate = metrics.value("false_intervention_rate", 0.0f);
        report.control_retention = metrics.value("control_retention", 0.0f);
        report.transfer_gain = metrics.value("transfer_gain", 0.0f);
        report.semantically_helped = metrics.value("semantically_helped", false);
        report.safe_to_continue = metrics.value("safe_to_continue", false);
        const auto observations = value.value("observations", json::array());
        if (!observations.is_array()) {
            error = "FlyDelta Oracle-suite observations are not an array";
            return false;
        }
        for (const auto & item : observations) {
            if (!item.is_object()) {
                error = "FlyDelta Oracle-suite observation is not an object";
                return false;
            }
            common_flydelta_oracle_suite_fixture_observation observation;
            observation.probe_id = item.value("probe_id", "");
            observation.suite_kind = item.value("suite_kind", "");
            observation.fixture_ref = item.value("fixture_ref", "");
            observation.verifier_revision = item.value("verifier_revision", "");
            observation.outcome = item.value("outcome", "");
            observation.baseline_known = item.value("baseline_known", false);
            observation.baseline_passed = item.value("baseline_passed", false);
            observation.candidate_known = item.value("candidate_known", false);
            observation.candidate_passed = item.value("candidate_passed", false);
            report.observations.push_back(std::move(observation));
        }
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta Oracle-suite report JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_oracle_suite_report_validate(report, error);
}

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
