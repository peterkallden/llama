#include "agent/adaptation/flydelta/flydelta-experiment.h"

#include <cmath>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

const char * trial_outcome(bool passed) {
    return passed ? "passed" : "failed";
}

common_flydelta_counterfactual_outcome parse_outcome(const std::string & value) {
    if (value == "helped") return common_flydelta_counterfactual_outcome::helped;
    if (value == "neutral") return common_flydelta_counterfactual_outcome::neutral;
    if (value == "harmed") return common_flydelta_counterfactual_outcome::harmed;
    return common_flydelta_counterfactual_outcome::unknown;
}

json trial_to_json(const common_flydelta_counterfactual_trial & trial) {
    return json{
        {"executed", trial.executed},
        {"verifier_known", trial.verifier_known},
        {"status", trial_outcome(trial.passed)},
        {"quality", trial.quality},
        {"overlay_applied", trial.overlay_applied},
        {"intervention_count", trial.intervention_count},
        {"evidence_ref", trial.evidence_ref},
    };
}

void trial_from_json(const json & value, common_flydelta_counterfactual_trial & trial) {
    trial = {};
    trial.executed = value.value("executed", false);
    trial.verifier_known = value.value("verifier_known", false);
    trial.passed = value.value("status", "failed") == "passed";
    trial.quality = value.value("quality", 0.0f);
    trial.overlay_applied = value.value("overlay_applied", false);
    trial.intervention_count = value.value("intervention_count", 0U);
    trial.evidence_ref = value.value("evidence_ref", "");
}

} // namespace

const char * common_flydelta_counterfactual_outcome_name(
        common_flydelta_counterfactual_outcome outcome) {
    switch (outcome) {
        case common_flydelta_counterfactual_outcome::unknown: return "unknown";
        case common_flydelta_counterfactual_outcome::helped: return "helped";
        case common_flydelta_counterfactual_outcome::neutral: return "neutral";
        case common_flydelta_counterfactual_outcome::harmed: return "harmed";
    }
    return "unknown";
}

bool common_flydelta_experiment_fixture_validate(
        const common_flydelta_experiment_fixture & fixture,
        std::string & error) {
    error.clear();
    if (fixture.schema_version != 1 || !nonempty_bounded(fixture.id) ||
            !nonempty_bounded(fixture.task_fingerprint) ||
            !nonempty_bounded(fixture.model_profile_fingerprint) ||
            !nonempty_bounded(fixture.tokenizer_fingerprint) ||
            !nonempty_bounded(fixture.template_fingerprint) ||
            !nonempty_bounded(fixture.tool_catalog_fingerprint) ||
            !nonempty_bounded(fixture.resource_snapshot_fingerprint) ||
            !nonempty_bounded(fixture.verifier_revision)) {
        error = "FlyDelta experiment fixture identity is incomplete";
        return false;
    }
    return true;
}

bool common_flydelta_counterfactual_trial_validate(
        const common_flydelta_counterfactual_trial & trial,
        std::string & error) {
    error.clear();
    if (!trial.executed) {
        error = "FlyDelta counterfactual trial did not execute";
        return false;
    }
    if (!std::isfinite(trial.quality) || trial.quality < 0.0f || trial.quality > 1.0f) {
        error = "FlyDelta counterfactual trial quality is invalid";
        return false;
    }
    if (trial.verifier_known && !nonempty_bounded(trial.evidence_ref)) {
        error = "FlyDelta verified trial requires evidence";
        return false;
    }
    return true;
}

common_flydelta_counterfactual_outcome common_flydelta_classify_counterfactual(
        const common_flydelta_counterfactual_trial & baseline,
        const common_flydelta_counterfactual_trial & candidate) {
    if (!baseline.executed || !candidate.executed ||
            !baseline.verifier_known || !candidate.verifier_known) {
        return common_flydelta_counterfactual_outcome::unknown;
    }
    if (!baseline.passed && candidate.passed) return common_flydelta_counterfactual_outcome::helped;
    if (baseline.passed && !candidate.passed) return common_flydelta_counterfactual_outcome::harmed;
    if (baseline.passed && candidate.passed) return common_flydelta_counterfactual_outcome::neutral;
    return common_flydelta_counterfactual_outcome::unknown;
}

bool common_flydelta_counterfactual_report_validate(
        const common_flydelta_counterfactual_report & report,
        std::string & error) {
    error.clear();
    if (report.schema_version != 1 || !nonempty_bounded(report.experiment_id) ||
            !nonempty_bounded(report.fixture_id) || !nonempty_bounded(report.candidate_id) ||
            !nonempty_bounded(report.baseline_profile_id) ||
            !nonempty_bounded(report.candidate_profile_id) ||
            report.baseline_profile_id == report.candidate_profile_id) {
        error = "FlyDelta counterfactual identity is incomplete";
        return false;
    }
    if (!common_flydelta_counterfactual_trial_validate(report.baseline, error) ||
            !common_flydelta_counterfactual_trial_validate(report.candidate, error)) {
        return false;
    }
    const auto expected = common_flydelta_classify_counterfactual(report.baseline, report.candidate);
    if (report.outcome != expected || !std::isfinite(report.quality_delta) ||
            report.quality_delta < -1.0f || report.quality_delta > 1.0f) {
        error = "FlyDelta counterfactual outcome is inconsistent with trials";
        return false;
    }
    return true;
}

bool common_flydelta_run_counterfactual(
        const std::string & experiment_id,
        const std::string & candidate_id,
        const std::string & baseline_profile_id,
        const std::string & candidate_profile_id,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_counterfactual_runner & runner,
        common_flydelta_counterfactual_report & report,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !nonempty_bounded(experiment_id) || !nonempty_bounded(candidate_id) ||
            !nonempty_bounded(baseline_profile_id) || !nonempty_bounded(candidate_profile_id) ||
            baseline_profile_id == candidate_profile_id || !runner) {
        if (error.empty()) error = "FlyDelta counterfactual runner identity is invalid";
        return false;
    }
    report = {};
    report.experiment_id = experiment_id;
    report.fixture_id = fixture.id;
    report.candidate_id = candidate_id;
    report.baseline_profile_id = baseline_profile_id;
    report.candidate_profile_id = candidate_profile_id;
    if (!runner(fixture, false, report.baseline, error)) return false;
    if (!runner(fixture, true, report.candidate, error)) return false;
    report.outcome = common_flydelta_classify_counterfactual(report.baseline, report.candidate);
    report.quality_delta = report.candidate.quality - report.baseline.quality;
    return common_flydelta_counterfactual_report_validate(report, error);
}

std::string common_flydelta_counterfactual_report_to_json(
        const common_flydelta_counterfactual_report & report) {
    return json{
        {"schema_version", report.schema_version},
        {"experiment_id", report.experiment_id},
        {"fixture_id", report.fixture_id},
        {"candidate_id", report.candidate_id},
        {"baseline_profile_id", report.baseline_profile_id},
        {"candidate_profile_id", report.candidate_profile_id},
        {"baseline", trial_to_json(report.baseline)},
        {"candidate", trial_to_json(report.candidate)},
        {"outcome", common_flydelta_counterfactual_outcome_name(report.outcome)},
        {"quality_delta", report.quality_delta},
    }.dump();
}

bool common_flydelta_counterfactual_report_from_json(
        const std::string & text,
        common_flydelta_counterfactual_report & report,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        report = {};
        report.schema_version = value.value("schema_version", 0);
        report.experiment_id = value.value("experiment_id", "");
        report.fixture_id = value.value("fixture_id", "");
        report.candidate_id = value.value("candidate_id", "");
        report.baseline_profile_id = value.value("baseline_profile_id", "");
        report.candidate_profile_id = value.value("candidate_profile_id", "");
        trial_from_json(value.value("baseline", json::object()), report.baseline);
        trial_from_json(value.value("candidate", json::object()), report.candidate);
        report.outcome = parse_outcome(value.value("outcome", "unknown"));
        report.quality_delta = value.value("quality_delta", 0.0f);
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta counterfactual JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_counterfactual_report_validate(report, error);
}
