#include "agent/adaptation/evaluation-contract.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool nonempty(const std::string & value) { return !value.empty() && value.size() <= 512; }

} // namespace

bool common_learning_validate_evaluation_report(
        const common_learning_evaluation_report & report,
        std::string & error) {
    error.clear();
    if (report.schema_version != 1) { error = "unsupported evaluation report schema"; return false; }
    if (!nonempty(report.revision_id) || !nonempty(report.candidate_adapter_id) ||
            !nonempty(report.baseline_profile_id) || !nonempty(report.candidate_profile_id) ||
            !nonempty(report.corpus_revision_id) || !nonempty(report.corpus_bundle_hash) ||
            !nonempty(report.base_training_fingerprint) || !nonempty(report.test_suite_revision) ||
            !nonempty(report.evaluated_at)) {
        error = "evaluation report identity is incomplete";
        return false;
    }
    if (report.baseline_profile_id == report.candidate_profile_id) {
        error = "evaluation baseline and candidate profiles must differ";
        return false;
    }
    if (report.runtime_interventions > report.evaluated_turns) {
        error = "evaluation interventions exceed evaluated turns";
        return false;
    }
    if (report.baseline_runtime_interventions > report.evaluated_turns) {
        error = "baseline evaluation interventions exceed evaluated turns";
        return false;
    }
    if (report.status != "passed" && report.status != "failed") {
        error = "evaluation report status is invalid";
        return false;
    }
    const bool all_gates_passed = report.intended_behavior_passed &&
        report.retention_passed && report.agent_regression_passed;
    if ((report.status == "passed") != all_gates_passed) {
        error = "evaluation status does not match gate results";
        return false;
    }
    return true;
}

double common_learning_runtime_intervention_rate(
        const common_learning_evaluation_report & report) {
    if (report.evaluated_turns == 0) return 0.0;
    return static_cast<double>(report.runtime_interventions) /
        static_cast<double>(report.evaluated_turns);
}

bool common_learning_evaluate_candidate(
        const common_learning_evaluation_fixture & fixture,
        const std::string & candidate_adapter_id,
        const std::string & baseline_profile_id,
        const std::string & candidate_profile_id,
        const std::string & revision_id,
        const std::string & evaluated_at,
        const common_learning_profile_evaluation_runner & runner,
        common_learning_evaluation_report & report,
        std::string & error) {
    error.clear();
    if (!nonempty(fixture.corpus_revision_id) || !nonempty(fixture.corpus_bundle_hash) ||
            !nonempty(fixture.base_training_fingerprint) || !nonempty(fixture.test_suite_revision) ||
            !nonempty(candidate_adapter_id) || !nonempty(baseline_profile_id) ||
            !nonempty(candidate_profile_id) || !nonempty(revision_id) || !nonempty(evaluated_at) ||
            baseline_profile_id == candidate_profile_id || !runner) {
        error = "evaluation fixture or runner is incomplete";
        return false;
    }

    common_learning_profile_evaluation_result baseline;
    if (!runner(baseline_profile_id, fixture, baseline, error)) return false;
    common_learning_profile_evaluation_result candidate;
    if (!runner(candidate_profile_id, fixture, candidate, error)) return false;

    if (baseline.profile_id != baseline_profile_id || candidate.profile_id != candidate_profile_id ||
            baseline.evaluated_turns == 0 || candidate.evaluated_turns == 0 ||
            baseline.evaluated_turns != candidate.evaluated_turns ||
            baseline.runtime_interventions > baseline.evaluated_turns ||
            candidate.runtime_interventions > candidate.evaluated_turns) {
        error = "baseline and candidate evaluation results are not comparable";
        return false;
    }

    report = {};
    report.revision_id = revision_id;
    report.candidate_adapter_id = candidate_adapter_id;
    report.baseline_profile_id = baseline_profile_id;
    report.candidate_profile_id = candidate_profile_id;
    report.corpus_revision_id = fixture.corpus_revision_id;
    report.corpus_bundle_hash = fixture.corpus_bundle_hash;
    report.base_training_fingerprint = fixture.base_training_fingerprint;
    report.test_suite_revision = fixture.test_suite_revision;
    report.intended_behavior_passed = candidate.intended_behavior_passed;
    report.retention_passed = candidate.retention_passed;
    report.agent_regression_passed = candidate.agent_regression_passed;
    report.evaluated_turns = candidate.evaluated_turns;
    report.baseline_runtime_interventions = baseline.runtime_interventions;
    report.runtime_interventions = candidate.runtime_interventions;
    report.status = report.intended_behavior_passed && report.retention_passed &&
        report.agent_regression_passed && baseline.intended_behavior_passed &&
        baseline.retention_passed && baseline.agent_regression_passed
        ? "passed" : "failed";
    report.evaluated_at = evaluated_at;
    return common_learning_validate_evaluation_report(report, error);
}

std::string common_learning_evaluation_report_to_json(
        const common_learning_evaluation_report & report) {
    return json{
        {"schema_version", report.schema_version},
        {"revision_id", report.revision_id},
        {"candidate_adapter_id", report.candidate_adapter_id},
        {"baseline_profile_id", report.baseline_profile_id},
        {"candidate_profile_id", report.candidate_profile_id},
        {"corpus_revision_id", report.corpus_revision_id},
        {"corpus_bundle_hash", report.corpus_bundle_hash},
        {"base_training_fingerprint", report.base_training_fingerprint},
        {"test_suite_revision", report.test_suite_revision},
        {"gates", {
            {"intended_behavior", report.intended_behavior_passed},
            {"retention", report.retention_passed},
            {"agent_regression", report.agent_regression_passed},
        }},
        {"evaluated_turns", report.evaluated_turns},
        {"baseline_runtime_interventions", report.baseline_runtime_interventions},
        {"baseline_intervention_rate", report.evaluated_turns == 0 ? 0.0 :
            static_cast<double>(report.baseline_runtime_interventions) /
                static_cast<double>(report.evaluated_turns)},
        {"runtime_interventions", report.runtime_interventions},
        {"runtime_intervention_rate", common_learning_runtime_intervention_rate(report)},
        {"status", report.status},
        {"evaluated_at", report.evaluated_at},
    }.dump();
}

bool common_learning_evaluation_report_from_json(
        const std::string & text,
        common_learning_evaluation_report & report,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        const auto gates = value.value("gates", json::object());
        if (!value.is_object() || !gates.is_object()) {
            error = "evaluation report is not an object";
            return false;
        }
        report = {};
        report.schema_version = value.value("schema_version", 0);
        report.revision_id = value.value("revision_id", "");
        report.candidate_adapter_id = value.value("candidate_adapter_id", "");
        report.baseline_profile_id = value.value("baseline_profile_id", "");
        report.candidate_profile_id = value.value("candidate_profile_id", "");
        report.corpus_revision_id = value.value("corpus_revision_id", "");
        report.corpus_bundle_hash = value.value("corpus_bundle_hash", "");
        report.base_training_fingerprint = value.value("base_training_fingerprint", "");
        report.test_suite_revision = value.value("test_suite_revision", "");
        report.intended_behavior_passed = gates.value("intended_behavior", false);
        report.retention_passed = gates.value("retention", false);
        report.agent_regression_passed = gates.value("agent_regression", false);
        report.evaluated_turns = value.value("evaluated_turns", size_t{0});
        report.baseline_runtime_interventions = value.value(
            "baseline_runtime_interventions", size_t{0});
        report.runtime_interventions = value.value("runtime_interventions", size_t{0});
        report.status = value.value("status", "failed");
        report.evaluated_at = value.value("evaluated_at", "");
        return common_learning_validate_evaluation_report(report, error);
    } catch (const std::exception & exception) {
        error = std::string("invalid evaluation report JSON: ") + exception.what();
        return false;
    }
}
