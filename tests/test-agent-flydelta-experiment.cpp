#include "agent/adaptation/flydelta/flydelta-experiment.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture fixture() {
    common_flydelta_experiment_fixture value;
    value.id = "flydelta://fixture/1";
    value.task_fingerprint = "sha256:task";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.verifier_revision = "verifier:v1";
    return value;
}

static common_flydelta_counterfactual_trial trial(bool passed, bool overlay, const char * evidence) {
    common_flydelta_counterfactual_trial value;
    value.executed = true;
    value.host_evaluated = true;
    value.verifier_known = true;
    value.passed = passed;
    value.quality = passed ? 1.0f : 0.25f;
    value.overlay_applied = overlay;
    value.intervention_count = overlay ? 1 : 0;
    value.evidence_ref = evidence;
    return value;
}

int main() {
    std::string error;
    const auto experiment_fixture = fixture();
    CHECK(common_flydelta_experiment_fixture_validate(experiment_fixture, error));

    common_flydelta_counterfactual_report report;
    int calls = 0;
    CHECK(common_flydelta_run_counterfactual(
        "flydelta://experiment/1", "flydelta://candidate/1", "base", "overlay",
        experiment_fixture,
        [&](const auto & received, bool apply_overlay, auto & result, auto &) {
            if (received.id != experiment_fixture.id) return false;
            ++calls;
            result = trial(apply_overlay, apply_overlay, apply_overlay ? "evidence:candidate" : "evidence:baseline");
            return true;
        }, report, error));
    CHECK(calls == 2);
    CHECK(report.outcome == common_flydelta_counterfactual_outcome::helped);
    CHECK(report.quality_delta > 0.0f);

    const auto json = common_flydelta_counterfactual_report_to_json(report);
    common_flydelta_counterfactual_report parsed;
    CHECK(common_flydelta_counterfactual_report_from_json(json, parsed, error));
    CHECK(parsed.outcome == common_flydelta_counterfactual_outcome::helped);
    CHECK(parsed.baseline.host_evaluated && parsed.candidate.host_evaluated);

    const auto baseline_pass = trial(true, false, "evidence:baseline");
    const auto candidate_fail = trial(false, true, "evidence:candidate");
    CHECK(common_flydelta_classify_counterfactual(baseline_pass, candidate_fail) ==
        common_flydelta_counterfactual_outcome::harmed);
    auto unknown = candidate_fail;
    unknown.verifier_known = false;
    CHECK(common_flydelta_classify_counterfactual(baseline_pass, unknown) ==
        common_flydelta_counterfactual_outcome::unknown);
    CHECK(common_flydelta_classify_counterfactual(baseline_pass, baseline_pass) ==
        common_flydelta_counterfactual_outcome::neutral);

    common_flydelta_semantic_decision expected;
    expected.operation = "aggregate";
    expected.dataset = "dataset://local/sales";
    expected.group_by = {"region"};
    expected.aggregate_function = "sum";
    expected.aggregate_field = "amount";
    common_flydelta_semantic_progress_observation before;
    common_flydelta_semantic_progress_observation after;
    CHECK(common_flydelta_observe_semantic_progress(
        R"({"name":"statistics.describe","arguments":{"dataset":"dataset://local/sales","columns":["amount"]}})",
        expected, before, error));
    CHECK(common_flydelta_observe_semantic_progress(
        R"({"name":"data.aggregate","arguments":{"dataset":"dataset://local/sales","groupby":["region"],"columns":["amount"]}})",
        expected, after, error));
    const auto progress = common_flydelta_compare_semantic_progress(before, after, false, false);
    CHECK(progress.outcome == common_flydelta_semantic_progress_outcome::improved);
    CHECK(progress.candidate_score > progress.baseline_score);
    CHECK(!progress.residual_dimensions.empty());

    const auto failed_baseline = trial(false, false, "evidence:baseline-failed");
    const auto partial_candidate = trial(false, true, "evidence:partial-candidate");
    CHECK(common_flydelta_classify_counterfactual(failed_baseline, partial_candidate) ==
        common_flydelta_counterfactual_outcome::unknown);
    return 0;
}
