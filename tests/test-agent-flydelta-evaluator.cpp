#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_job base_job(
        common_flydelta_experiment_job_kind kind, const std::string & id) {
    common_flydelta_experiment_job job;
    job.id = id;
    job.kind = kind;
    job.seed.id = "evidence://flydelta/evaluator";
    job.seed.behavior_key = "tool_use/diagnostics/missing-argument";
    job.seed.source = common_adaptation_evidence_source::tool_repair;
    job.seed.scope.namespace_id = "local";
    job.seed.scope.project_id = "project";
    job.seed.scope.session_id = "session";
    job.seed.task_fingerprint = "sha256:task";
    job.seed.model_profile_fingerprint = "sha256:model";
    job.seed.tokenizer_fingerprint = "sha256:tokenizer";
    job.seed.template_fingerprint = "sha256:template";
    job.seed.execution_context_fingerprint = "sha256:execution-context";
    job.seed.baseline_ref = "execution:failed";
    job.seed.candidate_ref = "execution:repaired";
    job.seed.verifier_ref = "verifier:v1";
    job.seed.evidence_ref = "evidence://flydelta/evaluator";
    job.seed.transaction_ids = {"learning://failed", "learning://repaired"};
    job.learning_rate = 0.5f;
    job.decay = 1.0f;
    job.code_revision = "evaluator-test:v1";
    return job;
}

static common_flydelta_counterfactual_report report(const std::string & id) {
    common_flydelta_counterfactual_report value;
    value.experiment_id = id;
    value.fixture_id = "flydelta://fixture/evaluator";
    value.candidate_id = "flydelta://candidate/evaluator";
    value.baseline_profile_id = "base";
    value.candidate_profile_id = "overlay";
    value.baseline.executed = true;
    value.baseline.verifier_known = true;
    value.baseline.passed = false;
    value.baseline.evidence_ref = "evidence:baseline";
    value.candidate.executed = true;
    value.candidate.verifier_known = true;
    value.candidate.passed = true;
    value.candidate.overlay_applied = true;
    value.candidate.evidence_ref = "evidence:candidate";
    value.outcome = common_flydelta_counterfactual_outcome::helped;
    value.quality_delta = 1.0f;
    return value;
}

static common_flydelta_behavior_delta behavior_delta() {
    common_flydelta_behavior_delta value;
    value.id = "flydelta://behavior/evaluator";
    value.capture_manifest_id = "flydelta://capture/evaluator";
    value.host_evidence_ref = "evidence://repair/evaluator";
    value.model_profile_fingerprint = "sha256:model";
    value.capture_layout_revision = "layout:v1";
    value.layer_index = 4;
    value.values = {1.0f, 0.0f, 0.0f};
    return value;
}

static common_flydelta_training_example training_example(
        common_flydelta_training_split split) {
    common_flydelta_training_example value;
    value.id = "flydelta://training/evaluator";
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.context_fingerprint = "sha256:context";
    value.basis_revision = "flydelta://basis/evaluator";
    value.evidence_ref = "evidence://training/evaluator";
    value.split = split;
    value.context.expansion_dim = 8;
    value.context.indices = {1};
    value.context.values = {1.0f};
    value.target_coefficients = {0.25f, -0.1f};
    value.outcome = common_flydelta_counterfactual_outcome::helped;
    value.confidence = 1.0f;
    return value;
}

int main() {
    std::string error;
    common_flydelta_evaluator_config config;
    config.basis.dimension = 3;
    config.basis.max_directions = 4;
    config.basis.model_profile_fingerprint = "sha256:model";
    config.basis.capture_layout_revision = "layout:v1";
    config.memory = {8, 2, 1.0f};

    common_flydelta_evaluator_result result;
    auto counterfactual = base_job(
            common_flydelta_experiment_job_kind::counterfactual,
            "flydelta://job/counterfactual");
    counterfactual.capture_manifest_ids = {"flydelta://capture/evaluator"};
    counterfactual.alpha_search.candidates = {0.05f};
    counterfactual.alpha_search.max_candidates = 1;
    common_flydelta_evaluator_callbacks callbacks;
    callbacks.run_counterfactual = [&](const auto & job, auto & reports, std::string &) {
        reports.push_back(report(job.id));
        return true;
    };
    CHECK(common_flydelta_evaluate_job(counterfactual, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && result.counterfactual_reports.size() == 1);

    auto basis = base_job(common_flydelta_experiment_job_kind::basis,
            "flydelta://job/basis");
    basis.behavior_delta_ids = {"flydelta://behavior/evaluator"};
    callbacks = {};
    callbacks.resolve_behavior_delta = [](const auto &, auto & delta, auto & credit, std::string &) {
        delta = behavior_delta();
        credit.experiment_id = "flydelta://experiment/evaluator";
        credit.candidate_id = "flydelta://candidate/evaluator";
        credit.fixture_id = "flydelta://fixture/evaluator";
        credit.outcome = common_flydelta_counterfactual_outcome::helped;
        credit.quality_delta = 1.0f;
        credit.eligible_for_learning = true;
        return true;
    };
    CHECK(common_flydelta_evaluate_job(basis, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && result.basis_directions.size() == 1);

    auto memory = base_job(common_flydelta_experiment_job_kind::delta_memory,
            "flydelta://job/memory");
    memory.training_example_ids = {"flydelta://training/evaluator"};
    callbacks = {};
    callbacks.resolve_training_example = [](const auto &, auto & example, std::string &) {
        example = training_example(common_flydelta_training_split::train);
        return true;
    };
    CHECK(common_flydelta_evaluate_job(memory, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && !result.delta_memory_weights.empty());

    callbacks.resolve_training_example = [](const auto &, auto & example, std::string &) {
        example = training_example(common_flydelta_training_split::holdout);
        return true;
    };
    CHECK(!common_flydelta_evaluate_job(memory, config, callbacks, result, error));
    CHECK(error.find("validation or holdout") != std::string::npos);
    return 0;
}
