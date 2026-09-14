#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-promotion.h"
#include "agent/adaptation/flydelta/flydelta.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "check failed at line " << __LINE__ << ": " << #condition \
                  << " error=" << error << '\n'; \
        return 1; \
    } \
} while (false)

namespace {

common_agent_scope smoke_scope() {
    common_agent_scope scope;
    scope.namespace_id = "local";
    scope.project_id = "flydelta-tool-repair";
    scope.session_id = "flydelta-tool-repair-smoke";
    return scope;
}

common_learning_transaction transaction(
        const std::string & id,
        const std::string & turn_id,
        common_learning_signal_type signal_type,
        const std::string & tool_name,
        const std::string & evidence_id,
        const std::string & summary) {
    common_learning_transaction value;
    value.id = id;
    value.created_at = "2026-09-13T00:00:00Z";
    value.observation.id = id;
    value.observation.scope = smoke_scope();
    value.observation.scope.turn_id = turn_id;
    value.observation.source_turn_id = turn_id;
    value.observation.source_plan_id = "plan:tool-repair";
    value.observation.signals.push_back({
        signal_type,
        "plan:tool-repair",
        signal_type == common_learning_signal_type::tool_failure ? "step:wrong" : "step:repair",
        tool_name,
        evidence_id,
        summary,
        "data",
        "native",
    });
    value.observation.evidence_ids = {evidence_id};
    value.observation.cause = common_learning_cause::model_behavior;
    value.observation.verification = common_learning_verification::host_verified;
    value.observation.idempotency_key = id + ":idempotency";
    value.observation.collection_allowed = true;
    value.observation.content_hash = common_learning_observation_hash(value.observation);
    return value;
}

common_flydelta_experiment_fixture fixture() {
    common_flydelta_experiment_fixture value;
    value.id = "flydelta://fixture/tool-repair-1";
    value.task_fingerprint = "sha256:inspect-selected-dataset";
    value.model_profile_fingerprint = "profile:qwen-small";
    value.tokenizer_fingerprint = "tokenizer:qwen-small";
    value.template_fingerprint = "template:qwen-instruct";
    value.execution_context_fingerprint = "sha256:data-tool-resource-context-v1";
    value.verifier_revision = "verifier:tool-contract-v1";
    return value;
}

} // namespace

int main() {
    std::string error;
    const auto failed = transaction(
        "learning://transaction/wrong-tool",
        "turn:wrong-tool",
        common_learning_signal_type::tool_failure,
        "data.describe",
        "evidence:wrong-tool",
        "host rejected the wrong inspection tool for the selected dataset");
    const auto repaired = transaction(
        "learning://transaction/repaired-tool",
        "turn:repaired-tool",
        common_learning_signal_type::successful_recovery,
        "data.inspect",
        "evidence:repaired-tool",
        "host verified data.inspect for the selected dataset");
    CHECK(common_learning_transaction_validate(failed, 16, error));
    CHECK(common_learning_transaction_validate(repaired, 16, error));

    common_flydelta_behavior_transition transition;
    CHECK(common_flydelta_tool_repair_transition_from_transactions(
        failed, repaired, "sha256:inspect-selected-dataset",
        "tool_use/diagnostics/missing-argument",
        "execution:wrong-tool", "execution:repaired-tool", "verifier:tool-contract-v1",
        transition, error));

    common_flydelta_contrast_set contrast_set;
    CHECK(common_flydelta_contrast_set_from_transitions(
        "flydelta://contrast/tool-repair-1", "structured_tool_selection",
        {transition}, 8, contrast_set, error));
    CHECK(contrast_set.negative_transaction_ids.front() == failed.id);
    CHECK(contrast_set.positive_transaction_ids.front() == repaired.id);

    common_flydelta_counterfactual_report report;
    CHECK(common_flydelta_run_counterfactual(
        "flydelta://experiment/tool-repair-1",
        "flydelta://candidate/tool-repair-1",
        "profile:qwen-small",
        "profile:qwen-small+flydelta",
        fixture(),
        [](const auto &, bool apply_overlay, auto & trial, auto &) {
            // Deterministic host fixture: the baseline emits data.describe,
            // while the candidate overlay selects the verified data.inspect.
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.passed = apply_overlay;
            trial.quality = apply_overlay ? 1.0f : 0.0f;
            trial.overlay_applied = apply_overlay;
            trial.intervention_count = apply_overlay ? 1 : 0;
            trial.evidence_ref = apply_overlay
                ? "evidence:repaired-tool"
                : "evidence:wrong-tool";
            return true;
        }, report, error));
    CHECK(report.outcome == common_flydelta_counterfactual_outcome::helped);

    common_flydelta_intervention_credit credit;
    CHECK(common_flydelta_intervention_credit_from_report(report, credit, error));
    CHECK(credit.eligible_for_learning);

    common_flydelta_capture_manifest manifest;
    manifest.id = "flydelta://capture/tool-repair-1";
    manifest.observation_id = repaired.id;
    manifest.behavior_key = "tool_use/diagnostics/missing-argument";
    manifest.model_profile_fingerprint = fixture().model_profile_fingerprint;
    manifest.template_fingerprint = fixture().template_fingerprint;
    manifest.positive_execution_ref = "execution:repaired-tool";
    manifest.negative_execution_ref = "execution:wrong-tool";
    manifest.capture_layout_revision = "layout:cvec-v1";
    manifest.evidence_hash = "sha256:tool-repair-evidence";
    manifest.redaction_attested = true;
    manifest.captured_bytes = 2 * 4 * sizeof(float);

    common_flydelta_hidden_state_capture failed_capture;
    failed_capture.captured = true;
    failed_capture.model_profile_fingerprint = fixture().model_profile_fingerprint;
    failed_capture.capture_layout_revision = "layout:cvec-v1";
    failed_capture.layer_indices = {2};
    failed_capture.n_embd = 4;
    failed_capture.token_index = 3;
    failed_capture.values = {0.0f, 0.0f, 0.0f, 0.0f};
    common_flydelta_hidden_state_capture repaired_capture = failed_capture;
    repaired_capture.values = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<common_flydelta_behavior_delta> behavior_deltas;
    CHECK(common_flydelta_behavior_deltas_from_captures(
        manifest, failed_capture, repaired_capture, "evidence:repaired-tool",
        64 * 1024, 64 * 1024, behavior_deltas, error));
    CHECK(behavior_deltas.size() == 1 && behavior_deltas.front().layer_index == 2);
    common_flydelta_basis_config basis_config;
    basis_config.dimension = 4;
    basis_config.max_directions = 4;
    basis_config.cluster_similarity = 0.85f;
    basis_config.source = common_adaptation_evidence_source::tool_repair;
    basis_config.behavior_key = "tool_use/diagnostics/missing-argument";
    basis_config.model_profile_fingerprint = fixture().model_profile_fingerprint;
    basis_config.capture_layout_revision = "layout:cvec-v1";
    common_flydelta_basis_builder basis(basis_config);
    CHECK(basis.add(behavior_deltas.front(), credit, error));
    CHECK(basis.directions().size() == 1);

    common_flydelta_promotion_policy promotion_policy;
    promotion_policy.min_trials = 1;
    promotion_policy.min_known_trials = 1;
    promotion_policy.min_helped_trials = 1;
    promotion_policy.min_help_confidence = 1.0f;
    promotion_policy.max_unknown_ratio = 0.0f;
    common_flydelta_promotion_summary promotion;
    CHECK(common_flydelta_promotion_summary_from_reports(
        "flydelta://promotion/tool-repair-1", {report}, promotion_policy, promotion, error));
    CHECK(promotion.status == common_flydelta_candidate_status::eligible);

    common_flydelta_encoder encoder({0x1234, 4, 16, 2, 2});
    common_flydelta_sparse_code code;
    CHECK(encoder.encode({1.0f, 0.0f, 0.0f, 0.0f}, code, error));
    common_flydelta_recognition_memory recognition(8);
    CHECK(recognition.remember(code, error));
    CHECK(recognition.familiarity(code) > 0.99f);

    common_flydelta_delta_memory memory({16, 1, 1.0f});
    std::vector<float> before;
    CHECK(memory.predict(code, before, error));
    CHECK(before.size() == 1 && std::fabs(before[0]) < 0.001f);
    CHECK(memory.learn(code, {1.0f}, 1.0f, 1.0f, 1.0f, error));
    std::vector<float> coefficients;
    CHECK(memory.predict(code, coefficients, error));
    CHECK(coefficients.size() == 1 && coefficients[0] > 0.5f);

    const auto choose_tool = [](float correction) {
        return correction > 0.05f ? std::string("data.inspect") : std::string("data.describe");
    };
    CHECK(choose_tool(before[0]) == "data.describe");
    CHECK(choose_tool(coefficients[0]) == "data.inspect");

    common_flydelta_gate_config gate_config;
    gate_config.enabled = true;
    gate_config.max_scale = 0.25f;
    common_flydelta_activation_request activation_request;
    activation_request.candidate_id = "flydelta://candidate/tool-repair-1";
    activation_request.artifact_id = "flydelta://artifact/tool-repair-1";
    activation_request.model_profile_fingerprint = fixture().model_profile_fingerprint;
    activation_request.capture_layout_revision = "layout:cvec-v1";
    activation_request.model_n_embd = 4;
    activation_request.model_n_layers = 4;
    activation_request.il_start = 1;
    activation_request.il_end = 3;
    activation_request.directions = basis.directions();
    activation_request.coefficients = coefficients;
    activation_request.gate_request.explicit_opt_in = true;
    // Eligibility is not approval. This is the explicit host approval step.
    activation_request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
    activation_request.gate_request.basis_available = true;
    activation_request.gate_request.familiarity = recognition.familiarity(code);
    activation_request.gate_request.novelty = recognition.novelty(code);
    activation_request.gate_request.requested_scale = 0.2f;
    common_flydelta_activation_result activation;
    CHECK(common_flydelta_prepare_activation(
        gate_config, activation_request, 64 * 1024, activation, error));
    CHECK(activation.gate.apply && activation.overlay.enabled);
    CHECK(std::fabs(activation.overlay.data[4]) > 0.0f);

    activation_request.gate_request.familiarity = 0.0f;
    activation_request.gate_request.novelty = 1.0f;
    CHECK(common_flydelta_prepare_activation(
        gate_config, activation_request, 64 * 1024, activation, error));
    CHECK(!activation.gate.apply && !activation.overlay.enabled);

    std::cout << "wrong_tool_before_learning=data.describe\n"
              << "repaired_tool=data.inspect\n"
              << "contrastive_outcome=helped\n"
              << "promotion=eligible_then_host_approved\n"
              << "learned_tool=data.inspect\n"
              << "activation=applied_for_familiar_context\n"
              << "novel_context=no_op\n";
    return 0;
}
