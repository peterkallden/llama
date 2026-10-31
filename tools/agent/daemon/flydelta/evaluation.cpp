#include "../agent-daemon-flydelta-internal.h"
#include "agent/adaptation/flydelta/flydelta-search-budget.h"

#include "agent/agent-residual-patch.h"
#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "state.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_run_counterfactual(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_counterfactual_report> & reports,
        std::string & error) {
    error.clear();
    reports.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    if (job.seed.baseline_ref.empty() || job.seed.candidate_ref.empty() ||
            job.alpha_search.candidates.empty()) {
        error = "FlyDelta counterfactual requires baseline, candidate and alpha references";
        return false;
    }

    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref,
            directions, error) || directions.empty()) return false;
    // Counterfactual jobs are the existing rank-one refinement contract. A
    // multi-direction candidate must go through its existing search phase;
    // silently selecting one direction here would change experiment meaning.
    if (directions.size() != 1) {
        error = "FlyDelta counterfactual requires one direction; use search_pipeline for a basis";
        return false;
    }
    const uint32_t layer = static_cast<uint32_t>(directions.front().layer_index);

    for (const float alpha : job.alpha_search.candidates) {
        const std::string alpha_id = job.seed.candidate_ref + ":alpha:" +
            std::to_string(alpha);
        const common_flydelta_counterfactual_runner runner =
            [provider, &job, layer, alpha](
                    const common_flydelta_experiment_fixture &,
                    bool apply_overlay,
                    common_flydelta_counterfactual_trial & trial,
                    std::string & runner_error) {
                common_flydelta_arm_batch_request batch;
                batch.batch_id = job.id + ":counterfactual:" + std::to_string(alpha);
                batch.wave_id = "counterfactual";
                common_flydelta_arm_request arm;
                arm.job_id = job.id;
                arm.wave_id = batch.wave_id;
                arm.proposal_index = apply_overlay ? 1 : 0;
                arm.arm_id = batch.batch_id + (apply_overlay ? ":candidate" : ":baseline");
                arm.context_ref = job.seed.baseline_ref;
                arm.fixture_ref = job.seed.verifier_ref;
                arm.intervention_ref = job.seed.candidate_ref;
                arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
                arm.layer_indices = {layer};
                arm.coefficients = {1.0f};
                arm.alpha = apply_overlay ? alpha : 0.0f;
                arm.apply_overlay = apply_overlay;
                arm.fresh_context = true;
                arm.request_generation = true;
                arm.request_host_verification = true;
                arm.max_capture_bytes = 4U * 1024U * 1024U;
                const size_t runtime_token_limit = provider->n_predict > 0
                    ? static_cast<size_t>(provider->n_predict) : 256U;
                arm.max_generated_tokens = job.kind ==
                        common_flydelta_experiment_job_kind::evaluation
                    ? std::min(runtime_token_limit, job.evaluation_limits.max_generated_tokens)
                    : runtime_token_limit;
                if (!common_flydelta_arm_request_validate(arm, runner_error)) return false;
                batch.arms.push_back(std::move(arm));
                common_flydelta_arm_batch_result executed;
                if (!daemon_flydelta_execute_batch(provider, batch, executed, runner_error) ||
                        executed.arms.size() != 1) return false;
                const auto & arm_result = executed.arms.front();
                trial = {};
                trial.executed = arm_result.executed;
                trial.host_evaluated = arm_result.host_evaluated;
                trial.verifier_known = arm_result.verifier_known;
                // finalize_arm deliberately reports a single semantic pass as
                // NEUTRAL. At this layer that pass is the trial predicate;
                // common_flydelta_classify_counterfactual() then derives the
                // relational HELPED outcome from baseline versus candidate.
                trial.passed = daemon_flydelta_arm_semantic_passed(arm_result);
                trial.quality = trial.passed ? 1.0f : 0.0f;
                trial.overlay_applied = arm_result.overlay_applied;
                trial.intervention_count = apply_overlay ? 1 : 0;
                trial.semantic_progress_observation = arm_result.semantic_progress_observation;
                trial.observed_decision_summary = arm_result.observed_decision_summary;
                trial.expected_decision_summary = arm_result.expected_decision_summary;
                trial.verifier_reason = arm_result.verifier_reason;
                trial.evidence_ref = arm_result.generation_ref;
                return common_flydelta_counterfactual_trial_validate(trial, runner_error);
            };
        common_flydelta_counterfactual_report report;
        const std::string report_candidate_id = job.evaluation_candidate_id.empty()
            ? alpha_id : job.evaluation_candidate_id;
        if (!common_flydelta_run_counterfactual(
                job.id, report_candidate_id, job.seed.baseline_ref,
                report_candidate_id, fixture, runner, report, error)) return false;
        reports.push_back(std::move(report));
    }
    return !reports.empty();
}

bool daemon_flydelta_run_causal_diagnostic(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        common_flydelta_causal_diagnostic_report & report,
        std::string & error) {
    error.clear();
    report = {};
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!provider || job.causal_diagnostic_manifest_ref.empty()) {
        error = "FlyDelta causal diagnostic requires a scoped manifest";
        return false;
    }
    json manifest;
    if (!daemon_flydelta_read_json(*provider, job.causal_diagnostic_manifest_ref,
            manifest, error)) return false;
    if (manifest.value("kind", "") != "flydelta_causal_diagnostic_manifest" ||
            manifest.value("schema_version", 0) != 1) {
        error = "FlyDelta causal diagnostic manifest kind or schema is invalid";
        return false;
    }
    const std::string baseline_ref = manifest.value("baseline_capture_ref", "");
    const std::string repair_ref = manifest.value("repair_capture_ref", "");
    const std::string control_ref = manifest.value("control_capture_ref", "");
    const std::string direction_ref = manifest.value("direction_ref", "");
    const int32_t layer = manifest.value("layer", -1);
    const int32_t absolute_position = manifest.value("absolute_position", -1);
    if (baseline_ref.empty() || repair_ref.empty() || layer < 0 || absolute_position < 0) {
        error = "FlyDelta causal diagnostic manifest lacks aligned capture site";
        return false;
    }
    std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
    std::shared_ptr<const common_flydelta_hidden_state_capture> repair_capture;
    if (!daemon_flydelta_capture_from_reference(provider, baseline_ref, baseline_capture, error) ||
            !daemon_flydelta_capture_from_reference(provider, repair_ref, repair_capture, error)) {
        return false;
    }
    std::vector<float> baseline_values;
    std::vector<float> repair_values;
    if (!daemon_flydelta_capture_layer(*baseline_capture, layer, baseline_values, error) ||
            !daemon_flydelta_capture_layer(*repair_capture, layer, repair_values, error) ||
            baseline_values.size() != repair_values.size() || baseline_values.empty()) {
        error = "FlyDelta causal diagnostic captures are not dimension-aligned";
        return false;
    }
    std::vector<float> control_values;
    if (!control_ref.empty()) {
        std::shared_ptr<const common_flydelta_hidden_state_capture> control_capture;
        if (!daemon_flydelta_capture_from_reference(provider, control_ref, control_capture, error) ||
                !daemon_flydelta_capture_layer(*control_capture, layer, control_values, error) ||
                control_values.size() != baseline_values.size()) {
            error = "FlyDelta causal diagnostic control capture is not dimension-aligned";
            return false;
        }
    }
    std::vector<float> direction_values;
    if (!direction_ref.empty()) {
        std::vector<common_flydelta_basis_direction> directions;
        if (!daemon_flydelta_parse_directions(*provider, direction_ref, directions, error)) {
            return false;
        }
        const auto it = std::find_if(directions.begin(), directions.end(),
            [layer](const auto & value) { return value.layer_index == layer; });
        if (it != directions.end()) direction_values = it->values;
        if (!direction_values.empty() && direction_values.size() != baseline_values.size()) {
            error = "FlyDelta causal diagnostic direction dimension does not match captures";
            return false;
        }
    }
    const auto requested_kinds = manifest.value(
        "patch_kinds", std::vector<std::string>{
            "self_replacement", "exact_replacement", "scaled_exact_delta",
            "existing_direction", "norm_matched_control"});
    if (requested_kinds.empty() || requested_kinds.size() > 8) {
        error = "FlyDelta causal diagnostic patch family budget is invalid";
        return false;
    }
    const float scaled_alpha = manifest.value("scaled_alpha", 0.5f);
    const size_t max_full_generation = std::min<size_t>(2,
        manifest.value("max_full_generation", 2U));
    if (!std::isfinite(scaled_alpha) || scaled_alpha <= 0.0f || scaled_alpha > 1.0f) {
        error = "FlyDelta causal diagnostic scaled alpha is invalid";
        return false;
    }
    common_flydelta_arm_batch_request diagnostic_batch;
    diagnostic_batch.batch_id = job.id + ":causal-diagnostic";
    diagnostic_batch.wave_id = "causal-diagnostic";
    std::vector<common_flydelta_causal_patch_kind> kinds;
    std::vector<std::shared_ptr<const common_agent_residual_patch_request>> patches;
    const auto make_patch = [&](const common_flydelta_causal_patch_kind kind,
            std::shared_ptr<const common_agent_residual_patch_request> & patch) {
        auto value = std::make_shared<common_agent_residual_patch_request>();
        value->site = common_agent_residual_patch_site::layer_input_residual;
        value->layer = static_cast<uint32_t>(layer);
        value->absolute_position = absolute_position;
        value->sequence_id = manifest.value("sequence_id", -1);
        value->observe_applied_vector = true;
        value->identity = job.id + ":" + common_flydelta_causal_patch_kind_name(kind);
        switch (kind) {
            case common_flydelta_causal_patch_kind::self_replacement:
                value->operation = common_agent_residual_patch_operation::replace;
                value->values = baseline_values;
                break;
            case common_flydelta_causal_patch_kind::exact_replacement:
                value->operation = common_agent_residual_patch_operation::replace;
                value->values = repair_values;
                break;
            case common_flydelta_causal_patch_kind::scaled_exact_delta:
                value->operation = common_agent_residual_patch_operation::add;
                value->values.resize(baseline_values.size());
                for (size_t index = 0; index < baseline_values.size(); ++index) {
                    value->values[index] = scaled_alpha *
                        (repair_values[index] - baseline_values[index]);
                }
                break;
            case common_flydelta_causal_patch_kind::existing_direction:
                if (direction_values.empty()) return false;
                value->operation = common_agent_residual_patch_operation::add;
                value->values = direction_values;
                break;
            case common_flydelta_causal_patch_kind::norm_matched_control:
                value->operation = common_agent_residual_patch_operation::add;
                value->values.resize(baseline_values.size());
                for (size_t index = 0; index < baseline_values.size(); ++index) {
                    value->values[index] = std::sin(static_cast<float>(
                        (index + 1) * (layer + 3)) * 0.017f);
                }
                break;
            case common_flydelta_causal_patch_kind::wrong_concept:
                if (control_values.empty()) return false;
                value->operation = common_agent_residual_patch_operation::replace;
                value->values = control_values;
                break;
        }
        float patch_norm = 0.0f;
        float target_norm = 0.0f;
        for (size_t index = 0; index < value->values.size(); ++index) {
            patch_norm += value->values[index] * value->values[index];
            const float delta = repair_values[index] - baseline_values[index];
            target_norm += delta * delta;
        }
        if (kind == common_flydelta_causal_patch_kind::norm_matched_control &&
                patch_norm > 1.0e-12f && target_norm > 0.0f) {
            const float scale = std::sqrt(target_norm / patch_norm);
            for (float & item : value->values) item *= scale;
        }
        if (value->values.size() != baseline_values.size()) return false;
        patch = std::move(value);
        return true;
    };
    for (const auto & kind_name : requested_kinds) {
        common_flydelta_causal_patch_kind kind;
        if (!common_flydelta_causal_patch_kind_from_name(kind_name, kind)) {
            error = "FlyDelta causal diagnostic patch family is unknown";
            return false;
        }
        std::shared_ptr<const common_agent_residual_patch_request> patch;
        if (!make_patch(kind, patch)) continue;
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = diagnostic_batch.wave_id;
        arm.proposal_index = diagnostic_batch.arms.size();
        arm.arm_id = job.id + ":causal:" + std::to_string(arm.proposal_index);
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = job.seed.verifier_ref;
        arm.intervention_ref = job.causal_diagnostic_manifest_ref;
        arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
        arm.fresh_context = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = false;
        arm.request_host_verification = false;
        arm.residual_patch = patch;
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        diagnostic_batch.arms.push_back(std::move(arm));
        kinds.push_back(kind);
        patches.push_back(std::move(patch));
    }
    if (diagnostic_batch.arms.empty()) {
        error = "FlyDelta causal diagnostic produced no compatible patch arms";
        return false;
    }
    common_flydelta_arm_batch_result diagnostic_result;
    if (!daemon_flydelta_execute_batch(provider, diagnostic_batch, diagnostic_result, error) ||
            diagnostic_result.arms.size() != diagnostic_batch.arms.size()) return false;
    report.experiment_id = job.id;
    report.manifest_ref = job.causal_diagnostic_manifest_ref;
    report.model_profile_fingerprint = provider->model_profile_fingerprint;
    report.capture_layout_revision = provider->capture_layout_revision;
    report.arms.reserve(diagnostic_result.arms.size());
    float baseline_margin = 0.0f;
    bool baseline_available = false;
    for (size_t index = 0; index < diagnostic_result.arms.size(); ++index) {
        const auto & result = diagnostic_result.arms[index];
        common_flydelta_causal_diagnostic_arm arm;
        arm.arm_id = result.arm_id;
        arm.patch_kind = kinds[index];
        arm.layer = static_cast<uint32_t>(layer);
        arm.absolute_position = absolute_position;
        arm.sequence_id = patches[index]->sequence_id;
        arm.executed = result.executed;
        arm.patch_attempted = true;
        arm.patch_applied = result.executed;
        arm.margin_available = result.margin_available;
        arm.candidate_margin = result.margin_total;
        if (kinds[index] == common_flydelta_causal_patch_kind::self_replacement &&
                result.margin_available) {
            baseline_margin = result.margin_total;
            baseline_available = true;
        }
        if (baseline_available && result.margin_available) {
            arm.baseline_margin = baseline_margin;
            arm.margin_delta = arm.candidate_margin - baseline_margin;
        }
        report.arms.push_back(std::move(arm));
    }
    std::vector<size_t> full_indices;
    for (size_t index = 0; index < report.arms.size(); ++index) {
        if (report.arms[index].margin_available && report.arms[index].margin_delta > 0.0f) {
            full_indices.push_back(index);
        }
    }
    std::sort(full_indices.begin(), full_indices.end(), [&](const size_t left, const size_t right) {
        return report.arms[left].margin_delta > report.arms[right].margin_delta;
    });
    if (full_indices.size() > max_full_generation) full_indices.resize(max_full_generation);
    if (!full_indices.empty()) {
        common_flydelta_arm_batch_request full_batch;
        full_batch.batch_id = job.id + ":causal-diagnostic-full";
        full_batch.wave_id = "causal-diagnostic-full";
        for (const size_t index : full_indices) {
            auto arm = diagnostic_batch.arms[index];
            arm.wave_id = full_batch.wave_id;
            arm.request_generation = true;
            arm.request_host_verification = true;
            arm.max_generated_tokens = std::max<size_t>(
                64U, provider->n_predict > 0 ? provider->n_predict : 64U);
            full_batch.arms.push_back(std::move(arm));
        }
        common_flydelta_arm_batch_result full_result;
        if (!daemon_flydelta_execute_batch(provider, full_batch, full_result, error) ||
                full_result.arms.size() != full_batch.arms.size()) return false;
        for (size_t index = 0; index < full_result.arms.size(); ++index) {
            const auto & full = full_result.arms[index];
            auto & target = report.arms[full_indices[index]];
            target.host_evaluated = full.host_evaluated;
            target.verifier_known = full.verifier_known;
            target.host_outcome = full.host_outcome;
            target.evidence_ref = full.generation_ref;
        }
    }
    return common_flydelta_causal_diagnostic_report_validate(report, error);
}

bool daemon_flydelta_run_evaluation(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        common_flydelta_evaluation_report & report,
        std::vector<common_flydelta_evaluation_fixture_result> & fixture_results,
        std::string & error) {
    error.clear();
    report = {};
    fixture_results.clear();
    const auto scoped_provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!common_flydelta_evaluation_limits_validate(job.evaluation_limits, error)) return false;
    json candidate_artifact;
    if (!daemon_flydelta_read_json_bounded(
            *scoped_provider, job.seed.candidate_ref, 4U * 1024U * 1024U,
            candidate_artifact, error) || candidate_artifact.value("kind", "") != "flydelta") {
        error = "FlyDelta evaluation candidate reference must resolve to a flydelta artifact";
        return false;
    }
    common_flydelta_artifact artifact;
    if (!common_flydelta_artifact_from_json(
            candidate_artifact.dump(), 1U << 20, 4U * 1024U * 1024U,
            artifact, error) || artifact.id != job.evaluation_candidate_id ||
            (!job.seed.evidence_ref.empty() && artifact.content_hash != job.seed.evidence_ref)) {
        if (error.empty()) error = "FlyDelta evaluation candidate artifact identity is incompatible";
        return false;
    }
    json suite;
    if (!daemon_flydelta_read_json(*scoped_provider, job.evaluation_suite_ref, suite, error)) return false;
    if (suite.value("kind", "") != "flydelta_evaluation_suite" ||
            suite.value("revision", "") != job.evaluation_revision ||
            !suite.contains("fixtures") || !suite["fixtures"].is_array() ||
            suite["fixtures"].empty() ||
            suite["fixtures"].size() > job.evaluation_limits.max_fixtures) {
        error = "FlyDelta evaluation suite is missing a matching revision or bounded fixtures";
        return false;
    }
    report.schema_version = 1;
    report.revision_id = job.evaluation_revision;
    report.candidate_id = job.evaluation_candidate_id;
    report.baseline_profile_id = job.seed.model_profile_fingerprint + ":baseline";
    report.candidate_profile_id = job.evaluation_candidate_id;
    report.test_suite_revision = job.evaluation_revision;
    report.intended_behavior_passed = true;
    report.retention_passed = true;
    report.agent_regression_passed = true;
    std::set<common_flydelta_evaluation_suite_kind> seen_kinds;
    size_t model_calls = 0;
    std::set<std::string> seen_fixture_refs;
    for (const auto & item : suite["fixtures"]) {
        if (!item.is_object()) {
            error = "FlyDelta evaluation suite contains a malformed fixture entry";
            return false;
        }
        common_flydelta_evaluation_suite_kind suite_kind;
        if (!common_flydelta_evaluation_suite_kind_from_name(
                item.value("suite_kind", ""), suite_kind)) {
            error = "FlyDelta evaluation fixture has an unknown suite kind";
            return false;
        }
        const std::string fixture_ref = item.value("fixture_ref", "");
        const std::string context_ref = item.value("context_ref", "");
        if (fixture_ref.empty() || context_ref.empty()) {
            error = "FlyDelta evaluation fixture requires fixture_ref and context_ref";
            return false;
        }
        if (!seen_fixture_refs.insert(fixture_ref).second) {
            error = "FlyDelta evaluation suite contains a duplicate fixture reference";
            return false;
        }
        auto fixture_job = job;
        fixture_job.id = job.id + ":fixture:" + std::to_string(fixture_results.size());
        fixture_job.seed.id = fixture_job.id;
        fixture_job.seed.baseline_ref = context_ref;
        fixture_job.seed.verifier_ref = fixture_ref;
        fixture_job.alpha_search.candidates = {1.0f};
        fixture_job.alpha_search.max_candidates = 1;
        std::vector<common_flydelta_counterfactual_report> reports;
        bool completed = false;
        std::string last_error;
        for (size_t attempt = 0; attempt <= job.evaluation_limits.max_retries; ++attempt) {
            if (model_calls + 2 > job.evaluation_limits.max_model_calls) {
                error = "FlyDelta evaluation model-call limit exhausted";
                return false;
            }
            reports.clear();
            std::string attempt_error;
            if (daemon_flydelta_run_counterfactual(
                    scoped_provider, fixture_job, reports, attempt_error) &&
                    reports.size() == 1) {
                completed = true;
                model_calls += 2;
                break;
            }
            model_calls += 2;
            last_error = attempt_error;
        }
        if (!completed) {
            error = last_error.empty()
                ? "FlyDelta evaluation fixture returned no report" : last_error;
            return false;
        }
        const auto & counterfactual = reports.front();
        common_flydelta_evaluation_fixture_result fixture_result;
        fixture_result.candidate_id = job.evaluation_candidate_id;
        fixture_result.suite_kind = suite_kind;
        fixture_result.fixture_ref = fixture_ref;
        fixture_result.verifier_revision = job.evaluation_revision;
        fixture_result.baseline_known = counterfactual.baseline.verifier_known;
        fixture_result.baseline_passed = counterfactual.baseline.passed;
        fixture_result.candidate_known = counterfactual.candidate.verifier_known;
        fixture_result.candidate_passed = counterfactual.candidate.passed;
        fixture_result.passed = fixture_result.baseline_known &&
            fixture_result.candidate_known && fixture_result.candidate_passed;
        fixture_result.report_ref = job.id + ":fixture:" + std::to_string(fixture_results.size());
        fixture_result.counterfactual = counterfactual;
        if (!common_flydelta_evaluation_fixture_result_validate(fixture_result, error)) return false;
        seen_kinds.insert(suite_kind);
        ++report.evaluated_turns;
        report.baseline_successes += fixture_result.baseline_passed ? 1 : 0;
        report.candidate_successes += fixture_result.candidate_passed ? 1 : 0;
        ++report.candidate_interventions;
        if (fixture_result.baseline_passed && !fixture_result.candidate_passed) ++report.false_interventions;
        if (suite_kind == common_flydelta_evaluation_suite_kind::intended ||
                suite_kind == common_flydelta_evaluation_suite_kind::holdout ||
                suite_kind == common_flydelta_evaluation_suite_kind::paraphrase ||
                suite_kind == common_flydelta_evaluation_suite_kind::transfer) {
            report.intended_behavior_passed = report.intended_behavior_passed && fixture_result.passed;
        } else if (suite_kind == common_flydelta_evaluation_suite_kind::retention ||
                suite_kind == common_flydelta_evaluation_suite_kind::control) {
            report.retention_passed = report.retention_passed && fixture_result.passed;
        } else {
            report.agent_regression_passed = report.agent_regression_passed && fixture_result.passed;
        }
        fixture_results.push_back(std::move(fixture_result));
    }
    if (seen_kinds.size() != 4) {
        error = "FlyDelta evaluation suite must contain intended, holdout, retention and agent_regression";
        return false;
    }
    report.status = report.intended_behavior_passed && report.retention_passed &&
        report.agent_regression_passed ? "passed" : "failed";

    // Persist the detailed Oracle observations as a separate immutable
    // resource.  The lifecycle evaluation report keeps only this reference
    // and provenance revisions, so promotion continues to consume the same
    // bounded gates as before without embedding a second evidence system.
    common_flydelta_oracle_suite_report oracle_report;
    const std::string oracle_identity = job.id + "\n" + job.evaluation_candidate_id +
        "\n" + job.evaluation_revision;
    oracle_report.id = "flydelta://oracle-suite/" +
        hash_sha256_hex(oracle_identity.data(), oracle_identity.size()).substr(0, 32);
    oracle_report.candidate_id = job.evaluation_candidate_id;
    oracle_report.evaluation_revision = job.evaluation_revision;
    oracle_report.model_profile_id = report.candidate_profile_id;
    oracle_report.oracle_ref = "flydelta://oracle/host-counterfactual";
    oracle_report.oracle_revision = "daemon:host-verifier-v1";
    oracle_report.policy_revision = "daemon:evaluation-gates-v1";
    oracle_report.observations.reserve(fixture_results.size());
    size_t controls = 0;
    size_t retained_controls = 0;
    size_t transfers = 0;
    int transfer_delta = 0;
    bool no_harm = true;
    for (const auto & fixture : fixture_results) {
        common_flydelta_oracle_suite_fixture_observation observation;
        observation.probe_id = fixture.report_ref;
        observation.suite_kind = common_flydelta_evaluation_suite_kind_name(fixture.suite_kind);
        observation.fixture_ref = fixture.fixture_ref;
        observation.verifier_revision = fixture.verifier_revision;
        observation.outcome = common_flydelta_counterfactual_outcome_name(
            fixture.counterfactual.outcome);
        observation.baseline_known = fixture.baseline_known;
        observation.baseline_passed = fixture.baseline_passed;
        observation.candidate_known = fixture.candidate_known;
        observation.candidate_passed = fixture.candidate_passed;
        oracle_report.observations.push_back(std::move(observation));
        if (fixture.counterfactual.outcome == common_flydelta_counterfactual_outcome::harmed) {
            no_harm = false;
        }
        if (fixture.suite_kind == common_flydelta_evaluation_suite_kind::retention ||
                fixture.suite_kind == common_flydelta_evaluation_suite_kind::control) {
            ++controls;
            if (fixture.counterfactual.outcome == common_flydelta_counterfactual_outcome::neutral) {
                ++retained_controls;
            }
        }
        if (fixture.suite_kind == common_flydelta_evaluation_suite_kind::transfer) {
            ++transfers;
            if (fixture.candidate_passed != fixture.baseline_passed) {
                transfer_delta += fixture.candidate_passed ? 1 : -1;
            }
        }
    }
    const float evaluated = static_cast<float>(std::max<size_t>(1, report.evaluated_turns));
    oracle_report.baseline_success_rate =
        static_cast<float>(report.baseline_successes) / evaluated;
    oracle_report.candidate_success_rate =
        static_cast<float>(report.candidate_successes) / evaluated;
    oracle_report.intervention_gain = oracle_report.candidate_success_rate -
        oracle_report.baseline_success_rate;
    oracle_report.false_intervention_rate = report.candidate_interventions == 0
        ? 0.0f : static_cast<float>(report.false_interventions) /
            static_cast<float>(report.candidate_interventions);
    oracle_report.control_retention = controls == 0
        ? 1.0f : static_cast<float>(retained_controls) / static_cast<float>(controls);
    oracle_report.transfer_gain = transfers == 0
        ? 0.0f : static_cast<float>(transfer_delta) /
            static_cast<float>(transfers);
    oracle_report.semantically_helped = oracle_report.intervention_gain > 0.0f && no_harm;
    oracle_report.safe_to_continue = no_harm &&
        oracle_report.candidate_success_rate >= oracle_report.baseline_success_rate;
    if (!common_flydelta_oracle_suite_report_validate(oracle_report, error)) return false;
    if (!daemon_flydelta_put_json_resource(
            scoped_provider,
            "flydelta-oracle-suite-" +
                hash_sha256_hex(oracle_identity.data(), oracle_identity.size()).substr(0, 24) +
                ".json",
            "flydelta_oracle_suite_report",
            job.evaluation_suite_ref,
            common_flydelta_oracle_suite_report_to_json(oracle_report),
            report.oracle_suite_report_ref,
            error)) return false;
    report.oracle_ref = oracle_report.oracle_ref;
    report.oracle_revision = oracle_report.oracle_revision;
    report.oracle_policy_revision = oracle_report.policy_revision;
    return common_flydelta_evaluation_report_validate(report, error);
}

bool daemon_flydelta_register_evaluator(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        common_flydelta_evaluator_config & config,
        common_flydelta_evaluator_callbacks & callbacks,
        std::string & callback_error) {
    config = {};
    config.pipeline.dimension = provider->model_n_embd;
    config.pipeline.max_directions = 1;
    config.pipeline.whirlpool_max_rounds = 1;
    config.pipeline.whirlpool_probes_per_round = 4;
    config.pipeline.whirlpool_max_trials = 4;
    config.pipeline.region_max_trials = 4;
    config.direction.dimension = provider->model_n_embd;
    config.direction.model_profile_fingerprint = provider->model_profile_fingerprint;
    config.direction.capture_layout_revision = provider->capture_layout_revision;
    config.direction_identity_from_job = true;
    config.initial_search_model_call_limit = provider->search_model_call_budget;
    callbacks = {};
    callbacks.run_evaluation = [provider](
            const common_flydelta_experiment_job & job,
            common_flydelta_evaluation_report & report,
            std::vector<common_flydelta_evaluation_fixture_result> & fixtures,
            std::string & callback_error) {
        return daemon_flydelta_run_evaluation(
            provider, job, report, fixtures, callback_error);
    };
    callbacks.run_causal_diagnostic = [provider](
            const common_flydelta_experiment_job & job,
            common_flydelta_causal_diagnostic_report & report,
            std::string & callback_error) {
        return daemon_flydelta_run_causal_diagnostic(
            provider, job, report, callback_error);
    };
    const auto pipeline = config.pipeline;
    callbacks.resolve_behavior_delta = [provider](const std::string & reference,
            common_flydelta_behavior_delta & delta,
            common_flydelta_intervention_credit & credit,
            std::string & resolver_error) {
        return daemon_flydelta_parse_behavior_delta(
            *provider, reference, delta, credit, resolver_error);
    };
    // ConceptSynthesis produces ordinary experimental direction
    // material. Persist it in the same immutable direction registry
    // used by the normal search path; the scheduler passes the
    // returned reference as the next job's candidate_ref. This seam
    // never approves, activates or promotes the concept direction.
    callbacks.persist_experimental_direction = [provider](
            const common_flydelta_direction_candidate & candidate,
            std::string & direction_ref,
            std::string & persist_error) {
        if (!common_flydelta_direction_candidate_validate(
                candidate, provider->model_n_embd, persist_error)) return false;
        common_flydelta_basis_direction direction;
        direction.layer_index = candidate.layer_index;
        direction.values = candidate.values;
        const std::string source_ref = candidate.extraction_id.empty()
            ? "flydelta://concept-graft/anonymous"
            : candidate.extraction_id;
        return daemon_flydelta_register_composed_directions(
            provider, source_ref, {std::move(direction)},
            "flydelta:concept-graft", direction_ref, persist_error);
    };
    callbacks.persist_experimental_direction_for_job = [provider](
            const common_flydelta_experiment_job & job,
            const common_flydelta_direction_candidate & candidate,
            std::string & direction_ref,
            std::string & persist_error) {
        const auto scoped_provider = daemon_flydelta_provider_for_scope(
            provider, job.seed.scope);
        if (!scoped_provider) {
            persist_error = "FlyDelta concept graft requires a scoped provider";
            return false;
        }
        if (!common_flydelta_direction_candidate_validate(
                candidate, scoped_provider->model_n_embd, persist_error)) return false;
        common_flydelta_basis_direction direction;
        direction.layer_index = candidate.layer_index;
        direction.values = candidate.values;
        const std::string source_ref = candidate.extraction_id.empty()
            ? "flydelta://concept-graft/anonymous"
            : candidate.extraction_id;
        return daemon_flydelta_register_composed_directions(
            scoped_provider, source_ref, {std::move(direction)},
            "flydelta:concept-graft", direction_ref, persist_error);
    };
    callbacks.persist_experimental_paired_basis_for_job = [provider](
            const common_flydelta_experiment_job & job,
            const common_flydelta_paired_intervention_proposal & proposal,
            const common_flydelta_low_rank_basis & basis,
            std::string & direction_ref,
            std::string & persist_error) {
        const auto scoped_provider = daemon_flydelta_provider_for_scope(
            provider, job.seed.scope);
        if (!scoped_provider) {
            persist_error = "FlyDelta paired concept graft requires a scoped provider";
            return false;
        }
        if (!common_flydelta_paired_intervention_proposal_validate(
                    proposal, persist_error) ||
                !common_flydelta_low_rank_basis_validate(basis, 2, persist_error) ||
                basis.vectors.size() != 2) {
            if (persist_error.empty()) {
                persist_error = "FlyDelta paired concept graft basis is invalid";
            }
            return false;
        }
        std::vector<common_flydelta_basis_direction> directions;
        directions.reserve(basis.vectors.size());
        for (const auto & values : basis.vectors) {
            common_flydelta_basis_direction direction;
            direction.layer_index = basis.layer_index;
            direction.values = values;
            directions.push_back(std::move(direction));
        }
        return daemon_flydelta_register_composed_directions(
            scoped_provider, proposal.proposal_id, std::move(directions),
            "flydelta:paired-concept-graft", direction_ref, persist_error);
    };
    callbacks.persist_experimental_layer_family_for_job = [provider](
            const common_flydelta_experiment_job & job,
            const std::vector<common_flydelta_concept_candidate> & candidates,
            const std::string & assessment_ref, std::string & direction_ref,
            std::string & persist_error) {
        const auto scoped = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
        if (!scoped || candidates.empty() || candidates.size() > 8 || assessment_ref.empty() ||
                !common_flydelta_concept_localization_request_validate(job.concept_localization, persist_error) ||
                job.concept_localization.layers.empty()) {
            if (persist_error.empty()) persist_error = "FlyDelta layer-family persistence has no authorized material";
            return false;
        }
        json provenance = {{"schema_version", 1}, {"domain", "multi_layer"},
            {"transferability", "unknown"}, {"where_status", "inherited"},
            {"capture_set_ref", job.concept_localization.capture_set_ref},
            {"material_key", common_flydelta_concept_localization_material_key(job)},
            {"origin_state_ref", job.concept_localization.origin_state_ref},
            {"origin_direction_ref", job.concept_localization.origin_direction_ref},
            {"material_group_ref", job.teaching_material_group_ref},
            {"assessment_ref", assessment_ref}, {"baseline_ref", job.seed.baseline_ref},
            {"model_profile_fingerprint", job.seed.model_profile_fingerprint},
            {"model_n_embd", scoped->model_n_embd}, {"model_n_layers", scoped->model_n_layers},
            {"tokenizer_fingerprint", job.seed.tokenizer_fingerprint},
            {"template_fingerprint", job.seed.template_fingerprint},
            {"execution_context_fingerprint", job.seed.execution_context_fingerprint},
            {"capture_layout_revision", scoped->capture_layout_revision},
            {"authorized_layers", job.concept_localization.layers},
            {"layer_directions", json::array()}, {"experimental_only", true}};
        std::vector<common_flydelta_basis_direction> directions;
        std::set<int32_t> layers;
        for (const auto & candidate : candidates) {
            if (!common_flydelta_concept_candidate_validate(candidate, scoped->model_n_embd, persist_error) ||
                    !candidate.experimental_only || !layers.insert(candidate.layer_index).second ||
                    candidate.model_profile_fingerprint != job.seed.model_profile_fingerprint ||
                    candidate.capture_layout_revision != scoped->capture_layout_revision ||
                    candidate.kind != candidates.front().kind ||
                    candidate.synthesis_semantics != candidates.front().synthesis_semantics ||
                    candidate.concept_key != candidates.front().concept_key ||
                    candidate.behavior_key != job.seed.behavior_key ||
                    std::find(job.concept_localization.layers.begin(), job.concept_localization.layers.end(),
                        static_cast<uint32_t>(candidate.layer_index)) == job.concept_localization.layers.end()) {
                if (persist_error.empty()) persist_error = "FlyDelta layer family contains incompatible direction material";
                return false;
            }
            common_flydelta_basis_direction direction;
            direction.layer_index = candidate.layer_index;
            direction.values = candidate.values;
            json binding = provenance;
            binding["domain"] = "layer_bound";
            binding["anchor_layer"] = candidate.layer_index;
            binding.erase("layer_directions");
            std::string reference;
            if (!daemon_flydelta_register_composed_directions(scoped, candidate.extraction_id, {direction},
                    "flydelta:layer-bound-concept", reference, persist_error, binding)) return false;
            provenance["layer_directions"].push_back({{"layer_index", candidate.layer_index},
                {"direction_ref", reference}, {"extraction_id", candidate.extraction_id},
                {"kind", common_flydelta_concept_candidate_kind_name(candidate.kind)},
                {"synthesis_semantics", common_flydelta_concept_synthesis_semantics_name(candidate.synthesis_semantics)}});
            directions.push_back(std::move(direction));
        }
        return daemon_flydelta_register_composed_directions(scoped, job.concept_localization.capture_set_ref,
            std::move(directions), "flydelta:concept-relocalization", direction_ref, persist_error, provenance);
    };
    callbacks.run_concept_capture = [provider](
            const common_flydelta_experiment_job & job,
            std::vector<std::string> & trajectory_refs,
            std::string & capture_error) {
        return daemon_flydelta_run_concept_capture(
            provider, job, trajectory_refs, capture_error);
    };
    callbacks.run_concept_synthesis = [provider](
            const common_flydelta_experiment_job & job,
            std::vector<common_flydelta_concept_candidate> & candidates,
            std::string & synthesis_error) {
        return daemon_flydelta_run_concept_synthesis(
            provider, job, candidates, synthesis_error);
    };
    callbacks.run_concept_localization_synthesis = [provider](
            const common_flydelta_experiment_job & job,
            std::vector<common_flydelta_concept_candidate> & candidates,
            std::string & assessment_ref, std::string & synthesis_error) {
        return daemon_flydelta_run_concept_synthesis(provider, job, candidates, synthesis_error, &assessment_ref);
    };
    callbacks.run_donor_capture = [provider](
            const common_flydelta_experiment_job & job,
            std::vector<common_flydelta_capture_manifest> & manifests,
            std::string & capture_error) {
        return daemon_flydelta_run_donor_capture(
            provider, job, manifests, capture_error);
    };
    callbacks.run_representation_augmentation_with_state = [provider, pipeline](
            const common_flydelta_experiment_job & job,
            const common_flydelta_representation_augmentation_state * resume,
            common_flydelta_search_pipeline_result & result,
            common_flydelta_representation_augmentation_state & next,
            std::string & augmentation_error) {
        return daemon_flydelta_run_representation_augmentation(
            provider, job, pipeline, resume, result, next, augmentation_error);
    };
    callbacks.run_search_pipeline = [provider, pipeline](
            const common_flydelta_experiment_job & job,
            common_flydelta_search_pipeline_result & result,
            std::string & runner_error) {
        return daemon_flydelta_run_search_pipeline(provider, job, pipeline, result, runner_error);
    };
    callbacks.run_search_pipeline_with_state = [provider, pipeline](
            const common_flydelta_experiment_job & job,
            const common_flydelta_bootstrap_zoom_state * resume,
            common_flydelta_search_pipeline_result & result,
            common_flydelta_bootstrap_zoom_state & next,
            std::string & runner_error) {
        const auto scoped_provider = daemon_flydelta_provider_for_scope(
            provider, job.seed.scope);
        if (!scoped_provider) {
            runner_error = "FlyDelta search runner could not create a scoped provider";
            return false;
        }
        if (resume != nullptr) {
            if (!daemon_flydelta_run_bootstrap_zoom_slice(
                    scoped_provider, job, *resume, result, next, runner_error)) return false;
            if (job.concept_localization.layers.empty()) {
                json direction;
                std::string ignored;
                if (daemon_flydelta_read_json(*scoped_provider, job.seed.candidate_ref, direction, ignored) &&
                        (direction.value("derivation", "") == "flydelta:concept-graft" ||
                         direction.value("derivation", "") == "flydelta:paired-concept-graft")) {
                    next.where_origin_ref = job.bootstrap_zoom_state_ref;
                    next.where_status = "inherited";
                    if (next.selection.selected && next.selection.trial_index >= resume->completed_trials.size() &&
                            next.selection.trial_index < next.completed_trials.size()) {
                        const auto & trial = next.completed_trials[next.selection.trial_index];
                        if (trial.host_evaluated && trial.verifier_known &&
                                trial.outcome == common_flydelta_counterfactual_outcome::helped) {
                            next.where_status = "locally_revalidated";
                        }
                    }
                }
            }
            return true;
        }
        if (!daemon_flydelta_run_search_pipeline(
                scoped_provider, job, pipeline, result, runner_error)) {
            return false;
        }
        std::vector<common_flydelta_basis_direction> directions;
        if (!daemon_flydelta_parse_directions(*scoped_provider, job.seed.candidate_ref,
                directions, runner_error)) return false;
        common_flydelta_search_continuation continuation;
        if (!common_flydelta_select_search_continuation(result, continuation, runner_error)) {
            return false;
        }
        next = {};
        next.behavior_key = job.seed.behavior_key;
        next.model_profile_fingerprint = job.seed.model_profile_fingerprint;
        next.capture_layout_revision = scoped_provider->capture_layout_revision;
        next.anchor_layer = continuation.region.anchor_layer_index == 0
            ? static_cast<uint32_t>(directions.front().layer_index)
            : continuation.region.anchor_layer_index;
        next.selected_scale = continuation.region.total_scale > 0.0f
            ? continuation.region.total_scale : pipeline.scale.initial_scale;
        next.best_search_score = continuation.search_score;
        next.best_margin_delta = continuation.search_score;
        next.local_layers = continuation.region.layer_indices;
        if (next.local_layers.empty()) next.local_layers = {
            static_cast<uint32_t>(directions.front().layer_index)};
        next.surface_origin = "bootstrap_rank1";
        if (!job.concept_localization.layers.empty()) {
            next.where_origin_ref = job.concept_localization.origin_state_ref;
            next.where_status = result.selection.selected ? "relocalized" : "challenged_unverified";
            next.surface_origin = "concept_localization";
        }
        return common_flydelta_bootstrap_zoom_state_validate(next, runner_error);
    };
    // Reuse the append-only lifecycle journal already used by the
    // adaptation subsystem.  The worker receives opaque refs only;
    // model material stays in the daemon resource provider.
    if (!daemon_flydelta_bind_state_callbacks(provider, callbacks, callback_error)) return false;
    const auto resolve_bootstrap_state_for_job =
        callbacks.resolve_bootstrap_zoom_state_for_job;
    const auto resolve_orchestration_state_for_job =
        callbacks.resolve_search_orchestration_state_for_job;
    const auto persist_bootstrap_state_for_job =
        callbacks.persist_bootstrap_zoom_state_for_job;
    callbacks.run_search_pipeline_with_search_state = [provider,
            resolve_bootstrap_state_for_job, resolve_orchestration_state_for_job,
            persist_bootstrap_state_for_job](const common_flydelta_experiment_job & job,
            const std::string & state_ref,
            common_flydelta_search_pipeline_result & result,
            std::string & next_state_ref,
            std::string & state_error) {
        common_flydelta_experiment_plan plan;
        common_flydelta_utility_history history;
        if (!resolve_orchestration_state_for_job ||
                !resolve_orchestration_state_for_job(
                    job, state_ref, plan, history, state_error)) return false;
        if (plan.phase != common_flydelta_experiment_phase::bootstrap) {
            if (plan.phase == common_flydelta_experiment_phase::shallow_controls ||
                    plan.phase == common_flydelta_experiment_phase::deep_controls) {
                if (!daemon_flydelta_run_post_bootstrap_slice(
                        provider, job, plan, result, state_error)) return false;
                next_state_ref = state_ref;
                return true;
            }
            state_error = "FlyDelta daemon received an unsupported post-Bootstrap phase";
            return false;
        }
        std::string bootstrap_ref;
        {
            std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
            const auto it = provider->bootstrap_state_by_orchestration_ref.find(state_ref);
            if (it != provider->bootstrap_state_by_orchestration_ref.end()) bootstrap_ref = it->second;
        }
        if (bootstrap_ref.empty()) bootstrap_ref = job.bootstrap_zoom_state_ref;
        if (bootstrap_ref.empty()) {
            state_error = "FlyDelta Bootstrap continuation requires a BootstrapZoom state ref";
            return false;
        }
        common_flydelta_bootstrap_zoom_state bootstrap;
        if (!resolve_bootstrap_state_for_job ||
                !resolve_bootstrap_state_for_job(
                    job, bootstrap_ref, bootstrap, state_error)) return false;
        if (plan.plateau_evidence_rank_available) {
            bootstrap.evidence_rank_available = true;
            bootstrap.evidence_rank = static_cast<float>(plan.plateau_evidence_rank);
        }
        common_flydelta_bootstrap_zoom_state next_bootstrap;
        if (plan.run_orthogonal_search) {
            constexpr size_t rows = common_flydelta_orthogonal_control_rows;
            const size_t limit = plan.search_model_calls_used == 0 && plan.search_anchor_ref.empty()
                ? provider->search_model_call_budget : plan.search_model_call_limit;
            if (!provider->lifecycle_store || plan.search_model_calls_used > limit ||
                    limit - plan.search_model_calls_used < rows) {
                state_error = "FlyDelta orthogonal search exceeds its remaining execution budget";
                return false;
            }
            std::vector<std::string> logical_ids;
            const auto identity = hash_sha256_hex(job.id.data(), job.id.size()).substr(0, 32);
            for (size_t index = 0; index < rows; ++index) {
                logical_ids.push_back("orthogonal:" + identity + ":" + std::to_string(index));
            }
            common_flydelta_search_consumption consumption;
            std::string attempt;
            if (!common_flydelta_begin_search_execution(*provider->lifecycle_store, job,
                    plan.search_model_calls_used, limit, logical_ids, "orthogonal-controls", 3,
                    attempt, consumption, state_error)) return false;
            const bool ran = daemon_flydelta_run_orthogonal_slice(
                provider, job, bootstrap, result, next_bootstrap, state_error);
            std::string finish_error;
            if (!common_flydelta_finish_search_execution(*provider->lifecycle_store, job, attempt,
                    ran ? rows : 0, ran, finish_error)) {
                state_error = finish_error;
                return false;
            }
            if (!ran) return false;
            if (!common_flydelta_load_search_consumption(*provider->lifecycle_store, job,
                    plan.search_model_calls_used, limit, consumption, state_error)) return false;
            result.search_model_call_limit = limit;
            result.search_model_calls_used = consumption.logical_used;
            result.physical_rows_started = consumption.physical_rows_started;
            result.physical_rows_completed = consumption.physical_rows_completed;
            result.physical_rows_unresolved = consumption.physical_rows_unresolved;
        } else if (bootstrap.phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha) {
            if (!daemon_flydelta_run_adaptive_alpha_slice(
                    provider, job, bootstrap, result, next_bootstrap, state_error)) return false;
        } else if (!daemon_flydelta_run_bootstrap_zoom_slice(
                provider, job, bootstrap, result, next_bootstrap, state_error)) {
            return false;
        }
        bootstrap = std::move(next_bootstrap);
        if (!plan.run_orthogonal_search && !bootstrap.completed_trials.empty()) {
            common_flydelta_orthogonal_search_config orthogonal_config;
            common_flydelta_orthogonal_search_input input;
            common_flydelta_orthogonal_search_result orthogonal;
            std::string orthogonal_error;
            if (common_flydelta_prepare_orthogonal_search_input(
                    orthogonal_config, bootstrap, input, orthogonal_error) &&
                    common_flydelta_build_orthogonal_search_direction(
                        orthogonal_config, input.rank1_intervention, input.arms, orthogonal, orthogonal_error)) {
                result.orthogonal_direction_available = orthogonal.available;
            }
        }
        std::string persisted_bootstrap_ref;
        if (!persist_bootstrap_state_for_job || !persist_bootstrap_state_for_job(
                job, bootstrap, persisted_bootstrap_ref, state_error)) return false;
        {
            std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
            provider->bootstrap_state_by_continuation[
                daemon_flydelta_scope_fingerprint(job.seed.scope) + "\n" +
                daemon_flydelta_continuation_key(plan)] = persisted_bootstrap_ref;
        }
        // The typed Bootstrap state is persisted by the evaluator's
        // regular state callback.  Preserve the orchestration ref for
        // the pure policy decision that follows this slice.
        next_state_ref = state_ref;
        return true;
    };
    callback_error.clear();
    return true;
}

} // namespace agent_daemon_flydelta_internal
