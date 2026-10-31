#include "tools/agent/daemon/flydelta/state.h"
#include "tools/agent/daemon/agent-daemon-flydelta-internal.h"
#include "agent/adaptation/flydelta/flydelta-collection.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { std::cerr << "plateau queue smoke line " << __LINE__ << ": " << error << '\n'; return 1; } } while (false)

using namespace agent_daemon_flydelta_internal;

static common_flydelta_search_pipeline_result observed_slice() {
    common_flydelta_search_pipeline_result result;
    common_flydelta_search_pipeline_direction_result direction;
    direction.direction.layer_index = 2;
    direction.direction.values = {1.0f, 0.0f};
    direction.direction.source_samples = 1;
    direction.direction.retained_samples = 1;
    common_flydelta_intervention_region_trial trial;
    trial.candidate.layer_indices = {2};
    trial.candidate.anchor_layer_index = 2;
    trial.candidate.total_scale = 0.1f;
    trial.candidate.per_layer_scale = 0.1f;
    trial.executed = true;
    trial.host_evaluated = true;
    trial.verifier_known = true;
    trial.safe_to_continue = true;
    trial.promising = true;
    trial.search_score = 0.4f;
    trial.evidence_ref = "evidence:smoke";
    trial.margin_comparison.available = true;
    trial.margin_comparison.baseline.available = true;
    trial.margin_comparison.baseline.positive_total_logprob = -2.0f;
    trial.margin_comparison.baseline.negative_total_logprob = -1.0f;
    trial.margin_comparison.baseline.positive_token_count = 1;
    trial.margin_comparison.baseline.negative_token_count = 1;
    trial.margin_comparison.candidate = trial.margin_comparison.baseline;
    trial.margin_comparison.candidate.positive_total_logprob = -1.6f;
    direction.region_trials = {trial, trial};
    direction.region_selection = {true, 0, 0.4f};
    result.directions.push_back(std::move(direction));
    result.selection.selected = true;
    result.selection.intervention_region = true;
    result.selection.scale = 0.1f;
    result.selection.score = 0.4f;
    result.orthogonal_direction_available = true;
    return result;
}

int main() {
    std::string error;
    const auto root = std::filesystem::temp_directory_path() /
        ("flydelta-plateau-smoke-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const auto journal_path = root / "lifecycle.jsonl";
    const auto queue_path = root / "queue";
    common_flydelta_experiment_job job;
    job.id = "flydelta://job/plateau-smoke";
    job.kind = common_flydelta_experiment_job_kind::search_pipeline;
    job.seed.id = "evidence://plateau-smoke";
    job.seed.behavior_key = "tool_use/smoke";
    job.seed.scope.namespace_id = "local";
    job.seed.scope.project_id = "plateau-smoke";
    job.seed.scope.session_id = "session";
    job.seed.task_fingerprint = "task:smoke";
    job.seed.model_profile_fingerprint = "model:smoke";
    job.seed.tokenizer_fingerprint = "tokenizer:smoke";
    job.seed.template_fingerprint = "template:smoke";
    job.seed.execution_context_fingerprint = "context:smoke";
    job.seed.baseline_ref = "execution:baseline";
    job.seed.candidate_ref = "execution:candidate";
    job.seed.verifier_ref = "verifier:smoke";
    job.seed.evidence_ref = "evidence:smoke";
    job.seed.transaction_ids = {"learning:baseline", "learning:candidate"};
    job.capture_manifest_ids = {"capture:smoke"};
    job.behavior_delta_ids = {"delta:smoke"};
    job.alpha_search.candidates = {0.05f};
    job.alpha_search.max_candidates = 1;
    job.code_revision = "plateau-smoke:v1";

    auto journal = std::make_shared<common_learning_jsonl_lifecycle_store>();
    CHECK(journal->open(journal_path, error));
    auto provider = std::make_shared<daemon_flydelta_resource_provider>();
    provider->lifecycle_store = journal;
    provider->authority.namespace_id = "local";
    provider->authority.session_id = "session";
    common_flydelta_evaluator_callbacks callbacks;
    CHECK(daemon_flydelta_bind_state_callbacks(provider, callbacks, error));
    common_flydelta_bootstrap_zoom_state bootstrap;
    bootstrap.behavior_key = job.seed.behavior_key;
    bootstrap.model_profile_fingerprint = job.seed.model_profile_fingerprint;
    bootstrap.capture_layout_revision = "layout:smoke";
    bootstrap.anchor_layer = 2;
    bootstrap.local_layers = {1, 2, 3};
    bootstrap.selected_scale = 0.1f;
    bootstrap.evidence_rank_available = true;
    bootstrap.evidence_rank = 0.0f;
    common_flydelta_bootstrap_zoom_config zoom_config;
    std::vector<common_flydelta_bootstrap_zoom_candidate> candidates;
    CHECK(common_flydelta_propose_bootstrap_profile_zoom(
        bootstrap.local_layers, bootstrap.anchor_layer, bootstrap.selected_scale, zoom_config, candidates, error));
    for (const auto & candidate : candidates) {
        common_flydelta_bootstrap_zoom_trial trial;
        trial.candidate = candidate;
        trial.host_evaluated = true;
        trial.verifier_known = true;
        trial.outcome = common_flydelta_counterfactual_outcome::neutral;
        trial.margin_available = true;
        trial.margin_delta = 0.4f;
        for (size_t index = 0; index < candidate.layer_indices.size(); ++index) {
            if (candidate.layer_indices[index] == 1) trial.margin_delta -= candidate.layer_weights[index];
            if (candidate.layer_indices[index] == 3) trial.margin_delta += candidate.layer_weights[index];
        }
        trial.diagnostics_available = true;
        trial.diagnostics = {1, 2, 0.8f, 0.2f, 0.1f, 0.2f};
        bootstrap.completed_trials.push_back(std::move(trial));
    }
    bootstrap.extra_model_trials = bootstrap.completed_trials.size();
    bootstrap.next_candidate_index = bootstrap.completed_trials.size();
    bootstrap.selection = {true, 0, 0.4f};
    common_flydelta_orthogonal_search_config orthogonal_config;
    common_flydelta_orthogonal_search_input orthogonal_input;
    common_flydelta_orthogonal_search_result orthogonal;
    CHECK(common_flydelta_prepare_orthogonal_search_input(
        orthogonal_config, bootstrap, orthogonal_input, error));
    CHECK(common_flydelta_build_orthogonal_search_direction(
        orthogonal_config, orthogonal_input.rank1_intervention, orthogonal_input.arms, orthogonal, error));
    CHECK(orthogonal.available && orthogonal.experimental_only);
    CHECK(callbacks.persist_bootstrap_zoom_state_for_job(job, bootstrap, job.bootstrap_zoom_state_ref, error));
    common_flydelta_experiment_plan plan;
    plan.continuation.region.layer_indices = {1, 2, 3};
    plan.continuation.region.anchor_layer_index = 2;
    plan.continuation.region.total_scale = 0.1f;
    plan.continuation.region.per_layer_scale = 0.1f / std::sqrt(3.0f);
    plan.budget = common_flydelta_search_budget_for_depth(common_flydelta_search_depth::bootstrap);
    plan.plateau_evidence_rank_available = true;
    plan.plateau_evidence_rank = 0;
    plan.plateau_rounds = {{true, 2, 2, 0.4f}, {true, 2, 2, 0.4f}};
    CHECK(callbacks.persist_search_orchestration_state_for_job(job, plan, {}, job.search_state_ref, error));
    common_flydelta_evaluator_config config;
    config.pipeline.dimension = 2;
    callbacks.run_search_pipeline_with_search_state = [](const auto &, const auto & ref, auto & result,
            auto & next_ref, std::string &) {
        result = observed_slice();
        next_ref = ref;
        return true;
    };
    CHECK(common_flydelta_experiment_queue_enqueue(queue_path, job, {}, error));
    common_flydelta_experiment_worker_report report;
    CHECK(common_flydelta_experiment_worker_run_evaluator_once(queue_path, {}, config, callbacks, report, error));
    CHECK(report.state == common_flydelta_experiment_queue_state::succeeded && report.has_next_action &&
        report.next_action == common_flydelta_next_action::run_orthogonal_search);
    const auto saved_ref = report.search_state_ref;
    common_flydelta_experiment_collection_result collected;
    CHECK(common_flydelta_collect_next_action_job(queue_path, {}, report.completed_job, report.next_action,
        report.bootstrap_zoom_state_ref, saved_ref, "", "", collected, error));
    CHECK(collected == common_flydelta_experiment_collection_result::enqueued);
    CHECK(common_flydelta_collect_next_action_job(queue_path, {}, report.completed_job, report.next_action,
        report.bootstrap_zoom_state_ref, saved_ref, "", "", collected, error));
    CHECK(collected == common_flydelta_experiment_collection_result::already_present);

    callbacks = {};
    provider.reset();
    journal.reset();
    auto reopened = std::make_shared<common_learning_jsonl_lifecycle_store>();
    CHECK(reopened->open(journal_path, error));
    provider = std::make_shared<daemon_flydelta_resource_provider>();
    provider->lifecycle_store = reopened;
    provider->authority.namespace_id = "local";
    provider->authority.session_id = "session";
    CHECK(daemon_flydelta_bind_state_callbacks(provider, callbacks, error));
    common_flydelta_bootstrap_zoom_state restored_bootstrap;
    CHECK(callbacks.resolve_bootstrap_zoom_state_for_job(
        job, job.bootstrap_zoom_state_ref, restored_bootstrap, error));
    CHECK(restored_bootstrap.evidence_rank_available && restored_bootstrap.evidence_rank == 0.0f);
    bool resumed_orthogonal = false;
    callbacks.run_search_pipeline_with_search_state = [&](const auto & next_job, const auto & ref, auto & result,
            auto & next_ref, std::string & callback_error) {
        common_flydelta_experiment_plan restored;
        common_flydelta_utility_history history;
        if (!callbacks.resolve_search_orchestration_state_for_job(next_job, ref, restored, history, callback_error)) return false;
        resumed_orthogonal = restored.run_orthogonal_search && restored.plateau_rounds.size() == 3 &&
            restored.plateau_evidence_rank_available && restored.plateau_evidence_rank == 0 &&
            history.qualifying_streak == 1 &&
            provider->bootstrap_state_by_orchestration_ref[ref] == job.bootstrap_zoom_state_ref;
        result = observed_slice();
        next_ref = ref;
        return true;
    };
    CHECK(common_flydelta_experiment_worker_run_evaluator_once(queue_path, {}, config, callbacks, report, error));
    CHECK(resumed_orthogonal && report.state == common_flydelta_experiment_queue_state::succeeded &&
        report.has_experiment_plan && report.experiment_plan.plateau_escape_completed &&
        !report.experiment_plan.run_orthogonal_search &&
        report.experiment_plan.depth == common_flydelta_search_depth::bootstrap);
    std::cout << "plateau_queue=passed\nreopened_journal=passed\nobservations=synthetic\n";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    return 0;
}
