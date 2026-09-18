#pragma once

#include "agent/adaptation/flydelta/flydelta-queue.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-aggregation.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"

#include <functional>
#include <string>
#include <vector>

struct common_flydelta_evaluator_config;
struct common_flydelta_evaluator_callbacks;

// Bounded, machine-readable diagnostics for one FlyDelta worker slice. This
// is search history, not learning or promotion state. It deliberately keeps
// references and scalar measurements, never prompts, tool output or tensors.
struct common_flydelta_trace_arm {
    std::string phase;
    std::string arm_id;
    std::vector<uint32_t> layer_indices;
    float scale = 0.0f;
    std::vector<float> coefficients;
    bool margin_available = false;
    float margin_total = 0.0f;
    float margin_normalized = 0.0f;
    float margin_delta_total = 0.0f;
    float margin_delta_normalized = 0.0f;
    bool geometry_available = false;
    float cosine = 0.0f;
    float progress = 0.0f;
    float leakage = 0.0f;
    float shift_norm = 0.0f;
    float search_score = 0.0f;
    bool promising = false;
    bool safe_to_continue = false;
    bool host_evaluated = false;
    bool verifier_known = false;
    bool candidate_passed = false;
    bool has_baseline = false;
    bool baseline_executed = false;
    bool baseline_verifier_known = false;
    bool baseline_passed = false;
    common_flydelta_counterfactual_outcome host_outcome =
        common_flydelta_counterfactual_outcome::unknown;
    std::string evidence_ref;
};

struct common_flydelta_trace {
    int schema_version = 1;
    std::string job_id;
    std::string phase;
    std::string behavior_key;
    std::string fixture_baseline_ref;
    std::string surface_parent_best_ref;
    size_t evidence_rank = 0;
    size_t search_rank = 0;
    size_t model_evaluations = 0;
    size_t region_budget = 0;
    size_t coefficient_budget = 0;
    bool tfo_lite_allowed = false;
    std::string bootstrap_refinement;
    bool alpha_response_available = false;
    common_flydelta_alpha_response_status alpha_response_status =
        common_flydelta_alpha_response_status::inconclusive;
    bool alpha_range_not_exhausted = false;
    float alpha_last_scale = 0.0f;
    float alpha_utility_slope = 0.0f;
    float alpha_best_margin_delta_normalized = 0.0f;
    bool has_next_action = false;
    common_flydelta_next_action next_action = common_flydelta_next_action::retain;
    std::string next_action_reason;
    std::vector<common_flydelta_trace_arm> arms;
    std::vector<common_flydelta_whirlpool_trace> whirlpool;
};

bool common_flydelta_trace_validate(
        const common_flydelta_trace & trace, std::string & error);
std::string common_flydelta_trace_to_json(const common_flydelta_trace & trace);

struct common_flydelta_experiment_worker_result {
    std::string safe_summary;
    common_flydelta_trace trace;
    std::vector<common_flydelta_capture_manifest> capture_manifests;
    std::vector<common_flydelta_counterfactual_report> counterfactual_reports;
    std::vector<common_flydelta_direction_candidate> direction_candidates;
    std::vector<common_flydelta_basis_direction> basis_directions;
    std::vector<common_flydelta_search_pipeline_result> search_pipeline_results;
    std::vector<common_flydelta_search_continuation> search_continuations;
    std::vector<float> delta_memory_weights;
    common_flydelta_aggregation_snapshot aggregation;
    common_flydelta_evidence_depth_result evidence_depth;
    common_flydelta_search_budget search_budget;
    bool has_experiment_plan = false;
    common_flydelta_experiment_plan experiment_plan;
    bool has_bootstrap_zoom_state = false;
    common_flydelta_bootstrap_zoom_state bootstrap_zoom_state;
    std::string bootstrap_zoom_state_ref;
    std::string search_state_ref;
    bool has_next_action = false;
    common_flydelta_next_action next_action = common_flydelta_next_action::retain;
    common_flydelta_utility_gate_decision utility_decision;
    std::string next_action_reason;
    bool has_representation_augmentation_state = false;
    common_flydelta_representation_augmentation_state representation_augmentation_state;
    std::string representation_augmentation_state_ref;
};

struct common_flydelta_experiment_worker_report {
    common_flydelta_experiment_queue_state state = common_flydelta_experiment_queue_state::pending;
    std::string job_id;
    std::string safe_summary;
    common_flydelta_trace trace;
    std::string trace_json;
    size_t report_count = 0;
    std::vector<common_flydelta_capture_manifest> capture_manifests;
    common_flydelta_evidence_depth_result evidence_depth;
    common_flydelta_search_budget search_budget;
    bool has_experiment_plan = false;
    common_flydelta_experiment_plan experiment_plan;
    bool has_bootstrap_zoom_state = false;
    common_flydelta_bootstrap_zoom_state bootstrap_zoom_state;
    std::string bootstrap_zoom_state_ref;
    std::string search_state_ref;
    bool has_next_action = false;
    common_flydelta_next_action next_action = common_flydelta_next_action::retain;
    common_flydelta_utility_gate_decision utility_decision;
    std::string next_action_reason;
    bool has_representation_augmentation_state = false;
    common_flydelta_representation_augmentation_state representation_augmentation_state;
    std::string representation_augmentation_state_ref;
};

// The worker owns queue lifecycle and result validation. The callback owns
// resolving reference IDs and performing bounded host-side work; it must not
// return raw prompts, tool output, credentials or activation tensors here.
using common_flydelta_experiment_worker_callback = std::function<bool(
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_worker_result & result,
        std::string & error)>;

// Claims and processes at most one job. Callback failures become a terminal
// failed job without exposing callback diagnostics in queue state. An empty
// queue returns true with state=pending and no job ID.
bool common_flydelta_experiment_worker_run_once(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_experiment_worker_callback & callback,
        common_flydelta_experiment_worker_report & report,
        std::string & error);

// Adapts the queue worker to the common evaluator. The evaluator callbacks
// still own reference resolution, fresh model contexts and host verification;
// this bridge only transports typed evaluator results into the worker result.
bool common_flydelta_experiment_worker_run_evaluator_once(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_experiment_worker_report & report,
        std::string & error);
