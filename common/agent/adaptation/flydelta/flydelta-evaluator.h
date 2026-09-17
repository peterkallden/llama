#pragma once

#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-job.h"
#include "agent/adaptation/flydelta/flydelta-training.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-aggregation.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// Host-owned bounds for one offline FlyDelta evaluation. The evaluator never
// resolves paths or reads raw prompt/tool data itself.
struct common_flydelta_evaluator_config {
    common_flydelta_basis_config basis;
    common_flydelta_direction_search_config direction;
    common_flydelta_search_pipeline_config pipeline;
    common_flydelta_memory_config memory;
    common_flydelta_evidence_depth_config evidence_depth;
    common_flydelta_representation_augmentation_config representation_augmentation;
    size_t aggregation_max_retained_samples = 32;
    size_t max_references = 128;
};

struct common_flydelta_evaluator_callbacks {
    // Resolves one immutable behavior delta and its host-derived intervention
    // credit. The callback owns the evidence/artifact store.
    std::function<bool(
            const std::string & id,
            common_flydelta_behavior_delta & delta,
            common_flydelta_intervention_credit & credit,
            std::string & error)> resolve_behavior_delta;

    // Resolves one bounded training example. The evaluator enforces that it
    // is a train split example before passing it to DeltaMemory.
    std::function<bool(
            const std::string & id,
            common_flydelta_training_example & example,
            std::string & error)> resolve_training_example;

    // Executes host-owned baseline/candidate work for a counterfactual job.
    // The callback must return verifier-backed reports only.
    std::function<bool(
            const common_flydelta_experiment_job & job,
            std::vector<common_flydelta_counterfactual_report> & reports,
            std::string & error)> run_counterfactual;

    // Executes the composed direction/layer/scale search. The callback owns
    // reference resolution, fresh inference contexts and host verification.
    std::function<bool(
            const common_flydelta_experiment_job & job,
            common_flydelta_search_pipeline_result & result,
            std::string & error)> run_search_pipeline;

    // Optional state-aware variant used by the real worker to resume and
    // advance BootstrapZoom without rerunning completed arms. The legacy
    // callback above remains valid for callers that do not persist state.
    std::function<bool(
            const common_flydelta_experiment_job & job,
            const common_flydelta_bootstrap_zoom_state * resume_state,
            common_flydelta_search_pipeline_result & result,
            common_flydelta_bootstrap_zoom_state & next_state,
            std::string & error)> run_search_pipeline_with_state;
    std::function<bool(
            const std::string & state_ref,
            common_flydelta_bootstrap_zoom_state & state,
            std::string & error)> resolve_bootstrap_zoom_state;
    std::function<bool(
            const common_flydelta_bootstrap_zoom_state & state,
            std::string & state_ref,
            std::string & error)> persist_bootstrap_zoom_state;

    // Typed continuation seam for the representation-augmentation escape.
    // The queue still carries only search_state_ref; the host owns the typed
    // state, fresh inference contexts and all donor/capture resolution.
    std::function<bool(
            const std::string & state_ref,
            common_flydelta_representation_augmentation_state & state,
            std::string & error)> resolve_representation_augmentation_state;
    std::function<bool(
            const common_flydelta_representation_augmentation_state & state,
            std::string & state_ref,
            std::string & error)> persist_representation_augmentation_state;
    std::function<bool(
            const common_flydelta_experiment_job & job,
            const common_flydelta_representation_augmentation_state * resume_state,
            common_flydelta_search_pipeline_result & result,
            common_flydelta_representation_augmentation_state & next_state,
            std::string & error)> run_representation_augmentation_with_state;

    // Optional generic state-aware runner for post-Bootstrap phases. The host
    // owns the typed state behind the opaque reference and returns the next
    // reference after one bounded slice.
    std::function<bool(
            const common_flydelta_experiment_job & job,
            const std::string & resume_state_ref,
            common_flydelta_search_pipeline_result & result,
            std::string & next_state_ref,
            std::string & error)> run_search_pipeline_with_search_state;
};

struct common_flydelta_evaluator_result {
    std::vector<common_flydelta_counterfactual_report> counterfactual_reports;
    std::vector<common_flydelta_basis_direction> basis_directions;
    std::vector<common_flydelta_direction_candidate> direction_candidates;
    std::vector<common_flydelta_search_pipeline_result> search_pipeline_results;
    std::vector<common_flydelta_search_continuation> search_continuations;
    std::vector<float> delta_memory_weights;
    common_flydelta_aggregation_snapshot aggregation;
    common_flydelta_evidence_depth_result evidence_depth;
    common_flydelta_search_budget search_budget;
    bool has_experiment_plan = false;
    common_flydelta_experiment_plan experiment_plan;
    size_t processed_references = 0;
    bool has_bootstrap_zoom_state = false;
    common_flydelta_bootstrap_zoom_state bootstrap_zoom_state;
    std::string bootstrap_zoom_state_ref;
    std::string search_state_ref;
    bool has_representation_augmentation_state = false;
    common_flydelta_representation_augmentation_state representation_augmentation_state;
    std::string representation_augmentation_state_ref;
};

// Evaluates exactly one already-validated job. This is an orchestration seam,
// not a model trainer: callbacks own reference resolution, inference and host
// verification. No result may contain raw prompts, tool output or credentials.
bool common_flydelta_evaluate_job(
        const common_flydelta_experiment_job & job,
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_evaluator_result & result,
        std::string & error);
