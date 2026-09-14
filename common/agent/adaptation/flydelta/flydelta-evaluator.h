#pragma once

#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-job.h"
#include "agent/adaptation/flydelta/flydelta-training.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// Host-owned bounds for one offline FlyDelta evaluation. The evaluator never
// resolves paths or reads raw prompt/tool data itself.
struct common_flydelta_evaluator_config {
    common_flydelta_basis_config basis;
    common_flydelta_memory_config memory;
    size_t max_references = 128;
};

struct common_flydelta_evaluator_callbacks {
    // Resolves one immutable repair delta and its host-derived intervention
    // credit. The callback owns the evidence/artifact store.
    std::function<bool(
            const std::string & id,
            common_flydelta_repair_delta & delta,
            common_flydelta_intervention_credit & credit,
            std::string & error)> resolve_repair_delta;

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
};

struct common_flydelta_evaluator_result {
    std::vector<common_flydelta_counterfactual_report> counterfactual_reports;
    std::vector<common_flydelta_basis_direction> basis_directions;
    std::vector<float> delta_memory_weights;
    size_t processed_references = 0;
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
