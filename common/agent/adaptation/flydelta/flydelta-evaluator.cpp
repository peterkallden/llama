#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <utility>

namespace {

bool valid_reference_count(size_t count, size_t max_references) {
    return count != 0 && count <= max_references;
}

} // namespace

bool common_flydelta_evaluate_job(
        const common_flydelta_experiment_job & job,
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_evaluator_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (config.max_references == 0 ||
            !common_flydelta_experiment_job_validate(
                job, config.max_references, error)) return false;

    switch (job.kind) {
        case common_flydelta_experiment_job_kind::counterfactual: {
            if (!callbacks.run_counterfactual) {
                error = "FlyDelta counterfactual evaluator requires a host runner";
                return false;
            }
            if (!callbacks.run_counterfactual(
                    job, result.counterfactual_reports, error)) return false;
            if (result.counterfactual_reports.empty()) {
                error = "FlyDelta counterfactual evaluator returned no reports";
                return false;
            }
            for (const auto & report : result.counterfactual_reports) {
                if (!common_flydelta_counterfactual_report_validate(report, error)) return false;
                if (report.experiment_id != job.id) {
                    error = "FlyDelta evaluator report belongs to another job";
                    return false;
                }
            }
            result.processed_references = result.counterfactual_reports.size();
            return true;
        }
        case common_flydelta_experiment_job_kind::basis: {
            if (!callbacks.resolve_behavior_delta ||
                    !common_flydelta_basis_config_validate(config.basis, error)) {
                if (error.empty()) error = "FlyDelta basis evaluator requires a resolver and basis config";
                return false;
            }
            if (config.basis.execution_context_fingerprint !=
                    job.seed.execution_context_fingerprint) {
                error = "FlyDelta basis evaluator context fingerprint does not match job";
                return false;
            }
            common_flydelta_basis_builder builder(config.basis);
            for (const auto & id : job.behavior_delta_ids) {
                common_flydelta_behavior_delta delta;
                common_flydelta_intervention_credit credit;
                if (!callbacks.resolve_behavior_delta(id, delta, credit, error) ||
                        !builder.add(delta, credit, error)) return false;
                ++result.processed_references;
            }
            result.basis_directions = builder.directions();
            return true;
        }
        case common_flydelta_experiment_job_kind::direction: {
            if (!callbacks.resolve_behavior_delta ||
                    !common_flydelta_direction_search_config_validate(config.direction, error)) {
                if (error.empty()) error = "FlyDelta direction evaluator requires a resolver and direction config";
                return false;
            }
            if (config.direction.execution_context_fingerprint !=
                    job.seed.execution_context_fingerprint) {
                error = "FlyDelta direction evaluator context fingerprint does not match job";
                return false;
            }
            std::vector<common_flydelta_contrast_sample> samples;
            samples.reserve(job.behavior_delta_ids.size());
            for (const auto & id : job.behavior_delta_ids) {
                common_flydelta_contrast_sample sample;
                if (!callbacks.resolve_behavior_delta(
                        id, sample.delta, sample.credit, error)) return false;
                samples.push_back(std::move(sample));
            }
            if (!common_flydelta_build_direction_candidates(
                    config.direction, samples, result.direction_candidates, error)) return false;
            result.processed_references = samples.size();
            return true;
        }
        case common_flydelta_experiment_job_kind::delta_memory: {
            if (!callbacks.resolve_training_example ||
                    !common_flydelta_validate_memory_config(config.memory, error)) {
                if (error.empty()) error = "FlyDelta DeltaMemory evaluator requires a resolver and memory config";
                return false;
            }
            if (!valid_reference_count(job.training_example_ids.size(), config.max_references)) {
                error = "FlyDelta DeltaMemory evaluator reference count is invalid";
                return false;
            }
            std::vector<common_flydelta_training_example> examples;
            examples.reserve(job.training_example_ids.size());
            for (const auto & id : job.training_example_ids) {
                common_flydelta_training_example example;
                if (!callbacks.resolve_training_example(id, example, error)) return false;
                if (example.split != common_flydelta_training_split::train) {
                    error = "FlyDelta evaluator refuses validation or holdout training input";
                    return false;
                }
                examples.push_back(std::move(example));
            }
            common_flydelta_delta_memory memory(config.memory);
            if (!common_flydelta_train_delta_memory(
                    memory, examples, job.learning_rate, job.decay,
                    result.processed_references, error)) return false;
            result.delta_memory_weights = memory.weights();
            return true;
        }
    }
    error = "FlyDelta evaluator job kind is unsupported";
    return false;
}
