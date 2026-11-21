#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <utility>

bool common_flydelta_run_low_rank_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
    if (!runner) {
        error = "FlyDelta coefficient search runner is invalid";
        return false;
    }
    const common_flydelta_coefficient_search_batch_runner batch_runner =
        [&](const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & batch_trials,
            std::vector<common_flydelta_decision_margin> & batch_margins,
            std::vector<common_flydelta_representation_diagnostics> & batch_geometries,
            std::vector<bool> & batch_geometry_available,
            std::string & batch_error) {
            batch_trials.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            for (const auto & values : coefficients) {
                common_flydelta_counterfactual_trial trial;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics geometry;
                bool geometry_available = false;
                if (!runner(current_fixture, current_basis, values, true, trial, margin,
                        geometry, geometry_available, batch_error)) return false;
                batch_trials.push_back(std::move(trial));
                batch_margins.push_back(std::move(margin));
                batch_geometries.push_back(std::move(geometry));
                batch_geometry_available.push_back(geometry_available);
            }
            return true;
        };
    return common_flydelta_run_low_rank_coefficient_search_batched(
        fixture, basis, config, runner, batch_runner, trials, selection, error);
}
