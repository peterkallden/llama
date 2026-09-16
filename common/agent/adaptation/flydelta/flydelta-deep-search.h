#pragma once

#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <cstddef>
#include <string>
#include <vector>

// The deep search is the continuation of the normal region scan. It does not
// discover layers or decide host truth; it consumes compatible WHAT
// candidates for one already selected layer and searches their low-rank MIX.
struct common_flydelta_deep_search_config {
    int schema_version = 1;
    size_t max_rank = 2;
    size_t max_directions = 4;
    size_t full_generation_top_k = 3;
    common_flydelta_coefficient_search_config coefficients;
};

struct common_flydelta_deep_search_direction {
    common_flydelta_direction_candidate direction;
    // A margin/geometry score supplied by the host. It only ranks which
    // directions enter the experimental basis; it is never host evidence.
    bool decision_score_available = false;
    float decision_score = 0.0f;
};

struct common_flydelta_deep_search_result {
    int schema_version = 1;
    int32_t layer_index = -1;
    std::vector<common_flydelta_deep_search_direction> selected_directions;
    common_flydelta_low_rank_basis basis;
    std::vector<common_flydelta_coefficient_trial> coefficient_trials;
    common_flydelta_coefficient_selection coefficient_selection;
};

bool common_flydelta_deep_search_config_validate(
        const common_flydelta_deep_search_config & config,
        std::string & error);

// Selects the highest-scoring compatible WHAT candidates for one layer and
// builds the existing bounded low-rank basis. Missing decision scores fall
// back to candidate alignment, so this seam also works before model-facing
// margin capture is available.
bool common_flydelta_select_deep_search_directions(
        const common_flydelta_deep_search_config & config,
        const std::vector<common_flydelta_deep_search_direction> & directions,
        std::vector<common_flydelta_deep_search_direction> & selected,
        std::string & error);

// Runs the selected rank-N basis through the existing coordinate/TFO-lite
// coefficient search. Full generation and host verification remain owned by
// the runner; diagnostic margins only rank arms.
bool common_flydelta_run_deep_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_deep_search_config & config,
        const std::vector<common_flydelta_deep_search_direction> & directions,
        const common_flydelta_coefficient_search_runner & diagnostic_runner,
        const common_flydelta_coefficient_search_runner & full_generation_runner,
        common_flydelta_deep_search_result & result,
        std::string & error);

// Records the diagnostic and full-generation coefficient trials through the
// existing experimental lifecycle. This is reference-only: it never updates
// DeltaMemory, activates a sideband or promotes a challenger.
bool common_flydelta_append_deep_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_deep_search_config & config,
        const common_flydelta_deep_search_result & result,
        const std::string & experimental_artifact_id,
        std::string & error);
