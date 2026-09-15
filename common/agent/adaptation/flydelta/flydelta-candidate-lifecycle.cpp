#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"

#include <cmath>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool finite_nonnegative(float value) {
    return std::isfinite(value) && value >= 0.0f;
}

} // namespace

const char * common_flydelta_search_disposition_name(
        common_flydelta_search_disposition disposition) {
    switch (disposition) {
        case common_flydelta_search_disposition::none: return "none";
        case common_flydelta_search_disposition::retain: return "retain";
        case common_flydelta_search_disposition::refine: return "refine";
        case common_flydelta_search_disposition::validate_repeatability:
            return "validate_repeatability";
        case common_flydelta_search_disposition::reject: return "reject";
    }
    return "none";
}

bool common_flydelta_candidate_lineage_validate(
        const common_flydelta_candidate_lineage & lineage,
        size_t max_layers,
        std::string & error) {
    error.clear();
    if (lineage.schema_version != 1 || !nonempty_bounded(lineage.candidate_id) ||
            !nonempty_bounded(lineage.mutation_kind) ||
            lineage.parent_candidate_id.size() > 512 || lineage.direction_id.size() > 512 ||
            lineage.layer_indices.size() > max_layers ||
            !finite_nonnegative(lineage.scale) ||
            !finite_nonnegative(lineage.intervention_budget)) {
        error = "FlyDelta candidate lineage is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_search_observation_validate(
        const common_flydelta_search_observation & observation,
        std::string & error) {
    error.clear();
    if (observation.schema_version != 1 || !nonempty_bounded(observation.experiment_id) ||
            !nonempty_bounded(observation.candidate_id) || !observation.host_verified ||
            (observation.diagnostics_available &&
                !common_flydelta_representation_diagnostics_validate(observation.diagnostics, error)) ||
            (observation.sequence_margin_available &&
                !std::isfinite(observation.sequence_margin_delta))) {
        if (error.empty()) error = "FlyDelta search observation is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_decide_search_disposition(
        const common_flydelta_search_observation & observation,
        common_flydelta_search_decision & decision,
        std::string & error) {
    if (!common_flydelta_search_observation_validate(observation, error)) return false;

    decision = {};
    decision.candidate_id = observation.candidate_id;

    const bool geometry_is_safe = observation.diagnostics_available &&
        observation.diagnostics.cosine >= 0.3f && observation.diagnostics.progress > 0.0f &&
        observation.diagnostics.leakage <= 1.0f && observation.diagnostics.shift_norm <= 1.0f;
    const bool margin_is_promising = observation.sequence_margin_available &&
        observation.sequence_margin_delta > 0.0f;
    const bool search_signal = geometry_is_safe || margin_is_promising;

    switch (observation.outcome) {
        case common_flydelta_counterfactual_outcome::helped:
            decision.disposition = common_flydelta_search_disposition::validate_repeatability;
            decision.search_priority = 0.5f;
            decision.evidence_score = 1.0f;
            decision.reason = "host verified HELPED; repeatability is the next gate";
            break;
        case common_flydelta_counterfactual_outcome::harmed:
            decision.disposition = common_flydelta_search_disposition::reject;
            decision.search_priority = 0.0f;
            decision.evidence_score = -1.0f;
            decision.reason = "host verified HARMED; reject candidate";
            break;
        case common_flydelta_counterfactual_outcome::neutral:
            decision.disposition = search_signal && observation.budget_remaining
                ? common_flydelta_search_disposition::refine
                : common_flydelta_search_disposition::retain;
            decision.search_priority = search_signal ? 0.6f : 0.2f;
            decision.evidence_score = 0.0f;
            decision.reason = search_signal
                ? "NEUTRAL has bounded diagnostic signal; refine candidate"
                : "NEUTRAL has no bounded follow-up signal; retain without promotion";
            break;
        case common_flydelta_counterfactual_outcome::unknown:
            decision.disposition = search_signal && observation.budget_remaining
                ? common_flydelta_search_disposition::refine
                : common_flydelta_search_disposition::retain;
            decision.search_priority = search_signal ? 1.0f : 0.1f;
            decision.evidence_score = 0.0f;
            decision.reason = search_signal
                ? "UNKNOWN has bounded diagnostic signal; refine candidate"
                : "UNKNOWN remains unproven; retain without promotion";
            break;
    }
    return true;
}

bool common_flydelta_experiment_champion_validate(
        const common_flydelta_experiment_champion & champion,
        std::string & error) {
    error.clear();
    if (champion.schema_version != 1 || !nonempty_bounded(champion.experiment_id) ||
            !nonempty_bounded(champion.candidate_id) || !nonempty_bounded(champion.model_profile_id) ||
            !nonempty_bounded(champion.fixture_set_revision) ||
            !nonempty_bounded(champion.verifier_revision) ||
            !std::isfinite(champion.objective_score) || champion.evaluated_turns == 0 ||
            champion.harmed_trials > champion.evaluated_turns) {
        error = "FlyDelta experiment champion is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_select_experiment_champion(
        const common_flydelta_experiment_champion & current,
        const common_flydelta_experiment_champion & challenger,
        common_flydelta_experiment_champion & selected,
        std::string & error) {
    if (!common_flydelta_experiment_champion_validate(current, error) ||
            !common_flydelta_experiment_champion_validate(challenger, error)) return false;
    if (current.experiment_id != challenger.experiment_id ||
            current.model_profile_id != challenger.model_profile_id ||
            current.fixture_set_revision != challenger.fixture_set_revision ||
            current.verifier_revision != challenger.verifier_revision) {
        error = "FlyDelta champions are not comparable";
        return false;
    }

    selected = current;
    if (challenger.host_verified && challenger.holdout_passed && challenger.no_regression &&
            challenger.helped_trials > 0 && challenger.harmed_trials == 0 &&
            challenger.objective_score > current.objective_score) {
        selected = challenger;
    }
    return true;
}
