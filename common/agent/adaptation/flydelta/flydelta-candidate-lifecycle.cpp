#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"

#include <nlohmann/json.hpp>

#include <cmath>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool finite_nonnegative(float value) {
    return std::isfinite(value) && value >= 0.0f;
}

bool lifecycle_context_valid(
        const common_flydelta_lifecycle_event_context & context,
        std::string & error) {
    if (!nonempty_bounded(context.event_id) || !nonempty_bounded(context.idempotency_key) ||
            !nonempty_bounded(context.source_id) || !nonempty_bounded(context.content_hash) ||
            !nonempty_bounded(context.created_at) || context.scope.namespace_id.empty()) {
        error = "FlyDelta lifecycle event context is invalid";
        return false;
    }
    return true;
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

bool common_flydelta_append_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_search_observation & observation,
        const common_flydelta_search_decision & decision,
        const common_flydelta_candidate_lineage * lineage,
        std::string & error) {
    error.clear();
    common_flydelta_search_decision expected;
    if (!lifecycle_context_valid(context, error) ||
            !common_flydelta_search_observation_validate(observation, error) ||
            !common_flydelta_decide_search_disposition(observation, expected, error) ||
            decision.schema_version != 1 || decision.candidate_id != observation.candidate_id ||
            decision.disposition != expected.disposition ||
            !nonempty_bounded(decision.reason) || !std::isfinite(decision.search_priority) ||
            decision.search_priority < 0.0f || decision.search_priority > 1.0f ||
            !std::isfinite(decision.evidence_score) || decision.evidence_score < -1.0f ||
            decision.evidence_score > 1.0f) {
        if (error.empty()) error = "FlyDelta search lifecycle payload is invalid";
        return false;
    }
    if (lineage && (!common_flydelta_candidate_lineage_validate(*lineage, 64, error) ||
            lineage->candidate_id != observation.candidate_id)) return false;

    using json = nlohmann::ordered_json;
    json payload = {
        {"record_type", "candidate_search_decision"},
        {"experiment_id", observation.experiment_id},
        {"candidate_id", observation.candidate_id},
        {"outcome", common_flydelta_counterfactual_outcome_name(observation.outcome)},
        {"host_verified", observation.host_verified},
        {"disposition", common_flydelta_search_disposition_name(decision.disposition)},
        {"search_priority", decision.search_priority},
        {"evidence_score", decision.evidence_score},
        {"reason", decision.reason},
    };
    if (observation.diagnostics_available) {
        payload["diagnostics"] = {
            {"layer_index", observation.diagnostics.layer_index},
            {"cosine", observation.diagnostics.cosine},
            {"progress", observation.diagnostics.progress},
            {"leakage", observation.diagnostics.leakage},
            {"shift_norm", observation.diagnostics.shift_norm},
        };
    }
    if (observation.sequence_margin_available) {
        payload["sequence_margin_delta"] = observation.sequence_margin_delta;
    }
    if (lineage) {
        payload["lineage"] = {
            {"candidate_id", lineage->candidate_id},
            {"parent_candidate_id", lineage->parent_candidate_id},
            {"mutation_kind", lineage->mutation_kind},
            {"generation", lineage->generation},
            {"direction_id", lineage->direction_id},
            {"layer_indices", lineage->layer_indices},
            {"scale", lineage->scale},
            {"intervention_budget", lineage->intervention_budget},
        };
    }
    common_learning_lifecycle_record record;
    record.event_id = context.event_id;
    record.subject_id = observation.candidate_id;
    record.kind = common_learning_lifecycle_kind::flydelta_result;
    record.status = decision.disposition == common_flydelta_search_disposition::reject
        ? common_learning_lifecycle_status::rejected
        : common_learning_lifecycle_status::observed;
    record.idempotency_key = context.idempotency_key;
    record.source_id = context.source_id;
    record.namespace_id = context.scope.namespace_id;
    record.project_id = context.scope.project_id;
    record.session_id = context.scope.session_id;
    record.content_hash = context.content_hash;
    record.created_at = context.created_at;
    record.payload_json = payload.dump();
    return store.append(record, error);
}

bool common_flydelta_append_champion_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_champion & current,
        const common_flydelta_experiment_champion & challenger,
        const common_flydelta_experiment_champion & selected,
        std::string & error) {
    error.clear();
    if (!lifecycle_context_valid(context, error) ||
            !common_flydelta_experiment_champion_validate(current, error) ||
            !common_flydelta_experiment_champion_validate(challenger, error) ||
            !common_flydelta_experiment_champion_validate(selected, error) ||
            current.experiment_id != challenger.experiment_id ||
            challenger.experiment_id != selected.experiment_id ||
            (selected.candidate_id != current.candidate_id &&
                selected.candidate_id != challenger.candidate_id)) {
        if (error.empty()) error = "FlyDelta champion lifecycle payload is invalid";
        return false;
    }
    using json = nlohmann::ordered_json;
    const auto snapshot = [](const auto & value) {
        return nlohmann::ordered_json{
            {"candidate_id", value.candidate_id},
            {"objective_score", value.objective_score},
            {"evaluated_turns", value.evaluated_turns},
            {"helped_trials", value.helped_trials},
            {"harmed_trials", value.harmed_trials},
            {"host_verified", value.host_verified},
            {"holdout_passed", value.holdout_passed},
            {"no_regression", value.no_regression},
        };
    };
    const json payload = {
        {"record_type", "experiment_champion_selection"},
        {"experiment_id", selected.experiment_id},
        {"current", snapshot(current)},
        {"challenger", snapshot(challenger)},
        {"selected_candidate_id", selected.candidate_id},
    };
    common_learning_lifecycle_record record;
    record.event_id = context.event_id;
    record.subject_id = selected.experiment_id;
    record.kind = common_learning_lifecycle_kind::flydelta_result;
    record.status = common_learning_lifecycle_status::observed;
    record.idempotency_key = context.idempotency_key;
    record.source_id = context.source_id;
    record.namespace_id = context.scope.namespace_id;
    record.project_id = context.scope.project_id;
    record.session_id = context.scope.session_id;
    record.content_hash = context.content_hash;
    record.created_at = context.created_at;
    record.payload_json = payload.dump();
    return store.append(record, error);
}
