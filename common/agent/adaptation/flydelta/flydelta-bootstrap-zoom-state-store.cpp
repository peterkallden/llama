#include "agent/adaptation/flydelta/flydelta-bootstrap-zoom-state-store.h"

#include "hash/hash.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

std::string hash_text(const std::string & value) {
    return "sha256:" + hash_sha256_hex(value.data(), value.size());
}

const char * phase_name(common_flydelta_bootstrap_zoom_phase phase) {
    return common_flydelta_bootstrap_zoom_phase_name(phase);
}

bool parse_phase(const std::string & value, common_flydelta_bootstrap_zoom_phase & phase) {
    if (value == "alpha_zoom") phase = common_flydelta_bootstrap_zoom_phase::alpha_zoom;
    else if (value == "profile_zoom") phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
    else if (value == "sign_control") phase = common_flydelta_bootstrap_zoom_phase::sign_control;
    else if (value == "adaptive_alpha") phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    else return false;
    return true;
}

const char * refinement_kind_name(common_flydelta_bootstrap_refinement_kind kind) {
    return common_flydelta_bootstrap_refinement_kind_name(kind);
}

bool parse_refinement_kind(
        const std::string & value,
        common_flydelta_bootstrap_refinement_kind & kind) {
    if (value == "bootstrap_zoom") kind = common_flydelta_bootstrap_refinement_kind::bootstrap_zoom;
    else if (value == "adaptive_alpha") kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    else return false;
    return true;
}

const char * alpha_status_name(common_flydelta_alpha_response_status status) {
    return common_flydelta_alpha_response_status_name(status);
}

bool parse_alpha_status(
        const std::string & value,
        common_flydelta_alpha_response_status & status) {
    for (const auto candidate : {
            common_flydelta_alpha_response_status::inconclusive,
            common_flydelta_alpha_response_status::helped,
            common_flydelta_alpha_response_status::saturated,
            common_flydelta_alpha_response_status::safety_limited,
            common_flydelta_alpha_response_status::budget_limited,
            common_flydelta_alpha_response_status::upper_bound_reached}) {
        if (value == alpha_status_name(candidate)) {
            status = candidate;
            return true;
        }
    }
    return false;
}

const char * outcome_name(common_flydelta_counterfactual_outcome outcome) {
    return common_flydelta_counterfactual_outcome_name(outcome);
}

bool parse_outcome(const std::string & value, common_flydelta_counterfactual_outcome & outcome) {
    if (value == "helped") outcome = common_flydelta_counterfactual_outcome::helped;
    else if (value == "neutral") outcome = common_flydelta_counterfactual_outcome::neutral;
    else if (value == "harmed") outcome = common_flydelta_counterfactual_outcome::harmed;
    else if (value == "unknown") outcome = common_flydelta_counterfactual_outcome::unknown;
    else return false;
    return true;
}

json candidate_json(const common_flydelta_bootstrap_zoom_candidate & candidate) {
    return {
        {"schema_version", candidate.schema_version},
        {"phase", phase_name(candidate.phase)},
        {"layer_indices", candidate.layer_indices},
        {"layer_weights", candidate.layer_weights},
        {"total_scale", candidate.total_scale},
        {"opposite_sign_control", candidate.opposite_sign_control},
    };
}

json trial_json(const common_flydelta_bootstrap_zoom_trial & trial) {
    json value = {
        {"candidate", candidate_json(trial.candidate)},
        {"outcome", outcome_name(trial.outcome)},
        {"host_evaluated", trial.host_evaluated},
        {"verifier_known", trial.verifier_known},
        {"margin_available", trial.margin_available},
        {"margin_delta", trial.margin_delta},
        {"diagnostics_available", trial.diagnostics_available},
    };
    if (trial.diagnostics_available) {
        value["diagnostics"] = {
            {"schema_version", trial.diagnostics.schema_version},
            {"layer_index", trial.diagnostics.layer_index},
            {"cosine", trial.diagnostics.cosine},
            {"progress", trial.diagnostics.progress},
            {"leakage", trial.diagnostics.leakage},
            {"shift_norm", trial.diagnostics.shift_norm},
        };
    }
    return value;
}

bool parse_candidate(const json & value, common_flydelta_bootstrap_zoom_candidate & candidate) {
    candidate = {};
    candidate.schema_version = value.value("schema_version", 0);
    if (!parse_phase(value.value("phase", ""), candidate.phase)) return false;
    candidate.layer_indices = value.value("layer_indices", std::vector<uint32_t>{});
    candidate.layer_weights = value.value("layer_weights", std::vector<float>{});
    candidate.total_scale = value.value("total_scale", 0.0f);
    candidate.opposite_sign_control = value.value("opposite_sign_control", false);
    return true;
}

bool parse_trial(const json & value, common_flydelta_bootstrap_zoom_trial & trial) {
    trial = {};
    if (!value.is_object() || !parse_candidate(
            value.value("candidate", json::object()), trial.candidate) ||
            !parse_outcome(value.value("outcome", ""), trial.outcome)) return false;
    trial.host_evaluated = value.value("host_evaluated", false);
    trial.verifier_known = value.value("verifier_known",
        value.value("host_verified", false));
    trial.margin_available = value.value("margin_available", false);
    trial.margin_delta = value.value("margin_delta", 0.0f);
    trial.diagnostics_available = value.value("diagnostics_available", false);
    if (trial.diagnostics_available) {
        const auto diagnostics = value.value("diagnostics", json::object());
        trial.diagnostics.schema_version = diagnostics.value("schema_version", 0);
        trial.diagnostics.layer_index = diagnostics.value("layer_index", 0);
        trial.diagnostics.cosine = diagnostics.value("cosine", 0.0f);
        trial.diagnostics.progress = diagnostics.value("progress", 0.0f);
        trial.diagnostics.leakage = diagnostics.value("leakage", 0.0f);
        trial.diagnostics.shift_norm = diagnostics.value("shift_norm", 0.0f);
    }
    return true;
}

std::string state_to_json(const common_flydelta_bootstrap_zoom_state & state) {
    json value = {
        {"kind", "flydelta_bootstrap_zoom_state"},
        {"schema_version", state.schema_version},
        {"state_ref", state.state_ref},
        {"behavior_key", state.behavior_key},
        {"model_profile_fingerprint", state.model_profile_fingerprint},
        {"capture_layout_revision", state.capture_layout_revision},
        {"phase", phase_name(state.phase)},
        {"refinement_kind", refinement_kind_name(state.refinement_kind)},
        {"anchor_layer", state.anchor_layer},
        {"selected_scale", state.selected_scale},
        {"best_margin_delta", state.best_margin_delta},
        {"best_search_score", state.best_search_score},
        {"extra_model_trials", state.extra_model_trials},
        {"next_candidate_index", state.next_candidate_index},
        {"surface_revision", state.surface_revision},
        {"parent_surface_revision", state.parent_surface_revision},
        {"search_rank", state.search_rank},
        {"evidence_rank", state.evidence_rank},
        {"surface_origin", state.surface_origin},
        {"parent_surface_ref", state.parent_surface_ref},
        {"local_layers", state.local_layers},
        {"completed_trials", json::array()},
        {"surface_trials", json::array()},
        {"selection", {
            {"selected", state.selection.selected},
            {"trial_index", state.selection.trial_index},
            {"search_score", state.selection.search_score},
        }},
        {"alpha_response_available", state.alpha_response_available},
        {"alpha_response", {
            {"selected", state.alpha_response.selected},
            {"scale", state.alpha_response.scale},
            {"utility", state.alpha_response.utility},
            {"trial_index", state.alpha_response.trial_index},
            {"response_status", alpha_status_name(state.alpha_response.response_status)},
            {"last_scale", state.alpha_response.last_scale},
            {"last_utility", state.alpha_response.last_utility},
            {"utility_slope", state.alpha_response.utility_slope},
            {"max_reachable_scale", state.alpha_response.max_reachable_scale},
            {"range_not_exhausted", state.alpha_response.range_not_exhausted},
            {"minimum_effective_available", state.alpha_response.minimum_effective_available},
            {"minimum_effective_scale", state.alpha_response.minimum_effective_scale},
            {"minimum_effective_trial_index", state.alpha_response.minimum_effective_trial_index},
            {"best_margin_delta_total", state.alpha_response.best_margin_delta_total},
            {"best_margin_delta_normalized", state.alpha_response.best_margin_delta_normalized},
            {"best_margin_available", state.alpha_response.best_margin_available},
        }},
    };
    for (const auto & trial : state.completed_trials) {
        value["completed_trials"].push_back(trial_json(trial));
    }
    for (const auto & trial : state.surface_trials) {
        value["surface_trials"].push_back(trial_json(trial));
    }
    return value.dump();
}

bool state_from_json(const std::string & text,
        common_flydelta_bootstrap_zoom_state & state, std::string & error) {
    try {
        const auto value = json::parse(text);
        if (!value.is_object() || value.value("kind", "") != "flydelta_bootstrap_zoom_state") {
            error = "lifecycle record is not a BootstrapZoom state";
            return false;
        }
        state = {};
        state.schema_version = value.value("schema_version", 0);
        state.state_ref = value.value("state_ref", "");
        state.behavior_key = value.value("behavior_key", "");
        state.model_profile_fingerprint = value.value("model_profile_fingerprint", "");
        state.capture_layout_revision = value.value("capture_layout_revision", "");
        if (!parse_phase(value.value("phase", ""), state.phase)) {
            error = "BootstrapZoom lifecycle state phase is invalid";
            return false;
        }
        if (!parse_refinement_kind(value.value("refinement_kind", "bootstrap_zoom"),
                state.refinement_kind)) {
            error = "BootstrapZoom lifecycle refinement kind is invalid";
            return false;
        }
        state.anchor_layer = value.value("anchor_layer", 0U);
        state.selected_scale = value.value("selected_scale", 0.0f);
        state.best_margin_delta = value.value("best_margin_delta", 0.0f);
        state.best_search_score = value.value("best_search_score", 0.0f);
        state.extra_model_trials = value.value("extra_model_trials", size_t{0});
        state.next_candidate_index = value.value("next_candidate_index", size_t{0});
        state.surface_revision = value.value("surface_revision", 1U);
        state.parent_surface_revision = value.value("parent_surface_revision", 0U);
        state.search_rank = value.value("search_rank", size_t{1});
        state.evidence_rank = value.value("evidence_rank", 1.0f);
        state.surface_origin = value.value("surface_origin", "bootstrap_rank1");
        state.parent_surface_ref = value.value("parent_surface_ref", "");
        state.local_layers = value.value("local_layers", std::vector<uint32_t>{});
        for (const auto & item : value.value("completed_trials", json::array())) {
            common_flydelta_bootstrap_zoom_trial trial;
            if (!parse_trial(item, trial)) {
                error = "BootstrapZoom lifecycle trial is invalid";
                return false;
            }
            state.completed_trials.push_back(std::move(trial));
        }
        for (const auto & item : value.value("surface_trials", json::array())) {
            common_flydelta_bootstrap_zoom_trial trial;
            if (!parse_trial(item, trial)) {
                error = "BootstrapZoom lifecycle surface trial is invalid";
                return false;
            }
            state.surface_trials.push_back(std::move(trial));
        }
        const auto selection = value.value("selection", json::object());
        state.selection.selected = selection.value("selected", false);
        state.selection.trial_index = selection.value("trial_index", size_t{0});
        state.selection.search_score = selection.value("search_score", 0.0f);
        state.alpha_response_available = value.value("alpha_response_available", false);
        const auto alpha = value.value("alpha_response", json::object());
        state.alpha_response.selected = alpha.value("selected", false);
        state.alpha_response.scale = alpha.value("scale", 0.0f);
        state.alpha_response.utility = alpha.value("utility", 0.0f);
        state.alpha_response.trial_index = alpha.value("trial_index", size_t{0});
        if (!parse_alpha_status(alpha.value("response_status", "inconclusive"),
                state.alpha_response.response_status)) {
            error = "AdaptiveAlpha lifecycle response status is invalid";
            return false;
        }
        state.alpha_response.last_scale = alpha.value("last_scale", 0.0f);
        state.alpha_response.last_utility = alpha.value("last_utility", 0.0f);
        state.alpha_response.utility_slope = alpha.value("utility_slope", 0.0f);
        state.alpha_response.max_reachable_scale = alpha.value("max_reachable_scale", 0.0f);
        state.alpha_response.range_not_exhausted = alpha.value("range_not_exhausted", false);
        state.alpha_response.minimum_effective_available =
            alpha.value("minimum_effective_available", false);
        state.alpha_response.minimum_effective_scale =
            alpha.value("minimum_effective_scale", 0.0f);
        state.alpha_response.minimum_effective_trial_index =
            alpha.value("minimum_effective_trial_index", size_t{0});
        state.alpha_response.best_margin_delta_total =
            alpha.value("best_margin_delta_total", 0.0f);
        state.alpha_response.best_margin_delta_normalized =
            alpha.value("best_margin_delta_normalized", 0.0f);
        state.alpha_response.best_margin_available =
            alpha.value("best_margin_available", false);
    } catch (const std::exception & exception) {
        error = std::string("invalid BootstrapZoom lifecycle state JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_bootstrap_zoom_state_validate(state, error);
}

} // namespace

bool common_flydelta_bootstrap_zoom_lifecycle_context_validate(
        const common_flydelta_bootstrap_zoom_lifecycle_context & context,
        std::string & error) {
    error.clear();
    if (!bounded(context.namespace_id) || context.project_id.size() > 512 ||
            context.session_id.size() > 512 || !bounded(context.source_id) ||
            !bounded(context.created_at)) {
        error = "FlyDelta BootstrapZoom lifecycle context is incomplete or unbounded";
        return false;
    }
    return true;
}

bool common_flydelta_configure_bootstrap_zoom_lifecycle_callbacks(
        common_learning_lifecycle_store & store,
        const common_flydelta_bootstrap_zoom_lifecycle_context & context,
        common_flydelta_evaluator_callbacks & callbacks,
        std::string & error) {
    if (!common_flydelta_bootstrap_zoom_lifecycle_context_validate(context, error)) return false;
    callbacks.resolve_bootstrap_zoom_state = [&store](const std::string & state_ref,
            common_flydelta_bootstrap_zoom_state & state, std::string & resolve_error) {
        if (!bounded(state_ref)) { resolve_error = "BootstrapZoom state reference is invalid"; return false; }
        const auto records = store.list(resolve_error);
        if (!resolve_error.empty()) return false;
        for (auto it = records.rbegin(); it != records.rend(); ++it) {
            if (it->kind != common_learning_lifecycle_kind::flydelta_experiment ||
                    it->subject_id != state_ref) continue;
            if (!state_from_json(it->payload_json, state, resolve_error)) return false;
            if (state.state_ref != state_ref) {
                resolve_error = "BootstrapZoom lifecycle state reference does not match subject";
                return false;
            }
            return true;
        }
        resolve_error = "BootstrapZoom lifecycle state was not found";
        return false;
    };
    callbacks.persist_bootstrap_zoom_state = [&store, context](
            const common_flydelta_bootstrap_zoom_state & input,
            std::string & state_ref, std::string & persist_error) {
        auto state = input;
        if (!common_flydelta_bootstrap_zoom_state_validate(state, persist_error)) return false;
        if (state.state_ref.empty()) {
            const auto identity = state_to_json(state);
            state.state_ref = "flydelta://state/bootstrap-zoom/" +
                hash_text(identity).substr(7, 32);
        }
        if (!common_flydelta_bootstrap_zoom_state_validate(state, persist_error)) return false;
        const auto payload = state_to_json(state);
        common_learning_lifecycle_record record;
        record.event_id = "flydelta://event/bootstrap-zoom/" +
            hash_text(state.state_ref).substr(7, 32);
        record.subject_id = state.state_ref;
        record.kind = common_learning_lifecycle_kind::flydelta_experiment;
        record.status = common_learning_lifecycle_status::running;
        record.idempotency_key = "flydelta/bootstrap-zoom/" + hash_text(state.state_ref).substr(7, 32);
        record.source_id = context.source_id;
        record.namespace_id = context.namespace_id;
        record.project_id = context.project_id;
        record.session_id = context.session_id;
        record.content_hash = hash_text(payload);
        record.created_at = context.created_at;
        record.payload_json = payload;
        if (!store.append(record, persist_error)) return false;
        state_ref = state.state_ref;
        return true;
    };
    return true;
}
