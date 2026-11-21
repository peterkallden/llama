#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

using json = nlohmann::ordered_json;

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

bool bounded(const std::string & value, size_t maximum = 512) {
    return !value.empty() && value.size() <= maximum;
}

float norm(const std::vector<float> & values) {
    float sum = 0.0f;
    for (const float value : values) sum += value * value;
    return std::sqrt(sum);
}

float dot(const std::vector<float> & left, const std::vector<float> & right) {
    float value = 0.0f;
    for (size_t index = 0; index < left.size(); ++index) value += left[index] * right[index];
    return value;
}

bool valid_phase(common_flydelta_representation_augmentation_phase phase) {
    return phase >= common_flydelta_representation_augmentation_phase::discover_donors &&
        phase <= common_flydelta_representation_augmentation_phase::done;
}

bool parse_phase(const std::string & value,
        common_flydelta_representation_augmentation_phase & phase) {
    for (int index = 0; index <= static_cast<int>(
            common_flydelta_representation_augmentation_phase::done); ++index) {
        const auto candidate = static_cast<common_flydelta_representation_augmentation_phase>(index);
        if (value == common_flydelta_representation_augmentation_phase_name(candidate)) {
            phase = candidate;
            return true;
        }
    }
    return false;
}

} // namespace

const char * common_flydelta_representation_augmentation_phase_name(
        common_flydelta_representation_augmentation_phase phase) {
    switch (phase) {
        case common_flydelta_representation_augmentation_phase::discover_donors:
            return "discover_donors";
        case common_flydelta_representation_augmentation_phase::qualify_donor:
            return "qualify_donor";
        case common_flydelta_representation_augmentation_phase::capture_donor:
            return "capture_donor";
        case common_flydelta_representation_augmentation_phase::build_latent_delta:
            return "build_latent_delta";
        case common_flydelta_representation_augmentation_phase::run_controls:
            return "run_controls";
        case common_flydelta_representation_augmentation_phase::localize_surface:
            return "localize_surface";
        case common_flydelta_representation_augmentation_phase::full_generation:
            return "full_generation";
        case common_flydelta_representation_augmentation_phase::verify:
            return "verify";
        case common_flydelta_representation_augmentation_phase::done:
            return "done";
    }
    return "done";
}

const char * common_flydelta_representation_augmentation_action_name(
        common_flydelta_representation_augmentation_action action) {
    switch (action) {
        case common_flydelta_representation_augmentation_action::stop: return "stop";
        case common_flydelta_representation_augmentation_action::retain: return "retain";
        case common_flydelta_representation_augmentation_action::refine_bootstrap:
            return "refine_bootstrap";
        case common_flydelta_representation_augmentation_action::run_controls:
            return "run_controls";
        case common_flydelta_representation_augmentation_action::recenter_augmented_surface:
            return "recenter_augmented_surface";
        case common_flydelta_representation_augmentation_action::prepare_concept_material:
            return "prepare_concept_material";
        case common_flydelta_representation_augmentation_action::run_concept_synthesis:
            return "run_concept_synthesis";
        case common_flydelta_representation_augmentation_action::allow_tfo_lite:
            return "allow_tfo_lite";
    }
    return "stop";
}

bool common_flydelta_representation_donor_candidate_validate(
        const common_flydelta_representation_donor_candidate & candidate,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || !bounded(candidate.donor_id) ||
            !bounded(candidate.source_type, 128) || !bounded(candidate.source_ref) ||
            !bounded(candidate.behavior_key) || !bounded(candidate.context_payload_ref) ||
            !bounded(candidate.qualification_policy, 128) ||
            !bounded(candidate.provenance, 1024)) {
        error = "FlyDelta representation donor candidate is invalid";
        return false;
    }
    if (candidate.promotable) {
        error = "FlyDelta representation donor must not be promotable";
        return false;
    }
    return true;
}

bool common_flydelta_representation_donor_qualification_validate(
        const common_flydelta_representation_donor_qualification & qualification,
        std::string & error) {
    error.clear();
    if (qualification.schema_version != 1 || !bounded(qualification.donor_id) ||
            (qualification.verifier_known && !qualification.host_evaluated) ||
            !finite(qualification.margin_gain) || qualification.reason.size() > 512 ||
            (qualification.geometry_available &&
                !common_flydelta_representation_diagnostics_validate(
                    qualification.geometry, error))) {
        if (error.empty()) error = "FlyDelta representation donor qualification is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_qualify_representation_donor(
        const common_flydelta_representation_donor_candidate & candidate,
        const common_flydelta_representation_donor_qualification & observation,
        float minimum_margin_gain,
        float max_leakage,
        float max_shift_norm,
        common_flydelta_representation_donor_qualification & qualified,
        std::string & error) {
    error.clear();
    qualified = observation;
    if (!common_flydelta_representation_donor_candidate_validate(candidate, error) ||
            !common_flydelta_representation_donor_qualification_validate(observation, error) ||
            observation.donor_id != candidate.donor_id || !finite(minimum_margin_gain) ||
            minimum_margin_gain < 0.0f || !finite(max_leakage) || max_leakage < 0.0f ||
            !finite(max_shift_norm) || max_shift_norm <= 0.0f) {
        if (error.empty()) error = "FlyDelta donor qualification bounds are invalid";
        return false;
    }
    const bool geometry_ok = !observation.geometry_available ||
        (observation.geometry.cosine >= 0.0f && observation.geometry.progress > 0.0f &&
         observation.geometry.leakage <= max_leakage &&
         observation.geometry.shift_norm <= max_shift_norm);
    const bool verified_helped = observation.host_evaluated && observation.verifier_known &&
        observation.host_outcome ==
        common_flydelta_counterfactual_outcome::helped;
    const bool margin_useful = observation.decision_margin_available &&
        observation.margin_gain >= minimum_margin_gain;
    qualified.search_qualified = observation.safe_to_continue && geometry_ok &&
        (verified_helped || margin_useful);
    if (!qualified.search_qualified) {
        qualified.reason = !observation.safe_to_continue ? "unsafe" :
            !geometry_ok ? "geometry_out_of_bounds" : "no_search_signal";
    } else if (verified_helped) {
        qualified.reason = "host_verified_contextual_help";
    } else {
        qualified.reason = "margin_guided_experimental_search";
    }
    return true;
}

bool common_flydelta_representation_latent_delta_validate(
        const common_flydelta_representation_latent_delta & delta,
        size_t expected_dimension,
        std::string & error) {
    error.clear();
    if (delta.schema_version != 1 || !bounded(delta.donor_id) || delta.layer_index < 0 ||
            !finite(delta.raw_norm) || !finite(delta.residual_norm) ||
            !finite(delta.removed_norm) || delta.raw_norm < 0.0f ||
            delta.residual_norm < 0.0f || delta.removed_norm < 0.0f ||
            (delta.available && (delta.values.size() != expected_dimension ||
                delta.residual_norm <= 0.0f)) ||
            (!delta.available && !delta.values.empty())) {
        error = "FlyDelta representation latent delta is invalid";
        return false;
    }
    for (const float value : delta.values) {
        if (!finite(value)) {
            error = "FlyDelta representation latent delta contains non-finite values";
            return false;
        }
    }
    return true;
}

bool common_flydelta_build_residualized_latent_delta(
        const std::string & donor_id,
        int32_t layer_index,
        const std::vector<float> & target_plus_donor,
        const std::vector<float> & target_only,
        const std::vector<std::vector<float>> & current_surface,
        float minimum_residual_norm,
        common_flydelta_representation_latent_delta & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!bounded(donor_id) || layer_index < 0 || target_plus_donor.empty() ||
            target_plus_donor.size() != target_only.size() || !finite(minimum_residual_norm) ||
            minimum_residual_norm <= 0.0f || current_surface.size() > 8) {
        error = "FlyDelta latent-delta input is invalid";
        return false;
    }
    for (const auto & vector : current_surface) {
        if (vector.size() != target_plus_donor.size() || norm(vector) <= 0.0f) {
            error = "FlyDelta current surface is incompatible with donor captures";
            return false;
        }
    }
    std::vector<float> residual(target_plus_donor.size());
    for (size_t index = 0; index < residual.size(); ++index) {
        residual[index] = target_plus_donor[index] - target_only[index];
        if (!finite(residual[index])) {
            error = "FlyDelta latent-delta captures contain non-finite values";
            return false;
        }
    }
    const float raw_norm = norm(residual);
    if (raw_norm <= 0.0f) {
        result = {1, donor_id, layer_index, {}, true, false, raw_norm, 0.0f, 0.0f};
        return true;
    }

    // Gram-Schmidt creates Q locally. The caller need not store an orthonormal
    // basis, and no dimension-sized matrix is allocated.
    std::vector<std::vector<float>> orthonormal;
    for (const auto & source : current_surface) {
        std::vector<float> candidate = source;
        for (const auto & basis : orthonormal) {
            const float projection = dot(candidate, basis);
            for (size_t index = 0; index < candidate.size(); ++index) {
                candidate[index] -= projection * basis[index];
            }
        }
        const float candidate_norm = norm(candidate);
        if (candidate_norm <= std::numeric_limits<float>::epsilon()) continue;
        for (float & value : candidate) value /= candidate_norm;
        orthonormal.push_back(std::move(candidate));
    }
    for (const auto & basis : orthonormal) {
        const float projection = dot(residual, basis);
        for (size_t index = 0; index < residual.size(); ++index) {
            residual[index] -= projection * basis[index];
        }
    }
    const float residual_norm = norm(residual);
    result.schema_version = 1;
    result.donor_id = donor_id;
    result.layer_index = layer_index;
    result.residualized = !current_surface.empty();
    result.raw_norm = raw_norm;
    result.residual_norm = residual_norm;
    result.removed_norm = std::sqrt(std::max(0.0f,
        raw_norm * raw_norm - residual_norm * residual_norm));
    if (residual_norm < minimum_residual_norm) return true;
    for (float & value : residual) value /= residual_norm;
    result.values = std::move(residual);
    result.available = true;
    result.residual_norm = residual_norm;
    return true;
}

bool common_flydelta_representation_augmentation_config_validate(
        const common_flydelta_representation_augmentation_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_donor_candidates == 0 ||
            config.max_donor_candidates > 16 || config.max_qualified_donors == 0 ||
            config.max_qualified_donors > config.max_donor_candidates ||
            config.max_latent_deltas == 0 || config.max_latent_deltas > config.max_qualified_donors ||
            config.max_controls != 4 || config.max_local_whirlpool_probes == 0 ||
            config.max_local_whirlpool_probes > 16 || config.max_full_generation == 0 ||
            config.max_full_generation > 8 || !finite(config.minimum_residual_norm) ||
            config.minimum_residual_norm <= 0.0f || !finite(config.minimum_margin_gain) ||
            config.minimum_margin_gain < 0.0f || !finite(config.max_leakage) ||
            config.max_leakage < 0.0f || !finite(config.max_shift_norm) ||
            config.max_shift_norm <= 0.0f) {
        error = "FlyDelta representation augmentation configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_propose_representation_augmentation_controls(
        const common_flydelta_representation_augmentation_config & config,
        std::vector<std::vector<float>> & coefficients,
        std::string & error) {
    error.clear();
    coefficients.clear();
    if (!common_flydelta_representation_augmentation_config_validate(config, error)) return false;
    const float diagonal = 1.0f / std::sqrt(2.0f);
    coefficients = {{1.0f, 0.0f}, {0.0f, 1.0f},
                    {diagonal, diagonal}, {diagonal, -diagonal}};
    return true;
}

bool common_flydelta_representation_augmentation_state_validate(
        const common_flydelta_representation_augmentation_state & state,
        const common_flydelta_representation_augmentation_config & config,
        std::string & error) {
    error.clear();
    if (!common_flydelta_representation_augmentation_config_validate(config, error) ||
            state.schema_version != 1 || state.state_ref.size() > 512 ||
            !bounded(state.model_fingerprint) || !bounded(state.behavior_key) ||
            !bounded(state.direction_family_id) || state.parent_surface_revision == 0 ||
            state.parent_search_state_ref.size() > 512 || !finite(state.parent_evidence_rank) ||
            state.parent_evidence_rank <= 0.0f || !finite(state.evidence_rank) ||
            state.evidence_rank != state.parent_evidence_rank || state.search_rank == 0 ||
            state.search_rank > 4 ||
            state.selected_region.empty() || state.selected_region.size() > 4 ||
            !bounded(state.target_fixture_ref) ||
            state.donor_candidate_refs.size() > config.max_donor_candidates ||
            state.qualified_donor_refs.size() > config.max_qualified_donors ||
            state.evaluated_donor_refs.size() > config.max_donor_candidates ||
            !valid_phase(state.phase) || !finite(state.best_margin_gain) ||
            state.remaining_budget > config.max_donor_candidates + config.max_controls +
                config.max_local_whirlpool_probes + config.max_full_generation ||
            state.surface_revision < state.parent_surface_revision ||
            state.next_action.size() > 64) {
        if (error.empty()) error = "FlyDelta representation augmentation state is invalid";
        return false;
    }
    for (const uint32_t layer : state.selected_region) {
        if (layer == 0) {
            error = "FlyDelta representation augmentation region contains layer zero";
            return false;
        }
    }
    if (!std::is_sorted(state.selected_region.begin(), state.selected_region.end()) ||
            std::adjacent_find(state.selected_region.begin(), state.selected_region.end()) !=
                state.selected_region.end()) {
        error = "FlyDelta representation augmentation region is not unique and sorted";
        return false;
    }
    for (const auto & ref : state.donor_candidate_refs) if (!bounded(ref)) {
        error = "FlyDelta donor candidate reference is invalid";
        return false;
    }
    for (const auto & ref : state.qualified_donor_refs) if (!bounded(ref)) {
        error = "FlyDelta qualified donor reference is invalid";
        return false;
    }
    for (const auto & ref : state.evaluated_donor_refs) if (!bounded(ref)) {
        error = "FlyDelta evaluated donor reference is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_apply_representation_augmentation(
        common_flydelta_representation_augmentation_state & state,
        const common_flydelta_representation_latent_delta & latent,
        std::string & error) {
    error.clear();
    common_flydelta_representation_augmentation_config config;
    if (!common_flydelta_representation_augmentation_state_validate(state, config, error) ||
            !common_flydelta_representation_latent_delta_validate(
                latent, latent.values.size(), error) || !latent.available ||
            latent.layer_index <= 0 || state.remaining_budget == 0) {
        if (error.empty()) error = "FlyDelta augmentation cannot apply latent delta";
        return false;
    }
    if (state.search_rank >= 4) {
        error = "FlyDelta augmentation search rank is at its bound";
        return false;
    }
    if (std::find(state.evaluated_donor_refs.begin(), state.evaluated_donor_refs.end(),
            latent.donor_id) == state.evaluated_donor_refs.end()) {
        state.evaluated_donor_refs.push_back(latent.donor_id);
    }
    ++state.search_rank;
    ++state.surface_revision;
    --state.remaining_budget;
    state.phase = common_flydelta_representation_augmentation_phase::run_controls;
    state.next_action = common_flydelta_representation_augmentation_action_name(
        common_flydelta_representation_augmentation_action::run_controls);
    return true;
}

bool common_flydelta_decide_representation_augmentation(
        const common_flydelta_representation_augmentation_config & config,
        const common_flydelta_representation_augmentation_state & state,
        const std::vector<common_flydelta_representation_donor_qualification> & observations,
        bool controls_have_positive_utility,
        bool tfo_has_positive_utility,
        common_flydelta_representation_augmentation_action & action,
        std::string & error) {
    error.clear();
    action = common_flydelta_representation_augmentation_action::stop;
    if (!common_flydelta_representation_augmentation_state_validate(state, config, error) ||
            observations.size() > config.max_donor_candidates) return false;
    bool qualified = false;
    for (const auto & observation : observations) {
        if (!common_flydelta_representation_donor_qualification_validate(observation, error)) {
            return false;
        }
        qualified = qualified || observation.search_qualified;
    }
    if (state.phase == common_flydelta_representation_augmentation_phase::done ||
            state.remaining_budget == 0) return true;
    switch (state.phase) {
        case common_flydelta_representation_augmentation_phase::discover_donors:
            action = qualified ? common_flydelta_representation_augmentation_action::run_controls :
                common_flydelta_representation_augmentation_action::retain;
            break;
        case common_flydelta_representation_augmentation_phase::qualify_donor:
        case common_flydelta_representation_augmentation_phase::capture_donor:
        case common_flydelta_representation_augmentation_phase::build_latent_delta:
            action = qualified ? common_flydelta_representation_augmentation_action::run_controls :
                common_flydelta_representation_augmentation_action::retain;
            break;
        case common_flydelta_representation_augmentation_phase::run_controls:
            action = controls_have_positive_utility
                ? common_flydelta_representation_augmentation_action::recenter_augmented_surface
                : common_flydelta_representation_augmentation_action::retain;
            break;
        case common_flydelta_representation_augmentation_phase::localize_surface:
            action = tfo_has_positive_utility
                ? common_flydelta_representation_augmentation_action::allow_tfo_lite
                : common_flydelta_representation_augmentation_action::retain;
            break;
        case common_flydelta_representation_augmentation_phase::full_generation:
        case common_flydelta_representation_augmentation_phase::verify:
            action = common_flydelta_representation_augmentation_action::retain;
            break;
        case common_flydelta_representation_augmentation_phase::done:
            break;
    }
    return true;
}

bool common_flydelta_select_concept_synthesis_escape(
        bool augmentation_terminal,
        bool concept_material_available,
        common_flydelta_representation_augmentation_action & action,
        std::string & error) {
    error.clear();
    action = common_flydelta_representation_augmentation_action::retain;
    if (!augmentation_terminal) return true;
    if (concept_material_available) {
        action = common_flydelta_representation_augmentation_action::run_concept_synthesis;
    }
    return true;
}

bool common_flydelta_select_concept_material_escape(
        bool augmentation_terminal,
        bool relation_set_available,
        bool trajectory_material_available,
        common_flydelta_representation_augmentation_action & action,
        std::string & error) {
    error.clear();
    action = common_flydelta_representation_augmentation_action::retain;
    if (!augmentation_terminal) return true;
    if (trajectory_material_available) {
        action = common_flydelta_representation_augmentation_action::run_concept_synthesis;
    } else if (relation_set_available) {
        action = common_flydelta_representation_augmentation_action::prepare_concept_material;
    }
    return true;
}

std::string common_flydelta_representation_augmentation_state_to_json(
        const common_flydelta_representation_augmentation_state & state) {
    json value = {
        {"kind", "flydelta_representation_augmentation_state"},
        {"schema_version", state.schema_version}, {"state_ref", state.state_ref},
        {"model_fingerprint", state.model_fingerprint}, {"behavior_key", state.behavior_key},
        {"direction_family_id", state.direction_family_id},
        {"parent_surface_revision", state.parent_surface_revision},
        {"parent_search_state_ref", state.parent_search_state_ref},
        {"parent_evidence_rank", state.parent_evidence_rank},
        {"evidence_rank", state.evidence_rank}, {"search_rank", state.search_rank},
        {"selected_region", state.selected_region}, {"target_fixture_ref", state.target_fixture_ref},
        {"donor_candidate_refs", state.donor_candidate_refs},
        {"qualified_donor_refs", state.qualified_donor_refs},
        {"evaluated_donor_refs", state.evaluated_donor_refs},
        {"phase", common_flydelta_representation_augmentation_phase_name(state.phase)},
        {"best_donor_ref", state.best_donor_ref}, {"best_margin_gain", state.best_margin_gain},
        {"remaining_budget", state.remaining_budget}, {"surface_revision", state.surface_revision},
        {"next_action", state.next_action},
    };
    return value.dump();
}

bool common_flydelta_representation_augmentation_state_from_json(
        const std::string & text,
        common_flydelta_representation_augmentation_state & state,
        const common_flydelta_representation_augmentation_config & config,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (!value.is_object() || value.value("kind", "") !=
                "flydelta_representation_augmentation_state") {
            error = "lifecycle record is not a FlyDelta representation augmentation state";
            return false;
        }
        state = {};
        state.schema_version = value.value("schema_version", 0);
        state.state_ref = value.value("state_ref", "");
        state.model_fingerprint = value.value("model_fingerprint", "");
        state.behavior_key = value.value("behavior_key", "");
        state.direction_family_id = value.value("direction_family_id", "");
        state.parent_surface_revision = value.value("parent_surface_revision", uint64_t{0});
        state.parent_search_state_ref = value.value("parent_search_state_ref", "");
        state.parent_evidence_rank = value.value("parent_evidence_rank", 0.0f);
        state.evidence_rank = value.value("evidence_rank", 0.0f);
        state.search_rank = value.value("search_rank", size_t{0});
        state.selected_region = value.value("selected_region", std::vector<uint32_t>{});
        state.target_fixture_ref = value.value("target_fixture_ref", "");
        state.donor_candidate_refs = value.value("donor_candidate_refs", std::vector<std::string>{});
        state.qualified_donor_refs = value.value("qualified_donor_refs", std::vector<std::string>{});
        state.evaluated_donor_refs = value.value("evaluated_donor_refs", std::vector<std::string>{});
        if (!parse_phase(value.value("phase", ""), state.phase)) {
            error = "FlyDelta representation augmentation phase is invalid";
            return false;
        }
        state.best_donor_ref = value.value("best_donor_ref", "");
        state.best_margin_gain = value.value("best_margin_gain", 0.0f);
        state.remaining_budget = value.value("remaining_budget", size_t{0});
        state.surface_revision = value.value("surface_revision", uint64_t{0});
        state.next_action = value.value("next_action", "");
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta representation augmentation state JSON: ") +
            exception.what();
        return false;
    }
    return common_flydelta_representation_augmentation_state_validate(state, config, error);
}
