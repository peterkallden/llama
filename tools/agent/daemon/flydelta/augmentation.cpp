#include "../agent-daemon-flydelta-internal.h"
#include "agent/adaptation/flydelta/flydelta-search-budget.h"

#include "agent/agent-residual-patch.h"
#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "state.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_run_donor_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_capture_manifest> & manifests,
        std::string & error) {
    error.clear();
    manifests.clear();
    const auto scoped_provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    for (const auto & reference : job.capture_candidate_ids) {
        json value;
        if (!daemon_flydelta_read_json(*scoped_provider, reference, value, error)) return false;
        common_flydelta_capture_candidate candidate;
        try {
            candidate.schema_version = value.value("schema_version", 0);
            candidate.id = value.value("id", reference);
            candidate.transaction_id = value.value("transaction_id", reference);
            const auto source = common_adaptation_evidence_source_from_name(
                value.value("source", ""));
            if (!source) {
                error = "FlyDelta donor capture candidate source is invalid";
                return false;
            }
            candidate.source = *source;
            const auto scope = value.value("scope", json::object());
            candidate.scope.namespace_id = scope.value("namespace_id", "");
            candidate.scope.project_id = scope.value("project_id", "");
            candidate.scope.session_id = scope.value("session_id", "");
            candidate.scope.turn_id = scope.value("turn_id", "");
            candidate.behavior_key = value.value("behavior_key", job.seed.behavior_key);
            candidate.task_fingerprint = value.value("task_fingerprint", job.seed.task_fingerprint);
            candidate.baseline_ref = value.value("baseline_ref", "");
            candidate.candidate_ref = value.value("candidate_ref", "");
            candidate.verifier_ref = value.value("verifier_ref", "");
            candidate.evidence_refs = value.value("evidence_refs", std::vector<std::string>{});
            candidate.model_profile_fingerprint = value.value(
                "model_profile_fingerprint", scoped_provider->model_profile_fingerprint);
            candidate.capture_layout_revision = value.value(
                "capture_layout_revision", scoped_provider->capture_layout_revision);
            candidate.candidate_ready = value.value("candidate_ready", true);
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta donor capture candidate is malformed: ") + exception.what();
            return false;
        }
        if (!common_flydelta_capture_candidate_validate(candidate, error)) return false;
        const std::string baseline_ref = candidate.baseline_ref.empty()
            ? job.seed.baseline_ref : candidate.baseline_ref;
        const std::string candidate_ref = candidate.candidate_ref.empty()
            ? job.seed.candidate_ref : candidate.candidate_ref;
        const std::string fixture_ref = candidate.verifier_ref.empty()
            ? job.seed.verifier_ref : candidate.verifier_ref;
        if (baseline_ref.empty() || candidate_ref.empty() || fixture_ref.empty()) {
            error = "FlyDelta donor capture candidate must provide baseline_ref, candidate_ref and fixture_ref";
            return false;
        }
        common_flydelta_arm_batch_request batch;
        batch.batch_id = job.id + ":donor-capture:" + candidate.id;
        batch.wave_id = "donor-capture";
        for (size_t index = 0; index < 2; ++index) {
            common_flydelta_arm_request arm;
            arm.job_id = job.id;
            arm.wave_id = batch.wave_id;
            arm.proposal_index = index;
            arm.arm_id = batch.batch_id + (index == 0 ? ":baseline" : ":candidate");
            arm.context_ref = index == 0 ? baseline_ref : candidate_ref;
            arm.fixture_ref = fixture_ref;
            arm.intervention_ref = job.seed.candidate_ref;
            arm.fresh_context = true;
            arm.request_capture = true;
            arm.request_generation = true;
            arm.max_capture_bytes = 4U * 1024U * 1024U;
            arm.max_generated_tokens = 1;
            if (!common_flydelta_arm_request_validate(arm, error)) return false;
            batch.arms.push_back(std::move(arm));
        }
        common_flydelta_arm_batch_result result;
        if (!daemon_flydelta_execute_batch(scoped_provider, batch, result, error) ||
                result.arms.size() != batch.arms.size() ||
                result.arms[0].capture_ref.empty() || result.arms[1].capture_ref.empty()) {
            if (error.empty()) error = "FlyDelta donor capture did not return both manifests";
            return false;
        }
        common_flydelta_capture_manifest manifest;
        manifest.id = candidate.id + "/manifest";
        manifest.observation_id = candidate.transaction_id;
        manifest.source = candidate.source;
        manifest.behavior_key = candidate.behavior_key.empty()
            ? job.seed.behavior_key : candidate.behavior_key;
        manifest.model_profile_fingerprint = candidate.model_profile_fingerprint;
        manifest.template_fingerprint = job.seed.template_fingerprint.empty()
            ? scoped_provider->model_profile_fingerprint + ":template"
            : job.seed.template_fingerprint;
        manifest.execution_context_fingerprint = job.seed.execution_context_fingerprint;
        manifest.positive_execution_ref = candidate_ref;
        manifest.negative_execution_ref = baseline_ref;
        manifest.capture_layout_revision = candidate.capture_layout_revision;
        const std::string evidence_identity = baseline_ref + "\n" + candidate_ref + "\n" +
            candidate.id;
        manifest.evidence_hash = "sha256:" + hash_sha256_hex(
            evidence_identity.data(), evidence_identity.size());
        manifest.redaction_attested = true;
        manifest.captured_bytes = 1;
        if (!common_flydelta_capture_manifest_validate(
                manifest, 4U * 1024U * 1024U, error)) return false;
        std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
        std::shared_ptr<const common_flydelta_hidden_state_capture> candidate_capture;
        if (!daemon_flydelta_capture_from_reference(
                scoped_provider, result.arms[0].capture_ref, baseline_capture, error) ||
                !daemon_flydelta_capture_from_reference(
                scoped_provider, result.arms[1].capture_ref, candidate_capture, error)) return false;
        manifest.captured_bytes = (baseline_capture->values.size() +
            candidate_capture->values.size()) * sizeof(float);
        if (!common_flydelta_capture_manifest_validate(
                manifest, 4U * 1024U * 1024U, error)) return false;
        std::string manifest_ref;
        if (!daemon_flydelta_put_json_resource(
                scoped_provider, "flydelta-donor-manifest-" +
                    hash_sha256_hex(manifest.id.data(), manifest.id.size()).substr(0, 24) + ".json",
                "flydelta_donor_capture_manifest", candidate.id,
                common_flydelta_capture_manifest_to_json(manifest), manifest_ref, error)) return false;
        manifests.push_back(std::move(manifest));
    }
    return !manifests.empty();
}

bool daemon_flydelta_run_representation_augmentation(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_search_pipeline_config & pipeline_config,
        const common_flydelta_representation_augmentation_state * resume,
        common_flydelta_search_pipeline_result & result,
        common_flydelta_representation_augmentation_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (resume == nullptr) {
        error = "FlyDelta representation augmentation requires a resumed typed state";
        return false;
    }
    next = *resume;
    daemon_flydelta_augmentation_material material;
    if (!daemon_flydelta_load_augmentation_material(
            provider, resume->state_ref, material, error)) return false;
    if (material.parent_direction_ref.empty()) material.parent_direction_ref = job.seed.candidate_ref;

    const auto seed_result = [&]() {
        common_flydelta_search_pipeline_result value;
        std::string seed_error;
        if (!daemon_flydelta_augmentation_seed_result(provider, job, next, value, seed_error)) {
            error = std::move(seed_error);
        }
        return value;
    };

    switch (resume->phase) {
        case common_flydelta_representation_augmentation_phase::discover_donors:
            if (next.donor_candidate_refs.empty()) {
                error = "FlyDelta representation augmentation has no host-owned donor candidates";
                return false;
            }
            next.phase = common_flydelta_representation_augmentation_phase::qualify_donor;
            next.next_action = "run_representation_augmentation";
            result = seed_result();
            return error.empty();
        case common_flydelta_representation_augmentation_phase::qualify_donor:
            for (const auto & reference : next.donor_candidate_refs) {
                if (std::none_of(material.donors.begin(), material.donors.end(),
                        [&](const auto & donor) { return donor.candidate.donor_id == reference; })) {
                    daemon_flydelta_augmentation_donor_material donor;
                    if (!daemon_flydelta_parse_augmentation_donor(
                            provider, reference, next, job, donor, error)) return false;
                    material.donors.push_back(std::move(donor));
                }
            }
            next.phase = common_flydelta_representation_augmentation_phase::capture_donor;
            next.next_action = "run_representation_augmentation";
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            result = seed_result();
            return error.empty();
        case common_flydelta_representation_augmentation_phase::capture_donor: {
            auto donor = std::find_if(material.donors.begin(), material.donors.end(),
                [](const auto & value) { return value.donor_capture_ref.empty(); });
            if (donor == material.donors.end()) {
                next.phase = common_flydelta_representation_augmentation_phase::build_latent_delta;
                next.next_action = "run_representation_augmentation";
            } else {
                if (!daemon_flydelta_capture_augmentation_pair(provider, job, *donor, error)) return false;
                common_flydelta_representation_donor_qualification qualified;
                if (!common_flydelta_qualify_representation_donor(
                        donor->candidate, donor->qualification,
                        common_flydelta_representation_augmentation_config{}.minimum_margin_gain,
                        common_flydelta_representation_augmentation_config{}.max_leakage,
                        common_flydelta_representation_augmentation_config{}.max_shift_norm,
                        qualified, error)) return false;
                donor->qualification = qualified;
                if (qualified.search_qualified) {
                    if (std::find(next.qualified_donor_refs.begin(), next.qualified_donor_refs.end(),
                            donor->candidate.donor_id) == next.qualified_donor_refs.end()) {
                        next.qualified_donor_refs.push_back(donor->candidate.donor_id);
                    }
                    next.phase = common_flydelta_representation_augmentation_phase::build_latent_delta;
                } else {
                    next.phase = common_flydelta_representation_augmentation_phase::capture_donor;
                }
                next.next_action = "run_representation_augmentation";
            }
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            result = seed_result();
            return error.empty();
        }
        case common_flydelta_representation_augmentation_phase::build_latent_delta: {
            auto donor = std::find_if(material.donors.begin(), material.donors.end(),
                [&](const auto & value) {
                    return value.qualification.search_qualified &&
                        !value.target_capture_ref.empty() && !value.donor_capture_ref.empty() &&
                        std::find(next.evaluated_donor_refs.begin(), next.evaluated_donor_refs.end(),
                            value.candidate.donor_id) == next.evaluated_donor_refs.end();
                });
            if (donor == material.donors.end()) {
                next.phase = common_flydelta_representation_augmentation_phase::done;
                next.remaining_budget = 0;
                next.next_action = "retain";
                result = seed_result();
                return error.empty();
            }
            std::shared_ptr<const common_flydelta_hidden_state_capture> target;
            std::shared_ptr<const common_flydelta_hidden_state_capture> donor_capture;
            if (!daemon_flydelta_capture_from_reference(
                    provider, donor->target_capture_ref, target, error) ||
                    !daemon_flydelta_capture_from_reference(
                    provider, donor->donor_capture_ref, donor_capture, error)) return false;
            std::vector<float> target_values;
            std::vector<float> donor_values;
            if (!daemon_flydelta_capture_layer(*target, donor->layer_index, target_values, error) ||
                    !daemon_flydelta_capture_layer(*donor_capture, donor->layer_index, donor_values, error)) return false;
            std::vector<common_flydelta_basis_direction> parent_directions;
            if (!daemon_flydelta_parse_directions(
                    *provider, material.parent_direction_ref, parent_directions, error)) return false;
            const auto parent = std::find_if(parent_directions.begin(), parent_directions.end(),
                [&](const auto & value) { return value.layer_index == donor->layer_index; });
            if (parent == parent_directions.end()) {
                error = "FlyDelta augmentation donor layer is absent from the parent surface";
                return false;
            }
            common_flydelta_representation_latent_delta latent;
            if (!common_flydelta_build_residualized_latent_delta(
                    donor->candidate.donor_id, donor->layer_index, donor_values, target_values,
                    {parent->values}, common_flydelta_representation_augmentation_config{}.minimum_residual_norm,
                    latent, error)) return false;
            donor->latent = latent;
            if (!latent.available || !common_flydelta_apply_representation_augmentation(next, latent, error)) {
                next.phase = common_flydelta_representation_augmentation_phase::done;
                next.remaining_budget = 0;
                next.next_action = "retain";
            } else {
                next.best_donor_ref = donor->candidate.donor_id;
                next.phase = common_flydelta_representation_augmentation_phase::run_controls;
                next.next_action = "run_representation_augmentation";
            }
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            result = seed_result();
            return error.empty();
        }
        case common_flydelta_representation_augmentation_phase::run_controls: {
            auto donor = std::find_if(material.donors.begin(), material.donors.end(),
                [&](const auto & value) { return value.candidate.donor_id == next.best_donor_ref; });
            if (donor == material.donors.end() || !donor->latent.available) {
                error = "FlyDelta augmentation controls have no persisted latent delta";
                return false;
            }
            std::vector<common_flydelta_basis_direction> parent_directions;
            if (!daemon_flydelta_parse_directions(
                    *provider, material.parent_direction_ref, parent_directions, error)) return false;
            const auto parent = std::find_if(parent_directions.begin(), parent_directions.end(),
                [&](const auto & value) { return value.layer_index == donor->latent.layer_index; });
            if (parent == parent_directions.end()) {
                error = "FlyDelta augmentation controls lack the parent direction";
                return false;
            }
            common_flydelta_direction_candidate parent_candidate;
            parent_candidate.layer_index = parent->layer_index;
            parent_candidate.values = parent->values;
            parent_candidate.origin = "runtime_augmentation_parent";
            parent_candidate.experimental_only = true;
            common_flydelta_direction_candidate latent_candidate;
            latent_candidate.layer_index = donor->latent.layer_index;
            latent_candidate.values = donor->latent.values;
            latent_candidate.origin = "runtime_augmentation_residual";
            latent_candidate.experimental_only = true;
            if (!common_flydelta_direction_candidate_validate(
                    parent_candidate, provider->model_n_embd, error) ||
                    !common_flydelta_direction_candidate_validate(
                    latent_candidate, provider->model_n_embd, error)) return false;
            common_flydelta_low_rank_basis basis;
            if (!common_flydelta_build_low_rank_basis(
                    provider->model_n_embd, 2, {parent_candidate, latent_candidate}, basis, error)) return false;
            common_flydelta_experiment_fixture fixture;
            if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
            std::vector<std::vector<float>> coefficients;
            if (!common_flydelta_propose_representation_augmentation_controls(
                    common_flydelta_representation_augmentation_config{}, coefficients, error)) return false;
            std::vector<common_flydelta_counterfactual_trial> baseline_trials;
            std::vector<common_flydelta_decision_margin> baseline_margins;
            std::vector<common_flydelta_representation_diagnostics> baseline_geometry;
            std::vector<bool> baseline_geometry_available;
            if (!daemon_flydelta_run_coefficient_arm_batch(
                    provider, job, fixture, basis, {{0.0f, 0.0f}}, false,
                    baseline_trials, baseline_margins, baseline_geometry,
                    baseline_geometry_available, error, false, "augmentation-diagnostics")) return false;
            std::vector<common_flydelta_counterfactual_trial> trials;
            std::vector<common_flydelta_decision_margin> margins;
            std::vector<common_flydelta_representation_diagnostics> geometries;
            std::vector<bool> geometry_available;
            if (!daemon_flydelta_run_coefficient_arm_batch(
                    provider, job, fixture, basis, coefficients, true,
                    trials, margins, geometries, geometry_available, error,
                    false, "augmentation-diagnostics")) return false;
            common_flydelta_search_pipeline_direction_result direction;
            direction.direction.kind = common_flydelta_direction_kind::raw_repair;
            direction.direction.layer_index = basis.layer_index;
            direction.direction.values = basis.vectors.front();
            direction.direction.origin = "runtime_representation_augmentation";
            direction.direction.extraction_id = donor->candidate.donor_id;
            direction.direction.source_samples = 2;
            direction.direction.retained_samples = 2;
            direction.direction.experimental_only = true;
            bool positive = false;
            float best_gain = next.best_margin_gain;
            size_t best_index = 0;
            for (size_t index = 0; index < trials.size(); ++index) {
                auto geometry = geometries[index];
                if (geometry_available[index]) {
                    geometry.layer_index = static_cast<uint32_t>(basis.layer_index);
                    geometry.schema_version = 1;
                }
                common_flydelta_intervention_region_trial trial;
                trial.candidate.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
                trial.candidate.anchor_layer_index = static_cast<uint32_t>(basis.layer_index);
                trial.candidate.total_scale = 1.0f;
                trial.candidate.per_layer_scale = trial.candidate.total_scale;
                trial.requested_total_scale = trial.candidate.total_scale;
                trial.executed_total_scale = trial.candidate.total_scale;
                trial.outcome = trials[index].passed
                    ? common_flydelta_counterfactual_outcome::helped
                    : common_flydelta_counterfactual_outcome::unknown;
                trial.quality_delta = trials[index].quality;
                trial.margin = margins[index];
                trial.margin_comparison.available = baseline_margins.front().available && margins[index].available;
                trial.margin_comparison.baseline = baseline_margins.front();
                trial.margin_comparison.candidate = margins[index];
                trial.executed = trials[index].executed;
                trial.verifier_known = trials[index].verifier_known;
                trial.geometry_available = geometry_available[index];
                trial.geometry = geometry;
                trial.safe_to_continue = !trial.geometry_available ||
                    (geometry.leakage <= 1.0f && geometry.shift_norm <= 1.0f);
                trial.search_score = trial.margin_comparison.available
                    ? trial.margin_comparison.normalized_delta() : 0.0f;
                trial.promising = trial.safe_to_continue && trial.search_score > 0.0f;
                if (trial.promising && (!positive || trial.search_score > best_gain)) {
                    positive = true;
                    best_gain = trial.search_score;
                    best_index = index;
                }
                if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
                direction.region_trials.push_back(std::move(trial));
            }
            // Diagnostics establish the frontier using compact geometry and
            // teacher-forced margin only.  Spend generation/verifier budget
            // on the selected control after ranking, through the same batch
            // host and the same immutable arm identity contract.
            if (positive) {
                std::vector<common_flydelta_counterfactual_trial> frontier_trials;
                std::vector<common_flydelta_decision_margin> frontier_margins;
                std::vector<common_flydelta_representation_diagnostics> frontier_geometry;
                std::vector<bool> frontier_geometry_available;
                if (!daemon_flydelta_run_coefficient_arm_batch(
                        provider, job, fixture, basis, {coefficients[best_index]}, true,
                        frontier_trials, frontier_margins, frontier_geometry,
                        frontier_geometry_available, error, true, "augmentation-frontier")) {
                    return false;
                }
                if (frontier_trials.size() != 1 || frontier_margins.size() != 1 ||
                        frontier_geometry.size() != 1 || frontier_geometry_available.size() != 1) {
                    error = "FlyDelta augmentation frontier returned an incomplete arm";
                    return false;
                }
                auto & selected_trial = direction.region_trials[best_index];
                selected_trial.outcome = frontier_trials.front().passed
                    ? common_flydelta_counterfactual_outcome::helped
                    : frontier_trials.front().verifier_known
                        ? frontier_trials.front().quality > 0.0f
                            ? common_flydelta_counterfactual_outcome::neutral
                            : common_flydelta_counterfactual_outcome::harmed
                        : common_flydelta_counterfactual_outcome::unknown;
                selected_trial.quality_delta = frontier_trials.front().quality;
                selected_trial.margin = frontier_margins.front();
                selected_trial.executed = frontier_trials.front().executed;
                selected_trial.verifier_known = frontier_trials.front().verifier_known;
                selected_trial.geometry_available = frontier_geometry_available.front();
                selected_trial.geometry = frontier_geometry.front();
                selected_trial.evidence_ref = frontier_trials.front().evidence_ref;
                selected_trial.safe_to_continue = selected_trial.outcome !=
                    common_flydelta_counterfactual_outcome::harmed &&
                    selected_trial.safe_to_continue;
                selected_trial.margin_comparison.candidate = frontier_margins.front();
                selected_trial.search_score = selected_trial.margin_comparison.available
                    ? selected_trial.margin_comparison.normalized_delta() : selected_trial.search_score;
                selected_trial.promising = selected_trial.safe_to_continue &&
                    selected_trial.search_score > 0.0f;
                if (!common_flydelta_intervention_region_trial_validate(selected_trial, error)) {
                    return false;
                }
            }
            result = {};
            result.directions.push_back(std::move(direction));
            result.search_status = positive
                ? common_flydelta_search_status::candidate_available
                : common_flydelta_search_status::no_useful_utility;
            next.best_margin_gain = positive ? best_gain : next.best_margin_gain;
            if (positive) {
                next.phase = common_flydelta_representation_augmentation_phase::localize_surface;
                next.next_action = "recenter_augmented_surface";
                next.remaining_budget = next.remaining_budget > coefficients.size()
                    ? next.remaining_budget - coefficients.size() : 1;
                // The selected control is a normal immutable direction resource;
                // the next slice reuses Whirlpool on that localized surface.
                const auto selected = coefficients[best_index];
                if (!daemon_flydelta_register_composed_direction(
                        provider, material.parent_direction_ref, basis, selected,
                        material.selected_control_ref, error)) return false;
            } else {
                next.phase = common_flydelta_representation_augmentation_phase::done;
                next.remaining_budget = 0;
                next.next_action = "retain";
            }
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            return true;
        }
        case common_flydelta_representation_augmentation_phase::localize_surface: {
            if (material.selected_control_ref.empty()) {
                error = "FlyDelta augmentation recenter has no selected control surface";
                return false;
            }
            auto localized_job = job;
            localized_job.seed.candidate_ref = material.selected_control_ref;
            if (!daemon_flydelta_run_search_pipeline(
                    provider, localized_job, pipeline_config, result, error)) return false;
            const bool useful = result.selection.selected && result.selection.score > 0.0f;
            next.phase = common_flydelta_representation_augmentation_phase::done;
            next.remaining_budget = 0;
            next.next_action = useful ? "allow_tfo_lite" : "retain";
            return true;
        }
        case common_flydelta_representation_augmentation_phase::full_generation:
        case common_flydelta_representation_augmentation_phase::verify:
            next.phase = common_flydelta_representation_augmentation_phase::done;
            next.remaining_budget = 0;
            next.next_action = "retain";
            result = seed_result();
            return error.empty();
        case common_flydelta_representation_augmentation_phase::done:
            next.next_action = "retain";
            result = seed_result();
            return error.empty();
    }
    error = "FlyDelta representation augmentation phase is unsupported by the daemon binding";
    return false;
}


} // namespace agent_daemon_flydelta_internal
