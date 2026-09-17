#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation-state-store.h"
#include "agent/adaptation/lifecycle-store.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_representation_donor_candidate donor() {
    common_flydelta_representation_donor_candidate value;
    value.donor_id = "flydelta://donor/repair-1";
    value.source_type = "host_repair";
    value.source_ref = "evidence://repair-1";
    value.behavior_key = "tool_choice/data_query";
    value.context_payload_ref = "context://repair-1";
    value.qualification_policy = "host-certified-contextual";
    value.provenance = "fixture://tool-repair-1";
    return value;
}

static common_flydelta_representation_augmentation_state state() {
    common_flydelta_representation_augmentation_state value;
    value.state_ref = "flydelta://state/representation-augmentation/1";
    value.model_fingerprint = "sha256:model";
    value.behavior_key = "tool_choice/data_query";
    value.direction_family_id = "tool-choice/data-query";
    value.parent_surface_revision = 1;
    value.parent_search_state_ref = "flydelta://state/search/1";
    value.parent_evidence_rank = 1.0f;
    value.evidence_rank = 1.0f;
    value.search_rank = 1;
    value.selected_region = {24};
    value.target_fixture_ref = "fixture://target-1";
    value.donor_candidate_refs = {"flydelta://donor/repair-1"};
    value.remaining_budget = 8;
    value.surface_revision = 1;
    return value;
}

int main() {
    std::string error;
    const auto candidate = donor();
    CHECK(common_flydelta_representation_donor_candidate_validate(candidate, error));
    auto invalid_candidate = candidate;
    invalid_candidate.promotable = true;
    CHECK(!common_flydelta_representation_donor_candidate_validate(invalid_candidate, error));

    common_flydelta_representation_donor_qualification observation;
    observation.donor_id = candidate.donor_id;
    observation.host_verified = true;
    observation.safe_to_continue = true;
    observation.decision_margin_available = true;
    observation.margin_gain = 0.12f;
    common_flydelta_representation_donor_qualification qualified;
    CHECK(common_flydelta_qualify_representation_donor(
        candidate, observation, 0.05f, 1.0f, 1.0f, qualified, error));
    CHECK(qualified.search_qualified && qualified.reason == "margin_guided_experimental_search");

    common_flydelta_representation_latent_delta latent;
    CHECK(common_flydelta_build_residualized_latent_delta(
        candidate.donor_id, 24, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.0f},
        {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, 0.01f, latent, error));
    CHECK(latent.available && latent.residualized && latent.values.size() == 3);
    CHECK(std::fabs(latent.values[0]) < 0.001f && std::fabs(latent.values[1]) < 0.001f &&
        latent.values[2] > 0.99f);
    CHECK(latent.removed_norm >= 0.0f && latent.residual_norm > 0.0f);

    common_flydelta_representation_augmentation_config config;
    CHECK(common_flydelta_representation_augmentation_config_validate(config, error));
    std::vector<std::vector<float>> controls;
    CHECK(common_flydelta_propose_representation_augmentation_controls(config, controls, error));
    const std::vector<float> first_control = {1.0f, 0.0f};
    const std::vector<float> second_control = {0.0f, 1.0f};
    CHECK(controls.size() == 4 && controls[0] == first_control &&
        controls[1] == second_control);
    CHECK(std::fabs(controls[2][0] - controls[2][1]) < 0.001f &&
        std::fabs(controls[3][0] + controls[3][1]) < 0.001f);

    auto augmentation = state();
    CHECK(common_flydelta_representation_augmentation_state_validate(augmentation, config, error));
    CHECK(common_flydelta_apply_representation_augmentation(augmentation, latent, error));
    CHECK(augmentation.evidence_rank == 1.0f && augmentation.search_rank == 2 &&
        augmentation.surface_revision == 2 && augmentation.remaining_budget == 7 &&
        augmentation.phase == common_flydelta_representation_augmentation_phase::run_controls);
    CHECK(augmentation.evaluated_donor_refs.size() == 1);

    common_flydelta_representation_augmentation_action action;
    CHECK(common_flydelta_decide_representation_augmentation(
        config, augmentation, {qualified}, false, false, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::retain);
    augmentation.phase = common_flydelta_representation_augmentation_phase::run_controls;
    CHECK(common_flydelta_decide_representation_augmentation(
        config, augmentation, {qualified}, true, false, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::recenter_augmented_surface);
    augmentation.phase = common_flydelta_representation_augmentation_phase::localize_surface;
    CHECK(common_flydelta_decide_representation_augmentation(
        config, augmentation, {qualified}, true, true, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::allow_tfo_lite);

    const auto encoded = common_flydelta_representation_augmentation_state_to_json(augmentation);
    common_flydelta_representation_augmentation_state decoded;
    CHECK(common_flydelta_representation_augmentation_state_from_json(
        encoded, decoded, config, error));
    CHECK(decoded.evidence_rank == augmentation.evidence_rank &&
        decoded.search_rank == augmentation.search_rank &&
        decoded.selected_region == augmentation.selected_region);

    common_learning_in_memory_lifecycle_store lifecycle;
    common_flydelta_evaluator_callbacks callbacks;
    common_flydelta_lifecycle_event_context lifecycle_context;
    lifecycle_context.source_id = "test-agent";
    lifecycle_context.created_at = "2026-09-17T00:00:00Z";
    CHECK(common_flydelta_configure_representation_augmentation_lifecycle_callbacks(
        lifecycle, lifecycle_context, callbacks, error));
    std::string persisted_ref;
    CHECK(callbacks.persist_representation_augmentation_state(
        augmentation, persisted_ref, error));
    CHECK(persisted_ref == augmentation.state_ref);
    common_flydelta_representation_augmentation_state resolved;
    CHECK(callbacks.resolve_representation_augmentation_state(
        persisted_ref, resolved, error));
    CHECK(resolved.state_ref == persisted_ref && resolved.search_rank == augmentation.search_rank);
    decoded.evidence_rank = 2.0f;
    CHECK(!common_flydelta_representation_augmentation_state_validate(decoded, config, error));
    return 0;
}
