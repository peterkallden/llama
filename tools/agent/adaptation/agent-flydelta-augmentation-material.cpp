#include "agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

json daemon_flydelta_augmentation_material_json(
        const daemon_flydelta_augmentation_material & material) {
    json donors = json::array();
    for (const auto & donor : material.donors) {
        donors.push_back({
            {"candidate", {
                {"schema_version", donor.candidate.schema_version},
                {"donor_id", donor.candidate.donor_id},
                {"source_type", donor.candidate.source_type},
                {"source_ref", donor.candidate.source_ref},
                {"behavior_key", donor.candidate.behavior_key},
                {"context_payload_ref", donor.candidate.context_payload_ref},
                {"qualification_policy", donor.candidate.qualification_policy},
                {"provenance", donor.candidate.provenance},
                {"promotable", donor.candidate.promotable},
            }},
            {"target_context_ref", donor.target_context_ref},
            {"donor_context_ref", donor.donor_context_ref},
            {"fixture_ref", donor.fixture_ref},
            {"layer_index", donor.layer_index},
            {"target_capture_ref", donor.target_capture_ref},
            {"donor_capture_ref", donor.donor_capture_ref},
            {"qualification", {
                {"donor_id", donor.qualification.donor_id},
                {"host_evaluated", donor.qualification.host_evaluated},
                {"verifier_known", donor.qualification.verifier_known},
                {"safe_to_continue", donor.qualification.safe_to_continue},
                {"host_outcome", common_flydelta_counterfactual_outcome_name(
                    donor.qualification.host_outcome)},
                {"decision_margin_available", donor.qualification.decision_margin_available},
                {"margin_gain", donor.qualification.margin_gain},
                {"geometry_available", donor.qualification.geometry_available},
                {"geometry", {
                    {"schema_version", donor.qualification.geometry.schema_version},
                    {"layer_index", donor.qualification.geometry.layer_index},
                    {"cosine", donor.qualification.geometry.cosine},
                    {"progress", donor.qualification.geometry.progress},
                    {"leakage", donor.qualification.geometry.leakage},
                    {"shift_norm", donor.qualification.geometry.shift_norm},
                }},
                {"search_qualified", donor.qualification.search_qualified},
                {"reason", donor.qualification.reason},
            }},
            {"latent", {
                {"schema_version", donor.latent.schema_version},
                {"donor_id", donor.latent.donor_id},
                {"layer_index", donor.latent.layer_index},
                {"values", donor.latent.values},
                {"residualized", donor.latent.residualized},
                {"available", donor.latent.available},
                {"raw_norm", donor.latent.raw_norm},
                {"residual_norm", donor.latent.residual_norm},
                {"removed_norm", donor.latent.removed_norm},
            }},
        });
    }
    return {
        {"kind", "flydelta_representation_augmentation_material"},
        {"schema_version", 1},
        {"state_ref", material.state_ref},
        {"parent_direction_ref", material.parent_direction_ref},
        {"selected_control_ref", material.selected_control_ref},
        {"donors", std::move(donors)},
    };
}

bool daemon_flydelta_augmentation_material_from_json(
        const json & value,
        daemon_flydelta_augmentation_material & material,
        std::string & error) {
    error.clear();
    try {
        if (value.value("kind", "") != "flydelta_representation_augmentation_material" ||
                value.value("schema_version", 0) != 1) {
            error = "FlyDelta augmentation material kind is invalid";
            return false;
        }
        material = {};
        material.state_ref = value.value("state_ref", "");
        material.parent_direction_ref = value.value("parent_direction_ref", "");
        material.selected_control_ref = value.value("selected_control_ref", "");
        for (const auto & item : value.value("donors", json::array())) {
            daemon_flydelta_augmentation_donor_material donor;
            const auto candidate = item.value("candidate", json::object());
            donor.candidate.schema_version = candidate.value("schema_version", 0);
            donor.candidate.donor_id = candidate.value("donor_id", "");
            donor.candidate.source_type = candidate.value("source_type", "");
            donor.candidate.source_ref = candidate.value("source_ref", "");
            donor.candidate.behavior_key = candidate.value("behavior_key", "");
            donor.candidate.context_payload_ref = candidate.value("context_payload_ref", "");
            donor.candidate.qualification_policy = candidate.value("qualification_policy", "");
            donor.candidate.provenance = candidate.value("provenance", "");
            donor.candidate.promotable = candidate.value("promotable", false);
            donor.target_context_ref = item.value("target_context_ref", "");
            donor.donor_context_ref = item.value("donor_context_ref", "");
            donor.fixture_ref = item.value("fixture_ref", "");
            donor.layer_index = item.value("layer_index", -1);
            donor.target_capture_ref = item.value("target_capture_ref", "");
            donor.donor_capture_ref = item.value("donor_capture_ref", "");
            const auto qualification = item.value("qualification", json::object());
            donor.qualification.donor_id = qualification.value("donor_id", donor.candidate.donor_id);
            donor.qualification.host_evaluated = qualification.value("host_evaluated", false);
            donor.qualification.verifier_known = qualification.value("verifier_known", false);
            donor.qualification.safe_to_continue = qualification.value("safe_to_continue", false);
            donor.qualification.decision_margin_available =
                qualification.value("decision_margin_available", false);
            donor.qualification.margin_gain = qualification.value("margin_gain", 0.0f);
            const auto outcome = qualification.value("host_outcome", "unknown");
            donor.qualification.host_outcome = outcome == "helped"
                ? common_flydelta_counterfactual_outcome::helped
                : outcome == "neutral" ? common_flydelta_counterfactual_outcome::neutral
                : outcome == "harmed" ? common_flydelta_counterfactual_outcome::harmed
                : common_flydelta_counterfactual_outcome::unknown;
            donor.qualification.search_qualified = qualification.value("search_qualified", false);
            donor.qualification.reason = qualification.value("reason", "");
            const auto geometry = qualification.value("geometry", json::object());
            donor.qualification.geometry_available = qualification.value("geometry_available", false);
            donor.qualification.geometry.schema_version = geometry.value("schema_version", 1);
            donor.qualification.geometry.layer_index = geometry.value("layer_index", 0U);
            donor.qualification.geometry.cosine = geometry.value("cosine", 0.0f);
            donor.qualification.geometry.progress = geometry.value("progress", 0.0f);
            donor.qualification.geometry.leakage = geometry.value("leakage", 0.0f);
            donor.qualification.geometry.shift_norm = geometry.value("shift_norm", 0.0f);
            const auto latent = item.value("latent", json::object());
            donor.latent.schema_version = latent.value("schema_version", 0);
            donor.latent.donor_id = latent.value("donor_id", "");
            donor.latent.layer_index = latent.value("layer_index", -1);
            donor.latent.values = latent.value("values", std::vector<float>{});
            donor.latent.residualized = latent.value("residualized", false);
            donor.latent.available = latent.value("available", false);
            donor.latent.raw_norm = latent.value("raw_norm", 0.0f);
            donor.latent.residual_norm = latent.value("residual_norm", 0.0f);
            donor.latent.removed_norm = latent.value("removed_norm", 0.0f);
            material.donors.push_back(std::move(donor));
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta augmentation material is malformed: ") + exception.what();
        return false;
    }
    return !material.state_ref.empty() && material.state_ref.size() <= 512;
}

} // namespace agent_daemon_flydelta_internal
