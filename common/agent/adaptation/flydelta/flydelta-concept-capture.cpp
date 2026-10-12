#include "agent/adaptation/flydelta/flydelta-concept-capture.h"

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::ordered_json;

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}
} // namespace

bool common_flydelta_concept_capture_plan_validate(
        const common_flydelta_concept_capture_plan & plan,
        std::string & error) {
    error.clear();
    if (plan.schema_version != 1 || !bounded(plan.id) || !bounded(plan.group_ref) ||
            !bounded(plan.relation_ref) || !bounded(plan.baseline_ref) ||
            !bounded(plan.conditioned_ref) || !bounded(plan.control_ref) ||
            !bounded(plan.semantic_anchor) || plan.layer_index < 0 ||
            !common_flydelta_teaching_material_identity_validate(plan.identity, error)) {
        if (error.empty()) error = "FlyDelta concept capture plan is incomplete";
        return false;
    }
    return true;
}

std::string common_flydelta_concept_capture_plan_to_json(
        const common_flydelta_concept_capture_plan & plan) {
    return json{
        {"schema_version", plan.schema_version},
        {"id", plan.id},
        {"group_ref", plan.group_ref},
        {"relation_ref", plan.relation_ref},
        {"baseline_ref", plan.baseline_ref},
        {"conditioned_ref", plan.conditioned_ref},
        {"control_ref", plan.control_ref},
        {"semantic_anchor", plan.semantic_anchor},
        {"layer_index", plan.layer_index},
        {"identity", {
            {"schema_version", plan.identity.schema_version},
            {"model_profile_fingerprint", plan.identity.model_profile_fingerprint},
            {"tokenizer_fingerprint", plan.identity.tokenizer_fingerprint},
            {"template_fingerprint", plan.identity.template_fingerprint},
            {"capture_layout_revision", plan.identity.capture_layout_revision},
            {"execution_context_fingerprint", plan.identity.execution_context_fingerprint},
            {"scope_fingerprint", plan.identity.scope_fingerprint},
            {"verifier_revision", plan.identity.verifier_revision},
        }},
    }.dump();
}

bool common_flydelta_concept_capture_plan_from_json(
        const std::string & text,
        common_flydelta_concept_capture_plan & plan,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        plan = {};
        plan.schema_version = value.value("schema_version", 0);
        plan.id = value.value("id", "");
        plan.group_ref = value.value("group_ref", "");
        plan.relation_ref = value.value("relation_ref", "");
        plan.baseline_ref = value.value("baseline_ref", "");
        plan.conditioned_ref = value.value("conditioned_ref", "");
        plan.control_ref = value.value("control_ref", "");
        plan.semantic_anchor = value.value("semantic_anchor", "");
        plan.layer_index = value.value("layer_index", -1);
        const auto identity = value.value("identity", json::object());
        plan.identity.schema_version = identity.value("schema_version", 0);
        plan.identity.model_profile_fingerprint = identity.value("model_profile_fingerprint", "");
        plan.identity.tokenizer_fingerprint = identity.value("tokenizer_fingerprint", "");
        plan.identity.template_fingerprint = identity.value("template_fingerprint", "");
        plan.identity.capture_layout_revision = identity.value("capture_layout_revision", "");
        plan.identity.execution_context_fingerprint = identity.value("execution_context_fingerprint", "");
        plan.identity.scope_fingerprint = identity.value("scope_fingerprint", "");
        plan.identity.verifier_revision = identity.value("verifier_revision", "");
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta concept capture plan JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_concept_capture_plan_validate(plan, error);
}
