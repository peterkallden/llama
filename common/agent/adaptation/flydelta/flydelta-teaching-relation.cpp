#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"

#include "agent/adaptation/adaptation-evidence.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"

#include <algorithm>
#include <cmath>

namespace {

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool source_supported(common_adaptation_evidence_source source) {
    return common_adaptation_evidence_source_from_name(
        common_adaptation_evidence_source_name(source)).has_value();
}

} // namespace

const char * common_flydelta_teaching_relation_status_name(
        common_flydelta_teaching_relation_status status) {
    switch (status) {
        case common_flydelta_teaching_relation_status::resolved: return "resolved";
        case common_flydelta_teaching_relation_status::ambiguous: return "ambiguous";
        case common_flydelta_teaching_relation_status::insufficient_evidence: return "insufficient_evidence";
        case common_flydelta_teaching_relation_status::no_contrast: return "no_contrast";
        case common_flydelta_teaching_relation_status::not_reusable: return "not_reusable";
        case common_flydelta_teaching_relation_status::unsupported_behavior: return "unsupported_behavior";
    }
    return "unknown";
}

const char * common_flydelta_teaching_origin_name(
        common_flydelta_teaching_origin origin) {
    switch (origin) {
        case common_flydelta_teaching_origin::none: return "none";
        case common_flydelta_teaching_origin::observed: return "observed";
        case common_flydelta_teaching_origin::host_derived: return "host_derived";
        case common_flydelta_teaching_origin::user_supplied: return "user_supplied";
        case common_flydelta_teaching_origin::host_counterfactual: return "host_counterfactual";
    }
    return "unknown";
}

bool common_flydelta_teaching_relation_validate(
        const common_flydelta_teaching_relation & relation,
        std::string & error) {
    error.clear();
    const bool complete = relation.status == common_flydelta_teaching_relation_status::resolved;
    if (relation.schema_version != 1 || !bounded(relation.id) ||
            !source_supported(relation.source) || !bounded(relation.behavior_key) ||
            relation.scope.namespace_id.empty() || relation.scope.session_id.empty() ||
            !bounded(relation.task_fingerprint) || !bounded(relation.baseline_ref) ||
            !bounded(relation.conditioned_ref) || relation.baseline_ref == relation.conditioned_ref ||
            (!relation.control_ref.empty() && !bounded(relation.control_ref)) ||
            !bounded(relation.verifier_ref) || !bounded(relation.evidence_ref) ||
            !std::isfinite(relation.confidence) || relation.confidence < 0.0f ||
            relation.confidence > 1.0f || (complete && !relation.host_approved) ||
            (complete && relation.baseline_origin == common_flydelta_teaching_origin::none) ||
            (complete && relation.conditioned_origin == common_flydelta_teaching_origin::none) ||
            (!relation.control_ref.empty() && relation.control_origin == common_flydelta_teaching_origin::none)) {
        error = "FlyDelta teaching relation identity, provenance or bounds are invalid";
        return false;
    }
    return true;
}

bool common_flydelta_teaching_relation_from_host_relation(
        const common_adaptation_evidence_relation & relation,
        const std::string & evidence_ref,
        const std::string & control_ref,
        common_flydelta_teaching_relation_status status,
        common_flydelta_teaching_origin baseline_origin,
        common_flydelta_teaching_origin conditioned_origin,
        common_flydelta_teaching_origin control_origin,
        float confidence,
        bool host_approved,
        common_flydelta_teaching_relation & teaching_relation,
        std::string & error) {
    error.clear();
    if (!bounded(relation.id) || !bounded(evidence_ref) ||
            (relation.source == common_adaptation_evidence_source::tool_repair &&
                relation.baseline_ref.empty()) ||
            relation.baseline_ref == relation.candidate_ref) {
        error = "host teaching relation lacks a contrast or evidence reference";
        return false;
    }
    teaching_relation = {};
    teaching_relation.id = relation.id;
    teaching_relation.source = relation.source;
    teaching_relation.behavior_key = relation.behavior_key;
    teaching_relation.scope = relation.scope;
    teaching_relation.task_fingerprint = relation.task_fingerprint;
    teaching_relation.baseline_ref = relation.baseline_ref;
    teaching_relation.conditioned_ref = relation.candidate_ref;
    teaching_relation.control_ref = control_ref;
    teaching_relation.verifier_ref = relation.verifier_ref;
    teaching_relation.evidence_ref = evidence_ref;
    teaching_relation.status = status;
    teaching_relation.baseline_origin = baseline_origin;
    teaching_relation.conditioned_origin = conditioned_origin;
    teaching_relation.control_origin = control_origin;
    teaching_relation.confidence = confidence;
    teaching_relation.host_approved = host_approved;
    return common_flydelta_teaching_relation_validate(teaching_relation, error);
}

bool common_flydelta_teaching_relation_from_evidence(
        const common_adaptation_evidence & evidence,
        common_flydelta_teaching_relation & relation,
        std::string & error) {
    error.clear();
    if (!common_adaptation_evidence_validate(evidence, 64, error)) return false;
    if (!evidence.host_verified) {
        error = "teaching relation requires host-verified evidence";
        return false;
    }
    relation = {};
    relation.id = evidence.id + "/teaching";
    relation.source = evidence.source;
    relation.behavior_key = evidence.behavior_key;
    relation.scope = evidence.scope;
    relation.task_fingerprint = evidence.task_fingerprint;
    relation.baseline_ref = evidence.baseline_ref;
    relation.conditioned_ref = evidence.candidate_ref;
    relation.verifier_ref = evidence.verifier_ref;
    relation.evidence_ref = evidence.id;
    relation.status = common_flydelta_teaching_relation_status::resolved;
    relation.baseline_origin = common_flydelta_teaching_origin::observed;
    relation.conditioned_origin = common_flydelta_teaching_origin::observed;
    relation.confidence = 1.0f;
    relation.host_approved = true;
    return common_flydelta_teaching_relation_validate(relation, error);
}

bool common_flydelta_procedure_blueprint_teaching_relation_from_evidence(
        const common_adaptation_evidence & evidence,
        common_flydelta_teaching_relation & relation,
        std::string & error) {
    if (evidence.source != common_adaptation_evidence_source::procedure_blueprint) {
        error = "procedure/blueprint teaching adapter requires procedure_blueprint evidence";
        return false;
    }
    return common_flydelta_teaching_relation_from_evidence(evidence, relation, error);
}

bool common_flydelta_teaching_relation_to_transition(
        const common_flydelta_teaching_relation & relation,
        const common_adaptation_evidence & evidence,
        const std::string & baseline_transaction_id,
        const std::string & conditioned_transaction_id,
        common_flydelta_behavior_transition & transition,
        std::string & error) {
    error.clear();
    if (!common_flydelta_teaching_relation_validate(relation, error) ||
            relation.status != common_flydelta_teaching_relation_status::resolved ||
            !relation.host_approved || relation.id != evidence.id + "/teaching" ||
            relation.source != evidence.source || relation.behavior_key != evidence.behavior_key ||
            relation.baseline_ref != evidence.baseline_ref ||
            relation.conditioned_ref != evidence.candidate_ref ||
            relation.verifier_ref != evidence.verifier_ref ||
            relation.scope.namespace_id != evidence.scope.namespace_id ||
            relation.scope.session_id != evidence.scope.session_id ||
            std::find(evidence.transaction_ids.begin(), evidence.transaction_ids.end(),
                baseline_transaction_id) == evidence.transaction_ids.end() ||
            std::find(evidence.transaction_ids.begin(), evidence.transaction_ids.end(),
                conditioned_transaction_id) == evidence.transaction_ids.end()) {
        if (error.empty()) error = "resolved teaching relation is not aligned with host evidence";
        return false;
    }
    return common_flydelta_behavior_transition_from_evidence(
        evidence, baseline_transaction_id, conditioned_transaction_id, transition, error);
}
