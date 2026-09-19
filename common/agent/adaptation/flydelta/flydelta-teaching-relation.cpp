#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"

#include "agent/adaptation/adaptation-evidence.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"

#include <algorithm>
#include <cmath>
#include <utility>

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

const char * common_agent_teaching_build_status_name(
        common_agent_teaching_build_status status) {
    switch (status) {
        case common_agent_teaching_build_status::resolved: return "resolved";
        case common_agent_teaching_build_status::out_of_scope: return "out_of_scope";
        case common_agent_teaching_build_status::not_host_verified: return "not_host_verified";
        case common_agent_teaching_build_status::not_reusable: return "not_reusable";
        case common_agent_teaching_build_status::missing_behavior_key: return "missing_behavior_key";
        case common_agent_teaching_build_status::missing_verifier: return "missing_verifier";
        case common_agent_teaching_build_status::no_contrast: return "no_contrast";
        case common_agent_teaching_build_status::incompatible_control: return "incompatible_control";
        case common_agent_teaching_build_status::insufficient_evidence: return "insufficient_evidence";
    }
    return "unknown";
}

bool common_flydelta_teaching_relation_validate(
        const common_flydelta_teaching_relation & relation,
        std::string & error) {
    error.clear();
    const bool complete = relation.status == common_flydelta_teaching_relation_status::resolved;
    if (relation.schema_version != 1 || !bounded(relation.id) ||
            !bounded(relation.teaching_key) ||
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

common_agent_teaching_build_result
common_agent_build_procedure_teaching_relation(
        const common_agent_procedure_teaching_request & request) {
    common_agent_teaching_build_result result;
    auto reject = [&](common_agent_teaching_build_status status, const char * diagnostic) {
        result.status = status;
        result.diagnostic = diagnostic;
        result.relation.reset();
        return result;
    };

    if (request.schema_version != 1) {
        return reject(common_agent_teaching_build_status::insufficient_evidence,
            "procedure/blueprint request schema is unsupported");
    }
    if (!request.host_scope_admitted) {
        return reject(common_agent_teaching_build_status::out_of_scope,
            "procedure/blueprint relation was not admitted in the host scope");
    }
    if (!request.host_verified) {
        return reject(common_agent_teaching_build_status::not_host_verified,
            "procedure/blueprint relation is not host verified");
    }
    if (!request.reusable) {
        return reject(common_agent_teaching_build_status::not_reusable,
            "procedure/blueprint is not admitted as reusable behavior");
    }
    if (!bounded(request.behavior_key)) {
        return reject(common_agent_teaching_build_status::missing_behavior_key,
            "procedure/blueprint relation has no behavior key");
    }
    if (!bounded(request.verifier_ref)) {
        return reject(common_agent_teaching_build_status::missing_verifier,
            "procedure/blueprint relation has no verifier reference");
    }
    if (!bounded(request.relation_id) || !bounded(request.evidence_ref) ||
            request.scope.namespace_id.empty() || request.scope.session_id.empty() ||
            !bounded(request.task_fingerprint) ||
            (!bounded(request.procedure_ref) && !bounded(request.blueprint_ref))) {
        return reject(common_agent_teaching_build_status::insufficient_evidence,
            "procedure/blueprint relation is missing immutable host references");
    }
    if (!bounded(request.baseline_ref) || !bounded(request.conditioned_ref) ||
            request.baseline_ref == request.conditioned_ref) {
        return reject(common_agent_teaching_build_status::no_contrast,
            "procedure/blueprint relation has no explicit behavioral contrast");
    }
    if (request.require_control &&
            (!request.control_ref || !bounded(*request.control_ref))) {
        return reject(common_agent_teaching_build_status::incompatible_control,
            "procedure/blueprint relation requires a compatible control reference");
    }
    if (!std::isfinite(request.confidence) || request.confidence < 0.0f ||
            request.confidence > 1.0f) {
        return reject(common_agent_teaching_build_status::insufficient_evidence,
            "procedure/blueprint relation confidence is out of bounds");
    }

    common_flydelta_teaching_relation relation;
    relation.id = request.relation_id;
    relation.teaching_key = bounded(request.teaching_key)
        ? request.teaching_key : request.behavior_key;
    relation.source = common_adaptation_evidence_source::procedure_blueprint;
    relation.behavior_key = request.behavior_key;
    relation.scope = request.scope;
    relation.task_fingerprint = request.task_fingerprint;
    relation.baseline_ref = request.baseline_ref;
    relation.conditioned_ref = request.conditioned_ref;
    relation.control_ref = request.control_ref.value_or("");
    relation.verifier_ref = request.verifier_ref;
    relation.evidence_ref = request.evidence_ref;
    relation.procedure_ref = request.procedure_ref;
    relation.blueprint_ref = request.blueprint_ref;
    relation.status = common_flydelta_teaching_relation_status::resolved;
    relation.baseline_origin = request.baseline_origin;
    relation.conditioned_origin = request.conditioned_origin;
    relation.control_origin = request.control_ref
        ? request.control_origin : common_flydelta_teaching_origin::none;
    relation.confidence = request.confidence;
    relation.host_approved = true;

    std::string error;
    if (!common_flydelta_teaching_relation_validate(relation, error)) {
        return reject(common_agent_teaching_build_status::insufficient_evidence,
            error.c_str());
    }
    result.status = common_agent_teaching_build_status::resolved;
    result.relation = std::move(relation);
    result.diagnostic = "resolved host procedure/blueprint teaching relation";
    return result;
}

bool common_agent_procedure_teaching_relation_to_evidence_relation(
        const common_flydelta_teaching_relation & teaching_relation,
        const common_learning_transaction & transaction,
        common_adaptation_evidence_relation & relation,
        std::string & error) {
    error.clear();
    if (teaching_relation.source != common_adaptation_evidence_source::procedure_blueprint ||
            teaching_relation.status != common_flydelta_teaching_relation_status::resolved ||
            !teaching_relation.host_approved || transaction.id.empty()) {
        error = "procedure/blueprint teaching relation is not resolved for evidence";
        return false;
    }
    if (!common_flydelta_teaching_relation_validate(teaching_relation, error)) return false;
    relation = {};
    relation.id = teaching_relation.id;
    relation.source = common_adaptation_evidence_source::procedure_blueprint;
    relation.scope = teaching_relation.scope;
    relation.behavior_key = teaching_relation.behavior_key;
    relation.task_fingerprint = teaching_relation.task_fingerprint;
    relation.baseline_ref = teaching_relation.baseline_ref;
    relation.candidate_ref = teaching_relation.conditioned_ref;
    relation.verifier_ref = teaching_relation.verifier_ref;
    relation.transaction_ids = { transaction.id };
    relation.cause = transaction.observation.cause;
    relation.host_verified = true;
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
    teaching_relation.teaching_key = relation.behavior_key;
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
    relation.teaching_key = evidence.behavior_key;
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
