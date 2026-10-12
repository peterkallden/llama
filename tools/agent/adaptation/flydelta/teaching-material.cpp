#include "../../daemon/agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_parse_teaching_origin(
        const std::string & value,
        common_flydelta_teaching_origin & origin) {
    if (value == "none") origin = common_flydelta_teaching_origin::none;
    else if (value == "observed") origin = common_flydelta_teaching_origin::observed;
    else if (value == "host_derived") origin = common_flydelta_teaching_origin::host_derived;
    else if (value == "user_supplied") origin = common_flydelta_teaching_origin::user_supplied;
    else if (value == "host_counterfactual") origin = common_flydelta_teaching_origin::host_counterfactual;
    else return false;
    return true;
}

json daemon_flydelta_teaching_relation_json(
        const common_flydelta_teaching_relation & relation) {
    return {
        {"kind", "flydelta_teaching_relation"},
        {"schema_version", relation.schema_version},
        {"id", relation.id},
        {"teaching_key", relation.teaching_key},
        {"source", common_adaptation_evidence_source_name(relation.source)},
        {"behavior_key", relation.behavior_key},
        {"scope", {
            {"namespace_id", relation.scope.namespace_id},
            {"project_id", relation.scope.project_id},
            {"session_id", relation.scope.session_id},
            {"turn_id", relation.scope.turn_id},
        }},
        {"task_fingerprint", relation.task_fingerprint},
        {"baseline_ref", relation.baseline_ref},
        {"conditioned_ref", relation.conditioned_ref},
        {"control_ref", relation.control_ref},
        {"verifier_ref", relation.verifier_ref},
        {"evidence_ref", relation.evidence_ref},
        {"contrast_ref", relation.contrast_ref},
        {"procedure_ref", relation.procedure_ref},
        {"blueprint_ref", relation.blueprint_ref},
        {"status", common_flydelta_teaching_relation_status_name(relation.status)},
        {"baseline_origin", common_flydelta_teaching_origin_name(relation.baseline_origin)},
        {"conditioned_origin", common_flydelta_teaching_origin_name(relation.conditioned_origin)},
        {"control_origin", common_flydelta_teaching_origin_name(relation.control_origin)},
        {"confidence", relation.confidence},
        {"host_approved", relation.host_approved},
        // The relation describes WHAT should be taught. WHERE is supplied by
        // the persisted FlyDelta search surface when a capture plan is made.
        {"semantic_anchor", relation.task_fingerprint},
    };
}

bool daemon_flydelta_persist_teaching_relation(
        agent_resource_store * resources,
        const agent_resource_read_authority & authority,
        const common_flydelta_teaching_relation & relation,
        common_flydelta_teaching_relation & persisted_relation,
        std::string & error) {
    error.clear();
    if (resources == nullptr || relation.id.empty()) {
        error = "FlyDelta teaching relation persistence is not configured";
        return false;
    }
    const std::string identity = relation.id + "\n" +
        relation.teaching_key + "\n" + relation.behavior_key;
    agent_resource_put_request request;
    request.name = "flydelta-teaching-relation-" +
        hash_sha256_hex(identity.data(), identity.size()).substr(0, 24) + ".json";
    request.description = "Durable host-verified FlyDelta teaching relation";
    request.mime_type = "application/json";
    request.text = daemon_flydelta_teaching_relation_json(relation).dump();
    request.scope = common_runtime_resource_scope::session;
    request.namespace_id = authority.namespace_id;
    request.session_id = authority.session_id;
    request.source_provider = "flydelta";
    request.source_tool = "teaching-relation";
    request.created_at = std::time(nullptr);
    request.metadata.purpose = "flydelta_teaching_relation";
    request.metadata.content_summary =
        "Durable host-verified teaching relation for concept capture and synthesis.";
    request.metadata.processing_cache_key = identity;
    request.lineage.parent_uri = relation.evidence_ref;
    request.lineage.chunk_count = 1;
    request.lineage.chunk_index = 0;
    request.lineage.derivation = "host-teaching-relation";
    agent_resource_descriptor descriptor;
    if (!resources->put_text(request, descriptor, error) || descriptor.uri.empty()) {
        if (error.empty()) error = "FlyDelta teaching relation resource has no URI";
        return false;
    }
    persisted_relation = relation;
    persisted_relation.id = descriptor.uri;
    return true;
}

bool daemon_flydelta_parse_teaching_relation(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        daemon_flydelta_concept_relation_material & material,
        std::string & error) {
    json parsed;
    if (!daemon_flydelta_read_json_bounded(
            provider, reference, 4U * 1024U * 1024U, parsed, error)) return false;
    const json value = parsed.contains("relation") && parsed["relation"].is_object()
        ? parsed["relation"] : parsed;
    try {
        material = {};
        auto & relation = material.relation;
        relation.schema_version = value.value("schema_version", 0);
        relation.id = value.value("id", reference);
        relation.teaching_key = value.value("teaching_key", "");
        relation.behavior_key = value.value("behavior_key", "");
        relation.task_fingerprint = value.value("task_fingerprint", "");
        relation.baseline_ref = value.value("baseline_ref", "");
        relation.conditioned_ref = value.value("conditioned_ref", "");
        relation.control_ref = value.value("control_ref", "");
        relation.verifier_ref = value.value("verifier_ref", "");
        relation.evidence_ref = value.value("evidence_ref", "");
        relation.contrast_ref = value.value("contrast_ref", "");
        relation.procedure_ref = value.value("procedure_ref", "");
        relation.blueprint_ref = value.value("blueprint_ref", "");
        relation.confidence = value.value("confidence", 0.0f);
        relation.host_approved = value.value("host_approved", false);
        relation.status = common_flydelta_teaching_relation_status::resolved;
        const auto source = common_adaptation_evidence_source_from_name(
            value.value("source", ""));
        if (!source) {
            error = "FlyDelta teaching relation source is invalid";
            return false;
        }
        relation.source = *source;
        const auto scope = value.value("scope", json::object());
        relation.scope.namespace_id = scope.value("namespace_id", "");
        relation.scope.project_id = scope.value("project_id", "");
        relation.scope.session_id = scope.value("session_id", "");
        relation.scope.turn_id = scope.value("turn_id", "");
        if (!daemon_flydelta_parse_teaching_origin(
                value.value("baseline_origin", "none"), relation.baseline_origin) ||
            !daemon_flydelta_parse_teaching_origin(
                value.value("conditioned_origin", "none"), relation.conditioned_origin) ||
            !daemon_flydelta_parse_teaching_origin(
                value.value("control_origin", "none"), relation.control_origin)) {
            error = "FlyDelta teaching relation provenance is invalid";
            return false;
        }
        material.semantic_anchor = value.value("semantic_anchor", relation.task_fingerprint);
        // Kept only as legacy metadata. The capture planner overwrites this
        // from the localized search state and never trusts a relation layer.
        material.layer_index = value.value("layer_index", -1);
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta teaching relation resource is malformed: ") + exception.what();
        return false;
    }
    if (!common_flydelta_teaching_relation_validate(material.relation, error) ||
            material.semantic_anchor.empty() || material.semantic_anchor.size() > 512) {
        if (error.empty()) error = "FlyDelta teaching relation material has no semantic anchor";
        return false;
    }
    return true;
}

} // namespace agent_daemon_flydelta_internal
