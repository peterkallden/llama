#include "agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

// when any semantic material is absent.  It deliberately does not infer a
// prompt, repair, decision pair or verifier from a FlyDelta job.
bool daemon_flydelta_lifecycle_record_matches_scope(
        const common_learning_lifecycle_record & record,
        const common_agent_scope & scope) {
    return record.namespace_id == scope.namespace_id &&
        record.project_id == scope.project_id &&
        record.session_id == scope.session_id;
}

// Resource authority is execution-scoped, not a property of the resident
// model. Worker callbacks receive the immutable job scope and use this
// lightweight provider view so concurrent jobs cannot overwrite one another's
// authority while sharing the same host and resource store.
std::shared_ptr<daemon_flydelta_resource_provider>
daemon_flydelta_provider_for_scope(
        const std::shared_ptr<daemon_flydelta_resource_provider> & source,
        const common_agent_scope & scope) {
    if (!source) return {};
    auto scoped = std::make_shared<daemon_flydelta_resource_provider>();
    scoped->host = source->host;
    scoped->resources = source->resources;
    scoped->authority.namespace_id = scope.namespace_id.empty()
        ? source->authority.namespace_id : scope.namespace_id;
    scoped->authority.project_id = scope.project_id;
    scoped->authority.session_id = scope.session_id.empty()
        ? source->authority.session_id : scope.session_id;
    scoped->authority.turn_id = scope.turn_id;
    scoped->authority.now = source->authority.now;
    scoped->model_profile_fingerprint = source->model_profile_fingerprint;
    scoped->capture_layout_revision = source->capture_layout_revision;
    scoped->teaching_material_runtime = source->teaching_material_runtime;
    scoped->n_predict = source->n_predict;
    scoped->n_threads = source->n_threads;
    scoped->model_n_embd = source->model_n_embd;
    scoped->model_n_layers = source->model_n_layers;
    scoped->lifecycle_store = source->lifecycle_store;
    scoped->resolve_bootstrap_zoom_state = source->resolve_bootstrap_zoom_state;
    scoped->resolve_bootstrap_zoom_state_for_job =
        source->resolve_bootstrap_zoom_state_for_job;
    scoped->resolve_representation_augmentation_state =
        source->resolve_representation_augmentation_state;
    scoped->resolve_representation_augmentation_state_for_job =
        source->resolve_representation_augmentation_state_for_job;
    scoped->resolve_search_orchestration_state = source->resolve_search_orchestration_state;
    scoped->resolve_search_orchestration_state_for_job =
        source->resolve_search_orchestration_state_for_job;
    return scoped;
}

bool daemon_flydelta_read_json(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        json & parsed,
        std::string & error);
bool daemon_flydelta_read_json_bounded(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        size_t max_bytes,
        json & parsed,
        std::string & error);

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

bool daemon_flydelta_read_json(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        json & parsed,
        std::string & error) {
    return daemon_flydelta_read_json_bounded(
        provider, reference, 1024U * 1024U, parsed, error);
}

bool daemon_flydelta_read_json_bounded(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        size_t max_bytes,
        json & parsed,
        std::string & error) {
    error.clear();
    if (provider.resources == nullptr || reference.empty() || reference.size() > 512) {
        error = "FlyDelta resource provider received an invalid reference";
        return false;
    }
    std::string text;
    if (!provider.resources->read_text(reference, provider.authority, max_bytes, text, error)) {
        error = "FlyDelta resource read failed reference=" + reference +
            " scope=" + provider.authority.namespace_id + "/" +
            provider.authority.project_id + "/" + provider.authority.session_id +
            ": " + error;
        return false;
    }
    try {
        parsed = json::parse(text);
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta referenced resource is not valid JSON: ") + exception.what();
        return false;
    }
    if (!parsed.is_object()) {
        error = "FlyDelta referenced resource must be a JSON object";
        return false;
    }
    return true;
}

bool daemon_flydelta_parse_context(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        common_agent_generation_request & request,
        std::string & error) {
    json parsed;
    if (!daemon_flydelta_read_json(provider, reference, parsed, error)) return false;
    request = {};
    request.purpose = common_agent_generation_purpose::draft;
    request.options.n_predict = provider.n_predict;
    request.options.n_threads = provider.n_threads;
    if (parsed.contains("n_predict")) {
        if (!parsed["n_predict"].is_number_integer() || parsed["n_predict"].get<int>() <= 0) {
            error = "FlyDelta context n_predict must be positive";
            return false;
        }
        request.options.n_predict = std::min(provider.n_predict, parsed["n_predict"].get<int>());
    }
    try {
        if (parsed.contains("messages")) {
            if (!parsed["messages"].is_array() || parsed["messages"].empty()) {
                error = "FlyDelta context messages must be a non-empty array";
                return false;
            }
            request.messages = common_chat_msgs_parse_oaicompat(parsed["messages"]);
        } else if (parsed.contains("prompt") && parsed["prompt"].is_string() &&
                !parsed["prompt"].get<std::string>().empty()) {
            request.messages = {{"user", parsed["prompt"].get<std::string>()}};
        } else {
            error = "FlyDelta context requires messages or prompt";
            return false;
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta context messages are invalid: ") + exception.what();
        return false;
    }
    return true;
}

bool daemon_flydelta_parse_directions(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        std::vector<common_flydelta_basis_direction> & directions,
        std::string & error) {
    {
        std::lock_guard<std::mutex> lock(provider.composed_direction_mutex);
        const auto composed = provider.composed_directions.find(reference);
        if (composed != provider.composed_directions.end()) {
            directions = composed->second;
            if (directions.empty() || directions.size() > 64) {
                error = "FlyDelta composed intervention is invalid";
                return false;
            }
            return true;
        }
    }
    json parsed;
    if (!daemon_flydelta_read_json(provider, reference, parsed, error)) return false;
    if (parsed.value("kind", "") == "flydelta") {
        common_flydelta_artifact artifact;
        if (!common_flydelta_artifact_from_json(
                parsed.dump(), 1U << 20, 4U * 1024U * 1024U, artifact, error) ||
                artifact.model_n_embd != provider.model_n_embd ||
                artifact.model_n_layers != provider.model_n_layers) {
            if (error.empty()) error = "FlyDelta artifact is incompatible with the resident model";
            return false;
        }
        directions.clear();
        directions.reserve(artifact.steering_basis.size());
        std::set<uint32_t> layers;
        for (const auto & source : artifact.steering_basis) {
            common_flydelta_basis_direction direction;
            direction.layer_index = source.layer_index;
            direction.values = source.values;
            if (direction.layer_index < 1 ||
                    static_cast<size_t>(direction.layer_index) >= provider.model_n_layers ||
                    direction.values.size() != provider.model_n_embd ||
                    !layers.insert(static_cast<uint32_t>(direction.layer_index)).second) {
                error = "FlyDelta artifact direction is incompatible with the resident model";
                return false;
            }
            double squared = 0.0;
            for (const float value : direction.values) {
                if (!std::isfinite(value)) {
                    error = "FlyDelta artifact direction contains a non-finite value";
                    return false;
                }
                squared += static_cast<double>(value) * value;
            }
            if (!std::isfinite(squared) || squared <= 0.0) {
                error = "FlyDelta artifact direction has zero norm";
                return false;
            }
            const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared));
            for (float & value : direction.values) value *= inverse_norm;
            directions.push_back(std::move(direction));
        }
        if (directions.empty() || directions.size() > 64) {
            error = "FlyDelta artifact has no bounded steering basis";
            return false;
        }
        return true;
    }
    if (!parsed.contains("directions") || !parsed["directions"].is_array() ||
            parsed["directions"].empty() || parsed["directions"].size() > 64) {
        error = "FlyDelta intervention requires a bounded directions array";
        return false;
    }
    directions.clear();
    directions.reserve(parsed["directions"].size());
    std::set<uint32_t> layers;
    try {
        for (const auto & item : parsed["directions"]) {
            if (!item.is_object() || !item.contains("layer_index") ||
                    !item["layer_index"].is_number_unsigned() ||
                    (!item.contains("values") && !item.contains("sparse_values"))) {
                error = "FlyDelta intervention direction is malformed";
                return false;
            }
            common_flydelta_basis_direction direction;
            direction.layer_index = static_cast<int32_t>(item["layer_index"].get<uint32_t>());
            if (item.contains("values")) {
                if (!item["values"].is_array()) {
                    error = "FlyDelta intervention direction values must be an array";
                    return false;
                }
                direction.values = item["values"].get<std::vector<float>>();
            } else {
                if (!item["sparse_values"].is_array() || item["sparse_values"].empty() ||
                        item["sparse_values"].size() > provider.model_n_embd) {
                    error = "FlyDelta sparse intervention direction is malformed";
                    return false;
                }
                direction.values.assign(provider.model_n_embd, 0.0f);
                std::set<size_t> indices;
                for (const auto & entry : item["sparse_values"]) {
                    if (!entry.is_object() || !entry.contains("index") ||
                            !entry["index"].is_number_unsigned() || !entry.contains("value") ||
                            !entry["value"].is_number()) {
                        error = "FlyDelta sparse intervention entry is malformed";
                        return false;
                    }
                    const auto index = entry["index"].get<size_t>();
                    const auto value = entry["value"].get<float>();
                    if (index >= provider.model_n_embd || !std::isfinite(value) ||
                            !indices.insert(index).second) {
                        error = "FlyDelta sparse intervention entry is invalid";
                        return false;
                    }
                    direction.values[index] = value;
                }
            }
            if (direction.layer_index < 1 ||
                    static_cast<size_t>(direction.layer_index) >= provider.model_n_layers ||
                    direction.values.size() != provider.model_n_embd ||
                    !layers.insert(static_cast<uint32_t>(direction.layer_index)).second) {
                error = "FlyDelta intervention direction is incompatible with the resident model";
                return false;
            }
            double squared = 0.0;
            for (const float value : direction.values) {
                if (!std::isfinite(value)) {
                    error = "FlyDelta intervention direction contains a non-finite value";
                    return false;
                }
                squared += static_cast<double>(value) * value;
            }
            if (!std::isfinite(squared) || squared <= 0.0) {
                error = "FlyDelta intervention direction has zero norm";
                return false;
            }
            const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared));
            for (float & value : direction.values) value *= inverse_norm;
            directions.push_back(std::move(direction));
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta intervention direction values are invalid: ") + exception.what();
        return false;
    }
    return true;
}

bool daemon_flydelta_parse_behavior_delta(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        common_flydelta_behavior_delta & delta,
        common_flydelta_intervention_credit & credit,
        std::string & error) {
    json value;
    if (!daemon_flydelta_read_json(provider, reference, value, error)) return false;
    try {
        if (value.value("kind", "") != "flydelta_behavior_delta" ||
                !value.contains("values") || !value["values"].is_array() ||
                !value.contains("credit") || !value["credit"].is_object()) {
            error = "FlyDelta behavior-delta resource is missing typed delta or credit material";
            return false;
        }
        delta = {};
        delta.id = value.value("id", reference);
        delta.behavior_key = value.value("behavior_key", "");
        delta.capture_manifest_id = value.value("capture_manifest_id", "");
        delta.host_evidence_ref = value.value("host_evidence_ref", "");
        delta.scope_fingerprint = value.value("scope_fingerprint", "");
        delta.tokenizer_fingerprint = value.value("tokenizer_fingerprint", "");
        delta.template_fingerprint = value.value("template_fingerprint", "");
        delta.generation_semantics_fingerprint = value.value("generation_semantics_fingerprint", "");
        delta.model_profile_fingerprint = value.value("model_profile_fingerprint", "");
        delta.execution_context_fingerprint = value.value("execution_context_fingerprint", "");
        delta.capture_layout_revision = value.value("capture_layout_revision", "");
        delta.layer_index = value.value("layer_index", -1);
        delta.values = value.at("values").get<std::vector<float>>();
        const auto source = value.value("source", "tool_repair");
        if (source == "tool_repair") delta.source = common_adaptation_evidence_source::tool_repair;
        else if (source == "procedure_blueprint") delta.source = common_adaptation_evidence_source::procedure_blueprint;
        else if (source == "user_correction") delta.source = common_adaptation_evidence_source::user_correction;
        else if (source == "user_taught_concept") delta.source = common_adaptation_evidence_source::user_taught_concept;
        else {
            error = "FlyDelta behavior-delta source is unsupported";
            return false;
        }
        credit = {};
        const auto & credit_json = value.at("credit");
        credit.experiment_id = credit_json.value("experiment_id", "");
        credit.candidate_id = credit_json.value("candidate_id", "");
        credit.fixture_id = credit_json.value("fixture_id", "");
        credit.quality_delta = credit_json.value("quality_delta", 0.0f);
        credit.eligible_for_learning = credit_json.value("eligible_for_learning", false);
        const auto outcome = credit_json.value("outcome", "unknown");
        if (outcome == "helped") credit.outcome = common_flydelta_counterfactual_outcome::helped;
        else if (outcome == "neutral") credit.outcome = common_flydelta_counterfactual_outcome::neutral;
        else if (outcome == "harmed") credit.outcome = common_flydelta_counterfactual_outcome::harmed;
        else if (outcome == "unknown") credit.outcome = common_flydelta_counterfactual_outcome::unknown;
        else {
            error = "FlyDelta behavior-delta credit outcome is invalid";
            return false;
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta behavior-delta resource is malformed: ") + exception.what();
        return false;
    }
    return common_flydelta_behavior_delta_validate(
        delta, provider.model_n_embd, 4U * 1024U * 1024U, error) &&
        common_flydelta_intervention_credit_validate(credit, error);
}

bool daemon_flydelta_put_json_resource(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & name,
        const std::string & purpose,
        const std::string & parent_uri,
        const std::string & text,
        std::string & reference,
        std::string & error) {
    error.clear();
    if (!provider || provider->resources == nullptr || name.empty() || purpose.empty() ||
            text.empty() || text.size() > 16U * 1024U * 1024U) {
        error = "FlyDelta resource persistence request is invalid";
        return false;
    }
    agent_resource_put_request request;
    request.name = name;
    request.description = "Durable FlyDelta runtime material";
    request.mime_type = "application/json";
    request.text = text;
    request.scope = common_runtime_resource_scope::session;
    request.namespace_id = provider->authority.namespace_id;
    request.session_id = provider->authority.session_id;
    request.source_provider = "flydelta";
    request.source_tool = purpose;
    request.created_at = std::time(nullptr);
    request.metadata.purpose = purpose;
    request.metadata.content_summary = "Opaque FlyDelta material for bounded worker resume.";
    request.lineage.parent_uri = parent_uri;
    request.lineage.chunk_count = 1;
    request.lineage.chunk_index = 0;
    request.lineage.derivation = purpose;
    agent_resource_descriptor descriptor;
    if (!provider->resources->put_text(request, descriptor, error) || descriptor.uri.empty()) {
        if (error.empty()) error = "FlyDelta resource store returned no URI";
        return false;
    }
    reference = descriptor.uri;
    return true;
}

json daemon_flydelta_capture_json(
        const common_flydelta_hidden_state_capture & capture) {
    return {
        {"kind", "flydelta_hidden_state_capture"},
        {"schema_version", capture.schema_version},
        {"captured", capture.captured},
        {"model_profile_fingerprint", capture.model_profile_fingerprint},
        {"capture_layout_revision", capture.capture_layout_revision},
        {"layer_indices", capture.layer_indices},
        {"n_embd", capture.n_embd},
        {"position", common_flydelta_capture_position_name(capture.position)},
        {"token_index", capture.token_index},
        {"values", capture.values},
        {"failure_reason", capture.failure_reason},
    };
}

bool daemon_flydelta_capture_from_json(
        const json & value,
        common_flydelta_hidden_state_capture & capture,
        std::string & error) {
    error.clear();
    try {
        capture = {};
        capture.schema_version = value.value("schema_version", 0);
        capture.captured = value.value("captured", false);
        capture.model_profile_fingerprint = value.value("model_profile_fingerprint", "");
        capture.capture_layout_revision = value.value("capture_layout_revision", "");
        capture.layer_indices = value.value("layer_indices", std::vector<uint32_t>{});
        capture.n_embd = value.value("n_embd", 0U);
        const auto position = value.value("position", "prompt_row");
        capture.position = position == "generation_boundary"
            ? common_flydelta_capture_position::generation_boundary
            : common_flydelta_capture_position::prompt_row;
        capture.token_index = value.value("token_index", -1);
        capture.values = value.value("values", std::vector<float>{});
        capture.failure_reason = value.value("failure_reason", "");
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta capture resource is malformed: ") + exception.what();
        return false;
    }
    return common_flydelta_hidden_state_capture_validate(
        capture, 16U * 1024U * 1024U, error);
}

bool daemon_flydelta_load_persisted_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture) {
    if (!provider || provider->resources == nullptr || arm_id.empty()) return false;
    const std::string name = "flydelta-capture-" +
        hash_sha256_hex(arm_id.data(), arm_id.size()).substr(0, 24) + ".json";
    std::vector<agent_resource_descriptor> descriptors;
    std::string error;
    if (!provider->resources->list(provider->authority, descriptors, error)) return false;
    for (const auto & descriptor : descriptors) {
        if (descriptor.name != name || descriptor.mime_type != "application/json") continue;
        std::string text;
        if (!provider->resources->read_text(
                descriptor.uri, provider->authority, 16U * 1024U * 1024U, text, error)) {
            return false;
        }
        json value;
        try {
            value = json::parse(text);
        } catch (...) {
            return false;
        }
        common_flydelta_hidden_state_capture parsed;
        if (!daemon_flydelta_capture_from_json(value, parsed, error)) return false;
        capture = std::make_shared<const common_flydelta_hidden_state_capture>(
            std::move(parsed));
        std::lock_guard<std::mutex> lock(provider->capture_mutex);
        provider->captures[arm_id] = capture;
        return true;
    }
    return false;
}

bool daemon_flydelta_capture_layer(
        const common_flydelta_hidden_state_capture & capture,
        int32_t layer_index,
        std::vector<float> & values,
        std::string & error) {
    error.clear();
    const auto it = std::find(capture.layer_indices.begin(), capture.layer_indices.end(),
        static_cast<uint32_t>(layer_index));
    if (it == capture.layer_indices.end() || capture.n_embd == 0) {
        error = "FlyDelta concept capture does not contain the requested layer";
        return false;
    }
    const size_t offset = static_cast<size_t>(std::distance(capture.layer_indices.begin(), it)) *
        capture.n_embd;
    values.assign(capture.values.begin() + static_cast<std::ptrdiff_t>(offset),
        capture.values.begin() + static_cast<std::ptrdiff_t>(offset + capture.n_embd));
    return true;
}

// Resolve the latest localized search surface for concept capture. The
// relation supplies WHAT, while the persisted search/augmentation state
// supplies WHERE. If that state cannot identify one anchor, fail closed
// rather than silently reverting to a relation/default layer.
bool daemon_flydelta_resolve_concept_target(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        daemon_flydelta_concept_target & target,
        std::string & error) {
    error.clear();
    target = {};
    if (!provider) {
        error = "FlyDelta concept target has no resource provider";
        return false;
    }

    common_flydelta_representation_augmentation_state augmentation;
    bool has_augmentation = false;
    if (!job.representation_augmentation_state_ref.empty()) {
        const bool resolved = provider->resolve_representation_augmentation_state_for_job
            ? provider->resolve_representation_augmentation_state_for_job(
                job, job.representation_augmentation_state_ref, augmentation, error)
            : provider->resolve_representation_augmentation_state &&
                provider->resolve_representation_augmentation_state(
                    job.representation_augmentation_state_ref, augmentation, error);
        if (!resolved) {
            if (error.empty()) error = "FlyDelta concept target augmentation state is unavailable";
            return false;
        }
        has_augmentation = true;
        target.local_layers = augmentation.selected_region;
        target.parent_surface_ref = augmentation.parent_search_state_ref;
        target.parent_surface_revision = augmentation.parent_surface_revision;
        target.parent_evidence_rank = augmentation.parent_evidence_rank;
    }

    const auto apply_orchestration = [&](const std::string & reference) {
        if (reference.empty() ||
                (!provider->resolve_search_orchestration_state_for_job &&
                 !provider->resolve_search_orchestration_state)) return false;
        common_flydelta_experiment_plan plan;
        common_flydelta_utility_history history;
        std::string resolve_error;
        const bool resolved = provider->resolve_search_orchestration_state_for_job
            ? provider->resolve_search_orchestration_state_for_job(
                job, reference, plan, history, resolve_error)
            : provider->resolve_search_orchestration_state(
                reference, plan, history, resolve_error);
        if (!resolved) return false;
        target.layer_index = static_cast<int32_t>(
            plan.continuation.region.anchor_layer_index);
        target.local_layers = plan.continuation.region.layer_indices;
        target.parent_surface_ref = reference;
        return target.layer_index > 0;
    };
    const auto apply_bootstrap = [&](const std::string & reference) {
        if (reference.empty() ||
                (!provider->resolve_bootstrap_zoom_state_for_job &&
                 !provider->resolve_bootstrap_zoom_state)) return false;
        common_flydelta_bootstrap_zoom_state state;
        std::string resolve_error;
        const bool resolved = provider->resolve_bootstrap_zoom_state_for_job
            ? provider->resolve_bootstrap_zoom_state_for_job(
                job, reference, state, resolve_error)
            : provider->resolve_bootstrap_zoom_state(reference, state, resolve_error);
        if (!resolved) return false;
        target.layer_index = static_cast<int32_t>(state.anchor_layer);
        target.local_layers = state.local_layers;
        target.parent_surface_ref = reference;
        target.parent_surface_revision = state.surface_revision;
        target.parent_evidence_rank = state.evidence_rank;
        return target.layer_index > 0;
    };

    // The augmentation state's parent is the most localized explicit search
    // reference. Prefer it over the job envelope, then use the BootstrapZoom
    // ref carried through the same queued lineage.
    const std::string parent_search_ref = has_augmentation
        ? augmentation.parent_search_state_ref : job.search_state_ref;
    bool resolved = apply_orchestration(parent_search_ref);
    if (!resolved && parent_search_ref != job.bootstrap_zoom_state_ref) {
        resolved = apply_bootstrap(parent_search_ref);
    }
    if (!resolved) resolved = apply_orchestration(job.search_state_ref);
    if (!resolved) resolved = apply_bootstrap(job.bootstrap_zoom_state_ref);

    // A one-layer selected region is already an unambiguous localized
    // surface. Multi-layer regions require their parent state so we do not
    // invent an anchor by choosing an arbitrary member.
    if (!resolved && has_augmentation && augmentation.selected_region.size() == 1) {
        target.layer_index = static_cast<int32_t>(augmentation.selected_region.front());
        target.local_layers = augmentation.selected_region;
        resolved = true;
    }
    if (!resolved || target.layer_index <= 0 ||
            static_cast<size_t>(target.layer_index) >= provider->model_n_layers) {
        error = "FlyDelta concept capture requires a persisted localized anchor layer";
        return false;
    }
    if (!target.local_layers.empty() &&
            std::find(target.local_layers.begin(), target.local_layers.end(),
                static_cast<uint32_t>(target.layer_index)) == target.local_layers.end()) {
        error = "FlyDelta concept target anchor is outside its localized region";
        return false;
    }
    return true;
}

bool daemon_flydelta_persist_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        const std::shared_ptr<const common_flydelta_hidden_state_capture> & capture,
        std::string & reference,
        std::string & error) {
    if (!capture || !capture->captured) {
        error = "FlyDelta cannot persist an empty hidden-state capture";
        return false;
    }
    return daemon_flydelta_put_json_resource(
        provider,
        "flydelta-capture-" + hash_sha256_hex(arm_id.data(), arm_id.size()).substr(0, 24) + ".json",
        "flydelta_hidden_state_capture",
        arm_id,
        daemon_flydelta_capture_json(*capture).dump(),
        reference,
        error);
}

bool daemon_flydelta_persist_concept_trajectory(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const daemon_flydelta_concept_relation_material & material,
        const std::string & group_ref,
        const std::string & job_id,
        const std::string & baseline_capture_ref,
        const std::string & conditioned_capture_ref,
        const std::string & control_capture_ref,
        const common_flydelta_hidden_state_capture & baseline,
        const common_flydelta_hidden_state_capture & conditioned,
        const common_flydelta_hidden_state_capture & control,
        std::string & trajectory_ref,
        std::string & error) {
    std::vector<float> baseline_values;
    std::vector<float> conditioned_values;
    std::vector<float> control_values;
    if (!daemon_flydelta_capture_layer(baseline, material.layer_index, baseline_values, error) ||
            !daemon_flydelta_capture_layer(conditioned, material.layer_index, conditioned_values, error) ||
            !daemon_flydelta_capture_layer(control, material.layer_index, control_values, error)) {
        return false;
    }
    const auto & relation = material.relation;
    const std::string trajectory_identity = job_id + "\n" + relation.id;
    const std::string trajectory_id = "flydelta://trajectory/" +
        hash_sha256_hex(trajectory_identity.data(), trajectory_identity.size()).substr(0, 32);
    const json payload = {
        {"kind", "flydelta_concept_trajectory"},
        {"schema_version", 1},
        {"id", trajectory_id},
        {"group_ref", group_ref},
        {"fixture_ref", relation.verifier_ref},
        {"semantic_anchor", material.semantic_anchor},
        {"layer_index", material.layer_index},
        {"baseline_capture_ref", baseline_capture_ref},
        {"conditioned_capture_ref", conditioned_capture_ref},
        {"control_capture_ref", control_capture_ref},
        {"baseline", baseline_values},
        {"conditioned", conditioned_values},
        {"control", control_values},
        {"aligned", true},
        {"conditioned_host_verified", relation.host_approved},
        {"teaching_key", relation.teaching_key},
        {"concept_key", relation.teaching_key},
        {"extraction_id", "flydelta://extraction/" + relation.teaching_key},
        {"behavior_key", relation.behavior_key},
        {"source", common_adaptation_evidence_source_name(relation.source)},
        {"source_ref", relation.id},
        {"grounding_ref", relation.contrast_ref},
        {"verifier_ref", relation.verifier_ref},
        {"model_profile_fingerprint", provider->model_profile_fingerprint},
        {"tokenizer_fingerprint", provider->model_profile_fingerprint + ":tokenizer"},
        {"template_fingerprint", provider->model_profile_fingerprint + ":template"},
        {"capture_layout_revision", provider->capture_layout_revision},
        {"scope_fingerprint", relation.scope.namespace_id + ":" + relation.scope.session_id},
        {"parent_surface_ref", material.parent_surface_ref},
        {"parent_surface_revision", material.parent_surface_revision},
        {"parent_evidence_rank", material.parent_evidence_rank},
    };
    return daemon_flydelta_put_json_resource(
        provider,
        "flydelta-concept-trajectory-" + hash_sha256_hex(trajectory_id.data(), trajectory_id.size()).substr(0, 24) + ".json",
        "flydelta_concept_trajectory",
        relation.id,
        payload.dump(),
        trajectory_ref,
        error);
}

bool daemon_flydelta_register_composed_directions(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & source_ref,
        std::vector<common_flydelta_basis_direction> directions,
        const std::string & derivation,
        std::string & reference,
        std::string & error) {
    error.clear();
    if (directions.empty() || directions.size() > 64 || derivation.empty()) {
        error = "FlyDelta composed direction artifact is invalid";
        return false;
    }
    json serialized = {{"directions", json::array()}};
    std::string identity = source_ref + "\n" + derivation;
    for (const auto & direction : directions) {
        if (direction.layer_index < 1 || direction.values.size() != provider->model_n_embd) {
            error = "FlyDelta composed direction artifact has an invalid layer or dimension";
            return false;
        }
        serialized["directions"].push_back({
            {"layer_index", static_cast<uint32_t>(direction.layer_index)},
            {"values", direction.values},
        });
        identity += "\n" + std::to_string(direction.layer_index);
        for (const float value : direction.values) identity += ":" + std::to_string(value);
    }
    reference = "flydelta://runtime/composed/" +
        hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
    if (provider->resources != nullptr) {
        agent_resource_put_request request;
        request.name = "flydelta-composed-" +
            hash_sha256_hex(identity.data(), identity.size()).substr(0, 16) + ".json";
        request.description = "Durable composed FlyDelta intervention direction";
        request.mime_type = "application/json";
        request.text = serialized.dump();
        request.scope = common_runtime_resource_scope::session;
        request.namespace_id = provider->authority.namespace_id;
        request.session_id = provider->authority.session_id;
        request.source_provider = "flydelta";
        request.source_tool = "compose-direction";
        request.created_at = std::time(nullptr);
        request.metadata.purpose = "flydelta_composed_direction";
        request.metadata.content_summary =
            "Durable normalized composed FlyDelta direction for a resumed bounded slice.";
        request.metadata.processing_cache_key = reference;
        request.lineage.parent_uri = source_ref;
        request.lineage.chunk_count = 1;
        request.lineage.chunk_index = 0;
        request.lineage.derivation = derivation;
        agent_resource_descriptor descriptor;
        if (!provider->resources->put_text(request, descriptor, error)) return false;
        if (descriptor.uri.empty()) {
            error = "FlyDelta composed direction resource has no URI";
            return false;
        }
        reference = descriptor.uri;
    }
    {
        std::lock_guard<std::mutex> lock(provider->composed_direction_mutex);
        provider->composed_directions[reference] = std::move(directions);
    }
    return true;
}

bool daemon_flydelta_register_composed_direction(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & source_ref,
        const common_flydelta_low_rank_basis & basis,
        const std::vector<float> & coefficients,
        std::string & reference,
        std::string & error) {
    error.clear();
    if (basis.vectors.empty() || basis.vectors.size() != coefficients.size() ||
            basis.layer_index < 1 || basis.dimension == 0) {
        error = "FlyDelta composed coefficient direction has invalid basis shape";
        return false;
    }
    std::vector<float> values(basis.dimension, 0.0f);
    double squared = 0.0;
    std::string identity = source_ref + "\n" + std::to_string(basis.layer_index);
    for (size_t index = 0; index < coefficients.size(); ++index) {
        if (!std::isfinite(coefficients[index]) ||
                basis.vectors[index].size() != basis.dimension) {
            error = "FlyDelta composed coefficient direction contains invalid values";
            return false;
        }
        identity += "\n" + std::to_string(coefficients[index]);
        for (size_t value = 0; value < basis.dimension; ++value) {
            values[value] += coefficients[index] * basis.vectors[index][value];
        }
    }
    for (const float value : values) squared += static_cast<double>(value) * value;
    if (!std::isfinite(squared) || squared <= 0.0) {
        error = "FlyDelta composed coefficient direction has zero norm";
        return false;
    }
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared));
    for (float & value : values) value *= inverse_norm;
    common_flydelta_basis_direction direction;
    direction.layer_index = basis.layer_index;
    direction.values = std::move(values);
    return daemon_flydelta_register_composed_directions(
        provider, source_ref, {std::move(direction)},
        "flydelta:coefficient-composition", reference, error);
}

bool daemon_flydelta_resolve_evidence_direction_candidates(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const int32_t layer_index,
        const size_t required_count,
        std::vector<common_flydelta_direction_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    for (const auto & reference : job.behavior_delta_ids) {
        common_flydelta_behavior_delta delta;
        common_flydelta_intervention_credit credit;
        if (!daemon_flydelta_parse_behavior_delta(*provider, reference, delta, credit, error)) {
            return false;
        }
        // Natural Shallow/Deep capacity is based on verified positive
        // evidence. Experimental UNKNOWN/NEUTRAL material belongs to the
        // orthogonal/augmentation paths and must not raise this basis.
        if (credit.outcome != common_flydelta_counterfactual_outcome::helped ||
                !credit.eligible_for_learning || delta.layer_index != layer_index) {
            continue;
        }
        if (delta.behavior_key != job.seed.behavior_key ||
                delta.model_profile_fingerprint != job.seed.model_profile_fingerprint ||
                delta.execution_context_fingerprint != job.seed.execution_context_fingerprint ||
                delta.capture_layout_revision != provider->capture_layout_revision) {
            error = "FlyDelta post-Bootstrap evidence delta is incompatible with the resumed job";
            return false;
        }
        common_flydelta_direction_candidate candidate;
        candidate.kind = common_flydelta_direction_kind::raw_repair;
        candidate.layer_index = delta.layer_index;
        candidate.values = std::move(delta.values);
        candidate.origin = "runtime_verified_evidence";
        candidate.extraction_id = reference;
        candidate.source_samples = 1;
        candidate.retained_samples = 1;
        candidate.experimental_only = false;
        if (!common_flydelta_direction_candidate_validate(
                candidate, provider->model_n_embd, error)) return false;
        candidates.push_back(std::move(candidate));
    }
    if (candidates.size() < required_count) {
        error = "FlyDelta post-Bootstrap phase requires compatible HELPED evidence directions at the selected layer";
        return false;
    }
    return true;
}

bool daemon_flydelta_parse_augmentation_donor(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & reference,
        const common_flydelta_representation_augmentation_state & state,
        const common_flydelta_experiment_job & job,
        daemon_flydelta_augmentation_donor_material & material,
        std::string & error) {
    error.clear();
    json candidate_resource;
    if (!daemon_flydelta_read_json(*provider, reference, candidate_resource, error)) return false;
    const json candidate = candidate_resource.contains("candidate") &&
            candidate_resource["candidate"].is_object()
        ? candidate_resource["candidate"] : candidate_resource;
    const json payload = candidate_resource.contains("payload") &&
            candidate_resource["payload"].is_object()
        ? candidate_resource["payload"] : candidate_resource;
    try {
        material = {};
        auto & value = material.candidate;
        value.schema_version = candidate.value("schema_version", 1);
        value.donor_id = candidate.value("donor_id",
            candidate.value("id", reference));
        value.source_type = candidate.value("source_type", "host_owned");
        value.source_ref = candidate.value("source_ref", reference);
        value.behavior_key = candidate.value("behavior_key", state.behavior_key);
        value.context_payload_ref = candidate.value("context_payload_ref", reference);
        value.qualification_policy = candidate.value(
            "qualification_policy", "host-certified-contextual");
        value.provenance = candidate.value("provenance", reference);
        value.promotable = candidate.value("promotable", false);

        material.target_context_ref = payload.value("target_context_ref",
            payload.value("target_only_ref", job.seed.baseline_ref));
        material.donor_context_ref = payload.value("target_plus_donor_context_ref",
            payload.value("donor_context_ref", ""));
        material.fixture_ref = payload.value("fixture_ref",
            state.target_fixture_ref.empty() ? job.seed.verifier_ref : state.target_fixture_ref);
        material.layer_index = payload.value("layer_index",
            state.selected_region.empty() ? 0 : static_cast<int32_t>(state.selected_region.front()));

        material.qualification.donor_id = value.donor_id;
        const auto qualification = payload.value("qualification", json::object());
        material.qualification.host_evaluated = qualification.value("host_evaluated", false);
        material.qualification.verifier_known = qualification.value("verifier_known", false);
        material.qualification.safe_to_continue = qualification.value("safe_to_continue", false);
        material.qualification.margin_gain = qualification.value("margin_gain", 0.0f);
        material.qualification.decision_margin_available =
            qualification.value("decision_margin_available", false);
        const auto outcome = qualification.value("host_outcome", "unknown");
        if (outcome == "helped") {
            material.qualification.host_outcome =
                common_flydelta_counterfactual_outcome::helped;
        } else if (outcome == "neutral") {
            material.qualification.host_outcome =
                common_flydelta_counterfactual_outcome::neutral;
        } else if (outcome == "harmed") {
            material.qualification.host_outcome =
                common_flydelta_counterfactual_outcome::harmed;
        } else {
            material.qualification.host_outcome =
                common_flydelta_counterfactual_outcome::unknown;
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta augmentation donor resource is malformed: ") +
            exception.what();
        return false;
    }
    if (!common_flydelta_representation_donor_candidate_validate(
            material.candidate, error) || material.candidate.behavior_key != state.behavior_key ||
            material.target_context_ref.empty() || material.donor_context_ref.empty() ||
            material.fixture_ref.empty() || material.layer_index <= 0 ||
            static_cast<size_t>(material.layer_index) >= provider->model_n_layers) {
        if (error.empty()) error = "FlyDelta augmentation donor is incompatible with the resumed state";
        return false;
    }
    if (!common_flydelta_representation_donor_qualification_validate(
            material.qualification, error)) return false;
    return true;
}

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

bool daemon_flydelta_load_augmentation_material(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & state_ref,
        daemon_flydelta_augmentation_material & material,
        std::string & error) {
    error.clear();
    if (!provider || provider->resources == nullptr || state_ref.empty()) {
        error = "FlyDelta augmentation material store is not configured";
        return false;
    }
    const std::string name = "flydelta-augmentation-material-" +
        hash_sha256_hex(state_ref.data(), state_ref.size()).substr(0, 24) + ".json";
    std::vector<agent_resource_descriptor> descriptors;
    if (!provider->resources->list(provider->authority, descriptors, error)) return false;
    for (auto it = descriptors.rbegin(); it != descriptors.rend(); ++it) {
        if (it->name != name || it->mime_type != "application/json") continue;
        std::string text;
        if (!provider->resources->read_text(
                it->uri, provider->authority, 16U * 1024U * 1024U, text, error)) return false;
        json value;
        try {
            value = json::parse(text);
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta augmentation material JSON is invalid: ") + exception.what();
            return false;
        }
        return daemon_flydelta_augmentation_material_from_json(value, material, error);
    }
    material = {};
    material.state_ref = state_ref;
    return true;
}

bool daemon_flydelta_persist_augmentation_material(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const daemon_flydelta_augmentation_material & material,
        std::string & error) {
    std::string reference;
    return daemon_flydelta_put_json_resource(
        provider,
        "flydelta-augmentation-material-" +
            hash_sha256_hex(material.state_ref.data(), material.state_ref.size()).substr(0, 24) + ".json",
        "flydelta_representation_augmentation_material", material.state_ref,
        daemon_flydelta_augmentation_material_json(material).dump(), reference, error);
}

bool daemon_flydelta_capture_from_reference(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & reference,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture,
        std::string & error) {
    json value;
    if (!daemon_flydelta_read_json(*provider, reference, value, error)) return false;
    common_flydelta_hidden_state_capture parsed;
    if (!daemon_flydelta_capture_from_json(value, parsed, error)) return false;
    capture = std::make_shared<const common_flydelta_hidden_state_capture>(std::move(parsed));
    return true;
}

bool daemon_flydelta_augmentation_seed_result(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_representation_augmentation_state & state,
        common_flydelta_search_pipeline_result & result,
        std::string & error) {
    result = {};
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, directions, error) ||
            directions.empty()) return false;
    const int32_t layer = state.selected_region.empty()
        ? directions.front().layer_index : static_cast<int32_t>(state.selected_region.front());
    const auto it = std::find_if(directions.begin(), directions.end(), [&](const auto & value) {
        return value.layer_index == layer;
    });
    if (it == directions.end()) {
        error = "FlyDelta augmentation parent surface lacks the selected layer";
        return false;
    }
    common_flydelta_search_pipeline_direction_result direction;
    direction.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction.direction.layer_index = it->layer_index;
    direction.direction.values = it->values;
    direction.direction.origin = "runtime_representation_augmentation";
    direction.direction.extraction_id = job.seed.candidate_ref;
    direction.direction.source_samples = 1;
    direction.direction.retained_samples = 1;
    direction.direction.experimental_only = true;
    result.directions.push_back(std::move(direction));
    result.search_status = common_flydelta_search_status::no_useful_utility;
    return true;
}

bool daemon_flydelta_capture_augmentation_pair(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        daemon_flydelta_augmentation_donor_material & material,
        std::string & error) {
    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":augmentation-capture:" + material.candidate.donor_id;
    batch.wave_id = "augmentation-donor-capture";
    const std::array<std::string, 2> contexts = {
        material.target_context_ref, material.donor_context_ref};
    for (size_t index = 0; index < contexts.size(); ++index) {
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = batch.wave_id;
        arm.proposal_index = index;
        arm.arm_id = batch.batch_id + (index == 0 ? ":target" : ":target-plus-donor");
        arm.context_ref = contexts[index];
        arm.fixture_ref = material.fixture_ref;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.layer_indices = {static_cast<uint32_t>(material.layer_index)};
        arm.coefficients = {0.0f};
        arm.fresh_context = true;
        arm.request_capture = true;
        arm.request_generation = true;
        arm.max_capture_bytes = 4U * 1024U * 1024U;
        arm.max_generated_tokens = 1;
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result result;
    if (!daemon_flydelta_execute_batch(provider, batch, result, error) ||
            result.arms.size() != batch.arms.size()) {
        if (error.empty()) error = "FlyDelta augmentation donor capture returned incomplete arms";
        return false;
    }
    if (!result.arms[0].executed || !result.arms[1].executed ||
            result.arms[0].capture_ref.empty() || result.arms[1].capture_ref.empty()) {
        error = "FlyDelta augmentation donor capture did not produce both captures";
        return false;
    }
    material.target_capture_ref = result.arms[0].capture_ref;
    material.donor_capture_ref = result.arms[1].capture_ref;
    return true;
}

bool daemon_flydelta_capture_from_reference(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & reference,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture,
        std::string & error);



} // namespace agent_daemon_flydelta_internal
