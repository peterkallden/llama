#include "agent/adaptation/flydelta/flydelta-teaching-material.h"

#include "hash/hash.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace {

using json = nlohmann::ordered_json;

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool append_key_part(std::string & key, const std::string & value) {
    if (!bounded(value)) return false;
    key.append(std::to_string(value.size()));
    key.push_back(':');
    key.append(value);
    key.push_back('|');
    return true;
}

} // namespace

bool common_flydelta_teaching_material_identity_validate(
        const common_flydelta_teaching_material_identity & identity,
        std::string & error) {
    error.clear();
    if (identity.schema_version != 1 ||
            !bounded(identity.model_profile_fingerprint) ||
            !bounded(identity.tokenizer_fingerprint) ||
            !bounded(identity.template_fingerprint) ||
            !bounded(identity.capture_layout_revision) ||
            !bounded(identity.execution_context_fingerprint) ||
            !bounded(identity.scope_fingerprint) ||
            !bounded(identity.verifier_revision)) {
        error = "FlyDelta teaching material compatibility identity is incomplete";
        return false;
    }
    return true;
}

bool common_flydelta_teaching_material_execution_matches(
        const common_flydelta_teaching_material_identity & identity,
        const std::string & model, const std::string & tokenizer,
        const std::string & chat_template, const std::string & execution_context,
        const std::string & capture_layout, std::string & error) {
    error.clear();
    if (!common_flydelta_teaching_material_identity_validate(identity, error)) return false;
    const auto compare = [&](const char * field, const std::string & expected, const std::string & actual) {
        if (actual.empty() || expected != actual) {
            error = std::string("FlyDelta teaching material ") + field + " does not match the execution profile";
            return false;
        }
        return true;
    };
    return compare("model fingerprint", identity.model_profile_fingerprint, model) &&
        compare("tokenizer fingerprint", identity.tokenizer_fingerprint, tokenizer) &&
        compare("template fingerprint", identity.template_fingerprint, chat_template) &&
        compare("execution context fingerprint", identity.execution_context_fingerprint, execution_context) &&
        (capture_layout.empty() || compare("capture layout revision", identity.capture_layout_revision, capture_layout));
}

std::string common_flydelta_teaching_material_compatibility_key(
        const common_flydelta_teaching_material_identity & identity) {
    std::string key;
    if (!append_key_part(key, identity.model_profile_fingerprint) ||
            !append_key_part(key, identity.tokenizer_fingerprint) ||
            !append_key_part(key, identity.template_fingerprint) ||
            !append_key_part(key, identity.capture_layout_revision) ||
            !append_key_part(key, identity.execution_context_fingerprint) ||
            !append_key_part(key, identity.scope_fingerprint) ||
            !append_key_part(key, identity.verifier_revision)) {
        return {};
    }
    return key;
}

namespace {
std::string material_family_ref(const common_flydelta_teaching_material_group & group) {
    const auto key = group.compatibility_key + "\n" + group.teaching_key + "\n" + group.behavior_key +
        (group.positive_only ? "\npositive_only" : "") + "\nscope-v2\n" + group.scope.namespace_id +
        "\n" + group.scope.project_id + "\n" + group.scope.session_id;
    return "flydelta://teaching-material/" + hash_sha256_hex(key.data(), key.size());
}
}

bool common_flydelta_teaching_material_store::observe(
        const common_flydelta_teaching_relation & relation,
        const common_flydelta_teaching_material_identity & identity,
        std::string & error) {
    error.clear();
    if (!common_flydelta_teaching_relation_validate(relation, error) ||
            relation.status != common_flydelta_teaching_relation_status::resolved ||
            !relation.host_approved ||
            !common_flydelta_teaching_material_identity_validate(identity, error)) {
        if (error.empty()) error = "FlyDelta teaching material relation is not admissible";
        return false;
    }
    const auto recorded_matches = [&](const char * field, const std::string & recorded, const std::string & current) {
        if (!recorded.empty() && recorded != current) {
            error = std::string("FlyDelta source relation ") + field + " differs from the host execution profile";
            return false;
        }
        return true;
    };
    if (!recorded_matches("model fingerprint", relation.model_profile_fingerprint, identity.model_profile_fingerprint) ||
            !recorded_matches("tokenizer fingerprint", relation.tokenizer_fingerprint, identity.tokenizer_fingerprint) ||
            !recorded_matches("template fingerprint", relation.template_fingerprint, identity.template_fingerprint) ||
            !recorded_matches("capture layout", relation.capture_layout_revision, identity.capture_layout_revision)) return false;
    const std::string compatibility_key =
        common_flydelta_teaching_material_compatibility_key(identity);
    if (compatibility_key.empty()) {
        error = "FlyDelta teaching material compatibility key is invalid";
        return false;
    }
    const auto same_group = [&](const common_flydelta_teaching_material_group & group) {
        return group.scope.namespace_id == relation.scope.namespace_id &&
            group.scope.project_id == relation.scope.project_id && group.scope.session_id == relation.scope.session_id &&
            group.positive_only == relation.positive_only &&
            group.teaching_key == relation.teaching_key &&
            group.behavior_key == relation.behavior_key &&
            group.compatibility_key == compatibility_key;
    };
    auto it = std::find_if(groups_.begin(), groups_.end(), same_group);
    if (it == groups_.end()) {
        common_flydelta_teaching_material_group group;
        group.scope = relation.scope;
        group.scope.turn_id.clear(); // Independent turns may contribute within the authorized session.
        group.positive_only = relation.positive_only;
        if (group.positive_only) group.minimum_relations = group.minimum_trajectories = 3;
        group.teaching_key = relation.teaching_key;
        group.behavior_key = relation.behavior_key;
        group.compatibility_key = compatibility_key;
        group.identity = identity;
        group.group_ref = material_family_ref(group);
        it = groups_.insert(groups_.end(), std::move(group));
    }
    // A completed source supersedes its pending entry under the same identity.
    // Never leave the same reference in both collections after reassessment.
    if (!relation.positive_only && !relation.control_ref.empty()) {
        auto & pending = it->pending_relation_refs;
        pending.erase(std::remove(pending.begin(), pending.end(), relation.id), pending.end());
    }
    auto & source_refs = !relation.positive_only && relation.control_ref.empty()
        ? it->pending_relation_refs : it->relation_refs;
    if (std::find(source_refs.begin(), source_refs.end(), relation.id) == source_refs.end())
        source_refs.push_back(relation.id);
    it->relation_set_ready = it->relation_refs.size() >= it->minimum_relations;
    it->trajectory_material_ready =
        it->trajectory_refs.size() >= it->minimum_trajectories;
    return true;
}

bool common_flydelta_teaching_material_store::group_ready(
        const std::string & teaching_key,
        const std::string & behavior_key,
        const common_flydelta_teaching_material_identity & identity,
        size_t minimum_relations,
        common_flydelta_teaching_material_group & group,
        std::string & error) const {
    error.clear();
    group = {};
    if (minimum_relations < 2 ||
            !common_flydelta_teaching_material_identity_validate(identity, error)) {
        if (error.empty()) error = "FlyDelta teaching material minimum is invalid";
        return false;
    }
    const std::string key = common_flydelta_teaching_material_compatibility_key(identity);
    // Preserve the established contrast-based route when both kinds exist.
    // An incomplete group must not hide a ready group of the other kind.
    const common_flydelta_teaching_material_group * match = nullptr;
    for (const auto & candidate : groups_) {
        if (candidate.teaching_key != teaching_key || candidate.behavior_key != behavior_key ||
                candidate.compatibility_key != key ||
                candidate.relation_refs.size() < std::max(minimum_relations, candidate.minimum_relations)) continue;
        if (!match || (match->positive_only && !candidate.positive_only)) match = &candidate;
    }
    if (!match) return true;
    minimum_relations = std::max(minimum_relations, match->minimum_relations);
    group = *match;
    group.minimum_relations = minimum_relations;
    group.relation_set_ready = true;
    return true;
}

bool common_flydelta_teaching_material_store::observe_trajectory(
        const std::string & group_ref,
        const std::string & trajectory_ref,
        const common_flydelta_teaching_material_identity & identity,
        std::string & error) {
    error.clear();
    if (!bounded(group_ref) || !bounded(trajectory_ref) ||
            !common_flydelta_teaching_material_identity_validate(identity, error)) {
        if (error.empty()) error = "FlyDelta teaching trajectory reference is invalid";
        return false;
    }
    const std::string compatibility_key =
        common_flydelta_teaching_material_compatibility_key(identity);
    auto it = std::find_if(groups_.begin(), groups_.end(), [&](const auto & candidate) {
        return candidate.group_ref == group_ref;
    });
    if (it == groups_.end()) {
        error = "FlyDelta teaching material group was not found";
        return false;
    }
    if (it->compatibility_key != compatibility_key) {
        error = "FlyDelta teaching trajectory compatibility does not match group";
        return false;
    }
    if (std::find(it->trajectory_refs.begin(), it->trajectory_refs.end(), trajectory_ref) ==
            it->trajectory_refs.end()) {
        it->trajectory_refs.push_back(trajectory_ref);
    }
    it->trajectory_material_ready =
        it->trajectory_refs.size() >= it->minimum_trajectories;
    return true;
}

bool common_flydelta_teaching_material_store::observe_layer_trajectory(
        const std::string & group_ref, int32_t layer_index,
        const std::string & relation_ref, const std::string & trajectory_ref,
        const common_flydelta_teaching_material_identity & identity, std::string & error,
        const std::string & capture_set_ref) {
    error.clear();
    if (layer_index < 1 || (!capture_set_ref.empty() && !bounded(capture_set_ref)) ||
            !bounded(group_ref) || !bounded(relation_ref) ||
            !bounded(trajectory_ref) || !common_flydelta_teaching_material_identity_validate(identity, error)) {
        if (error.empty()) error = "FlyDelta layer trajectory identity is invalid";
        return false;
    }
    auto it = std::find_if(groups_.begin(), groups_.end(), [&](const auto & g) { return g.group_ref == group_ref; });
    if (it == groups_.end() || it->compatibility_key != common_flydelta_teaching_material_compatibility_key(identity) ||
            std::find(it->relation_refs.begin(), it->relation_refs.end(), relation_ref) == it->relation_refs.end()) {
        error = "FlyDelta layer trajectory group, compatibility, or relation is invalid";
        return false;
    }
    for (const auto & entry : it->layer_trajectories) {
        if (entry.layer_index == layer_index && entry.relation_ref == relation_ref &&
                entry.capture_set_ref == capture_set_ref) {
            if (entry.trajectory_ref == trajectory_ref) return true;
            error = "FlyDelta layer trajectory conflicts with existing material";
            return false;
        }
        if (entry.trajectory_ref == trajectory_ref) {
            error = "FlyDelta trajectory reference is already bound to another layer or relation";
            return false;
        }
    }
    if (it->layer_trajectories.size() >= 4096) {
        error = "FlyDelta layer trajectory limit exceeded";
        return false;
    }
    it->layer_trajectories.push_back({layer_index, relation_ref, trajectory_ref, capture_set_ref});
    return true;
}

bool common_flydelta_teaching_material_store::resolve_layer_group(
        const std::string & group_ref, int32_t layer_index,
        common_flydelta_teaching_material_group & group, std::string & error,
        const std::string & capture_set_ref) const {
    error.clear();
    group = {};
    if (!bounded(group_ref) || layer_index < 1 ||
            (!capture_set_ref.empty() && !bounded(capture_set_ref))) {
        error = "FlyDelta layer material identity is invalid";
        return false;
    }
    const auto it = std::find_if(groups_.begin(), groups_.end(), [&](const auto & g) { return g.group_ref == group_ref; });
    if (it == groups_.end() || !it->relation_set_ready) return true;
    std::vector<std::string> refs;
    for (const auto & entry : it->layer_trajectories) {
        if (entry.layer_index == layer_index && entry.capture_set_ref == capture_set_ref)
            refs.push_back(entry.trajectory_ref);
    }
    if (refs.size() < it->minimum_trajectories) return true;
    group = *it;
    group.trajectory_refs = std::move(refs);
    group.trajectory_material_ready = true;
    return true;
}

bool common_flydelta_teaching_material_runtime::observe_layer_trajectory(
        const std::string & group_ref, int32_t layer_index, const std::string & relation_ref,
        const std::string & trajectory_ref, std::string & error, const std::string & capture_set_ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    common_flydelta_teaching_material_store next = store_;
    if (!next.observe_layer_trajectory(group_ref, layer_index, relation_ref, trajectory_ref,
            identity_, error, capture_set_ref)) return false;
    if (!persist(next, error)) return false;
    store_ = std::move(next);
    return true;
}

std::string common_flydelta_teaching_material_revision(const common_flydelta_teaching_material_group & group) {
    auto refs = group.relation_refs;
    auto pending = group.pending_relation_refs;
    std::sort(refs.begin(), refs.end());
    std::sort(pending.begin(), pending.end());
    const auto bytes = nlohmann::ordered_json{{"schema_version", 1},
        {"group_ref", group.group_ref}, {"identity", group.compatibility_key},
        {"verifier_revision", group.identity.verifier_revision}, {"relation_refs", refs}, {"pending_relation_refs", pending}}.dump();
    return "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
}

void common_flydelta_teaching_material_runtime::bind_assessment_journal(
        std::shared_ptr<common_learning_lifecycle_store> journal) {
    std::lock_guard<std::mutex> lock(mutex_);
    assessment_journal_ = std::move(journal);
}

bool common_flydelta_teaching_material_runtime::assess_contrast(
        const std::string & group_ref, const std::string & verifier_ref,
        const std::vector<std::string> & evidence_refs, bool rejected, std::string & error,
        const std::string & expected_material_revision, const std::string & verifier_revision) {
    std::lock_guard<std::mutex> lock(mutex_);
    error.clear();
    const auto selected = std::find_if(store_.groups().begin(), store_.groups().end(),
        [&](const auto & candidate) { return candidate.group_ref == group_ref; });
    if (selected == store_.groups().end()) {
        error = "FlyDelta contrast assessment source group was not found";
        return false;
    }
    return append_contrast_assessment_locked(*selected, verifier_ref, evidence_refs,
        rejected, error, expected_material_revision, verifier_revision);
}

bool common_flydelta_teaching_material_runtime::append_contrast_assessment_locked(
        const common_flydelta_teaching_material_group & group, const std::string & verifier_ref,
        const std::vector<std::string> & evidence_refs, bool rejected, std::string & error,
        const std::string & expected_material_revision, const std::string & verifier_revision) {
    const auto & group_ref = group.group_ref;
    if (!assessment_journal_ || group.group_ref.empty() || group.positive_only ||
            !bounded(verifier_ref) || evidence_refs.empty() || evidence_refs.size() > 128 ||
            std::any_of(evidence_refs.begin(), evidence_refs.end(), [](const auto & ref) { return !bounded(ref); })) {
        error = "FlyDelta contrast assessment requires a journal, source group, verifier and evidence";
        return false;
    }
    const auto revision = common_flydelta_teaching_material_revision(group);
    if (expected_material_revision != revision || !bounded(verifier_revision)) {
        error = "FlyDelta contrast assessment material revision is stale or its verifier revision is missing";
        return false;
    }
    const auto payload = nlohmann::ordered_json{{"record_type", "flydelta_contrast_assessment"},
        {"schema_version", 1}, {"group_ref", group_ref}, {"material_revision", revision},
        {"material_refs", group.relation_refs}, {"pending_material_refs", group.pending_relation_refs},
        {"verifier_ref", verifier_ref}, {"verifier_revision", verifier_revision},
        {"material_verifier_revision", group.identity.verifier_revision},
        {"evidence_refs", evidence_refs}, {"rejected", rejected}}.dump();
    common_learning_lifecycle_record record;
    record.subject_id = group_ref;
    record.namespace_id = group.scope.namespace_id;
    record.project_id = group.scope.project_id;
    record.session_id = group.scope.session_id;
    record.kind = common_learning_lifecycle_kind::flydelta_result;
    record.status = rejected ? common_learning_lifecycle_status::rejected : common_learning_lifecycle_status::approved;
    record.content_hash = hash_sha256_hex(payload.data(), payload.size());
    auto records = assessment_journal_->list(error);
    if (!error.empty()) return false;
    const common_learning_lifecycle_record * previous = nullptr;
    for (const auto & prior : records) {
        if (prior.subject_id == group_ref && prior.kind == common_learning_lifecycle_kind::flydelta_result &&
                (!previous || prior.registration_order > previous->registration_order)) previous = &prior;
    }
    if (previous && previous->payload_json == payload) return true;
    const auto event_identity = (previous ? previous->event_id : group_ref) + "\n" + payload;
    record.event_id = "flydelta://contrast-assessment/" + hash_sha256_hex(event_identity.data(), event_identity.size());
    record.idempotency_key = record.event_id;
    record.source_id = verifier_ref;
    record.created_at = "contrast-assessment-v1";
    record.payload_json = payload;
    return assessment_journal_->append(record, error);
}

bool common_flydelta_teaching_material_runtime::contrast_admissible_locked(
        const common_flydelta_teaching_material_group & group, std::string & error,
        const std::string & verifier_revision) const {
    if (group.positive_only || !assessment_journal_) return true;
    auto records = assessment_journal_->list(error);
    if (!error.empty()) return false;
    try {
        const common_learning_lifecycle_record * latest = nullptr;
        for (const auto & record : records) {
            if (record.kind != common_learning_lifecycle_kind::flydelta_result || record.subject_id != group.group_ref) continue;
            const auto value = nlohmann::ordered_json::parse(record.payload_json, nullptr, false);
            if (!value.is_object() || value.value("record_type", "") != "flydelta_contrast_assessment") continue;
            if (record.namespace_id != group.scope.namespace_id || record.project_id != group.scope.project_id ||
                    record.session_id != group.scope.session_id || record.registration_order == 0 ||
                    record.content_hash != hash_sha256_hex(record.payload_json.data(), record.payload_json.size()) ||
                    value.value("group_ref", "") != group.group_ref || value.value("verifier_ref", "") != record.source_id ||
                    !value.contains("evidence_refs") || !value["evidence_refs"].is_array() || value["evidence_refs"].empty() ||
                    !value.contains("rejected") || !value["rejected"].is_boolean() ||
                    (record.status != common_learning_lifecycle_status::rejected && record.status != common_learning_lifecycle_status::approved) ||
                    ((record.status == common_learning_lifecycle_status::rejected) != value["rejected"].get<bool>())) {
                error = "FlyDelta contrast assessment provenance is invalid";
                return false;
            }
            const auto references_valid = [](const json & refs, bool allow_empty = false) {
                return refs.is_array() && (allow_empty || !refs.empty()) && refs.size() <= 128 &&
                    std::all_of(refs.begin(), refs.end(), [](const auto & ref) {
                        return ref.is_string() && bounded(ref.template get<std::string>());
                    });
            };
            auto snapshot = group;
            if (!value.contains("material_refs") || !references_valid(value.at("material_refs"), true) ||
                    !value.contains("pending_material_refs") || !references_valid(value.at("pending_material_refs"), true) ||
                    (value.at("material_refs").empty() && value.at("pending_material_refs").empty()) ||
                    !references_valid(value.at("evidence_refs"))) {
                error = "FlyDelta contrast assessment source/evidence references are invalid";
                return false;
            }
            snapshot.relation_refs = value.at("material_refs").get<std::vector<std::string>>();
            snapshot.pending_relation_refs = value.at("pending_material_refs").get<std::vector<std::string>>();
            if (common_flydelta_teaching_material_revision(snapshot) != value.value("material_revision", "")) {
                error = "FlyDelta contrast assessment material revision does not match its immutable source references";
                return false;
            }
            if (value.value("schema_version", 0) != 1 || value.value("material_verifier_revision", "") != group.identity.verifier_revision ||
                    value.value("verifier_revision", "").empty()) {
                error = "FlyDelta contrast assessment has an incompatible contract or verifier revision";
                return false;
            }
            if (!latest || record.registration_order > latest->registration_order) latest = &record;
        }
        if (!latest) return true;
        const auto value = nlohmann::ordered_json::parse(latest->payload_json, nullptr, false);
        // A revision change alone never clears a rejection. The host must append
        // an explicit assessment of the new material with its verifier/evidence.
        if (value.value("rejected", true) || (!verifier_revision.empty() && value.value("verifier_revision", "") != verifier_revision) ||
                ((!group.relation_refs.empty() || !group.pending_relation_refs.empty()) &&
                    value.value("material_revision", "") != common_flydelta_teaching_material_revision(group))) {
            error = "FlyDelta contrast was rejected or changed after assessment; explicit reassessment is required";
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        error = std::string("FlyDelta contrast assessment is malformed: ") + e.what();
        return false;
    }
}

bool common_flydelta_teaching_material_runtime::contrast_admissible(
        const common_flydelta_teaching_material_group & group, std::string & error,
        const std::string & verifier_revision) const {
    std::lock_guard<std::mutex> lock(mutex_);
    error.clear();
    return contrast_admissible_locked(group, error, verifier_revision);
}

bool common_flydelta_teaching_material_runtime::observe_relation(
        const common_flydelta_teaching_relation & relation,
        std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    error.clear();
    if (relation.status == common_flydelta_teaching_relation_status::rejected) {
        if (relation.schema_version != 1 || relation.positive_only || !relation.host_approved ||
                !bounded(relation.id) || !bounded(relation.teaching_key) || !bounded(relation.behavior_key) ||
                !bounded(relation.scope.namespace_id) || !bounded(relation.scope.session_id) ||
                !bounded(relation.verifier_ref) || !bounded(relation.evidence_ref) ||
                (!relation.model_profile_fingerprint.empty() && relation.model_profile_fingerprint != identity_.model_profile_fingerprint) ||
                (!relation.tokenizer_fingerprint.empty() && relation.tokenizer_fingerprint != identity_.tokenizer_fingerprint) ||
                (!relation.template_fingerprint.empty() && relation.template_fingerprint != identity_.template_fingerprint) ||
                (!relation.capture_layout_revision.empty() && relation.capture_layout_revision != identity_.capture_layout_revision)) {
            error = "FlyDelta contrast rejection requires a bounded host-authorized source assessment";
            return false;
        }
        common_flydelta_teaching_material_group rejected;
        rejected.scope = relation.scope;
        rejected.scope.turn_id.clear();
        rejected.teaching_key = relation.teaching_key;
        rejected.behavior_key = relation.behavior_key;
        rejected.identity = identity_;
        rejected.compatibility_key = common_flydelta_teaching_material_compatibility_key(identity_);
        rejected.group_ref = material_family_ref(rejected);
        rejected.relation_refs = {relation.id};
        return append_contrast_assessment_locked(rejected, relation.verifier_ref,
            {relation.id, relation.evidence_ref}, true, error, common_flydelta_teaching_material_revision(rejected),
            relation.oracle_revision.empty() ? identity_.verifier_revision : relation.oracle_revision);
    }
    common_flydelta_teaching_material_store next = store_;
    if (!next.observe(relation, identity_, error)) return false;
    if (!persist(next, error)) return false;
    store_ = std::move(next);
    return true;
}

bool common_flydelta_teaching_material_runtime::load(std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    error.clear();
    if (!persistence_.load) return true;
    std::string snapshot;
    if (!persistence_.load(snapshot, error)) return false;
    if (snapshot.empty()) return true;
    common_flydelta_teaching_material_store next;
    if (!common_flydelta_teaching_material_store_from_json(snapshot, next, error)) return false;
    store_ = std::move(next);
    return true;
}

bool common_flydelta_teaching_material_runtime::persist(
        const common_flydelta_teaching_material_store & store,
        std::string & error) const {
    error.clear();
    if (!persistence_.save) return true;
    return persistence_.save(common_flydelta_teaching_material_store_to_json(store), error);
}

bool common_flydelta_teaching_material_runtime::inspect_group(
        const std::string & group_ref,
        bool & relation_set_ready,
        bool & trajectory_material_ready,
        std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    relation_set_ready = false;
    trajectory_material_ready = false;
    common_flydelta_teaching_material_group group;
    if (!store_.resolve_relation_set(group_ref, group, error)) return false;
    relation_set_ready = group.relation_set_ready;
    if (!store_.resolve_ready_group(group_ref, group, error)) return false;
    trajectory_material_ready = group.trajectory_material_ready;
    return true;
}

bool common_flydelta_teaching_material_runtime::resolve_group_for_behavior(
        const std::string & behavior_key,
        common_flydelta_teaching_material_group & group,
        bool & available,
        std::string & error,
        common_flydelta_teaching_material_mode mode, const common_agent_scope * scope) const {
    std::lock_guard<std::mutex> lock(mutex_);
    error.clear();
    group = {};
    available = false;
    if (!bounded(behavior_key)) {
        error = "FlyDelta teaching-material family behavior key is invalid";
        return false;
    }
    const std::string compatibility_key =
        common_flydelta_teaching_material_compatibility_key(identity_);
    if (compatibility_key.empty()) {
        error = "FlyDelta teaching-material compatibility identity is invalid";
        return false;
    }
    const common_flydelta_teaching_material_group * match = nullptr;
    for (const auto & candidate : store_.groups()) {
        if (scope && (candidate.scope.namespace_id != scope->namespace_id ||
                candidate.scope.project_id != scope->project_id || candidate.scope.session_id != scope->session_id)) continue;
        if (mode == common_flydelta_teaching_material_mode::automatic && candidate.positive_only &&
                candidate.behavior_key == behavior_key && candidate.compatibility_key == compatibility_key) {
            auto contrast = candidate;
            contrast.positive_only = false;
            contrast.group_ref = material_family_ref(contrast);
            contrast.relation_refs.clear();
            contrast.pending_relation_refs.clear();
            if (!contrast_admissible_locked(contrast, error)) return false;
        }
        if (mode != common_flydelta_teaching_material_mode::positive_only && !candidate.positive_only &&
                candidate.behavior_key == behavior_key && candidate.compatibility_key == compatibility_key &&
                !contrast_admissible_locked(candidate, error)) return false;
        if (!candidate.relation_set_ready ||
                (mode == common_flydelta_teaching_material_mode::contrast && candidate.positive_only) ||
                (mode == common_flydelta_teaching_material_mode::positive_only && !candidate.positive_only) ||
                candidate.behavior_key != behavior_key ||
                candidate.compatibility_key != compatibility_key) continue;
        if (match != nullptr) {
            // The two synthesis modes for the same family are alternatives,
            // not an ambiguous family. Prefer the established contrast mode.
            if (match->teaching_key == candidate.teaching_key &&
                    match->positive_only != candidate.positive_only) {
                if (match->positive_only) match = &candidate;
                continue;
            }
            error = "FlyDelta teaching-material family is ambiguous for behavior key";
            return false;
        }
        match = &candidate;
    }
    if (match == nullptr) return true;
    group = *match;
    available = true;
    return true;
}

bool common_flydelta_teaching_material_runtime::resolve_group_for_family(
        const std::string & teaching_key,
        const std::string & behavior_key,
        common_flydelta_teaching_material_group & group,
        bool & available,
        std::string & error,
        common_flydelta_teaching_material_mode mode, const common_agent_scope * scope) const {
    std::lock_guard<std::mutex> lock(mutex_);
    error.clear();
    group = {};
    available = false;
    if (!bounded(teaching_key) || !bounded(behavior_key)) {
        error = "FlyDelta teaching-material family identity is invalid";
        return false;
    }
    const std::string compatibility_key =
        common_flydelta_teaching_material_compatibility_key(identity_);
    if (compatibility_key.empty()) {
        error = "FlyDelta teaching-material compatibility identity is invalid";
        return false;
    }
    const common_flydelta_teaching_material_group * match = nullptr;
    for (const auto & candidate : store_.groups()) {
        if (scope && (candidate.scope.namespace_id != scope->namespace_id ||
                candidate.scope.project_id != scope->project_id || candidate.scope.session_id != scope->session_id)) continue;
        if (mode == common_flydelta_teaching_material_mode::automatic && candidate.positive_only &&
                candidate.teaching_key == teaching_key && candidate.behavior_key == behavior_key && candidate.compatibility_key == compatibility_key) {
            auto contrast = candidate;
            contrast.positive_only = false;
            contrast.group_ref = material_family_ref(contrast);
            contrast.relation_refs.clear();
            contrast.pending_relation_refs.clear();
            if (!contrast_admissible_locked(contrast, error)) return false;
        }
        if (mode != common_flydelta_teaching_material_mode::positive_only && !candidate.positive_only &&
                candidate.teaching_key == teaching_key && candidate.behavior_key == behavior_key &&
                candidate.compatibility_key == compatibility_key &&
                !contrast_admissible_locked(candidate, error)) return false;
        if (!candidate.relation_set_ready || candidate.teaching_key != teaching_key ||
                (mode == common_flydelta_teaching_material_mode::contrast && candidate.positive_only) ||
                (mode == common_flydelta_teaching_material_mode::positive_only && !candidate.positive_only) ||
                candidate.behavior_key != behavior_key || candidate.compatibility_key != compatibility_key) continue;
        if (!match || (match->positive_only && !candidate.positive_only)) match = &candidate;
    }
    if (!match) return true;
    group = *match;
    available = true;
    return true;
}

bool common_flydelta_teaching_material_runtime::observe_trajectory(
        const std::string & group_ref,
        const std::string & trajectory_ref,
        std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    common_flydelta_teaching_material_store next = store_;
    if (!next.observe_trajectory(group_ref, trajectory_ref, identity_, error)) return false;
    if (!persist(next, error)) return false;
    store_ = std::move(next);
    return true;
}

bool common_flydelta_teaching_material_runtime::resolve_relation_set(
        const std::string & ref, common_flydelta_teaching_material_group & group, std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return store_.resolve_relation_set(ref, group, error);
}

bool common_flydelta_teaching_material_runtime::resolve_ready_group(
        const std::string & ref, common_flydelta_teaching_material_group & group, std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return store_.resolve_ready_group(ref, group, error);
}

bool common_flydelta_teaching_material_runtime::resolve_layer_group(
        const std::string & ref, int32_t layer, common_flydelta_teaching_material_group & group,
        std::string & error, const std::string & capture_set_ref) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return store_.resolve_layer_group(ref, layer, group, error, capture_set_ref);
}

bool common_flydelta_teaching_material_store::resolve_ready_group(
        const std::string & group_ref,
        common_flydelta_teaching_material_group & group,
        std::string & error) const {
    error.clear();
    group = {};
    if (!bounded(group_ref)) {
        error = "FlyDelta teaching material group reference is invalid";
        return false;
    }
    const auto it = std::find_if(groups_.begin(), groups_.end(), [&](const auto & candidate) {
        return candidate.group_ref == group_ref;
    });
    if (it == groups_.end() || !it->trajectory_material_ready ||
            it->trajectory_refs.size() < it->minimum_trajectories) return true;
    group = *it;
    return true;
}

bool common_flydelta_teaching_material_store::resolve_relation_set(
        const std::string & group_ref,
        common_flydelta_teaching_material_group & group,
        std::string & error) const {
    error.clear();
    group = {};
    if (!bounded(group_ref)) {
        error = "FlyDelta teaching material group reference is invalid";
        return false;
    }
    const auto it = std::find_if(groups_.begin(), groups_.end(), [&](const auto & candidate) {
        return candidate.group_ref == group_ref;
    });
    if (it == groups_.end() || !it->relation_set_ready ||
            it->relation_refs.size() < it->minimum_relations) return true;
    group = *it;
    return true;
}

std::string common_flydelta_teaching_material_store_to_json(
        const common_flydelta_teaching_material_store & store) {
    json groups = json::array();
    for (const auto & group : store.groups()) {
        json layer_trajectories = json::array();
        for (const auto & entry : group.layer_trajectories) {
            layer_trajectories.push_back({{"layer_index", entry.layer_index}, {"relation_ref", entry.relation_ref},
                {"trajectory_ref", entry.trajectory_ref}, {"capture_set_ref", entry.capture_set_ref}});
        }
        groups.push_back({
            {"group_ref", group.group_ref},
            {"scope", {{"namespace_id", group.scope.namespace_id}, {"project_id", group.scope.project_id},
                {"session_id", group.scope.session_id}}},
            {"teaching_key", group.teaching_key},
            {"behavior_key", group.behavior_key},
            {"compatibility_key", group.compatibility_key},
            {"identity", {
                {"schema_version", group.identity.schema_version},
                {"model_profile_fingerprint", group.identity.model_profile_fingerprint},
                {"tokenizer_fingerprint", group.identity.tokenizer_fingerprint},
                {"template_fingerprint", group.identity.template_fingerprint},
                {"capture_layout_revision", group.identity.capture_layout_revision},
                {"execution_context_fingerprint", group.identity.execution_context_fingerprint},
                {"scope_fingerprint", group.identity.scope_fingerprint},
                {"verifier_revision", group.identity.verifier_revision},
            }},
            {"relation_refs", group.relation_refs},
            {"pending_relation_refs", group.pending_relation_refs},
            {"trajectory_refs", group.trajectory_refs},
            {"layer_trajectories", std::move(layer_trajectories)},
            {"positive_only", group.positive_only},
            {"minimum_relations", group.minimum_relations},
            {"minimum_trajectories", group.minimum_trajectories},
            {"relation_set_ready", group.relation_set_ready},
            {"trajectory_material_ready", group.trajectory_material_ready},
        });
    }
    return json{{"kind", "flydelta_teaching_material_store"},
        {"schema_version", 1}, {"groups", std::move(groups)}}.dump();
}

bool common_flydelta_teaching_material_store_from_json(
        const std::string & text,
        common_flydelta_teaching_material_store & store,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (value.value("kind", "") != "flydelta_teaching_material_store" ||
                value.value("schema_version", 0) != 1 ||
                !value.at("groups").is_array()) {
            error = "FlyDelta teaching material store JSON identity is invalid";
            return false;
        }
        common_flydelta_teaching_material_store decoded;
        for (const auto & item : value.at("groups")) {
            common_flydelta_teaching_material_group group;
            group.group_ref = item.value("group_ref", "");
            const auto scope = item.value("scope", json::object());
            group.scope.namespace_id = scope.value("namespace_id", "");
            group.scope.project_id = scope.value("project_id", "");
            group.scope.session_id = scope.value("session_id", "");
            group.teaching_key = item.value("teaching_key", "");
            group.behavior_key = item.value("behavior_key", "");
            group.compatibility_key = item.value("compatibility_key", "");
            const auto identity = item.value("identity", json::object());
            group.identity.schema_version = identity.value("schema_version", 0);
            group.identity.model_profile_fingerprint = identity.value("model_profile_fingerprint", "");
            group.identity.tokenizer_fingerprint = identity.value("tokenizer_fingerprint", "");
            group.identity.template_fingerprint = identity.value("template_fingerprint", "");
            group.identity.capture_layout_revision = identity.value("capture_layout_revision", "");
            group.identity.execution_context_fingerprint = identity.value("execution_context_fingerprint", "");
            group.identity.scope_fingerprint = identity.value("scope_fingerprint", "");
            group.identity.verifier_revision = identity.value("verifier_revision", "");
            group.relation_refs = item.value("relation_refs", std::vector<std::string>{});
            group.pending_relation_refs = item.value("pending_relation_refs", std::vector<std::string>{});
            group.trajectory_refs = item.value("trajectory_refs", std::vector<std::string>{});
            group.positive_only = item.value("positive_only", false);
            group.minimum_relations = item.value("minimum_relations", 2U);
            // Snapshots written before the relation/trajectory split used the
            // single `ready` bit for relation-set readiness. Preserve that
            // meaning when resuming an older store, but never infer complete
            // trajectory material from it.
            group.minimum_trajectories = item.value("minimum_trajectories", 2U);
            group.relation_set_ready = item.value(
                "relation_set_ready", item.value("ready", false));
            group.trajectory_material_ready = item.value("trajectory_material_ready", false);
            if (group.relation_set_ready &&
                    group.relation_refs.size() < group.minimum_relations) {
                group.relation_set_ready = false;
            }
            if (group.trajectory_material_ready &&
                    group.trajectory_refs.size() < group.minimum_trajectories) {
                group.trajectory_material_ready = false;
            }
            if (!bounded(group.group_ref) || !bounded(group.teaching_key) ||
                    !bounded(group.behavior_key) || group.compatibility_key.empty() ||
                    (group.relation_refs.empty() && group.pending_relation_refs.empty()) || group.relation_refs.size() > 128 ||
                    group.pending_relation_refs.size() > 128 || (group.positive_only && !group.pending_relation_refs.empty()) ||
                    group.trajectory_refs.size() > 128 || group.minimum_relations < 2 ||
                    group.minimum_trajectories < 2 ||
                    (group.positive_only && (group.minimum_relations < 3 || group.minimum_trajectories < 3)) ||
                    !common_flydelta_teaching_material_identity_validate(group.identity, error)) {
                if (error.empty()) error = "FlyDelta teaching material store group is invalid";
                return false;
            }
            if (group.compatibility_key != common_flydelta_teaching_material_compatibility_key(group.identity)) {
                error = "FlyDelta teaching material store compatibility key differs from its identity";
                return false;
            }
            for (const auto & ref : group.pending_relation_refs) {
                if (!bounded(ref) || std::find(group.relation_refs.begin(), group.relation_refs.end(), ref) != group.relation_refs.end()) {
                    error = "FlyDelta pending material reference is invalid or already capture-ready";
                    return false;
                }
            }
            for (const auto & ref : group.relation_refs) {
                if (!bounded(ref)) {
                    error = "FlyDelta teaching material store relation reference is invalid";
                    return false;
                }
            }
            for (const auto & ref : group.trajectory_refs) {
                if (!bounded(ref)) {
                    error = "FlyDelta teaching material trajectory reference is invalid";
                    return false;
                }
            }
            const auto layer_entries = item.value("layer_trajectories", json::array());
            if (!layer_entries.is_array() || layer_entries.size() > 4096) {
                error = "FlyDelta layer trajectory snapshot is invalid";
                return false;
            }
            decoded.groups_.push_back(std::move(group));
            const auto & saved_group = decoded.groups_.back();
            const std::string saved_ref = saved_group.group_ref;
            const auto saved_identity = saved_group.identity;
            for (const auto & entry : layer_entries) {
                if (!decoded.observe_layer_trajectory(saved_ref, entry.value("layer_index", 0),
                        entry.value("relation_ref", ""), entry.value("trajectory_ref", ""), saved_identity,
                        error, entry.value("capture_set_ref", ""))) return false;
            }
        }
        store = std::move(decoded);
        return true;
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta teaching material store JSON: ") + exception.what();
        return false;
    }
}
