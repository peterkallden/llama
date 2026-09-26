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
    const std::string compatibility_key =
        common_flydelta_teaching_material_compatibility_key(identity);
    if (compatibility_key.empty()) {
        error = "FlyDelta teaching material compatibility key is invalid";
        return false;
    }
    const auto same_group = [&](const common_flydelta_teaching_material_group & group) {
        return group.teaching_key == relation.teaching_key &&
            group.behavior_key == relation.behavior_key &&
            group.compatibility_key == compatibility_key;
    };
    auto it = std::find_if(groups_.begin(), groups_.end(), same_group);
    if (it == groups_.end()) {
        common_flydelta_teaching_material_group group;
        group.teaching_key = relation.teaching_key;
        group.behavior_key = relation.behavior_key;
        group.compatibility_key = compatibility_key;
        group.identity = identity;
        const std::string group_identity = compatibility_key + "\n" +
            relation.teaching_key + "\n" + relation.behavior_key;
        group.group_ref = "flydelta://teaching-material/" +
            hash_sha256_hex(group_identity.data(), group_identity.size());
        it = groups_.insert(groups_.end(), std::move(group));
    }
    if (std::find(it->relation_refs.begin(), it->relation_refs.end(), relation.id) ==
            it->relation_refs.end()) {
        it->relation_refs.push_back(relation.id);
    }
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
    const auto it = std::find_if(groups_.begin(), groups_.end(), [&](const auto & candidate) {
        return candidate.teaching_key == teaching_key &&
            candidate.behavior_key == behavior_key && candidate.compatibility_key == key;
    });
    if (it == groups_.end() || it->relation_refs.size() < minimum_relations) return true;
    group = *it;
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

bool common_flydelta_teaching_material_runtime::observe_relation(
        const common_flydelta_teaching_relation & relation,
        std::string & error) {
    common_flydelta_teaching_material_store next = store_;
    if (!next.observe(relation, identity_, error)) return false;
    if (!persist(next, error)) return false;
    store_ = std::move(next);
    return true;
}

bool common_flydelta_teaching_material_runtime::load(std::string & error) {
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
        std::string & error) const {
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
        if (!candidate.relation_set_ready ||
                candidate.behavior_key != behavior_key ||
                candidate.compatibility_key != compatibility_key) continue;
        if (match != nullptr) {
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
        std::string & error) const {
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
    const auto it = std::find_if(store_.groups().begin(), store_.groups().end(), [&](const auto & candidate) {
        return candidate.relation_set_ready && candidate.teaching_key == teaching_key &&
            candidate.behavior_key == behavior_key && candidate.compatibility_key == compatibility_key;
    });
    if (it == store_.groups().end()) return true;
    group = *it;
    available = true;
    return true;
}

bool common_flydelta_teaching_material_runtime::observe_trajectory(
        const std::string & group_ref,
        const std::string & trajectory_ref,
        std::string & error) {
    common_flydelta_teaching_material_store next = store_;
    if (!next.observe_trajectory(group_ref, trajectory_ref, identity_, error)) return false;
    if (!persist(next, error)) return false;
    store_ = std::move(next);
    return true;
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
        groups.push_back({
            {"group_ref", group.group_ref},
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
            {"trajectory_refs", group.trajectory_refs},
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
            group.trajectory_refs = item.value("trajectory_refs", std::vector<std::string>{});
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
                    group.relation_refs.empty() || group.relation_refs.size() > 128 ||
                    group.trajectory_refs.size() > 128 || group.minimum_relations < 2 ||
                    group.minimum_trajectories < 2 ||
                    !common_flydelta_teaching_material_identity_validate(group.identity, error)) {
                if (error.empty()) error = "FlyDelta teaching material store group is invalid";
                return false;
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
            decoded.groups_.push_back(std::move(group));
        }
        store = std::move(decoded);
        return true;
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta teaching material store JSON: ") + exception.what();
        return false;
    }
}
