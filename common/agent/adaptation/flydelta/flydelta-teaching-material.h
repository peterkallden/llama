#pragma once

#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// Host-owned compatibility identity for reusable teaching material.  The
// store keeps references only; it never owns prompts, activations or model
// contexts.
struct common_flydelta_teaching_material_identity {
    int schema_version = 1;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string execution_context_fingerprint;
    std::string scope_fingerprint;
    std::string verifier_revision;
};

bool common_flydelta_teaching_material_identity_validate(
        const common_flydelta_teaching_material_identity & identity,
        std::string & error);

std::string common_flydelta_teaching_material_compatibility_key(
        const common_flydelta_teaching_material_identity & identity);

struct common_flydelta_teaching_material_group {
    std::string group_ref;
    std::string teaching_key;
    std::string behavior_key;
    std::string compatibility_key;
    common_flydelta_teaching_material_identity identity;
    std::vector<std::string> relation_refs;
    std::vector<std::string> trajectory_refs;
    size_t minimum_relations = 2;
    size_t minimum_trajectories = 2;
    bool relation_set_ready = false;
    bool trajectory_material_ready = false;
};

// A small idempotent host-side accumulator.  It is deliberately independent
// of inference and can later be backed by the existing artifact store.
class common_flydelta_teaching_material_store {
public:
    bool observe(
            const common_flydelta_teaching_relation & relation,
            const common_flydelta_teaching_material_identity & identity,
            std::string & error);

    bool group_ready(
            const std::string & teaching_key,
            const std::string & behavior_key,
            const common_flydelta_teaching_material_identity & identity,
            size_t minimum_relations,
            common_flydelta_teaching_material_group & group,
            std::string & error) const;

    bool observe_trajectory(
            const std::string & group_ref,
            const std::string & trajectory_ref,
            const common_flydelta_teaching_material_identity & identity,
            std::string & error);

    // Resolves only a group with complete trajectory material. Relation and
    // trajectory payloads remain in the host's existing artifact store.
    bool resolve_ready_group(
            const std::string & group_ref,
            common_flydelta_teaching_material_group & group,
            std::string & error) const;

    // Resolves a relation-ready group for the bounded capture-planning step.
    bool resolve_relation_set(
            const std::string & group_ref,
            common_flydelta_teaching_material_group & group,
            std::string & error) const;

    const std::vector<common_flydelta_teaching_material_group> & groups() const {
        return groups_;
    }

private:
    friend bool common_flydelta_teaching_material_store_from_json(
            const std::string &, common_flydelta_teaching_material_store &, std::string &);
    std::vector<common_flydelta_teaching_material_group> groups_;
};

// Host-owned façade used when the same material index must be observed by
// the runtime learning path and inspected by the FlyDelta model host. It
// keeps the compatibility identity in one place and still stores references,
// not prompts, tensors or model contexts.
class common_flydelta_teaching_material_runtime {
public:
    explicit common_flydelta_teaching_material_runtime(
            common_flydelta_teaching_material_identity identity)
        : identity_(std::move(identity)) {}

    bool observe_relation(
            const common_flydelta_teaching_relation & relation,
            std::string & error);
    bool inspect_group(
            const std::string & group_ref,
            bool & relation_set_ready,
            bool & trajectory_material_ready,
            std::string & error) const;
    bool observe_trajectory(
            const std::string & group_ref,
            const std::string & trajectory_ref,
            std::string & error);

    const common_flydelta_teaching_material_identity & identity() const {
        return identity_;
    }
    const common_flydelta_teaching_material_store & store() const {
        return store_;
    }
    common_flydelta_teaching_material_store & store() {
        return store_;
    }

private:
    common_flydelta_teaching_material_identity identity_;
    common_flydelta_teaching_material_store store_;
};

// Reference-only snapshot helpers. The host may place this JSON in the
// existing lifecycle/artifact persistence layer; this component does not
// create a second filesystem store.
std::string common_flydelta_teaching_material_store_to_json(
        const common_flydelta_teaching_material_store & store);
bool common_flydelta_teaching_material_store_from_json(
        const std::string & text,
        common_flydelta_teaching_material_store & store,
        std::string & error);
