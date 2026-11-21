#pragma once

#include "agent/adaptation/learning-transaction.h"
#include "agent/adaptation/lifecycle-store.h"
#include "agent/adaptation/semantic-teaching.h"

#include <cstddef>
#include <string>
#include <vector>

// A reference-only projection over the existing adaptation lifecycle journal.
// It is deliberately not a second evidence or learning store.
struct common_agent_concept_candidate {
    int schema_version = 1;
    size_t revision = 0;
    std::string candidate_id;
    std::string concept_key;
    std::string canonical_statement_ref;
    common_agent_scope scope;
    std::vector<std::string> source_refs;
    std::vector<std::string> supporting_evidence_refs;
    std::vector<std::string> disconfirming_evidence_refs;
    std::vector<std::string> relation_refs;
    std::vector<std::string> independent_support_keys;
    std::string grounding_state = "proposed";
    std::string novelty_state = "new_concept";
    std::string source_strength;
    std::string contrast_strength;
    std::string independence;
    // Host-owned compatibility metadata. These are identities only; raw
    // prompts, captures and tool output never enter the candidate index.
    std::string behavior_key;
    std::string applicability_scope_fingerprint;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string oracle_revision;
    bool synthesis_eligible = false;
};

struct common_agent_concept_candidate_family_query {
    std::string concept_key;
    std::string namespace_id;
    std::string project_id;
    std::string behavior_key;
    std::string applicability_scope_fingerprint;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string oracle_revision;
};

// A read-only projection over compatible session candidates. It deliberately
// contains only opaque refs and identity metadata, never source material.
struct common_agent_concept_candidate_family {
    std::string concept_key;
    std::string namespace_id;
    std::string project_id;
    std::vector<common_agent_concept_candidate> members;
};

bool common_agent_concept_candidate_validate(
        const common_agent_concept_candidate & candidate,
        std::string & error);
std::string common_agent_concept_candidate_to_json(
        const common_agent_concept_candidate & candidate);
bool common_agent_concept_candidate_from_json(
        const std::string & text,
        common_agent_concept_candidate & candidate,
        std::string & error);

class common_agent_concept_candidate_index {
public:
    explicit common_agent_concept_candidate_index(
            common_learning_lifecycle_store * lifecycle_store = nullptr);

    bool load(std::string & error);

    bool observe_hypothesis(
            const common_agent_concept_hypothesis & hypothesis,
            const common_learning_transaction & transaction,
            std::string & error);

    bool observe_relation(
            const common_agent_concept_hypothesis & hypothesis,
            const common_flydelta_teaching_relation & relation,
            const common_learning_transaction & transaction,
            std::string & error);

    // Index a host-resolved relation from an explicit adapter without
    // creating a second semantic hypothesis persistence path.
    bool observe_resolved_relation(
            const common_flydelta_teaching_relation & relation,
            const common_learning_transaction & transaction,
            std::string & error);

    bool observe_disconfirmation(
            const common_agent_concept_hypothesis & hypothesis,
            const std::string & evidence_ref,
            const common_learning_transaction & transaction,
            std::string & error);

    bool resolve(
            const std::string & concept_key,
            const common_agent_scope & scope,
            common_agent_concept_candidate & candidate,
            std::string & error) const;

    bool find_project_reusable_family(
            const common_agent_concept_candidate_family_query & query,
            common_agent_concept_candidate_family & family,
            std::string & error) const;

    const std::vector<common_agent_concept_candidate> & candidates() const {
        return candidates_;
    }

private:
    common_agent_concept_candidate & find_or_create(
            const common_agent_concept_hypothesis & hypothesis);
    bool persist(
            common_agent_concept_candidate & candidate,
            const common_learning_transaction & transaction,
            std::string & error);

    common_learning_lifecycle_store * lifecycle_store_ = nullptr;
    std::vector<common_agent_concept_candidate> candidates_;
};
