#include "agent/adaptation/concept-candidate-index.h"

#include <cassert>

static common_agent_concept_hypothesis hypothesis() {
    common_agent_concept_hypothesis value;
    value.id = "hypothesis:grouped-sum";
    value.concept_key = "dataset.grouped_sum";
    value.statement = "Group totals by the requested dimension.";
    value.canonical_statement_ref = "statement://dataset/grouped-sum";
    value.source_kind = common_agent_concept_source_kind::user_taught_concept;
    value.status = common_agent_concept_hypothesis_status::grounded;
    value.scope.namespace_id = "candidate-index-test";
    value.scope.session_id = "session";
    value.source_refs = {"turn://one"};
    value.confidence = 1.0f;
    value.host_grounded = true;
    value.reusable = true;
    return value;
}

static common_flydelta_teaching_relation relation(const std::string & id, const std::string & task) {
    common_flydelta_teaching_relation value;
    value.id = id;
    value.teaching_key = "dataset.grouped_sum";
    value.source = common_adaptation_evidence_source::user_taught_concept;
    value.behavior_key = "dataset/grouped_sum";
    value.scope.namespace_id = "candidate-index-test";
    value.scope.session_id = "session";
    value.task_fingerprint = task;
    value.baseline_ref = "capture://" + id + "/baseline";
    value.conditioned_ref = "capture://" + id + "/conditioned";
    value.control_ref = "capture://" + id + "/control";
    value.verifier_ref = "verifier://grouped-sum";
    value.evidence_ref = "evidence://candidate-index-test";
    value.contrast_ref = "contrast://" + id;
    value.status = common_flydelta_teaching_relation_status::resolved;
    value.baseline_origin = common_flydelta_teaching_origin::observed;
    value.conditioned_origin = common_flydelta_teaching_origin::host_derived;
    value.control_origin = common_flydelta_teaching_origin::host_counterfactual;
    value.confidence = 1.0f;
    value.host_approved = true;
    return value;
}

static common_agent_concept_grounding grounding() {
    common_agent_concept_grounding value;
    value.id = "evidence://candidate-index-test";
    value.hypothesis_ref = "hypothesis:grouped-sum";
    value.behavior_key = "dataset/grouped_sum";
    value.verifier_ref = "verifier://grouped-sum";
    value.changed_dimension = "aggregation";
    value.scope.namespace_id = "candidate-index-test";
    value.scope.session_id = "session";
    value.minimum_contrasts = 2;
    value.host_approved = true;
    value.reusable = true;
    return value;
}

static common_learning_transaction transaction(const std::string & id) {
    common_learning_transaction value;
    value.id = id;
    value.created_at = "2026-09-26T00:00:00Z";
    value.observation.id = id + ":observation";
    value.observation.scope.namespace_id = "candidate-index-test";
    value.observation.scope.session_id = "session";
    value.observation.content_hash = "hash:" + id;
    return value;
}

int main() {
    common_learning_in_memory_lifecycle_store lifecycle;
    common_agent_concept_candidate_index index(&lifecycle);
    std::string error;
    const auto concept = hypothesis();
    assert(index.observe_hypothesis(concept, transaction("turn:one"), error));
    assert(index.observe_hypothesis(concept, transaction("turn:one"), error));

    common_agent_concept_candidate candidate;
    assert(index.resolve(concept.concept_key, concept.scope, candidate, error));
    assert(candidate.source_refs.size() == 1);
    assert(candidate.relation_refs.empty());
    assert(!candidate.synthesis_eligible);

    const auto first_relation = relation("relation:one", "task:one");
    const auto second_relation = relation("relation:two", "task:two");
    assert(common_agent_validate_concept_teaching_relation(
        concept, grounding(), first_relation, error));
    assert(!common_agent_validate_concept_teaching_relations(
        concept, grounding(), {first_relation}, error));
    assert(common_agent_validate_concept_teaching_relations(
        concept, grounding(), {first_relation, second_relation}, error));

    assert(index.observe_relation(concept, first_relation,
        transaction("turn:one"), error));
    assert(index.resolve(concept.concept_key, concept.scope, candidate, error));
    assert(candidate.relation_refs.size() == 1);
    assert(!candidate.synthesis_eligible);

    assert(index.observe_relation(concept, first_relation,
        transaction("turn:one"), error));
    assert(index.resolve(concept.concept_key, concept.scope, candidate, error));
    assert(candidate.relation_refs.size() == 1);
    assert(index.observe_relation(concept, second_relation,
        transaction("turn:two"), error));
    assert(index.resolve(concept.concept_key, concept.scope, candidate, error));
    assert(candidate.relation_refs.size() == 2);
    assert(candidate.independent_support_keys.size() == 2);
    assert(candidate.synthesis_eligible);

    assert(index.observe_disconfirmation(concept, "evidence://counterexample",
        transaction("turn:three"), error));
    assert(index.resolve(concept.concept_key, concept.scope, candidate, error));
    assert(candidate.novelty_state == "conflict");
    assert(!candidate.synthesis_eligible);

    common_agent_concept_candidate_index restored(&lifecycle);
    assert(restored.load(error));
    assert(restored.resolve(concept.concept_key, concept.scope, candidate, error));
    assert(candidate.novelty_state == "conflict");
    return 0;
}
