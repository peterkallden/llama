#pragma once

#include "agent/agent-scope.h"
#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

// Semantic preparation is host-side evidence preparation.  It is deliberately
// separate from FlyDelta representation synthesis: a hypothesis is not an
// observation, and a grounded hypothesis is not a TeachingRelation until the
// host has verified a concrete contrast.
enum class common_agent_concept_semantic_kind {
    fact,
    procedure,
    decision_rule,
    heuristic,
    concept,
};

const char * common_agent_concept_semantic_kind_name(
        common_agent_concept_semantic_kind kind);

enum class common_agent_concept_source_kind {
    conversation,
    research,
    user_correction,
    user_taught_concept,
    reflection,
};

const char * common_agent_concept_source_kind_name(
        common_agent_concept_source_kind kind);

enum class common_agent_concept_hypothesis_status {
    proposed,
    grounded,
    rejected,
    insufficient_evidence,
};

const char * common_agent_concept_hypothesis_status_name(
        common_agent_concept_hypothesis_status status);

struct common_agent_concept_transition {
    std::string state_before;
    std::string condition;
    std::string required_transition;
    std::string state_after;
};

// A bounded, host-persistable hypothesis. It may be produced from research,
// reflection or conversation, but it is never itself adaptation evidence.
struct common_agent_concept_hypothesis {
    int schema_version = 1;
    std::string id;
    std::string concept_key;
    std::string statement;
    common_agent_concept_semantic_kind semantic_kind =
        common_agent_concept_semantic_kind::concept;
    common_agent_concept_source_kind source_kind =
        common_agent_concept_source_kind::conversation;
    common_agent_concept_hypothesis_status status =
        common_agent_concept_hypothesis_status::proposed;
    common_agent_scope scope;
    std::vector<std::string> preconditions;
    std::vector<common_agent_concept_transition> transitions;
    std::vector<std::string> invariants;
    std::vector<std::string> counterexamples;
    std::vector<std::string> source_refs;
    float confidence = 0.0f;
    bool host_grounded = false;
    bool reusable = false;
};

bool common_agent_concept_hypothesis_validate(
        const common_agent_concept_hypothesis & hypothesis,
        std::string & error);

std::string common_agent_concept_hypothesis_to_json(
        const common_agent_concept_hypothesis & hypothesis);

bool common_agent_concept_hypothesis_from_json(
        const std::string & text,
        common_agent_concept_hypothesis & hypothesis,
        std::string & error);

// Host-approved grounding metadata. The grounding points to the semantic
// verifier and fixture family; prompts and model contexts remain opaque refs.
struct common_agent_concept_grounding {
    int schema_version = 1;
    std::string id;
    std::string hypothesis_ref;
    std::string behavior_key;
    std::string verifier_ref;
    std::string changed_dimension;
    std::vector<std::string> invariant_dimensions;
    common_agent_scope scope;
    size_t minimum_contrasts = 2;
    bool host_approved = false;
    bool reusable = false;
};

bool common_agent_concept_grounding_validate(
        const common_agent_concept_grounding & grounding,
        std::string & error);

// Fixture roles are semantic preparation inputs, not FlyDelta arms.
enum class common_agent_concept_fixture_role {
    baseline,
    conditioned,
    control,
    counterexample,
};

const char * common_agent_concept_fixture_role_name(
        common_agent_concept_fixture_role role);

struct common_agent_concept_fixture {
    int schema_version = 1;
    std::string id;
    std::string grounding_ref;
    std::string context_ref;
    std::string verifier_ref;
    std::string semantic_expectation_ref;
    std::string independent_key;
    common_agent_concept_fixture_role role =
        common_agent_concept_fixture_role::baseline;
    bool host_verified = false;
};

bool common_agent_concept_fixture_validate(
        const common_agent_concept_fixture & fixture,
        std::string & error);

bool common_agent_validate_concept_fixture_set(
        const common_agent_concept_grounding & grounding,
        const std::vector<common_agent_concept_fixture> & fixtures,
        std::string & error);

// A verified minimal contrast. The host owns semantic equivalence and
// independence checks; this object only carries their immutable references.
struct common_agent_concept_contrast {
    int schema_version = 1;
    std::string id;
    std::string grounding_ref;
    std::string behavior_key;
    std::string task_fingerprint;
    std::string baseline_ref;
    std::string conditioned_ref;
    std::string control_ref;
    std::string counterexample_ref;
    std::string verifier_ref;
    std::string changed_dimension;
    std::vector<std::string> invariant_dimensions;
    std::string independent_key;
    common_flydelta_teaching_origin baseline_origin =
        common_flydelta_teaching_origin::host_derived;
    common_flydelta_teaching_origin conditioned_origin =
        common_flydelta_teaching_origin::host_derived;
    common_flydelta_teaching_origin control_origin =
        common_flydelta_teaching_origin::host_counterfactual;
    float confidence = 0.0f;
    bool host_verified = false;
};

bool common_agent_concept_contrast_validate(
        const common_agent_concept_contrast & contrast,
        const common_agent_concept_grounding & grounding,
        std::string & error);

// Converts only a verified, grounded contrast into the existing generic
// TeachingRelation. This is the single bridge into the already implemented
// material store/capture/synthesis pipeline.
bool common_agent_concept_contrast_to_teaching_relation(
        const common_agent_concept_hypothesis & hypothesis,
        const common_agent_concept_grounding & grounding,
        const common_agent_concept_contrast & contrast,
        common_flydelta_teaching_relation & relation,
        std::string & error);

// Admission for a material group. The host must supply multiple independent
// verified contrasts; one successful fixture or one repeated task is not
// sufficient to make a concept ready for capture.
bool common_agent_validate_concept_teaching_relations(
        const common_agent_concept_hypothesis & hypothesis,
        const common_agent_concept_grounding & grounding,
        const std::vector<common_flydelta_teaching_relation> & relations,
        std::string & error);

// Optional host providers. They may return no hypothesis or no contrasts as a
// normal outcome. They must not manufacture host verification or learning
// credit; the grounding provider returns only already verified relations.
using common_agent_concept_hypothesis_provider = std::function<bool(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        const common_learning_transaction & transaction,
        std::optional<common_agent_concept_hypothesis> & hypothesis,
        std::string & error)>;

using common_agent_concept_grounding_provider = std::function<bool(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        const common_learning_transaction & transaction,
        const common_agent_concept_hypothesis & hypothesis,
        common_agent_concept_grounding & grounding,
        std::vector<common_flydelta_teaching_relation> & relations,
        std::string & error)>;
