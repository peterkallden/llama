#pragma once

#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"
#include "agent/adaptation/semi-oracle.h"

// Immutable host-owned record linking one live SemiOracle observation to the
// three model-facing continuations required by the ordinary FlyDelta capture
// path.  It contains no activations and never grants learning credit itself.
struct common_semi_oracle_interaction {
    int schema_version = 1;
    std::string id;
    std::string session_id;
    std::string ruleset_id;
    uint64_t ruleset_revision = 0;
    std::string action_contract_fingerprint;
    std::string domain;
    std::string contrast_kind;
    std::string evidence_ref;
    std::string verifier_ref;
    std::string verifier_revision;
    std::string task_fingerprint;
    // Host-owned serialized state.  The baseline and preferred action share
    // this state; a control may use a separately stored compatible state.
    std::string state_json;
    std::string control_state_json;
    std::string baseline_action_dsl;
    std::string preferred_action_dsl;
    std::string control_action_dsl;
    std::string baseline_result_json;
    std::string preferred_result_json;
    std::string contrast_ref;
};

bool common_semi_oracle_interaction_validate(
        const common_semi_oracle_interaction & interaction,
        std::string & error);
std::string common_semi_oracle_interaction_to_json(
        const common_semi_oracle_interaction & interaction);
bool common_semi_oracle_interaction_from_json(
        const std::string & text,
        common_semi_oracle_interaction & interaction,
        std::string & error);

// Builds the existing FlyDelta teaching relation only from a same-revision,
// host-proved rejected/accepted contrast. The references identify executions
// already retained by the caller; this function does not persist or capture.
bool common_semi_oracle_make_teaching_relation(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_result & rejected,
        const common_semi_oracle_result & accepted,
        const common_agent_scope & scope,
        const std::string & relation_id,
        const std::string & task_fingerprint,
        const std::string & rejected_execution_ref,
        const std::string & accepted_execution_ref,
        const std::string & evidence_ref,
        common_flydelta_teaching_relation & relation,
        std::string & error);

// Adds the matched host counterfactual required by the ordinary concept
// capture path to a proof-verified correctness relation.
bool common_semi_oracle_make_controlled_teaching_relation(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_result & rejected,
        const common_semi_oracle_result & accepted,
        const common_agent_scope & scope,
        const std::string & relation_id,
        const std::string & task_fingerprint,
        const std::string & rejected_execution_ref,
        const std::string & accepted_execution_ref,
        const std::string & control_execution_ref,
        const std::string & evidence_ref,
        common_flydelta_teaching_relation & relation,
        std::string & error);

// Builds a strategic preference relation only when both placements are
// legal, proven minimax values for the same state and the preferred action
// is strictly better. `control_execution_ref` must identify a host-verified
// matched control captured by the caller.
bool common_semi_oracle_make_preference_relation(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_placement_action_value & rejected,
        const common_semi_oracle_placement_action_value & preferred,
        const common_agent_scope & scope,
        const std::string & relation_id,
        const std::string & rejected_execution_ref,
        const std::string & preferred_execution_ref,
        const std::string & control_execution_ref,
        const std::string & evidence_ref,
        common_flydelta_teaching_relation & relation,
        std::string & error);
