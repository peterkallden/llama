#pragma once

#include "agent/adaptation/semi-oracle-store.h"
#include "agent/adaptation/semi-oracle-outcome-guidance.h"

struct common_training_candidate;
struct common_flydelta_teaching_relation;

// Host-side facade for the native `oracle.ruleset.define` and
// `oracle.action.verify` operations. It is intentionally not a general solver
// API exposed to the model: session identity, RuleSet identity/revision,
// provenance authority, budget and guidance are supplied by the host.
class common_semi_oracle_tool_service {
public:
    common_semi_oracle_tool_service(common_semi_oracle_store & store,
            std::string host_namespace, std::string session_id,
            common_semi_oracle_outcome_guidance_config guidance = {},
            std::string project_id = {});

    bool define_ruleset(const std::string & ruleset_dsl,
            common_semi_oracle_rule_authority host_asserted_authority,
            common_semi_oracle_ruleset & active, std::string & error);
    bool active_ruleset(common_semi_oracle_ruleset & ruleset,
            bool & found, std::string & error) const;
    bool action_contract(std::string & contract, std::string & error) const;
    // Host-only state admission for a newly established domain session. State
    // is validated against the active RuleSet and is never supplied by the
    // model's action call. Existing state is preserved unless replace is true.
    bool initialize_active_state(const std::string & state_json, bool replace,
            std::string & error);
    bool verify_action(const std::string & state_json, const std::string & action_dsl,
            common_semi_oracle_level level, const std::string & goal,
            common_semi_oracle_budget host_budget,
            const std::vector<std::string> & host_guidance,
            common_semi_oracle_result & result, std::string & error);
    // Production placement-game entry point: state and transition are owned
    // by the host. A successful proof advances the session board atomically
    // from the caller's perspective; rejected/unknown actions do not advance.
    bool verify_active_placement_action(const std::string & action_dsl,
            common_semi_oracle_result & result, std::string & state_before,
            std::string & state_after, std::string & error);
    // Domain-neutral production action entry point. It loads the host-owned
    // active state, verifies one action, and advances state only on a proven
    // SATISFIED transition with a host-verifiable next_state proof.
    bool verify_active_action(const std::string & action_dsl,
            common_semi_oracle_result & result, std::string & state_before,
            std::string & state_after, std::string & error);
    // Host-owned goal search for the active linear-algebra state. The model
    // supplies neither the target expression nor search policy; V1 solves
    // for x using the bounded exact-rewrite A* problem.
    bool search_active_algebra_goal(
            common_semi_oracle_result & result, std::string & error);
    // Host-only strategic ranking. The complete bounded ranking and proofs
    // are appended through the configured semi-oracle evaluation store.
    bool rank_placements(const std::string & state_json,
            common_semi_oracle_budget host_budget,
            std::vector<common_semi_oracle_placement_action_value> & values,
            std::string & error);
    bool make_preference_relation(
            const common_semi_oracle_placement_action_value & rejected,
            const common_semi_oracle_placement_action_value & preferred,
            const std::string & relation_id,
            const std::string & rejected_execution_ref,
            const std::string & preferred_execution_ref,
            const std::string & control_execution_ref,
            const std::string & evidence_ref,
            common_flydelta_teaching_relation & relation,
            std::string & error) const;
    bool make_sft_candidate(const common_semi_oracle_placement_action_value & preferred,
            const std::string & approved_prompt, const std::string & redaction_policy_id,
            const std::string & redaction_method, bool curator_approved,
            common_training_candidate & candidate, std::string & error) const;
    bool make_dpo_candidate(const common_semi_oracle_placement_action_value & chosen,
            const common_semi_oracle_placement_action_value & rejected,
            const std::string & approved_prompt, const std::string & redaction_policy_id,
            const std::string & redaction_method, bool curator_approved,
            common_training_candidate & candidate, std::string & error) const;
    bool promote_active_ruleset_to_host(std::string & error);

private:
    common_semi_oracle_store & store_;
    std::string host_namespace_;
    std::string session_id_;
    common_semi_oracle_outcome_guidance_config guidance_;
    std::string project_id_;
};
