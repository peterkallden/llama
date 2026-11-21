#pragma once

#include "agent/adaptation/semi-oracle.h"
#include "agent/adaptation/training-candidate.h"

// Builds an explicit offline SFT demonstration from a proven placement.
// It remains ineligible until curator approval and redaction attestation.
bool common_semi_oracle_make_sft_candidate(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_placement_action_value & preferred,
        const std::string & persisted_evaluation_payload_json,
        const common_agent_scope & scope,
        const std::string & approved_prompt,
        const std::string & redaction_policy_id,
        const std::string & redaction_method,
        bool curator_approved,
        common_training_candidate & candidate,
        std::string & error);

// Builds an offline DPO pair from a complete persisted minimax ranking. The
// chosen action must be strictly better than the rejected action; equal-value
// moves are intentionally not ordered by the solver.
bool common_semi_oracle_make_dpo_candidate(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_placement_action_value & chosen,
        const common_semi_oracle_placement_action_value & rejected,
        const std::string & persisted_evaluation_payload_json,
        const common_agent_scope & scope,
        const std::string & approved_prompt,
        const std::string & redaction_policy_id,
        const std::string & redaction_method,
        bool curator_approved,
        common_training_candidate & candidate,
        std::string & error);
