#pragma once

#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"
#include "agent/adaptation/semi-oracle.h"

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
