#pragma once

#include "agent/adaptation/adaptation-evidence.h"
#include "agent/contracts/agent-request.h"
#include "agent/contracts/agent-result.h"
#include "plan/plan-types.h"

#include <string>
#include <vector>

// A source match is an observation about the turn, not an adaptation
// decision.  candidate_ready only means that the turn contains the minimum
// host-visible relation for a later comparison; it never means that the
// candidate is correct or eligible for promotion.
struct common_adaptation_evidence_source_match {
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::vector<std::string> evidence_refs;
    bool candidate_ready = false;
};

// Derives only sources that can be established from the existing runtime
// result.  Sources such as workflow/code and an explicit planning revision
// need a host-created relation with its own verifier reference and are not
// inferred from broad flags such as result.revised.
std::vector<common_adaptation_evidence_source_match>
common_adaptation_evidence_sources_for_turn(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result);

