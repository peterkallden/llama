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

// Host-supplied comparison metadata. The runtime may discover a source, but
// only the host can supply the immutable task, execution and verifier refs.
struct common_adaptation_evidence_relation {
    std::string id;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::string behavior_key;
    std::string task_fingerprint;
    std::string baseline_ref;
    std::string candidate_ref;
    std::string verifier_ref;
    std::vector<std::string> transaction_ids;
    common_learning_cause cause = common_learning_cause::unknown;
    bool host_verified = false;
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

// Completes one discovered source into the shared reference-only evidence
// contract. It does not infer execution refs or accept model self-assessment.
bool common_adaptation_evidence_from_turn(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        const common_adaptation_evidence_relation & relation,
        common_adaptation_evidence & evidence,
        std::string & error);
