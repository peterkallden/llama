#pragma once

#include "agent/agent-scope.h"
#include "agent/adaptation/learning-observation.h"

#include <cstddef>
#include <string>
#include <vector>

// A transport-neutral source for a host-certified adaptation comparison.
// The source describes where a candidate came from; it does not assert that
// the candidate is better.  That decision belongs to the verifier.
enum class common_adaptation_evidence_source {
    tool_repair,
    reflection_alternative,
    planning_revision,
    research_alternative,
    dataset_resource,
    workflow_code,
    user_correction,
};

const char * common_adaptation_evidence_source_name(
        common_adaptation_evidence_source source);

// A derived, reference-only relation over existing learning transactions and
// host evidence.  It is deliberately not another store and contains no raw
// prompt, tool output, activation buffer or credential.
struct common_adaptation_evidence {
    int schema_version = 1;
    std::string id;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    common_agent_scope scope;
    std::string task_fingerprint;
    std::string baseline_ref;
    std::string candidate_ref;
    std::string verifier_ref;
    std::vector<std::string> transaction_ids;
    common_learning_cause cause = common_learning_cause::unknown;
    bool host_verified = false;
};

bool common_adaptation_evidence_validate(
        const common_adaptation_evidence & evidence,
        size_t max_transactions,
        std::string & error);

std::string common_adaptation_evidence_to_json(
        const common_adaptation_evidence & evidence);

bool common_adaptation_evidence_from_json(
        const std::string & text,
        common_adaptation_evidence & evidence,
        std::string & error);

