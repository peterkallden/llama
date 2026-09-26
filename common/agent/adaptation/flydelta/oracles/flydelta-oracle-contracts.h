#pragma once

#include "agent/adaptation/flydelta/flydelta-semantic-decision.h"

#include <functional>
#include <string>

// Oracle strength describes who can establish the semantic result. It is
// deliberately separate from FlyDelta geometry and lifecycle authority.
enum class common_flydelta_oracle_strength {
    deterministic,
    host_supported,
    model_supported,
};

enum class common_flydelta_oracle_verdict {
    satisfied,
    violated,
    not_applicable,
    unknown,
};

enum class common_flydelta_oracle_phase {
    concept,
    synthesis,
};

enum class common_flydelta_oracle_probe_kind {
    target,
    paraphrase,
    transfer,
    control,
    competing,
};

const char * common_flydelta_oracle_strength_name(common_flydelta_oracle_strength value);
const char * common_flydelta_oracle_verdict_name(common_flydelta_oracle_verdict value);
const char * common_flydelta_oracle_phase_name(common_flydelta_oracle_phase value);
const char * common_flydelta_oracle_probe_kind_name(common_flydelta_oracle_probe_kind value);

// This request is intentionally semantic rather than tool-registry-specific.
// A deterministic oracle can use expected_decision; a host/model oracle may
// use the refs and semantic_kind to resolve a richer contract.
struct common_flydelta_oracle_request {
    std::string oracle_ref;
    std::string oracle_revision;
    std::string policy_revision;
    common_flydelta_oracle_phase phase = common_flydelta_oracle_phase::concept;

    std::string concept_ref;
    std::string concept_key;
    std::string behavior_key;
    std::string semantic_kind;
    std::string task_ref;
    std::string execution_ref;
    std::string verifier_ref;

    bool applicable = true;
    bool expected_decision_available = false;
    common_flydelta_semantic_decision expected_decision;
};

struct common_flydelta_oracle_result {
    common_flydelta_oracle_verdict verdict = common_flydelta_oracle_verdict::unknown;
    common_flydelta_oracle_strength strength = common_flydelta_oracle_strength::deterministic;
    bool known = false;
    float confidence = 0.0f;
    std::string oracle_ref;
    std::string oracle_revision;
    std::string policy_revision;
    std::string evidence_ref;
    std::string reason;
};

// Returning false means this evaluator does not own the request. Returning
// true with known=false is a valid, explicit UNKNOWN result.
using common_flydelta_oracle_evaluator = std::function<bool(
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error)>;

struct common_flydelta_oracle_evaluator_chain {
    common_flydelta_oracle_evaluator deterministic;
    common_flydelta_oracle_evaluator host_supported;
    common_flydelta_oracle_evaluator model_supported;
};

bool common_flydelta_oracle_evaluate(
        const common_flydelta_oracle_evaluator_chain & evaluators,
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error);
