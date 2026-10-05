#pragma once

#include "agent/adaptation/flydelta/flydelta-semantic-decision.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

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

// A bounded, deterministic explanation for a known semantic violation.  This
// is provenance for host-side negative-material admission, never an outcome
// or lifecycle decision.  Only explicitly attributable values may become
// avoid_support material.
enum class common_flydelta_oracle_violation_kind {
    none,
    contract_violation,
    wrong_tool,
    missing_required_grouping,
    wrong_grouping_field,
    missing_measure,
    wrong_measure_column,
    wrong_measure_function,
    unattributable,
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
const char * common_flydelta_oracle_violation_kind_name(
        common_flydelta_oracle_violation_kind value);
const char * common_flydelta_oracle_phase_name(common_flydelta_oracle_phase value);
const char * common_flydelta_oracle_probe_kind_name(common_flydelta_oracle_probe_kind value);

// A bounded check keeps partial contract progress visible without changing
// the top-level Oracle verdict. It is provenance for the host evaluator, not
// lifecycle or learning authority.
struct common_flydelta_oracle_check {
    std::string code;
    common_flydelta_oracle_verdict verdict = common_flydelta_oracle_verdict::unknown;
    bool required = true;
    std::vector<std::string> dimensions;
    std::string evidence_ref;
};

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
    std::string expected_contract_kind;
    std::string expected_contract_ref;
    std::string expected_contract_revision;
    std::string expected_contract_fingerprint;
    // Model-facing text uses the shared output codec. Native callers pass a
    // pre-parsed call through the direct validator; registry evaluators use
    // jsonl or compact_dsl here.
    std::string observed_format = "jsonl";
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
    std::string evaluator_ref;
    std::string evaluator_revision;
    std::string evidence_ref;
    std::string reason;
    std::string violation_code;
    std::string normalized_arguments_json;
    common_flydelta_oracle_violation_kind violation_kind =
        common_flydelta_oracle_violation_kind::none;
    std::vector<std::string> violation_dimensions;
    std::vector<common_flydelta_oracle_check> checks;
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

// Registered evaluators are selected by host-owned semantic identity. The
// registry is an in-memory dispatch seam only; it is not a store, lifecycle
// authority or deployment path.
struct common_flydelta_oracle_evaluator_registration {
    common_flydelta_oracle_strength strength = common_flydelta_oracle_strength::deterministic;
    std::string evaluator_ref;
    std::string evaluator_revision;
    std::string semantic_kind;
    std::string expected_contract_kind;
    std::string expected_contract_ref;
    std::string expected_contract_revision;
    common_flydelta_oracle_evaluator evaluator;
};

struct common_flydelta_oracle_registry {
    std::vector<common_flydelta_oracle_evaluator_registration> evaluators;
};

bool common_flydelta_register_oracle_evaluator(
        common_flydelta_oracle_registry & registry,
        common_flydelta_oracle_evaluator_registration registration,
        std::string & error);

common_flydelta_oracle_registry common_flydelta_make_default_oracle_registry();

bool common_flydelta_oracle_evaluate(
        const common_flydelta_oracle_evaluator_chain & evaluators,
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error);

bool common_flydelta_oracle_evaluate(
        const common_flydelta_oracle_registry & registry,
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error);
