#pragma once

#include "agent/adaptation/adaptation-evidence-routing.h"
#include "agent/adaptation/learning-transaction.h"

#include <functional>
#include <optional>
#include <string>

struct common_flydelta_behavior_transition;

// A host-owned, reference-only relation that is ready to be rendered into
// FlyDelta trajectories. It is a view over the existing adaptation-evidence
// relation, not a second persistence format.
enum class common_flydelta_teaching_relation_status {
    resolved,
    ambiguous,
    insufficient_evidence,
    no_contrast,
    not_reusable,
    unsupported_behavior,
};

const char * common_flydelta_teaching_relation_status_name(
        common_flydelta_teaching_relation_status status);

enum class common_flydelta_teaching_origin {
    none,
    observed,
    host_derived,
    user_supplied,
    host_counterfactual,
};

const char * common_flydelta_teaching_origin_name(
        common_flydelta_teaching_origin origin);

// Result of the host-side procedure/blueprint admission boundary.  These are
// normal, inspectable outcomes; they are not worker failures and do not grant
// learning credit on their own.
enum class common_agent_teaching_build_status {
    resolved,
    out_of_scope,
    not_host_verified,
    not_reusable,
    missing_behavior_key,
    missing_verifier,
    no_contrast,
    incompatible_control,
    insufficient_evidence,
};

const char * common_agent_teaching_build_status_name(
        common_agent_teaching_build_status status);

struct common_flydelta_teaching_relation {
    int schema_version = 1;
    std::string id;
    // Stable semantic identity for aggregating independent executions.  It is
    // intentionally distinct from task_fingerprint, which identifies one
    // concrete execution situation.
    std::string teaching_key;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::string behavior_key;
    common_agent_scope scope;
    std::string task_fingerprint;
    std::string baseline_ref;
    std::string conditioned_ref;
    std::string control_ref;
    std::string verifier_ref;
    std::string evidence_ref;
    std::string procedure_ref;
    std::string blueprint_ref;
    common_flydelta_teaching_relation_status status =
        common_flydelta_teaching_relation_status::insufficient_evidence;
    common_flydelta_teaching_origin baseline_origin = common_flydelta_teaching_origin::none;
    common_flydelta_teaching_origin conditioned_origin = common_flydelta_teaching_origin::none;
    common_flydelta_teaching_origin control_origin = common_flydelta_teaching_origin::none;
    float confidence = 0.0f;
    bool host_approved = false;
};

bool common_flydelta_teaching_relation_validate(
        const common_flydelta_teaching_relation & relation,
        std::string & error);

// Host-only request for the V0 procedure/blueprint adapter.  All references
// are already resolved and immutable; the adapter never synthesizes a
// baseline, conditioned value or control.
struct common_agent_procedure_teaching_request {
    int schema_version = 1;
    std::string relation_id;
    std::string teaching_key;
    std::string procedure_ref;
    std::string blueprint_ref;
    common_agent_scope scope;
    std::string behavior_key;
    std::string task_fingerprint;
    std::string baseline_ref;
    std::string conditioned_ref;
    std::optional<std::string> control_ref;
    std::string verifier_ref;
    std::string evidence_ref;
    common_flydelta_teaching_origin baseline_origin = common_flydelta_teaching_origin::host_derived;
    common_flydelta_teaching_origin conditioned_origin = common_flydelta_teaching_origin::host_derived;
    common_flydelta_teaching_origin control_origin = common_flydelta_teaching_origin::host_counterfactual;
    float confidence = 0.0f;
    bool host_scope_admitted = false;
    bool host_verified = false;
    bool reusable = false;
    bool require_control = false;
};

struct common_agent_teaching_build_result {
    common_agent_teaching_build_status status =
        common_agent_teaching_build_status::insufficient_evidence;
    std::optional<common_flydelta_teaching_relation> relation;
    std::string diagnostic;
};

// Host-owned resolver for an explicit procedure/blueprint relation. A null
// request is a normal "not enough contrast for this turn" result; it is not a
// worker failure and must not synthesize a baseline or conditioned reference.
using common_agent_procedure_teaching_request_provider = std::function<bool(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        const common_learning_transaction & transaction,
        std::optional<common_agent_procedure_teaching_request> & teaching_request,
        std::string & error)>;

common_agent_teaching_build_result
common_agent_build_procedure_teaching_relation(
        const common_agent_procedure_teaching_request & request);

// Maps a resolved host teaching relation into the existing generic evidence
// relation used by the learning observer and capture queue.
bool common_agent_procedure_teaching_relation_to_evidence_relation(
        const common_flydelta_teaching_relation & teaching_relation,
        const common_learning_transaction & transaction,
        common_adaptation_evidence_relation & relation,
        std::string & error);

// Converts an already completed host relation. The caller supplies the
// semantic reusability decision and provenance; FlyDelta does not infer
// either from free text or model output.
bool common_flydelta_teaching_relation_from_host_relation(
        const common_adaptation_evidence_relation & relation,
        const std::string & evidence_ref,
        const std::string & control_ref,
        common_flydelta_teaching_relation_status status,
        common_flydelta_teaching_origin baseline_origin,
        common_flydelta_teaching_origin conditioned_origin,
        common_flydelta_teaching_origin control_origin,
        float confidence,
        bool host_approved,
        common_flydelta_teaching_relation & teaching_relation,
        std::string & error);

// Convenience source adapters. They only admit already host-verified
// evidence; execution/capture remains in the existing FlyDelta pipeline.
bool common_flydelta_teaching_relation_from_evidence(
        const common_adaptation_evidence & evidence,
        common_flydelta_teaching_relation & relation,
        std::string & error);

// Materializes the existing transition contract only after the teaching
// relation has been resolved and the host evidence carries both transaction
// sides. No new transition or persistence path is introduced here.
bool common_flydelta_teaching_relation_to_transition(
        const common_flydelta_teaching_relation & relation,
        const common_adaptation_evidence & evidence,
        const std::string & baseline_transaction_id,
        const std::string & conditioned_transaction_id,
        common_flydelta_behavior_transition & transition,
        std::string & error);

bool common_flydelta_procedure_blueprint_teaching_relation_from_evidence(
        const common_adaptation_evidence & evidence,
        common_flydelta_teaching_relation & relation,
        std::string & error);
