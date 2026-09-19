#pragma once

#include "agent/adaptation/adaptation-evidence-routing.h"

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

struct common_flydelta_teaching_relation {
    int schema_version = 1;
    std::string id;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::string behavior_key;
    common_agent_scope scope;
    std::string task_fingerprint;
    std::string baseline_ref;
    std::string conditioned_ref;
    std::string control_ref;
    std::string verifier_ref;
    std::string evidence_ref;
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
