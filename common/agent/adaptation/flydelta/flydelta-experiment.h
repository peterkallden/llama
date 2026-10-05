#pragma once

#include "agent/adaptation/flydelta/flydelta-semantic-decision.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

enum class common_flydelta_counterfactual_outcome {
    unknown,
    helped,
    neutral,
    harmed,
};

const char * common_flydelta_counterfactual_outcome_name(
        common_flydelta_counterfactual_outcome outcome);

// The fixture is immutable experiment identity. Both arms of a
// counterfactual must receive the same fixture; the host, not the model,
// owns these fingerprints.
struct common_flydelta_experiment_fixture {
    int schema_version = 1;
    std::string id;
    std::string task_fingerprint;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    // Host-composed identity for all external state relevant to this
    // comparison: tools/resources for tool work, or the corresponding
    // plan/research/project context for another behavior.
    std::string execution_context_fingerprint;
    std::string verifier_revision;
};

bool common_flydelta_experiment_fixture_validate(
        const common_flydelta_experiment_fixture & fixture,
        std::string & error);

// A trial contains host-verifier evidence, never a model self-claim of
// correctness. Unknown verification is deliberately preserved as UNKNOWN.
struct common_flydelta_counterfactual_trial {
    bool executed = false;
    // The host ran the requested evaluation/generation path. This is
    // intentionally independent from verifier_known: an attempted host
    // verification may still produce UNKNOWN when no semantic predicate
    // applies to the fixture.
    bool host_evaluated = false;
    bool verifier_known = false;
    bool passed = false;
    float quality = 0.0f;
    bool overlay_applied = false;
    size_t intervention_count = 0;
    // Compact host-verifier observation for operator-facing traces. These
    // fields explain a known outcome; they do not grant evidence or
    // promotion authority.
    std::string observed_decision_summary;
    std::string expected_decision_summary;
    std::string verifier_reason;
    std::string verifier_violation_code;
    std::string verifier_violation_kind;
    std::vector<std::string> verifier_violation_dimensions;
    common_flydelta_semantic_progress_observation semantic_progress_observation;
    std::string evidence_ref;
};

enum class common_flydelta_canary_observation_status {
    generation_failed,
    cancelled,
    oracle_unknown,
    evaluated,
};

const char * common_flydelta_canary_observation_status_name(
        common_flydelta_canary_observation_status status);
bool parse_common_flydelta_canary_observation_status(
        const std::string & value,
        common_flydelta_canary_observation_status & status,
        std::string & error);

// Immutable lineage captured at canary reservation time. It is carried by
// evaluation reports so replay never has to infer ownership from timestamps,
// artifact ids or the latest open envelope.
struct common_flydelta_canary_evaluation_context {
    int schema_version = 1;
    std::string canary_event_id;
    std::string observation_id;
    std::string binding_key;
    std::string candidate_revision_id;
    std::string allocation_id;
    std::string scope_step_id;
    std::string baseline_deployment_fingerprint;
    std::string candidate_deployment_fingerprint;
    std::string oracle_ref;
    std::string oracle_revision;
};

bool common_flydelta_canary_evaluation_context_validate(
        const common_flydelta_canary_evaluation_context & context,
        std::string & error);

bool common_flydelta_counterfactual_trial_validate(
        const common_flydelta_counterfactual_trial & trial,
        std::string & error);

struct common_flydelta_counterfactual_report {
    int schema_version = 1;
    std::string experiment_id;
    std::string fixture_id;
    std::string candidate_id;
    std::string baseline_profile_id;
    std::string candidate_profile_id;
    // Canonical host-composed identities.  They are intentionally separate
    // from profile ids because incremental evaluation may compare A against
    // A+B on the same base profile.
    std::string baseline_deployment_fingerprint;
    std::string candidate_deployment_fingerprint;
    bool has_canary_context = false;
    common_flydelta_canary_evaluation_context canary_context;
    common_flydelta_counterfactual_trial baseline;
    common_flydelta_counterfactual_trial candidate;
    common_flydelta_counterfactual_outcome outcome = common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    common_flydelta_semantic_progress semantic_progress;
};

bool common_flydelta_counterfactual_report_validate(
        const common_flydelta_counterfactual_report & report,
        std::string & error);

common_flydelta_counterfactual_outcome common_flydelta_classify_counterfactual(
        const common_flydelta_counterfactual_trial & baseline,
        const common_flydelta_counterfactual_trial & candidate);

using common_flydelta_counterfactual_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        std::string & error)>;

// Runs two host-owned arms over the same immutable fixture. This function
// does not perform inference, capture activations or persist learning.
bool common_flydelta_run_counterfactual(
        const std::string & experiment_id,
        const std::string & candidate_id,
        const std::string & baseline_profile_id,
        const std::string & candidate_profile_id,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_counterfactual_runner & runner,
        common_flydelta_counterfactual_report & report,
        std::string & error,
        const common_flydelta_canary_evaluation_context * canary_context = nullptr);

std::string common_flydelta_counterfactual_report_to_json(
        const common_flydelta_counterfactual_report & report);
bool common_flydelta_counterfactual_report_from_json(
        const std::string & text,
        common_flydelta_counterfactual_report & report,
        std::string & error);
