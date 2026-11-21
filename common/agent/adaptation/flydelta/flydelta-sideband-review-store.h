#pragma once

#include "agent/adaptation/flydelta/flydelta-promotion.h"
#include "agent/adaptation/lifecycle-store.h"

#include <string>
#include <vector>

enum class common_flydelta_review_source {
    operator_action,
    host_automation,
};

enum class common_flydelta_review_action {
    admit_experimental,
    promote_to_candidate,
    approve_canary,
    reject,
    stage_canary,
    activate,
    rollback,
    close_canary,
    reserve_canary_observation,
    complete_canary_observation,
    attach_canary_evaluation,
    retire,
    revoke,
};

const char * common_flydelta_review_source_name(common_flydelta_review_source source);
const char * common_flydelta_review_action_name(common_flydelta_review_action action);
bool parse_common_flydelta_review_source(
        const std::string & value, common_flydelta_review_source & source, std::string & error);
bool parse_common_flydelta_review_action(
        const std::string & value, common_flydelta_review_action & action, std::string & error);

// This is an operator/host decision envelope, not model output. The report
// references identify the durable inputs; the embedded snapshots keep replay
// independent of a mutable in-memory registry or a live model process.
struct common_flydelta_sideband_review {
    int schema_version = 1;
    std::string event_id;
    std::string actor_id;
    std::string policy_revision;
    std::string evaluation_revision;
    std::string promotion_summary_ref;
    std::string evaluation_report_ref;
    std::string reason;
    common_flydelta_review_source source = common_flydelta_review_source::operator_action;
    common_flydelta_review_action action = common_flydelta_review_action::admit_experimental;
    common_flydelta_sideband_manifest manifest;
    bool has_promotion_evidence = false;
    // A bounded canary may be admitted on host-observed iterative progress;
    // this is intentionally distinct from promotion evidence for `active`.
    bool has_canary_admission = false;
    common_flydelta_semantic_progress canary_progress;
    common_flydelta_promotion_policy promotion_policy;
    common_flydelta_promotion_summary promotion_summary;
    common_flydelta_evaluation_report evaluation;
    // Optional activation-binding fields are appended to preserve older
    // positional review literals.
    std::string binding_key;
    std::string expected_current_revision_id;
    // Canary state is projected exclusively from these append-only review
    // events.  close_canary names the stage_canary event it closes.
    bool has_canary_envelope = false;
    common_flydelta_canary_envelope canary_envelope;
    std::string canary_envelope_event_id;
    // Observation lineage and terminal/evaluation projection fields. These
    // are journal metadata; the referenced immutable reports remain the
    // semantic source of truth.
    std::string observation_id;
    std::string allocation_id;
    std::string scope_step_id;
    common_flydelta_canary_observation_status observation_status =
        common_flydelta_canary_observation_status::generation_failed;
    common_flydelta_counterfactual_outcome canary_outcome =
        common_flydelta_counterfactual_outcome::unknown;
    std::string counterfactual_report_ref;
    float target_gain = 0.0f;
    float control_regression = 0.0f;
    float competitor_regression = 0.0f;
};

bool common_flydelta_sideband_review_validate(
        const common_flydelta_sideband_review & review, std::string & error);
std::string common_flydelta_sideband_review_to_json(
        const common_flydelta_sideband_review & review);
bool common_flydelta_sideband_review_from_json(
        const std::string & text, common_flydelta_sideband_review & review, std::string & error);

// Durable review journal and registry replay seam. It deliberately reuses the
// existing lifecycle backend; it does not create another persistence system.
class common_flydelta_sideband_review_store final {
public:
    explicit common_flydelta_sideband_review_store(common_learning_lifecycle_store & journal)
        : journal_(journal) {}

    bool append(const common_flydelta_sideband_review & review, std::string & error);
    bool apply_and_append(
            common_flydelta_sideband_registry & registry,
            const common_flydelta_sideband_review & review,
            bool explicit_host_approval,
            std::string & error);
    std::vector<common_flydelta_sideband_review> list(std::string & error) const;
    std::vector<common_flydelta_sideband_review> open_canaries(std::string & error) const;
    bool replay(common_flydelta_sideband_registry & registry, std::string & error) const;

private:
    bool apply(
            common_flydelta_sideband_registry & registry,
            const common_flydelta_sideband_review & review,
            bool explicit_host_approval,
            std::string & error) const;

    common_learning_lifecycle_store & journal_;
};
