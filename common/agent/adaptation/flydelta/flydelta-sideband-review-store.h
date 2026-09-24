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
    common_flydelta_promotion_policy promotion_policy;
    common_flydelta_promotion_summary promotion_summary;
    common_flydelta_evaluation_report evaluation;
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
    bool replay(common_flydelta_sideband_registry & registry, std::string & error) const;

private:
    bool apply(
            common_flydelta_sideband_registry & registry,
            const common_flydelta_sideband_review & review,
            bool explicit_host_approval,
            std::string & error) const;

    common_learning_lifecycle_store & journal_;
};
