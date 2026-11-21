#pragma once

#include "flydelta-evidence.h"

#include <cstdint>
#include <string>
#include <vector>

// A request-scoped causal probe is deliberately not a FlyDelta artifact. It
// records whether an exact intervention could affect a model decision at one
// capture site; it never grants evidence, learning credit or activation
// authority.
enum class common_flydelta_causal_patch_kind : uint8_t {
    self_replacement,
    exact_replacement,
    scaled_exact_delta,
    existing_direction,
    norm_matched_control,
    wrong_concept,
};

const char * common_flydelta_causal_patch_kind_name(
        common_flydelta_causal_patch_kind kind);
bool common_flydelta_causal_patch_kind_from_name(
        const std::string & value,
        common_flydelta_causal_patch_kind & kind);

struct common_flydelta_causal_diagnostic_arm {
    int schema_version = 1;
    std::string arm_id;
    common_flydelta_causal_patch_kind patch_kind =
        common_flydelta_causal_patch_kind::self_replacement;
    uint32_t layer = 0;
    int32_t absolute_position = -1;
    int32_t sequence_id = -1;
    bool executed = false;
    bool patch_attempted = false;
    bool patch_applied = false;
    bool margin_available = false;
    float baseline_margin = 0.0f;
    float candidate_margin = 0.0f;
    float margin_delta = 0.0f;
    bool host_evaluated = false;
    bool verifier_known = false;
    common_flydelta_counterfactual_outcome host_outcome =
        common_flydelta_counterfactual_outcome::unknown;
    std::string evidence_ref;
};

struct common_flydelta_causal_diagnostic_report {
    int schema_version = 1;
    std::string experiment_id;
    std::string manifest_ref;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    std::vector<common_flydelta_causal_diagnostic_arm> arms;
    // These fields are intentionally fixed false in V0. Keeping them in the
    // report makes the lifecycle boundary machine-checkable.
    bool learning_eligible = false;
    bool promotion_eligible = false;
    bool activation_authority = false;
};

bool common_flydelta_causal_diagnostic_arm_validate(
        const common_flydelta_causal_diagnostic_arm & arm,
        std::string & error);

bool common_flydelta_causal_diagnostic_report_validate(
        const common_flydelta_causal_diagnostic_report & report,
        std::string & error);
