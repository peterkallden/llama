#pragma once

#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Representation augmentation is a bounded search escape. It adds a new,
// host-qualified context difference to the current experimental surface; it
// never invents evidence rank and never grants learning credit.
enum class common_flydelta_representation_augmentation_phase {
    discover_donors,
    qualify_donor,
    capture_donor,
    build_latent_delta,
    run_controls,
    localize_surface,
    full_generation,
    verify,
    done,
};

const char * common_flydelta_representation_augmentation_phase_name(
        common_flydelta_representation_augmentation_phase phase);

enum class common_flydelta_representation_augmentation_action {
    stop,
    retain,
    refine_bootstrap,
    run_controls,
    recenter_augmented_surface,
    allow_tfo_lite,
};

const char * common_flydelta_representation_augmentation_action_name(
        common_flydelta_representation_augmentation_action action);

// Donors are references to host-owned context material. Raw prompts, tool
// output, credentials and activation tensors remain outside this contract.
struct common_flydelta_representation_donor_candidate {
    int schema_version = 1;
    std::string donor_id;
    std::string source_type;
    std::string source_ref;
    std::string behavior_key;
    std::string context_payload_ref;
    std::string qualification_policy;
    std::string provenance;
    bool promotable = false;
};

bool common_flydelta_representation_donor_candidate_validate(
        const common_flydelta_representation_donor_candidate & candidate,
        std::string & error);

struct common_flydelta_representation_donor_qualification {
    int schema_version = 1;
    std::string donor_id;
    // Evaluation and decisiveness are separate. UNKNOWN is valid search
    // material, but only a known HELPED outcome can provide learning credit.
    bool host_evaluated = false;
    bool verifier_known = false;
    bool safe_to_continue = false;
    common_flydelta_counterfactual_outcome host_outcome =
        common_flydelta_counterfactual_outcome::unknown;
    bool decision_margin_available = false;
    float margin_gain = 0.0f;
    bool geometry_available = false;
    common_flydelta_representation_diagnostics geometry;
    bool search_qualified = false;
    std::string reason;
};

bool common_flydelta_representation_donor_qualification_validate(
        const common_flydelta_representation_donor_qualification & qualification,
        std::string & error);

// Qualifies a donor for bounded search. A positive margin or a host HELPED
// outcome can retain a donor, but qualification is deliberately distinct from
// HELPED evidence and never changes evidence_rank.
bool common_flydelta_qualify_representation_donor(
        const common_flydelta_representation_donor_candidate & candidate,
        const common_flydelta_representation_donor_qualification & observation,
        float minimum_margin_gain,
        float max_leakage,
        float max_shift_norm,
        common_flydelta_representation_donor_qualification & qualified,
        std::string & error);

struct common_flydelta_representation_latent_delta {
    int schema_version = 1;
    std::string donor_id;
    int32_t layer_index = -1;
    std::vector<float> values;
    bool residualized = false;
    bool available = false;
    float raw_norm = 0.0f;
    float residual_norm = 0.0f;
    float removed_norm = 0.0f;
};

bool common_flydelta_representation_latent_delta_validate(
        const common_flydelta_representation_latent_delta & delta,
        size_t expected_dimension,
        std::string & error);

// Builds c_perp = (I - P_Q)c, where Q is the permitted natural-evidence
// surface for the current augmentation policy. The surface is orthonormalized
// locally, so callers may pass raw compatible directions. In V0, callers must
// not include a failed orthogonal-search axis here: a rank-one parent becomes
// [natural d, donor residual], while rank-three surfaces require an explicit
// later surface policy. This prevents augmentation from silently combining
// two experimental escapes while preserving evidence_rank.
bool common_flydelta_build_residualized_latent_delta(
        const std::string & donor_id,
        int32_t layer_index,
        const std::vector<float> & target_plus_donor,
        const std::vector<float> & target_only,
        const std::vector<std::vector<float>> & current_surface,
        float minimum_residual_norm,
        common_flydelta_representation_latent_delta & result,
        std::string & error);

struct common_flydelta_representation_augmentation_config {
    int schema_version = 1;
    size_t max_donor_candidates = 4;
    size_t max_qualified_donors = 2;
    size_t max_latent_deltas = 2;
    size_t max_controls = 4;
    size_t max_local_whirlpool_probes = 5;
    size_t max_full_generation = 3;
    float minimum_residual_norm = 0.0001f;
    float minimum_margin_gain = 0.0f;
    float max_leakage = 1.0f;
    float max_shift_norm = 1.0f;
};

bool common_flydelta_representation_augmentation_config_validate(
        const common_flydelta_representation_augmentation_config & config,
        std::string & error);

// V0 augmentation controls: existing surface, donor residual, and their
// positive/negative combinations. These are experimental controls, not
// Shallow controls, and are evaluated before any TFO-lite escalation.
bool common_flydelta_propose_representation_augmentation_controls(
        const common_flydelta_representation_augmentation_config & config,
        std::vector<std::vector<float>> & coefficients,
        std::string & error);

struct common_flydelta_representation_augmentation_state {
    int schema_version = 1;
    std::string state_ref;
    std::string model_fingerprint;
    std::string behavior_key;
    std::string direction_family_id;
    uint64_t parent_surface_revision = 0;
    std::string parent_search_state_ref;
    float parent_evidence_rank = 0.0f;
    float evidence_rank = 0.0f;
    size_t search_rank = 0;
    std::vector<uint32_t> selected_region;
    std::string target_fixture_ref;
    std::vector<std::string> donor_candidate_refs;
    std::vector<std::string> qualified_donor_refs;
    std::vector<std::string> evaluated_donor_refs;
    common_flydelta_representation_augmentation_phase phase =
        common_flydelta_representation_augmentation_phase::discover_donors;
    std::string best_donor_ref;
    float best_margin_gain = 0.0f;
    size_t remaining_budget = 0;
    uint64_t surface_revision = 1;
    std::string next_action;
};

bool common_flydelta_representation_augmentation_state_validate(
        const common_flydelta_representation_augmentation_state & state,
        const common_flydelta_representation_augmentation_config & config,
        std::string & error);

// Evidence rank is immutable through augmentation. A usable residual may
// increase only search_rank, and only by one bounded experimental dimension.
bool common_flydelta_apply_representation_augmentation(
        common_flydelta_representation_augmentation_state & state,
        const common_flydelta_representation_latent_delta & latent,
        std::string & error);

// Chooses the next augmentation action from host/search observations. The
// decision is intentionally conservative: controls precede recentering and
// TFO; neither diagnostic utility nor donor qualification creates HELPED.
bool common_flydelta_decide_representation_augmentation(
        const common_flydelta_representation_augmentation_config & config,
        const common_flydelta_representation_augmentation_state & state,
        const std::vector<common_flydelta_representation_donor_qualification> & observations,
        bool controls_have_positive_utility,
        bool tfo_has_positive_utility,
        common_flydelta_representation_augmentation_action & action,
        std::string & error);

std::string common_flydelta_representation_augmentation_state_to_json(
        const common_flydelta_representation_augmentation_state & state);
bool common_flydelta_representation_augmentation_state_from_json(
        const std::string & text,
        common_flydelta_representation_augmentation_state & state,
        const common_flydelta_representation_augmentation_config & config,
        std::string & error);
