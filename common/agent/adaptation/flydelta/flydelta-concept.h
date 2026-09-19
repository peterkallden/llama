#pragma once

#include "agent/adaptation/adaptation-evidence.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Host-taught concept extraction is deliberately a builder, not a new
// runtime intervention path. The host supplies a bounded, redacted concept
// specification and verified baseline/conditioned/control trajectories; this
// module produces ordinary direction material for the existing FlyDelta
// WHERE/dose/search pipeline.
struct common_flydelta_concept_spec {
    int schema_version = 1;
    std::string concept_key;
    // One immutable extraction run for this concept. Multiple runs may later
    // be compared and aggregated only after their compatibility is checked.
    std::string extraction_id;
    std::string behavior_key;
    common_adaptation_evidence_source source =
        common_adaptation_evidence_source::procedure_blueprint;
    std::string procedure_ref;
    std::string verifier_ref;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string scope_fingerprint;
    bool host_approved = false;
    bool redaction_attested = false;
};

bool common_flydelta_concept_spec_validate(
        const common_flydelta_concept_spec & spec,
        std::string & error);

// One matched baseline/conditioned/control trajectory. The vectors are
// transient builder input; persisted worker state must carry only the refs.
struct common_flydelta_concept_trajectory {
    int schema_version = 1;
    std::string id;
    std::string fixture_ref;
    std::string baseline_capture_ref;
    std::string conditioned_capture_ref;
    std::string control_capture_ref;
    std::string semantic_anchor;
    int32_t layer_index = -1;
    std::vector<float> baseline;
    std::vector<float> conditioned;
    std::vector<float> control;
    bool aligned = false;
    bool conditioned_host_verified = false;
};

bool common_flydelta_concept_trajectory_validate(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_trajectory & trajectory,
        size_t expected_dimension,
        std::string & error);

struct common_flydelta_concept_build_config {
    int schema_version = 1;
    size_t dimension = 0;
    size_t min_trajectories = 2;
    size_t max_trajectories = 32;
    float trim_fraction = 0.20f;
    float variance_ridge = 0.001f;
    bool require_control = true;
};

bool common_flydelta_concept_build_config_validate(
        const common_flydelta_concept_build_config & config,
        std::string & error);

enum class common_flydelta_concept_candidate_kind {
    raw_mean,
    trimmed_mean,
    diagonal_whitened_mean,
};

const char * common_flydelta_concept_candidate_kind_name(
        common_flydelta_concept_candidate_kind kind);

struct common_flydelta_concept_candidate {
    int schema_version = 1;
    common_flydelta_concept_candidate_kind kind =
        common_flydelta_concept_candidate_kind::raw_mean;
    std::string concept_key;
    std::string extraction_id;
    std::string behavior_key;
    std::string origin = "host_taught_extracted";
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    int32_t layer_index = -1;
    std::vector<float> values;
    size_t source_trajectories = 0;
    size_t retained_trajectories = 0;
    float median_alignment = 0.0f;
    bool control_residualized = false;
    bool experimental_only = true;
    bool learning_eligible = false;
};

bool common_flydelta_concept_candidate_validate(
        const common_flydelta_concept_candidate & candidate,
        size_t expected_dimension,
        std::string & error);

// Admit an extracted concept into the existing direction/search contract.
// This preserves provenance but does not grant learning credit or bypass the
// normal evaluator, utility gate or host verifier.
bool common_flydelta_concept_candidate_to_direction(
        const common_flydelta_concept_candidate & candidate,
        common_flydelta_direction_candidate & direction,
        std::string & error);

// Builds CPU-cheap rank-one concept candidates. The residual for each
// trajectory is (conditioned - baseline) - (control - baseline), which keeps
// structurally similar but semantically irrelevant extra instruction text
// out of the concept direction when a control is supplied.
//
// This function never creates learning credit. Its candidates are always
// experimental until a later held-out host verification promotes them.
bool common_flydelta_build_concept_candidates(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_build_config & config,
        const std::vector<common_flydelta_concept_trajectory> & trajectories,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error);
