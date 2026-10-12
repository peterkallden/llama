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
    // Immutable host reference for the grounded teaching material. It may be
    // a procedure, a correction bundle or a concept-grounding record; concept
    // extraction must not depend on one specific teaching source.
    std::string source_ref;
    std::string grounding_ref;
    // Optional procedure provenance retained for existing blueprint callers.
    std::string procedure_ref;
    std::string verifier_ref;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string scope_fingerprint;
    bool host_approved = false;
    bool redaction_attested = false;
    bool require_control = true;
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
    // Optional explicitly host-labelled undesired behavior.  This is not a
    // fallback to baseline: a negative capture is valid only when the host
    // has verified that it represents the behavior to avoid.
    std::string negative_capture_ref;
    std::string semantic_anchor;
    int32_t layer_index = -1;
    std::vector<float> baseline;
    std::vector<float> conditioned;
    std::vector<float> control;
    std::vector<float> negative;
    bool aligned = false;
    bool conditioned_host_verified = false;
    bool negative_host_verified = false;
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

// The semantic source of a candidate is independent from the estimator used
// to construct its vector.  In particular, a positive prototype is not an
// observed repair pair and must retain that distinction through search and
// verification.
enum class common_flydelta_concept_synthesis_semantics {
    control_residualized,
    positive_prototype,
    negative_repulsion,
};

const char * common_flydelta_concept_synthesis_semantics_name(
        common_flydelta_concept_synthesis_semantics semantics);

const char * common_flydelta_concept_candidate_kind_name(
        common_flydelta_concept_candidate_kind kind);

struct common_flydelta_concept_candidate {
    int schema_version = 1;
    common_flydelta_concept_candidate_kind kind =
        common_flydelta_concept_candidate_kind::raw_mean;
    common_flydelta_concept_synthesis_semantics synthesis_semantics =
        common_flydelta_concept_synthesis_semantics::control_residualized;
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
    size_t control_trajectories = 0;
    size_t retained_control_trajectories = 0;
    size_t negative_trajectories = 0;
    size_t retained_negative_trajectories = 0;
    float median_alignment = 0.0f;
    bool control_residualized = false;
    bool experimental_only = true;
    bool learning_eligible = false;
};

// A synthesis component keeps its semantic role until the paired basis has
// been built.  The role is deliberately not added to the generic executable
// direction candidate: that type is also used by ordinary rank-one search.
enum class common_flydelta_intervention_component_role {
    prefer,
    avoid_support,
};

const char * common_flydelta_intervention_component_role_name(
        common_flydelta_intervention_component_role role);

// Reference-only identity for one bounded prefer/avoid composition.  Raw
// vectors remain in the immutable candidate/artifact stores; this proposal
// only carries the provenance needed to validate and replay the composition.
struct common_flydelta_paired_intervention_proposal {
    int schema_version = 1;
    std::string proposal_id;
    common_flydelta_intervention_component_role prefer_role =
        common_flydelta_intervention_component_role::prefer;
    common_flydelta_intervention_component_role avoid_role =
        common_flydelta_intervention_component_role::avoid_support;
    std::string prefer_candidate_ref;
    std::string avoid_candidate_ref;
    common_flydelta_direction_kind prefer_direction_kind =
        common_flydelta_direction_kind::raw_repair;
    common_flydelta_direction_kind avoid_direction_kind =
        common_flydelta_direction_kind::negative_repulsion;
    std::string concept_key;
    std::string behavior_key;
    std::string relation_ref;
    std::string fixture_ref;
    std::string oracle_ref;
    std::string oracle_revision;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string scope_fingerprint;
    int32_t layer_index = -1;
};

bool common_flydelta_paired_intervention_proposal_validate(
        const common_flydelta_paired_intervention_proposal & proposal,
        std::string & error);

// Builds a reference-only proposal from two already validated synthesis
// candidates.  The negative candidate is support material only; this helper
// never creates a rank-one frontier or learning evidence.
bool common_flydelta_build_paired_intervention_proposal(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_candidate & prefer,
        const common_flydelta_concept_candidate & avoid,
        const common_flydelta_direction_candidate & prefer_direction,
        const common_flydelta_direction_candidate & avoid_direction,
        const std::string & fixture_ref,
        const std::string & scope_fingerprint,
        common_flydelta_paired_intervention_proposal & proposal,
        std::string & error);

// Selects a bounded, deterministic comparison frontier across synthesis
// semantics. The selector compares at most one raw estimator per primary
// semantic source first (control_residualized, then positive_prototype), and
// only then fills remaining capacity in source order. Negative repulsion is
// intentionally never selected for an independent rank-1 frontier: it is
// retained as support material for a later explicitly paired contrast/basis.
// It does not rank behavior,
// assign evidence, or grant lifecycle authority; the selected candidates
// continue through the existing search and Oracle path.
std::vector<size_t> common_flydelta_select_concept_synthesis_frontier(
        const std::vector<common_flydelta_concept_candidate> & candidates,
        size_t max_candidates);

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

// An independently host-verified activation sample for prototype synthesis.
// Unlike a concept trajectory, it does not pretend that every positive sample
// has a matched baseline or repair event.  The two sample sets still must be
// compatible in model, scope, capture site and semantic anchor.
struct common_flydelta_concept_prototype_sample {
    int schema_version = 1;
    std::string id;
    std::string capture_ref;
    std::string semantic_anchor;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string scope_fingerprint;
    std::string verifier_ref;
    int32_t layer_index = -1;
    std::vector<float> values;
    bool host_verified = false;
    bool independent = false;
};

struct common_flydelta_concept_prototype_build_config {
    int schema_version = 1;
    size_t dimension = 0;
    size_t min_positive_samples = 2;
    size_t min_control_samples = 2;
    size_t max_samples = 32;
    float trim_fraction = 0.20f;
    float variance_ridge = 0.001f;
};

bool common_flydelta_concept_prototype_build_config_validate(
        const common_flydelta_concept_prototype_build_config & config,
        std::string & error);

bool common_flydelta_concept_prototype_sample_validate(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_prototype_sample & sample,
        size_t expected_dimension,
        std::string & error);

// Builds positive/control prototype candidates from independently verified
// capture sets.  This is CPU-only synthesis; the candidates remain
// experimental and must use the ordinary FlyDelta search and Oracle path.
bool common_flydelta_build_positive_prototype_candidates(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_prototype_build_config & config,
        const std::vector<common_flydelta_concept_prototype_sample> & positive,
        const std::vector<common_flydelta_concept_prototype_sample> & controls,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error);

// Builds negative-repulsion candidates from explicitly host-verified
// undesired captures and neutral controls.  The vector is control - negative,
// so it points away from the undesired behavior.  These candidates remain
// experimental-only and are useful only alongside a positive/repair basis
// member; they never create learning credit or promotion evidence alone.
bool common_flydelta_build_negative_repulsion_candidates(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_prototype_build_config & config,
        const std::vector<common_flydelta_concept_prototype_sample> & negative,
        const std::vector<common_flydelta_concept_prototype_sample> & controls,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error);
