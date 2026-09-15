#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/lifecycle-store.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Outcome answers what the host could establish. Disposition answers what the
// bounded experiment loop should do next; the two must not be conflated.
enum class common_flydelta_search_disposition {
    none,
    retain,
    refine,
    validate_repeatability,
    reject,
};

const char * common_flydelta_search_disposition_name(
        common_flydelta_search_disposition disposition);

// Search trials are retained as experimental artifacts until the host has
// explicitly reviewed their evidence. A HELPED result requests that review;
// it does not mutate the active registry.
enum class common_flydelta_experimental_artifact_action {
    retain_experimental,
    review_candidate,
    reject,
};

const char * common_flydelta_experimental_artifact_action_name(
        common_flydelta_experimental_artifact_action action);

struct common_flydelta_candidate_lineage {
    int schema_version = 1;
    std::string candidate_id;
    std::string parent_candidate_id;
    std::string mutation_kind;
    uint32_t generation = 0;
    std::string direction_id;
    std::vector<uint32_t> layer_indices;
    float scale = 0.0f;
    float intervention_budget = 0.0f;
};

bool common_flydelta_candidate_lineage_validate(
        const common_flydelta_candidate_lineage & lineage,
        size_t max_layers,
        std::string & error);

// This is deliberately a search record, not an evidence record. The host has
// already classified the outcome, while geometry and sequence margin only
// decide whether a bounded follow-up is worthwhile.
struct common_flydelta_search_observation {
    int schema_version = 1;
    std::string experiment_id;
    std::string candidate_id;
    // alpha, layer, coefficient, direction or another host-defined bounded
    // search kind. This keeps heterogeneous search records distinguishable
    // without creating separate lifecycle stores.
    std::string search_kind = "counterfactual";
    // One immutable experimental sideband may own many trial records. The
    // reference is optional for purely in-memory probes, but when present it
    // must point at a registry entry with status=experimental.
    std::string experimental_artifact_id;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    bool host_verified = false;
    bool diagnostics_available = false;
    common_flydelta_representation_diagnostics diagnostics;
    // Optional bounded arm parameters. Coefficient search uses this field;
    // direction/layer/scale searches keep their parameters in lineage.
    std::vector<float> coefficients;
    float search_fitness = 0.0f;
    bool sequence_margin_available = false;
    float sequence_margin_delta = 0.0f;
    bool budget_remaining = false;
};

bool common_flydelta_search_observation_validate(
        const common_flydelta_search_observation & observation,
        std::string & error);

struct common_flydelta_search_decision {
    int schema_version = 1;
    std::string candidate_id;
    common_flydelta_search_disposition disposition =
        common_flydelta_search_disposition::none;
    float search_priority = 0.0f;
    float evidence_score = 0.0f;
    common_flydelta_experimental_artifact_action artifact_action =
        common_flydelta_experimental_artifact_action::retain_experimental;
    std::string reason;
};

bool common_flydelta_decide_search_disposition(
        const common_flydelta_search_observation & observation,
        common_flydelta_search_decision & decision,
        std::string & error);

// The experiment champion is not necessarily the active sideband. It is the
// best verified candidate in one isolated experiment population.
struct common_flydelta_experiment_champion {
    int schema_version = 1;
    std::string experiment_id;
    std::string candidate_id;
    std::string model_profile_id;
    std::string fixture_set_revision;
    std::string verifier_revision;
    float objective_score = 0.0f;
    size_t evaluated_turns = 0;
    size_t helped_trials = 0;
    size_t harmed_trials = 0;
    bool host_verified = false;
    bool holdout_passed = false;
    bool no_regression = false;
};

bool common_flydelta_experiment_champion_validate(
        const common_flydelta_experiment_champion & champion,
        std::string & error);

// Selects a challenger only when it is host-verified, has a positive verified
// lift, passes holdout/no-regression gates and strictly beats the current
// experiment champion. The active sideband registry is not touched here.
bool common_flydelta_select_experiment_champion(
        const common_flydelta_experiment_champion & current,
        const common_flydelta_experiment_champion & challenger,
        common_flydelta_experiment_champion & selected,
        std::string & error);

// Common identity for append-only FlyDelta lifecycle records. The payload is
// reference-only; raw prompts, tool output, captures and credentials remain
// in their existing stores.
struct common_flydelta_lifecycle_event_context {
    std::string event_id;
    std::string idempotency_key;
    std::string source_id;
    common_agent_scope scope;
    std::string content_hash;
    std::string created_at;
};

bool common_flydelta_append_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_search_observation & observation,
        const common_flydelta_search_decision & decision,
        const common_flydelta_candidate_lineage * lineage,
        std::string & error);

// Records runtime discovery of a host-qualified capture candidate. Discovery
// is deliberately only an observed candidate lifecycle state; it is not a
// search result, training example, promotion decision or active overlay.
bool common_flydelta_append_capture_candidate_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_capture_candidate & candidate,
        const common_learning_transaction & transaction,
        std::string & error);

bool common_flydelta_append_champion_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_champion & current,
        const common_flydelta_experiment_champion & challenger,
        const common_flydelta_experiment_champion & selected,
        std::string & error);
