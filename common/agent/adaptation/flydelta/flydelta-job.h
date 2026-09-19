#pragma once

#include "agent/adaptation/flydelta/flydelta-evidence.h"

#include <cstddef>
#include <string>
#include <vector>

enum class common_flydelta_experiment_job_kind {
    basis,
    direction,
    counterfactual,
    search_pipeline,
    delta_memory,
    donor_capture,
};

const char * common_flydelta_experiment_job_kind_name(
        common_flydelta_experiment_job_kind kind);
bool common_flydelta_experiment_job_kind_from_name(
        const std::string & value,
        common_flydelta_experiment_job_kind & kind);

// Queue payload for offline FlyDelta work. It contains references and bounds,
// never prompts, tool output, activation buffers or credentials.
struct common_flydelta_experiment_job {
    int schema_version = 1;
    std::string id;
    common_flydelta_experiment_job_kind kind = common_flydelta_experiment_job_kind::basis;
    common_flydelta_experiment_seed seed;
    std::vector<std::string> capture_candidate_ids;
    std::vector<std::string> capture_manifest_ids;
    std::vector<std::string> behavior_delta_ids;
    std::vector<std::string> training_example_ids;
    // Reference to a host-persisted BootstrapZoom state. The queue carries
    // only this opaque reference, never the state payload or activations.
    std::string bootstrap_zoom_state_ref;
    // Reference to the host-persisted continuation state for later
    // rank-one plateau, orthogonal-search or augmentation phases. It is
    // deliberately separate from the legacy BootstrapZoom state type.
    std::string search_state_ref;
    // Reference to the host-persisted representation-augmentation state.
    // Keeping this separate prevents a resumed augmentation slice from being
    // mistaken for an ordinary rank-one search state.
    std::string representation_augmentation_state_ref;
    common_flydelta_alpha_search_config alpha_search;
    float learning_rate = 0.1f;
    float decay = 1.0f;
    std::string code_revision;
};

bool common_flydelta_experiment_job_validate(
        const common_flydelta_experiment_job & job,
        size_t max_references,
        std::string & error);
std::string common_flydelta_experiment_job_to_json(
        const common_flydelta_experiment_job & job);
bool common_flydelta_experiment_job_from_json(
        const std::string & text,
        common_flydelta_experiment_job & job,
        std::string & error);
