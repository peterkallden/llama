#pragma once

#include "agent/adaptation/flydelta/flydelta-queue.h"
#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

// The runtime observer can discover a source, but only a host verifier can
// supply the immutable comparison references needed for an experiment job.
// This request is therefore the explicit authority boundary between the
// learning ledger and the offline FlyDelta queue.
struct common_flydelta_experiment_collection_request {
    bool enabled = false;
    common_flydelta_experiment_job_kind kind = common_flydelta_experiment_job_kind::counterfactual;
    common_adaptation_evidence evidence;
    std::string behavior_key;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string execution_context_fingerprint;
    common_flydelta_training_split split = common_flydelta_training_split::train;
    std::vector<std::string> capture_candidate_ids;
    std::vector<std::string> capture_manifest_ids;
    std::vector<std::string> behavior_delta_ids;
    std::vector<std::string> training_example_ids;
    common_flydelta_alpha_search_config alpha_search;
    float learning_rate = 0.1f;
    float decay = 1.0f;
    std::string code_revision;
    // Optional variant identity for a follow-up search. It is part of the
    // job id only; it does not carry raw prompt or tool data.
    std::string job_variant_id;
    // Opaque host-owned resume reference. Collection transports it to the
    // next search-pipeline job; it never reads or writes the state payload.
    std::string bootstrap_zoom_state_ref;
    // Opaque state for post-Bootstrap continuation. The collection layer only
    // transports it; the host resolves its typed state.
    std::string search_state_ref;
    std::string representation_augmentation_state_ref;
};

enum class common_flydelta_experiment_collection_result {
    disabled,
    enqueued,
    already_present,
};

// Builds and enqueues one reference-only job. No prompt, tool output,
// credential, activation buffer or model inference enters the queue. A
// repeated request for the same evidence/kind is an idempotent no-op.
bool common_flydelta_collect_experiment_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_collection_request & request,
        common_flydelta_experiment_collection_result & result,
        std::string & error);

// Enqueues one bounded follow-up only after the host-side disposition has
// selected refine. The request supplies the same evidence and capture
// references as the original experiment; the variant prevents queue-id
// collisions between successive candidates.
bool common_flydelta_collect_refinement_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_collection_request & request,
        const common_flydelta_search_observation & observation,
        const common_flydelta_search_decision & decision,
        const common_flydelta_candidate_lineage & lineage,
        common_flydelta_experiment_collection_result & result,
        std::string & error);

// Enqueues a bounded follow-up while retaining the composed search pipeline
// job kind. This is used after a pipeline arm has useful UNKNOWN/NEUTRAL
// diagnostics; the queue still contains references and bounds only.
bool common_flydelta_collect_search_pipeline_refinement_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_collection_request & request,
        const common_flydelta_search_observation & observation,
        const common_flydelta_search_decision & decision,
        const common_flydelta_candidate_lineage & lineage,
        common_flydelta_experiment_collection_result & result,
        std::string & error);

// Enqueues exactly one reference-only follow-up slice from an already claimed
// search-pipeline job. This is the host scheduler seam: it carries forward
// the immutable seed and bounded references, updates only opaque state refs,
// and never evaluates the next phase recursively.
bool common_flydelta_collect_next_action_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_job & parent_job,
        common_flydelta_next_action next_action,
        const std::string & bootstrap_zoom_state_ref,
        const std::string & search_state_ref,
        const std::string & representation_augmentation_state_ref,
        common_flydelta_experiment_collection_result & result,
        std::string & error);
