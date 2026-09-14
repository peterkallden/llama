#pragma once

#include "agent/adaptation/flydelta/flydelta-queue.h"

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
    std::vector<std::string> capture_manifest_ids;
    std::vector<std::string> behavior_delta_ids;
    std::vector<std::string> training_example_ids;
    common_flydelta_alpha_search_config alpha_search;
    float learning_rate = 0.1f;
    float decay = 1.0f;
    std::string code_revision;
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
