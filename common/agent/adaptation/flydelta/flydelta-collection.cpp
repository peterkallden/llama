#include "agent/adaptation/flydelta/flydelta-collection.h"

#include "agent/adaptation/flydelta/flydelta-evidence.h"

#include <algorithm>

namespace {

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool references_empty_except(
        const common_flydelta_experiment_collection_request & request,
        common_flydelta_experiment_job_kind kind) {
    if (kind != common_flydelta_experiment_job_kind::counterfactual &&
            !request.capture_manifest_ids.empty()) return false;
    if (kind != common_flydelta_experiment_job_kind::basis &&
            !request.repair_delta_ids.empty()) return false;
    if (kind != common_flydelta_experiment_job_kind::delta_memory &&
            !request.training_example_ids.empty()) return false;
    return true;
}

} // namespace

bool common_flydelta_collect_experiment_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_collection_request & request,
        common_flydelta_experiment_collection_result & result,
        std::string & error) {
    error.clear();
    result = common_flydelta_experiment_collection_result::disabled;
    if (!request.enabled) return true;
    if (queue_root.empty() || !bounded(request.behavior_key) ||
            !bounded(request.model_profile_fingerprint) ||
            !bounded(request.tokenizer_fingerprint) ||
            !bounded(request.template_fingerprint) ||
            !bounded(request.execution_context_fingerprint) ||
            !bounded(request.code_revision) || !references_empty_except(request, request.kind)) {
        error = "FlyDelta experiment collection request is incomplete or mixes job references";
        return false;
    }

    common_flydelta_experiment_seed seed;
    if (!common_flydelta_experiment_seed_from_evidence(
            request.evidence, request.behavior_key,
            request.model_profile_fingerprint, request.tokenizer_fingerprint,
            request.template_fingerprint, request.execution_context_fingerprint,
            request.split, seed, error)) {
        return false;
    }

    common_flydelta_experiment_job job;
    job.id = seed.id + "/job/" + common_flydelta_experiment_job_kind_name(request.kind);
    job.kind = request.kind;
    job.seed = std::move(seed);
    job.capture_manifest_ids = request.capture_manifest_ids;
    job.repair_delta_ids = request.repair_delta_ids;
    job.training_example_ids = request.training_example_ids;
    job.alpha_search = request.alpha_search;
    job.learning_rate = request.learning_rate;
    job.decay = request.decay;
    job.code_revision = request.code_revision;
    if (!common_flydelta_experiment_job_validate(job, 128, error)) return false;

    bool contains = false;
    if (!common_flydelta_experiment_queue_contains(queue_root, job.id, contains, error)) return false;
    if (contains) {
        result = common_flydelta_experiment_collection_result::already_present;
        return true;
    }
    if (!common_flydelta_experiment_queue_enqueue(queue_root, job, queue_limits, error)) return false;
    result = common_flydelta_experiment_collection_result::enqueued;
    return true;
}
