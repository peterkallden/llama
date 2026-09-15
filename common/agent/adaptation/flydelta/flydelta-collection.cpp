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
            kind != common_flydelta_experiment_job_kind::search_pipeline &&
            !request.capture_manifest_ids.empty()) return false;
    if (kind != common_flydelta_experiment_job_kind::basis &&
            kind != common_flydelta_experiment_job_kind::direction &&
            kind != common_flydelta_experiment_job_kind::search_pipeline &&
            !request.behavior_delta_ids.empty()) return false;
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
            !bounded(request.code_revision) || request.job_variant_id.size() > 512 ||
            !references_empty_except(request, request.kind)) {
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

    if (request.evidence.behavior_key != request.behavior_key) {
        error = "FlyDelta experiment collection evidence behavior does not match request";
        return false;
    }

    common_flydelta_experiment_job job;
    job.id = seed.id + "/job/" + common_flydelta_experiment_job_kind_name(request.kind);
    if (!request.job_variant_id.empty()) job.id += "/" + request.job_variant_id;
    job.kind = request.kind;
    job.seed = std::move(seed);
    job.capture_manifest_ids = request.capture_manifest_ids;
    job.behavior_delta_ids = request.behavior_delta_ids;
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

bool common_flydelta_collect_refinement_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_collection_request & request,
        const common_flydelta_search_observation & observation,
        const common_flydelta_search_decision & decision,
        const common_flydelta_candidate_lineage & lineage,
        common_flydelta_experiment_collection_result & result,
        std::string & error) {
    error.clear();
    result = common_flydelta_experiment_collection_result::disabled;
    if (!request.enabled) return true;
    common_flydelta_search_decision expected;
    if (!common_flydelta_search_observation_validate(observation, error) ||
            !common_flydelta_decide_search_disposition(observation, expected, error) ||
            decision.disposition != expected.disposition ||
            decision.disposition != common_flydelta_search_disposition::refine ||
            !common_flydelta_candidate_lineage_validate(lineage, 64, error) ||
            lineage.candidate_id != observation.candidate_id) {
        if (error.empty()) error = "FlyDelta refinement requires a valid refine disposition";
        return false;
    }
    auto refinement = request;
    refinement.kind = common_flydelta_experiment_job_kind::counterfactual;
    refinement.behavior_delta_ids.clear();
    refinement.training_example_ids.clear();
    refinement.job_variant_id = lineage.candidate_id + "/generation/" +
        std::to_string(lineage.generation);
    return common_flydelta_collect_experiment_job(
        queue_root, queue_limits, refinement, result, error);
}

bool common_flydelta_collect_search_pipeline_refinement_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_collection_request & request,
        const common_flydelta_search_observation & observation,
        const common_flydelta_search_decision & decision,
        const common_flydelta_candidate_lineage & lineage,
        common_flydelta_experiment_collection_result & result,
        std::string & error) {
    error.clear();
    result = common_flydelta_experiment_collection_result::disabled;
    if (!request.enabled) return true;
    common_flydelta_search_decision expected;
    if (request.kind != common_flydelta_experiment_job_kind::search_pipeline ||
            !common_flydelta_search_observation_validate(observation, error) ||
            !common_flydelta_decide_search_disposition(observation, expected, error) ||
            decision.disposition != expected.disposition ||
            decision.disposition != common_flydelta_search_disposition::refine ||
            !common_flydelta_candidate_lineage_validate(lineage, 64, error) ||
            lineage.candidate_id != observation.candidate_id) {
        if (error.empty()) error = "FlyDelta pipeline refinement requires a valid refine disposition";
        return false;
    }
    auto refinement = request;
    refinement.job_variant_id = lineage.candidate_id + "/generation/" +
        std::to_string(lineage.generation);
    return common_flydelta_collect_experiment_job(
        queue_root, queue_limits, refinement, result, error);
}
