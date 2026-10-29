#include "agent/adaptation/flydelta/flydelta-collection.h"

#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "hash/hash.h"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace {

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool references_empty_except(
        const common_flydelta_experiment_collection_request & request,
        common_flydelta_experiment_job_kind kind) {
    if (kind != common_flydelta_experiment_job_kind::donor_capture &&
            !request.capture_candidate_ids.empty()) return false;
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
            (!request.bootstrap_zoom_state_ref.empty() &&
                !bounded(request.bootstrap_zoom_state_ref)) ||
            (!request.search_state_ref.empty() && !bounded(request.search_state_ref)) ||
            (!request.representation_augmentation_state_ref.empty() &&
                !bounded(request.representation_augmentation_state_ref)) ||
            (!request.teaching_material_group_ref.empty() &&
                !bounded(request.teaching_material_group_ref)) ||
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
    job.capture_candidate_ids = request.capture_candidate_ids;
    job.capture_manifest_ids = request.capture_manifest_ids;
    job.behavior_delta_ids = request.behavior_delta_ids;
    job.training_example_ids = request.training_example_ids;
    job.bootstrap_zoom_state_ref = request.bootstrap_zoom_state_ref;
    job.search_state_ref = request.search_state_ref;
    job.representation_augmentation_state_ref =
        request.representation_augmentation_state_ref;
    job.teaching_material_group_ref = request.teaching_material_group_ref;
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

bool common_flydelta_collect_concept_localization_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_job & parent_job,
        const common_flydelta_concept_localization_request & request,
        common_flydelta_experiment_collection_result & result, std::string & error) {
    result = common_flydelta_experiment_collection_result::disabled;
    error.clear();
    if (queue_root.empty() || parent_job.kind != common_flydelta_experiment_job_kind::search_pipeline ||
            !common_flydelta_experiment_job_validate(parent_job, 128, error) || request.layers.empty() ||
            !common_flydelta_concept_localization_request_validate(request, error) ||
            request.origin_direction_ref != parent_job.seed.candidate_ref) {
        if (error.empty()) error = "FlyDelta concept localization requires an explicit request and valid search parent";
        return false;
    }
    auto job = parent_job;
    job.kind = common_flydelta_experiment_job_kind::concept_capture;
    job.concept_localization = request;
    job.bootstrap_zoom_state_ref.clear();
    job.search_state_ref.clear();
    job.representation_augmentation_state_ref.clear();
    job.capture_candidate_ids.clear();
    job.training_example_ids.clear();
    const auto identity = common_flydelta_concept_localization_material_key(job) + "\n" + parent_job.id;
    job.id = "flydelta://job/concept-localization/" + hash_sha256_hex(identity.data(), identity.size());
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

bool common_flydelta_collect_next_action_job(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        const common_flydelta_experiment_job & parent_job,
        common_flydelta_next_action next_action,
        const std::string & bootstrap_zoom_state_ref,
        const std::string & search_state_ref,
        const std::string & representation_augmentation_state_ref,
        const std::string & graft_direction_ref,
        common_flydelta_experiment_collection_result & result,
        std::string & error) {
    error.clear();
    result = common_flydelta_experiment_collection_result::disabled;
    if (next_action == common_flydelta_next_action::stop ||
            next_action == common_flydelta_next_action::retain) {
        return true;
    }
    if (queue_root.empty() ||
            (parent_job.kind != common_flydelta_experiment_job_kind::search_pipeline &&
             parent_job.kind != common_flydelta_experiment_job_kind::concept_capture &&
             parent_job.kind != common_flydelta_experiment_job_kind::concept_synthesis) ||
            parent_job.id.empty() || parent_job.seed.id.empty()) {
        error = "FlyDelta next-action scheduling requires a bounded FlyDelta parent job";
        return false;
    }
    const std::string state_key = bootstrap_zoom_state_ref + "\n" +
        search_state_ref + "\n" + representation_augmentation_state_ref + "\n" +
        parent_job.teaching_material_group_ref + "\n" + graft_direction_ref;
    common_flydelta_experiment_job follow_up = parent_job;
    const std::string logical_identity = parent_job.id + "\n" +
        common_flydelta_next_action_name(next_action) + "\n" + state_key;
    const std::string legacy_id = parent_job.id + "/next/" +
        common_flydelta_next_action_name(next_action) + "/" +
        hash_sha256_hex(state_key.data(), state_key.size()).substr(0, 32);
    // Keep continuation identity bounded independently of lineage depth.
    follow_up.id = "flydelta://job/continuation/" +
        hash_sha256_hex(logical_identity.data(), logical_identity.size());
    follow_up.bootstrap_zoom_state_ref = bootstrap_zoom_state_ref;
    follow_up.search_state_ref = search_state_ref;
    follow_up.representation_augmentation_state_ref =
        representation_augmentation_state_ref;
    if (!graft_direction_ref.empty()) {
        if (!bounded(graft_direction_ref)) {
            error = "FlyDelta graft direction reference is invalid";
            return false;
        }
        follow_up.seed.candidate_ref = graft_direction_ref;
    }
    if (next_action == common_flydelta_next_action::prepare_concept_material ||
            next_action == common_flydelta_next_action::run_concept_synthesis) {
        if (parent_job.teaching_material_group_ref.empty()) {
            error = "FlyDelta concept material scheduling requires teaching material";
            return false;
        }
        follow_up.kind = next_action == common_flydelta_next_action::prepare_concept_material
            ? common_flydelta_experiment_job_kind::concept_capture
            : common_flydelta_experiment_job_kind::concept_synthesis;
        // Preserve the originating search material through the concept jobs.
        // Concept capture/synthesis consumes teaching material in addition to
        // the original captures and deltas; it must not strand those refs when
        // synthesis hands an experimental direction back to ordinary search.
        follow_up.capture_candidate_ids.clear();
        follow_up.training_example_ids.clear();
        // Keep the originating search/augmentation refs on concept jobs.
        // They identify the localized WHERE surface used by concept capture
        // and must survive the capture -> synthesis transition.
    } else if (parent_job.kind == common_flydelta_experiment_job_kind::concept_synthesis &&
            next_action == common_flydelta_next_action::run_bootstrap) {
        if (graft_direction_ref.empty()) {
            error = "FlyDelta graft search scheduling requires a persisted direction reference";
            return false;
        }
        follow_up.kind = common_flydelta_experiment_job_kind::search_pipeline;
        follow_up.capture_candidate_ids.clear();
        follow_up.training_example_ids.clear();
        // Keep BootstrapZoom state: the graft changes WHAT through
        // candidate_ref, but resumes HOW MUCH at the existing WHERE anchor.
        follow_up.search_state_ref.clear();
        follow_up.representation_augmentation_state_ref.clear();
        if (!follow_up.concept_localization.layers.empty()) {
            // Layer-specific material permits a fresh WHERE challenge. The
            // old origin remains immutable provenance, not a resume handle.
            follow_up.bootstrap_zoom_state_ref.clear();
        }
    }
    if (follow_up.id.size() > 512) {
        error = "FlyDelta next-action job identity exceeds its bound";
        return false;
    }
    if (!common_flydelta_experiment_job_validate(follow_up, 128, error)) {
        return false;
    }
    bool contains = false;
    if (legacy_id.size() <= 512) {
        if (!common_flydelta_experiment_queue_contains(queue_root, legacy_id, contains, error)) return false;
        if (contains) {
            result = common_flydelta_experiment_collection_result::already_present;
            return true;
        }
    }
    if (!common_flydelta_experiment_queue_contains(
            queue_root, follow_up.id, contains, error)) return false;
    if (contains) {
        result = common_flydelta_experiment_collection_result::already_present;
        return true;
    }
    if (!common_flydelta_experiment_queue_enqueue(
            queue_root, follow_up, queue_limits, error)) return false;
    result = common_flydelta_experiment_collection_result::enqueued;
    return true;
}

bool common_flydelta_recover_anchor_continuations(
        const common_learning_lifecycle_store & store,
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & queue_limits,
        std::string & error) {
    error.clear();
    const auto records = store.list(error);
    if (!error.empty()) return false;
    using json = nlohmann::ordered_json;
    for (const auto & record : records) {
        if (record.kind != common_learning_lifecycle_kind::flydelta_experiment) continue;
        const auto state = json::parse(record.payload_json, nullptr, false);
        if (!state.is_object() || !state.contains("anchor_continuation")) continue;
        try {
            const auto digest = hash_sha256_hex(record.payload_json.data(), record.payload_json.size());
            if (record.content_hash != "sha256:" + digest ||
                    record.subject_id != "flydelta://state/orchestration/" + digest.substr(0, 32) ||
                    state.value("kind", "") != "flydelta_daemon_orchestration_state" ||
                    !state.value("search_anchor_pending", false) || state.value("search_anchor_ref", "").empty() ||
                    state.value("phase", "") != "deep_controls") {
                error = "FlyDelta anchor continuation has invalid identity or publication state";
                return false;
            }
            const auto limit = state.value("search_model_call_limit", size_t{0});
            const auto used = state.value("search_model_calls_used", size_t{0});
            if (limit == 0 || limit > 65536 || used >= limit) {
                error = "FlyDelta anchor continuation has no remaining execution budget";
                return false;
            }
            const auto & intent = state.at("anchor_continuation");
            common_flydelta_experiment_job parent;
            if (!common_flydelta_experiment_job_from_json(intent.at("parent_job").dump(), parent, error)) return false;
            if (parent.kind != common_flydelta_experiment_job_kind::search_pipeline ||
                    parent.seed.scope.namespace_id != record.namespace_id ||
                    parent.seed.scope.project_id != record.project_id ||
                    parent.seed.scope.session_id != record.session_id) {
                error = "FlyDelta anchor continuation scope differs from its journal record";
                return false;
            }
            common_flydelta_experiment_queue_state parent_state;
            if (!common_flydelta_experiment_queue_job_state(
                    queue_root, parent, queue_limits, parent_state, error)) return false;
            if (parent_state == common_flydelta_experiment_queue_state::cancelled) continue;
            if (parent_state == common_flydelta_experiment_queue_state::failed) {
                error = "FlyDelta anchor parent failed; automatic recovery requires reconciliation";
                return false;
            }
            common_flydelta_experiment_collection_result result;
            if (!common_flydelta_collect_next_action_job(queue_root, queue_limits, parent,
                    common_flydelta_next_action::run_deep_controls,
                    intent.value("bootstrap_state_ref", ""), record.subject_id,
                    intent.value("representation_state_ref", ""), {}, result, error)) return false;
            // The checkpoint and its intent already commit this search slice.
            // Requeueing the parent as well would replay the old search and
            // produce a second branch beside the recovered continuation.
            if (!common_flydelta_experiment_queue_acknowledge_recovered_result(queue_root,
                    parent, "Search continued from durable checkpoint; full worker report may be incomplete",
                    queue_limits, error)) return false;
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta anchor continuation is malformed: ") + exception.what();
            return false;
        }
    }
    return true;
}
