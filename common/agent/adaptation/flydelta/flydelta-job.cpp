#include "agent/adaptation/flydelta/flydelta-job.h"

#include <cmath>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

} // namespace

const char * common_flydelta_experiment_job_kind_name(
        common_flydelta_experiment_job_kind kind) {
    switch (kind) {
        case common_flydelta_experiment_job_kind::basis: return "basis";
        case common_flydelta_experiment_job_kind::direction: return "direction";
        case common_flydelta_experiment_job_kind::counterfactual: return "counterfactual";
        case common_flydelta_experiment_job_kind::search_pipeline: return "search_pipeline";
        case common_flydelta_experiment_job_kind::delta_memory: return "delta_memory";
        case common_flydelta_experiment_job_kind::donor_capture: return "donor_capture";
    }
    return "basis";
}

bool common_flydelta_experiment_job_kind_from_name(
        const std::string & value,
        common_flydelta_experiment_job_kind & kind) {
    if (value == "basis") kind = common_flydelta_experiment_job_kind::basis;
    else if (value == "direction") kind = common_flydelta_experiment_job_kind::direction;
    else if (value == "counterfactual") kind = common_flydelta_experiment_job_kind::counterfactual;
    else if (value == "search_pipeline") kind = common_flydelta_experiment_job_kind::search_pipeline;
    else if (value == "delta_memory") kind = common_flydelta_experiment_job_kind::delta_memory;
    else if (value == "donor_capture") kind = common_flydelta_experiment_job_kind::donor_capture;
    else return false;
    return true;
}

bool common_flydelta_experiment_job_validate(
        const common_flydelta_experiment_job & job,
        size_t max_references,
        std::string & error) {
    error.clear();
    if (job.schema_version != 1 || !bounded(job.id) || !bounded(job.code_revision) ||
            (!job.bootstrap_zoom_state_ref.empty() && !bounded(job.bootstrap_zoom_state_ref)) ||
            (!job.search_state_ref.empty() && !bounded(job.search_state_ref)) ||
            (!job.representation_augmentation_state_ref.empty() &&
                !bounded(job.representation_augmentation_state_ref)) ||
            max_references == 0 || !common_flydelta_experiment_seed_validate(job.seed, error) ||
            !std::isfinite(job.learning_rate) || job.learning_rate <= 0.0f ||
            !std::isfinite(job.decay) || job.decay < 0.0f || job.decay > 1.0f) {
        if (error.empty()) error = "FlyDelta experiment job identity or parameters are invalid";
        return false;
    }
    const auto check_refs = [&](const std::vector<std::string> & refs) {
        if (refs.empty() || refs.size() > max_references) return false;
        for (const auto & ref : refs) if (!bounded(ref)) return false;
        return true;
    };
    if (job.kind != common_flydelta_experiment_job_kind::donor_capture &&
            !job.capture_candidate_ids.empty()) {
        error = "FlyDelta capture-candidate references are only valid for donor capture jobs";
        return false;
    }
    if (job.kind == common_flydelta_experiment_job_kind::donor_capture &&
            (!check_refs(job.capture_candidate_ids) ||
             !job.capture_manifest_ids.empty() || !job.behavior_delta_ids.empty() ||
             !job.training_example_ids.empty())) {
        error = "FlyDelta donor capture job requires bounded capture-candidate references";
        return false;
    }
    if ((job.kind == common_flydelta_experiment_job_kind::basis ||
         job.kind == common_flydelta_experiment_job_kind::direction) &&
            !check_refs(job.behavior_delta_ids)) {
        error = "FlyDelta direction/basis job requires bounded behavior-delta references";
        return false;
    }
    if (job.kind == common_flydelta_experiment_job_kind::counterfactual) {
        if (!check_refs(job.capture_manifest_ids) ||
                !common_flydelta_alpha_search_config_validate(job.alpha_search, error)) {
            if (error.empty()) error = "FlyDelta counterfactual job requires captures and alpha grid";
            return false;
        }
    }
    if (job.kind == common_flydelta_experiment_job_kind::search_pipeline &&
            (!check_refs(job.capture_manifest_ids) || !check_refs(job.behavior_delta_ids) ||
             !common_flydelta_alpha_search_config_validate(job.alpha_search, error))) {
        if (error.empty()) error = "FlyDelta search pipeline job requires captures, deltas and bounds";
        return false;
    }
    if (job.kind == common_flydelta_experiment_job_kind::delta_memory &&
            (!check_refs(job.training_example_ids) ||
             job.seed.split != common_flydelta_training_split::train)) {
        error = "FlyDelta DeltaMemory job requires train examples only";
        return false;
    }
    return true;
}

std::string common_flydelta_experiment_job_to_json(
        const common_flydelta_experiment_job & job) {
    return json{
        {"schema_version", job.schema_version},
        {"id", job.id},
        {"kind", common_flydelta_experiment_job_kind_name(job.kind)},
        {"seed", {
            {"id", job.seed.id},
            {"behavior_key", job.seed.behavior_key},
            {"source", common_adaptation_evidence_source_name(job.seed.source)},
            {"split", common_flydelta_training_split_name(job.seed.split)},
            {"scope", {
                {"namespace_id", job.seed.scope.namespace_id},
                {"project_id", job.seed.scope.project_id},
                {"session_id", job.seed.scope.session_id},
                {"turn_id", job.seed.scope.turn_id},
            }},
            {"task_fingerprint", job.seed.task_fingerprint},
            {"model_profile_fingerprint", job.seed.model_profile_fingerprint},
            {"tokenizer_fingerprint", job.seed.tokenizer_fingerprint},
            {"template_fingerprint", job.seed.template_fingerprint},
            {"execution_context_fingerprint", job.seed.execution_context_fingerprint},
            {"baseline_ref", job.seed.baseline_ref},
            {"candidate_ref", job.seed.candidate_ref},
            {"verifier_ref", job.seed.verifier_ref},
            {"evidence_ref", job.seed.evidence_ref},
            {"transaction_ids", job.seed.transaction_ids},
        }},
        {"capture_candidate_ids", job.capture_candidate_ids},
        {"capture_manifest_ids", job.capture_manifest_ids},
        {"behavior_delta_ids", job.behavior_delta_ids},
        {"training_example_ids", job.training_example_ids},
        {"bootstrap_zoom_state_ref", job.bootstrap_zoom_state_ref},
        {"search_state_ref", job.search_state_ref},
        {"representation_augmentation_state_ref", job.representation_augmentation_state_ref},
        {"alpha_search", {
            {"candidates", job.alpha_search.candidates},
            {"magnitude_penalty", job.alpha_search.magnitude_penalty},
            {"max_candidates", job.alpha_search.max_candidates},
        }},
        {"learning_rate", job.learning_rate},
        {"decay", job.decay},
        {"code_revision", job.code_revision},
    }.dump();
}

bool common_flydelta_experiment_job_from_json(
        const std::string & text,
        common_flydelta_experiment_job & job,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        job = {};
        job.schema_version = value.value("schema_version", 0);
        job.id = value.value("id", "");
        if (!common_flydelta_experiment_job_kind_from_name(value.value("kind", ""), job.kind)) {
            error = "FlyDelta experiment job kind is invalid";
            return false;
        }
        const auto seed = value.value("seed", json::object());
        job.seed.id = seed.value("id", "");
        job.seed.behavior_key = seed.value("behavior_key", "");
        const auto source = common_adaptation_evidence_source_from_name(
            seed.value("source", ""));
        if (!source || !common_flydelta_training_split_from_name(
                seed.value("split", ""), job.seed.split)) {
            error = "FlyDelta experiment job seed source or split is invalid";
            return false;
        }
        job.seed.source = *source;
        const auto scope = seed.value("scope", json::object());
        job.seed.scope.namespace_id = scope.value("namespace_id", "");
        job.seed.scope.project_id = scope.value("project_id", "");
        job.seed.scope.session_id = scope.value("session_id", "");
        job.seed.scope.turn_id = scope.value("turn_id", "");
        job.seed.task_fingerprint = seed.value("task_fingerprint", "");
        job.seed.model_profile_fingerprint = seed.value("model_profile_fingerprint", "");
        job.seed.tokenizer_fingerprint = seed.value("tokenizer_fingerprint", "");
        job.seed.template_fingerprint = seed.value("template_fingerprint", "");
        job.seed.execution_context_fingerprint = seed.value("execution_context_fingerprint", "");
        job.seed.baseline_ref = seed.value("baseline_ref", "");
        job.seed.candidate_ref = seed.value("candidate_ref", "");
        job.seed.verifier_ref = seed.value("verifier_ref", "");
        job.seed.evidence_ref = seed.value("evidence_ref", "");
        job.seed.transaction_ids = seed.value("transaction_ids", std::vector<std::string>{});
        job.capture_candidate_ids = value.value("capture_candidate_ids", std::vector<std::string>{});
        job.capture_manifest_ids = value.value("capture_manifest_ids", std::vector<std::string>{});
        job.behavior_delta_ids = value.value("behavior_delta_ids", std::vector<std::string>{});
        job.training_example_ids = value.value("training_example_ids", std::vector<std::string>{});
        job.bootstrap_zoom_state_ref = value.value("bootstrap_zoom_state_ref", "");
        job.search_state_ref = value.value("search_state_ref", "");
        job.representation_augmentation_state_ref = value.value(
            "representation_augmentation_state_ref", "");
        const auto alpha = value.value("alpha_search", json::object());
        job.alpha_search.candidates = alpha.value("candidates", std::vector<float>{});
        job.alpha_search.magnitude_penalty = alpha.value("magnitude_penalty", 0.0f);
        job.alpha_search.max_candidates = alpha.value("max_candidates", 0U);
        job.learning_rate = value.value("learning_rate", 0.0f);
        job.decay = value.value("decay", 0.0f);
        job.code_revision = value.value("code_revision", "");
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta experiment job JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_experiment_job_validate(job, 128, error);
}
