#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <utility>

namespace {

common_flydelta_counterfactual_trial arm_trial_from_result(
        const common_flydelta_arm_result & arm,
        bool apply_overlay,
        size_t intervention_count) {
    common_flydelta_counterfactual_trial trial;
    trial.executed = arm.executed;
    trial.verifier_known = arm.verifier_known;
    trial.passed = arm.host_outcome == common_flydelta_counterfactual_outcome::helped;
    trial.quality = arm.quality;
    trial.overlay_applied = apply_overlay;
    trial.intervention_count = intervention_count;
    trial.evidence_ref = arm.provenance_ref.empty() ? arm.generation_ref : arm.provenance_ref;
    return trial;
}

} // namespace

common_flydelta_model_capabilities common_flydelta_model_capabilities_from_primitives(
        const common_flydelta_model_capabilities & primitives,
        const bool has_bounded_arm,
        const bool has_search_runner,
        const bool has_stateful_search) {
    common_flydelta_model_capabilities result = primitives;
    result.bootstrap_zoom = has_bounded_arm && primitives.capture &&
        primitives.overlay && primitives.generation && has_search_runner;
    result.adaptive_alpha = result.bootstrap_zoom;
    result.teacher_forced_margin = has_bounded_arm && primitives.overlay &&
        primitives.generation && primitives.teacher_forced_scoring;
    result.orthogonal_search = has_bounded_arm && primitives.capture &&
        primitives.overlay && has_stateful_search;
    result.representation_augmentation = has_bounded_arm && primitives.capture &&
        primitives.overlay && has_stateful_search;
    return result;
}

common_flydelta_search_pipeline_runner common_flydelta_search_pipeline_runner_from_model_host(
        const common_flydelta_model_host & host,
        std::string job_id,
        std::string context_ref,
        std::string intervention_ref,
        const bool request_capture,
        const bool request_teacher_forced_margin,
        const bool request_generation,
        const bool request_host_verification,
        const size_t max_capture_bytes,
        const size_t max_generated_tokens) {
    return [
            &host,
            job_id = std::move(job_id),
            context_ref = std::move(context_ref),
            intervention_ref = std::move(intervention_ref),
            request_capture,
            request_teacher_forced_margin,
            request_generation,
            request_host_verification,
            max_capture_bytes,
            max_generated_tokens](
            const common_flydelta_experiment_fixture & fixture,
            const common_flydelta_direction_candidate & direction,
            const common_flydelta_layer_candidate * layer,
            const float scale,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_scale_geometry & geometry,
            std::string & error) {
        if (!host.run_bounded_arm) {
            error = "FlyDelta model host has no bounded arm callback";
            return false;
        }

        common_flydelta_arm_request request;
        request.job_id = job_id;
        request.context_ref = context_ref;
        request.fixture_ref = fixture.id;
        request.intervention_ref = intervention_ref;
        if (request.intervention_ref.empty()) {
            request.intervention_ref =
                std::string("direction:") + common_flydelta_direction_kind_name(direction.kind);
        }
        request.alpha = scale;
        request.apply_overlay = apply_overlay;
        request.fresh_context = true;
        request.request_capture = request_capture;
        request.request_teacher_forced_margin = request_teacher_forced_margin;
        request.request_generation = request_generation;
        request.request_host_verification = request_host_verification;
        request.max_capture_bytes = max_capture_bytes;
        request.max_generated_tokens = max_generated_tokens;
        if (layer != nullptr) {
            request.layer_indices = layer->layer_indices;
            request.coefficients.assign(layer->layer_indices.size(), 1.0f);
        }
        if (!apply_overlay) {
            request.layer_indices.clear();
            request.coefficients.clear();
            request.alpha = 0.0f;
        }

        common_flydelta_arm_result arm;
        if (!host.run_bounded_arm(request, arm, error)) return false;
        if (!arm.executed) {
            error = "FlyDelta model host returned an unexecuted bounded arm";
            return false;
        }

        trial = arm_trial_from_result(
            arm, apply_overlay, request.layer_indices.size());
        margin = arm.margin;
        if (!margin.available && arm.margin_available) {
            // Keep compatibility with early host implementations that only
            // returned scalar totals. Such a margin is intentionally marked
            // unavailable because token counts are required for normalization.
            margin = {};
        }
        geometry = {};
        geometry.available = arm.geometry_available;
        geometry.cosine = arm.cosine;
        geometry.progress = arm.progress;
        geometry.leakage = arm.leakage;
        geometry.shift_norm = arm.shift_norm;
        return true;
    };
}

bool common_flydelta_model_host_validate(
        const common_flydelta_model_host & host,
        std::string & error) {
    error.clear();
    if (!host.register_evaluator) {
        error = "FlyDelta model host has no evaluator registration callback";
        return false;
    }
    if (!host.run_bounded_arm) {
        error = "FlyDelta model host has no bounded arm callback";
        return false;
    }
    if (!host.capabilities.host_verification &&
            !host.capabilities.capture &&
            !host.capabilities.overlay &&
            !host.capabilities.generation &&
            !host.capabilities.teacher_forced_scoring) {
        error = "FlyDelta model host advertises no capabilities";
        return false;
    }
    return true;
}

bool common_flydelta_model_adapter_validate(
        const common_flydelta_model_adapter & adapter,
        std::string & error) {
    error.clear();
    if (!adapter.worker_callback) {
        error = "FlyDelta model adapter has no worker callback";
        return false;
    }
    if (!adapter.capabilities.bootstrap_zoom &&
            !adapter.capabilities.adaptive_alpha &&
            !adapter.capabilities.teacher_forced_margin &&
            !adapter.capabilities.orthogonal_search &&
            !adapter.capabilities.representation_augmentation &&
            !adapter.capabilities.host_verification) {
        error = "FlyDelta model adapter advertises no capabilities";
        return false;
    }
    return true;
}

bool common_flydelta_model_adapter_supports_search(
        const common_flydelta_model_adapter & adapter) {
    return static_cast<bool>(adapter.worker_callback) &&
        (adapter.capabilities.bootstrap_zoom ||
         adapter.capabilities.adaptive_alpha ||
         adapter.capabilities.orthogonal_search ||
         adapter.capabilities.representation_augmentation);
}

std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_evaluator(
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_model_capabilities capabilities,
        std::string & error) {
    auto adapter = std::make_shared<common_flydelta_model_adapter>();
    adapter->capabilities = capabilities;
    adapter->worker_callback = [config, callbacks](
            const common_flydelta_experiment_job & job,
            common_flydelta_experiment_worker_result & result,
            std::string & callback_error) {
        common_flydelta_evaluator_result evaluated;
        if (!common_flydelta_evaluate_job(
                job, config, callbacks, evaluated, callback_error)) {
            return false;
        }
        return common_flydelta_worker_result_from_evaluator(
            evaluated, result, callback_error);
    };
    if (!common_flydelta_model_adapter_validate(*adapter, error)) {
        return {};
    }
    error.clear();
    return adapter;
}

std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_host(
        const common_flydelta_model_host & host,
        std::string & error) {
    if (!common_flydelta_model_host_validate(host, error)) {
        return {};
    }

    common_flydelta_evaluator_config config;
    common_flydelta_evaluator_callbacks callbacks;
    if (!host.register_evaluator(config, callbacks, error)) {
        if (error.empty()) {
            error = "FlyDelta model host failed to register evaluator";
        }
        return {};
    }
    if (host.capabilities.orthogonal_search &&
            !callbacks.run_search_pipeline_with_search_state) {
        error = "FlyDelta orthogonal capability requires a post-Bootstrap state-aware runner";
        return {};
    }
    const auto capabilities = common_flydelta_model_capabilities_from_primitives(
        host.capabilities,
        static_cast<bool>(host.run_bounded_arm),
        static_cast<bool>(callbacks.run_search_pipeline ||
            callbacks.run_search_pipeline_with_state),
        static_cast<bool>(callbacks.run_search_pipeline_with_search_state));
    return common_flydelta_model_adapter_from_evaluator(
        config, callbacks, capabilities, error);
}
