#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "hash/hash.h"

#include <cmath>
#include <string>
#include <unordered_set>
#include <utility>

const char * common_flydelta_arm_execution_path_name(
        const common_flydelta_arm_execution_metrics::path value) {
    switch (value) {
        case common_flydelta_arm_execution_metrics::path::unknown: return "unknown";
        case common_flydelta_arm_execution_metrics::path::scalar: return "scalar";
        case common_flydelta_arm_execution_metrics::path::scalar_fallback: return "scalar_fallback";
        case common_flydelta_arm_execution_metrics::path::backend_batch: return "backend_batch";
        case common_flydelta_arm_execution_metrics::path::device_batch: return "device_batch";
    }
    return "unknown";
}

bool common_flydelta_arm_request_validate(
        const common_flydelta_arm_request & request,
        std::string & error) {
    error.clear();
    if (request.schema_version != 1) {
        error = "unsupported FlyDelta arm request schema";
        return false;
    }
    if (request.arm_id.empty() || request.arm_id.size() > 512) {
        error = "FlyDelta arm request arm id is invalid";
        return false;
    }
    if (!std::isfinite(request.alpha) || request.alpha < 0.0f) {
        error = "FlyDelta arm request alpha is invalid";
        return false;
    }
    if (request.layer_indices.size() != request.coefficients.size()) {
        error = "FlyDelta arm request layer/coefficient count mismatch";
        return false;
    }
    for (const float coefficient : request.coefficients) {
        if (!std::isfinite(coefficient)) {
            error = "FlyDelta arm request coefficient is invalid";
            return false;
        }
    }
    if (request.apply_overlay && request.layer_indices.empty()) {
        error = "FlyDelta overlay arm has no layers";
        return false;
    }
    if (request.apply_overlay && !request.fresh_context) {
        error = "FlyDelta overlay arm must request a fresh context";
        return false;
    }
    return true;
}

bool common_flydelta_arm_result_validate(
        const common_flydelta_arm_result & result,
        std::string & error) {
    error.clear();
    if (result.schema_version != 1) {
        error = "unsupported FlyDelta arm result schema";
        return false;
    }
    if (result.arm_id.size() > 512) {
        error = "FlyDelta arm result arm id is invalid";
        return false;
    }
    if (!std::isfinite(result.requested_alpha) ||
            !std::isfinite(result.executed_alpha) ||
            result.requested_alpha < 0.0f || result.executed_alpha < 0.0f) {
        error = "FlyDelta arm result alpha is invalid";
        return false;
    }
    if (result.host_evaluated && !result.executed) {
        error = "FlyDelta arm result cannot evaluate an unexecuted arm";
        return false;
    }
    if (result.verifier_known && !result.host_evaluated) {
        error = "FlyDelta arm result verifier state requires host evaluation";
        return false;
    }
    const float dose_epsilon = 1.0e-6f;
    if (!result.dose_safety_limited &&
            std::fabs(result.executed_alpha - result.requested_alpha) > dose_epsilon) {
        error = "FlyDelta arm result changed dose without safety-limit attribution";
        return false;
    }
    if (result.dose_safety_limited &&
            result.executed_alpha > result.requested_alpha + dose_epsilon) {
        error = "FlyDelta safety-limited arm increased the requested dose";
        return false;
    }
    if (result.geometry_available &&
            (!std::isfinite(result.cosine) || !std::isfinite(result.progress) ||
             !std::isfinite(result.leakage) || !std::isfinite(result.shift_norm))) {
        error = "FlyDelta arm result geometry is invalid";
        return false;
    }
    if (result.execution_metrics.schema_version != 1) {
        error = "unsupported FlyDelta arm execution metrics schema";
        return false;
    }
    if (result.execution_metrics.available &&
            (!std::isfinite(result.execution_metrics.model_ms) ||
             !std::isfinite(result.execution_metrics.teacher_forced_ms) ||
             !std::isfinite(result.execution_metrics.generation_ms) ||
             result.execution_metrics.model_ms < 0.0f ||
             result.execution_metrics.teacher_forced_ms < 0.0f ||
             result.execution_metrics.generation_ms < 0.0f)) {
        error = "FlyDelta arm execution metrics are invalid";
        return false;
    }
    if (result.execution_metrics.fallback_reason.size() > 512) {
        error = "FlyDelta arm execution fallback reason is too long";
        return false;
    }
    const auto execution_path = result.execution_metrics.execution_path;
    const bool batched_path =
        execution_path == common_flydelta_arm_execution_metrics::path::backend_batch ||
        execution_path == common_flydelta_arm_execution_metrics::path::device_batch;
    if (execution_path == common_flydelta_arm_execution_metrics::path::scalar_fallback &&
            (result.execution_metrics.batched_execution_used ||
             result.execution_metrics.fallback_reason.empty())) {
        error = "FlyDelta scalar fallback telemetry is inconsistent";
        return false;
    }
    if (batched_path && !result.execution_metrics.batched_execution_used) {
        error = "FlyDelta batched execution path is missing its batch flag";
        return false;
    }
    if (result.execution_metrics.device_reduction_used && result.geometry_available &&
            result.execution_metrics.diagnostics_bytes_to_host <
                common_flydelta_compact_geometry_bytes) {
        error = "FlyDelta device diagnostics transfer is not compact geometry sized";
        return false;
    }
    if (!common_flydelta_decision_margin_validate(result.margin, error) ||
            !common_flydelta_margin_comparison_validate(result.margin_comparison, error)) {
        return false;
    }
    return true;
}

namespace {

bool close_enough(const float left, const float right, const float tolerance) {
    return std::fabs(left - right) <= tolerance;
}

bool margin_replay_equivalent(
        const common_flydelta_decision_margin & expected,
        const common_flydelta_decision_margin & actual,
        const float tolerance) {
    return expected.available == actual.available &&
        close_enough(expected.positive_total_logprob,
                     actual.positive_total_logprob, tolerance) &&
        close_enough(expected.negative_total_logprob,
                     actual.negative_total_logprob, tolerance) &&
        expected.positive_token_count == actual.positive_token_count &&
        expected.negative_token_count == actual.negative_token_count;
}

} // namespace

bool common_flydelta_arm_result_replay_equivalent(
        const common_flydelta_arm_result & expected,
        const common_flydelta_arm_result & actual,
        const float absolute_tolerance,
        std::string & error) {
    error.clear();
    if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0f) {
        error = "FlyDelta replay tolerance is invalid";
        return false;
    }
    if (expected.arm_id != actual.arm_id || expected.executed != actual.executed ||
            expected.dose_safety_limited != actual.dose_safety_limited ||
            expected.geometry_available != actual.geometry_available ||
            expected.margin_available != actual.margin_available ||
            expected.generation_available != actual.generation_available ||
            expected.host_evaluated != actual.host_evaluated ||
            expected.verifier_known != actual.verifier_known ||
            expected.host_outcome != actual.host_outcome) {
        error = "FlyDelta replay result flags or identity differ";
        return false;
    }
    if (!close_enough(expected.requested_alpha, actual.requested_alpha, absolute_tolerance) ||
            !close_enough(expected.executed_alpha, actual.executed_alpha, absolute_tolerance) ||
            (expected.geometry_available &&
             (!close_enough(expected.cosine, actual.cosine, absolute_tolerance) ||
              !close_enough(expected.progress, actual.progress, absolute_tolerance) ||
              !close_enough(expected.leakage, actual.leakage, absolute_tolerance) ||
              !close_enough(expected.shift_norm, actual.shift_norm, absolute_tolerance)))) {
        error = "FlyDelta replay geometry or dose differs";
        return false;
    }
    if (!margin_replay_equivalent(expected.margin, actual.margin, absolute_tolerance) ||
            !margin_replay_equivalent(expected.margin_comparison.baseline,
                                      actual.margin_comparison.baseline,
                                      absolute_tolerance) ||
            !margin_replay_equivalent(expected.margin_comparison.candidate,
                                      actual.margin_comparison.candidate,
                                      absolute_tolerance) ||
            expected.margin_comparison.available != actual.margin_comparison.available ||
            !close_enough(expected.margin_total, actual.margin_total, absolute_tolerance) ||
            !close_enough(expected.margin_normalized, actual.margin_normalized, absolute_tolerance) ||
            !close_enough(expected.margin_delta_total, actual.margin_delta_total, absolute_tolerance) ||
            !close_enough(expected.margin_delta_normalized, actual.margin_delta_normalized,
                          absolute_tolerance) ||
            !close_enough(expected.quality, actual.quality, absolute_tolerance)) {
        error = "FlyDelta replay margin or quality differs";
        return false;
    }
    return true;
}

bool common_flydelta_arm_batch_request_validate(
        const common_flydelta_arm_batch_request & request,
        std::string & error) {
    error.clear();
    if (request.schema_version != 1 || request.arms.empty() || request.arms.size() > 256) {
        error = "FlyDelta arm batch request is invalid";
        return false;
    }
    std::unordered_set<std::string> arm_ids;
    for (const auto & arm : request.arms) {
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        if (!arm_ids.insert(arm.arm_id).second) {
            error = "FlyDelta arm batch request reuses an arm identity";
            return false;
        }
    }
    return true;
}

bool common_flydelta_arm_batch_result_validate(
        const common_flydelta_arm_batch_result & result,
        const common_flydelta_arm_batch_request & request,
        std::string & error) {
    error.clear();
    if (result.schema_version != 1 || result.arms.size() != request.arms.size()) {
        error = "FlyDelta arm batch result count or schema is invalid";
        return false;
    }
    for (size_t index = 0; index < result.arms.size(); ++index) {
        if (!common_flydelta_arm_result_validate(result.arms[index], error)) return false;
        if (result.arms[index].arm_id != request.arms[index].arm_id) {
            error = "FlyDelta arm batch result order or identity is invalid";
            return false;
        }
    }
    return true;
}

bool common_flydelta_run_bounded_arm_batch(
        const common_flydelta_model_host & host,
        const common_flydelta_arm_batch_request & request,
        common_flydelta_arm_batch_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_arm_batch_request_validate(request, error)) return false;
    if (host.run_bounded_arm_batch) {
        if (!host.run_bounded_arm_batch(request, result, error)) return false;
        for (auto & arm : result.arms) {
            if (arm.execution_metrics.execution_path ==
                    common_flydelta_arm_execution_metrics::path::unknown) {
                arm.execution_metrics.available = true;
                arm.execution_metrics.batched_execution_used = true;
                arm.execution_metrics.execution_path =
                    common_flydelta_arm_execution_metrics::path::backend_batch;
            }
        }
        return common_flydelta_arm_batch_result_validate(result, request, error);
    }
    if (!host.run_bounded_arm) {
        error = "FlyDelta model host has no bounded arm or batch callback";
        return false;
    }
    result.schema_version = request.schema_version;
    result.arms.reserve(request.arms.size());
    for (const auto & arm_request : request.arms) {
        common_flydelta_arm_result arm_result;
        if (!host.run_bounded_arm(arm_request, arm_result, error)) return false;
        if (!common_flydelta_arm_result_validate(arm_result, error)) return false;
        if (arm_result.arm_id != arm_request.arm_id) {
            error = "FlyDelta arm fallback result identity is invalid";
            return false;
        }
        if (arm_result.execution_metrics.execution_path ==
                common_flydelta_arm_execution_metrics::path::unknown) {
            arm_result.execution_metrics.available = true;
            arm_result.execution_metrics.batched_execution_used = false;
            arm_result.execution_metrics.execution_path =
                common_flydelta_arm_execution_metrics::path::scalar_fallback;
            arm_result.execution_metrics.fallback_reason = "batch_callback_unavailable";
        }
        result.arms.push_back(std::move(arm_result));
    }
    return common_flydelta_arm_batch_result_validate(result, request, error);
}

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
    // Algorithm flags are derived below. Do not carry stale phase flags from
    // a caller across runtime registration or a resumed search.
    result.bootstrap_zoom = false;
    result.adaptive_alpha = false;
    result.teacher_forced_margin = false;
    result.orthogonal_search = false;
    result.representation_augmentation = false;
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
        if (!host.run_bounded_arm && !host.run_bounded_arm_batch) {
            error = "FlyDelta model host has no bounded arm or batch callback";
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

        std::string arm_identity = request.job_id + "\n" + request.context_ref + "\n" +
            request.fixture_ref + "\n" + request.intervention_ref + "\n" +
            std::to_string(request.alpha) + "\n" +
            (request.apply_overlay ? "overlay" : "baseline") + "\n" +
            (request.request_capture ? "capture" : "no-capture") + "\n" +
            (request.request_teacher_forced_margin ? "margin" : "no-margin") + "\n" +
            (request.request_generation ? "generation" : "no-generation") + "\n" +
            (request.request_host_verification ? "verification" : "no-verification") + "\n" +
            std::to_string(request.max_capture_bytes) + "\n" +
            std::to_string(request.max_generated_tokens);
        for (const uint32_t layer_index : request.layer_indices) {
            arm_identity += "\n" + std::to_string(layer_index);
        }
        for (const float coefficient : request.coefficients) {
            arm_identity += "\n" + std::to_string(coefficient);
        }
        request.arm_id = "flydelta://arm/" +
            hash_sha256_hex(arm_identity.data(), arm_identity.size()).substr(0, 32);

        if (!common_flydelta_arm_request_validate(request, error)) return false;

        common_flydelta_arm_batch_request batch_request;
        batch_request.arms.push_back(request);
        common_flydelta_arm_batch_result batch_result;
        if (!common_flydelta_run_bounded_arm_batch(host, batch_request, batch_result, error)) {
            return false;
        }
        if (batch_result.arms.size() != 1) {
            error = "FlyDelta model host returned an invalid single-arm batch result";
            return false;
        }
        const common_flydelta_arm_result & arm = batch_result.arms.front();
        if (!arm.executed) {
            error = "FlyDelta model host returned an unexecuted bounded arm";
            return false;
        }
        if (arm.arm_id != request.arm_id) {
            error = "FlyDelta model host returned a mismatched arm id";
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

common_flydelta_search_pipeline_batch_runner
common_flydelta_search_pipeline_batch_runner_from_model_host(
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
            const std::vector<common_flydelta_intervention_region_candidate> & candidates,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_scale_geometry> & geometries,
            std::vector<bool> & geometry_available,
            std::string & error) {
        if (!host.run_bounded_arm && !host.run_bounded_arm_batch) {
            error = "FlyDelta model host has no bounded arm or batch callback";
            return false;
        }
        common_flydelta_arm_batch_request batch_request;
        batch_request.arms.reserve(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            const auto & candidate = candidates[index];
            common_flydelta_arm_request request;
            request.job_id = job_id;
            request.context_ref = context_ref;
            request.fixture_ref = fixture.id;
            request.intervention_ref = intervention_ref.empty()
                ? std::string("direction:") + common_flydelta_direction_kind_name(direction.kind)
                : intervention_ref;
            request.layer_indices = candidate.layer_indices;
            request.coefficients.assign(candidate.layer_indices.size(), 1.0f);
            request.alpha = candidate.total_scale;
            request.apply_overlay = true;
            request.fresh_context = true;
            request.request_capture = request_capture;
            request.request_teacher_forced_margin = request_teacher_forced_margin;
            request.request_generation = request_generation;
            request.request_host_verification = request_host_verification;
            request.max_capture_bytes = max_capture_bytes;
            request.max_generated_tokens = max_generated_tokens;
            std::string identity = request.job_id + "\n" + request.context_ref + "\n" +
                request.fixture_ref + "\n" + request.intervention_ref + "\n" +
                std::to_string(index) + "\n" + std::to_string(request.alpha);
            for (const uint32_t layer_index : request.layer_indices) {
                identity += "\n" + std::to_string(layer_index);
            }
            request.arm_id = "flydelta://arm/" +
                hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
            if (!common_flydelta_arm_request_validate(request, error)) return false;
            batch_request.arms.push_back(std::move(request));
        }

        common_flydelta_arm_batch_result batch_result;
        if (!common_flydelta_run_bounded_arm_batch(
                host, batch_request, batch_result, error)) return false;
        if (batch_result.arms.size() != candidates.size()) {
            error = "FlyDelta model host returned an incomplete region batch";
            return false;
        }
        trials.clear();
        margins.clear();
        geometries.clear();
        geometry_available.clear();
        trials.reserve(candidates.size());
        margins.reserve(candidates.size());
        geometries.reserve(candidates.size());
        geometry_available.reserve(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            const auto & arm = batch_result.arms[index];
            if (!arm.executed || arm.arm_id != batch_request.arms[index].arm_id) {
                error = "FlyDelta model host returned an invalid region batch arm";
                return false;
            }
            trials.push_back(arm_trial_from_result(
                arm, true, candidates[index].layer_indices.size()));
            margins.push_back(arm.margin);
            common_flydelta_scale_geometry geometry;
            geometry.available = arm.geometry_available;
            geometry.cosine = arm.cosine;
            geometry.progress = arm.progress;
            geometry.leakage = arm.leakage;
            geometry.shift_norm = arm.shift_norm;
            geometries.push_back(std::move(geometry));
            geometry_available.push_back(arm.geometry_available);
        }
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
    if (!host.run_bounded_arm && !host.run_bounded_arm_batch) {
        error = "FlyDelta model host has no bounded arm or batch callback";
        return false;
    }
    if (host.capabilities.bounded_arm_batch && !host.run_bounded_arm_batch) {
        error = "FlyDelta model host advertises batch execution without a batch callback";
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
    auto capabilities = common_flydelta_model_capabilities_from_primitives(
        host.capabilities,
        static_cast<bool>(host.run_bounded_arm || host.run_bounded_arm_batch),
        static_cast<bool>(callbacks.run_search_pipeline ||
            callbacks.run_search_pipeline_with_state),
        static_cast<bool>(callbacks.run_search_pipeline_with_search_state));
    // Registration is the source of truth for backend availability. A caller
    // cannot advertise a batch path that was not actually bound.
    capabilities.bounded_arm_batch = static_cast<bool>(host.run_bounded_arm_batch);
    return common_flydelta_model_adapter_from_evaluator(
        config, callbacks, capabilities, error);
}
