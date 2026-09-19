#pragma once

#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Generic model-facing primitive shared by every FlyDelta search phase. The
// request describes one bounded arm; it contains references and scalar bounds,
// never raw prompts, tensors or model contexts.
struct common_flydelta_arm_request {
    int schema_version = 1;
    std::string job_id;
    std::string context_ref;
    std::string fixture_ref;
    std::string intervention_ref;
    std::vector<uint32_t> layer_indices;
    float alpha = 0.0f;
    std::vector<float> coefficients;
    bool apply_overlay = false;
    bool fresh_context = true;
    bool request_capture = false;
    bool request_teacher_forced_margin = false;
    bool request_generation = false;
    bool request_host_verification = false;
    size_t max_capture_bytes = 0;
    size_t max_generated_tokens = 0;
};

struct common_flydelta_arm_result {
    int schema_version = 1;
    bool executed = false;
    float requested_alpha = 0.0f;
    float executed_alpha = 0.0f;
    bool dose_safety_limited = false;
    bool geometry_available = false;
    float cosine = 0.0f;
    float progress = 0.0f;
    float leakage = 0.0f;
    float shift_norm = 0.0f;
    bool margin_available = false;
    // The absolute candidate margin. The pipeline computes the comparison to
    // its immutable baseline; the host may also return the comparison when it
    // already owns that baseline.
    common_flydelta_decision_margin margin;
    common_flydelta_margin_comparison margin_comparison;
    float margin_total = 0.0f;
    float margin_normalized = 0.0f;
    float margin_delta_total = 0.0f;
    float margin_delta_normalized = 0.0f;
    bool generation_available = false;
    float quality = 0.0f;
    bool host_evaluated = false;
    bool verifier_known = false;
    common_flydelta_counterfactual_outcome host_outcome =
        common_flydelta_counterfactual_outcome::unknown;
    std::string capture_ref;
    std::string generation_ref;
    std::string provenance_ref;
};

struct common_flydelta_evaluator_config;
struct common_flydelta_evaluator_callbacks;

// Host/runtime capabilities for one concrete model-facing FlyDelta adapter.
// These flags describe what the adapter can execute; they do not grant search
// permission, evidence rank, learning credit or promotion authority.
struct common_flydelta_model_capabilities {
    // Primitive capabilities are the facts a runtime registration can prove.
    // Algorithm capabilities below are derived from these facts, not merely
    // copied from a caller's requested configuration.
    bool capture = false;
    bool overlay = false;
    bool generation = false;
    bool teacher_forced_scoring = false;
    bool host_verification = false;

    bool bootstrap_zoom = false;
    bool adaptive_alpha = false;
    bool teacher_forced_margin = false;
    bool orthogonal_search = false;
    bool representation_augmentation = false;
};

// Runtime-owned registration point for model-facing FlyDelta execution. The
// host owns model contexts, opaque reference resolution, fresh inference and
// host verification; the common worker only receives the resulting callback.
struct common_flydelta_model_host {
    common_flydelta_model_capabilities capabilities;
    // Executes one generic bounded model-facing arm. Search policy remains in
    // FlyDelta; the host owns contexts, reference resolution and verification.
    std::function<bool(
            const common_flydelta_arm_request & request,
            common_flydelta_arm_result & result,
            std::string & error)> run_bounded_arm;
    std::function<bool(
            common_flydelta_evaluator_config & config,
            common_flydelta_evaluator_callbacks & callbacks,
            std::string & error)> register_evaluator;
};

bool common_flydelta_model_host_validate(
        const common_flydelta_model_host & host,
        std::string & error);

// Computes the algorithm-level capabilities that are actually reachable from
// the registered model primitives. Host verification remains independent from
// model execution and is never implied by capture or generation.
common_flydelta_model_capabilities common_flydelta_model_capabilities_from_primitives(
        const common_flydelta_model_capabilities & primitives,
        bool has_bounded_arm,
        bool has_search_runner,
        bool has_stateful_search);

// Adapts the existing generic host arm contract to the common search-pipeline
// runner. This is the only model-facing mapping needed by search algorithms;
// the host still resolves fixture/intervention references and owns inference,
// fresh-context isolation and verification.
common_flydelta_search_pipeline_runner common_flydelta_search_pipeline_runner_from_model_host(
        const common_flydelta_model_host & host,
        std::string job_id,
        std::string context_ref,
        std::string intervention_ref,
        bool request_capture = true,
        bool request_teacher_forced_margin = true,
        bool request_generation = true,
        bool request_host_verification = true,
        size_t max_capture_bytes = 0,
        size_t max_generated_tokens = 0);

struct common_flydelta_model_adapter {
    common_flydelta_model_capabilities capabilities;
    common_flydelta_experiment_worker_callback worker_callback;
};

bool common_flydelta_model_adapter_validate(
        const common_flydelta_model_adapter & adapter,
        std::string & error);

// A capability is useful only when the adapter callback is registered too.
bool common_flydelta_model_adapter_supports_search(
        const common_flydelta_model_adapter & adapter);

// Builds the production-facing bridge around the existing bounded evaluator.
// The supplied callbacks remain host-owned: they resolve opaque references,
// create fresh inference contexts and perform host verification. This factory
// is model/backend-neutral and only composes evaluator execution with the
// generic worker callback.
std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_evaluator(
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_model_capabilities capabilities,
        std::string & error);

// Production composition seam. This remains backend- and model-neutral; the
// supplied host decides how to resolve refs and execute fresh model work.
std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_host(
        const common_flydelta_model_host & host,
        std::string & error);
