#pragma once

#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

inline constexpr size_t common_flydelta_compact_geometry_scalar_count = 4;
inline constexpr size_t common_flydelta_compact_geometry_bytes =
    common_flydelta_compact_geometry_scalar_count * sizeof(float);

// Generic model-facing primitive shared by every FlyDelta search phase. The
// request describes one bounded arm; it contains references and scalar bounds,
// never raw prompts, tensors or model contexts.
struct common_flydelta_arm_request {
    int schema_version = 1;
    std::string job_id;
    // Stable identity for one model-facing arm. The host uses this as the
    // retry/resume idempotency key; it is derived by the common runner from
    // the immutable job, fixture, intervention, layer and scale identity.
    std::string arm_id;
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

// Optional backend timing/transfer telemetry. It is deliberately diagnostic:
// search policy and lifecycle must not depend on these values. A device host
// can use it to prove whether reductions/batching actually remove transfers
// before a later optimization changes the runtime contract.
struct common_flydelta_arm_execution_metrics {
    int schema_version = 1;
    bool available = false;
    float model_ms = 0.0f;
    float teacher_forced_ms = 0.0f;
    float generation_ms = 0.0f;
    size_t overlay_bytes_to_device = 0;
    size_t capture_bytes_to_host = 0;
    size_t diagnostics_bytes_to_host = 0;
    bool device_reduction_used = false;
    bool batched_execution_used = false;
    // Execution provenance only; this must not affect utility, safety,
    // evidence depth or learning credit.
    enum class path : uint8_t {
        unknown,
        scalar,
        scalar_fallback,
        backend_batch,
        device_batch,
    } execution_path = path::unknown;
    std::string fallback_reason;
};

const char * common_flydelta_arm_execution_path_name(
        common_flydelta_arm_execution_metrics::path value);

struct common_flydelta_arm_result {
    int schema_version = 1;
    // Must echo the request arm_id when a host executes the arm.
    std::string arm_id;
    bool executed = false;
    float requested_alpha = 0.0f;
    float executed_alpha = 0.0f;
    bool dose_safety_limited = false;
    // These geometry fields are the compact reduction result. A device-aware
    // host may compute them without transferring the full hidden-state
    // capture; capture_ref is only needed when the vector itself is retained
    // as later basis/donor/augmentation material.
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
    common_flydelta_arm_execution_metrics execution_metrics;
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

// Transport validation only. These checks do not decide utility, promotion,
// learning credit or HELPED.
bool common_flydelta_arm_request_validate(
        const common_flydelta_arm_request & request,
        std::string & error);
bool common_flydelta_arm_result_validate(
        const common_flydelta_arm_result & result,
        std::string & error);

// Compares the observable arm result for scalar/device replay. Backend path,
// timings and temporary artifact references are deliberately excluded; those
// belong to execution telemetry and provenance, not model semantics.
bool common_flydelta_arm_result_replay_equivalent(
        const common_flydelta_arm_result & expected,
        const common_flydelta_arm_result & actual,
        float absolute_tolerance,
        std::string & error);

// A batch is only an execution optimization boundary. Each arm retains its
// own fresh-context, overlay, margin, geometry and host-outcome semantics.
// Backends may execute this as one device batch; the common fallback executes
// the same requests one by one without changing the results contract.
struct common_flydelta_arm_batch_request {
    int schema_version = 1;
    std::vector<common_flydelta_arm_request> arms;
};

struct common_flydelta_arm_batch_result {
    int schema_version = 1;
    std::vector<common_flydelta_arm_result> arms;
};

// A backend-neutral layer/profile proposal used by BootstrapZoom, Shallow
// controls and later coefficient search. The model host only executes these
// proposals; it does not decide which search phase produced them.
struct common_flydelta_layer_profile_arm {
    std::vector<uint32_t> layer_indices;
    std::vector<float> coefficients;
    float alpha = 0.0f;
    bool apply_overlay = true;
};

bool common_flydelta_arm_batch_request_validate(
        const common_flydelta_arm_batch_request & request,
        std::string & error);
bool common_flydelta_arm_batch_result_validate(
        const common_flydelta_arm_batch_result & result,
        const common_flydelta_arm_batch_request & request,
        std::string & error);

// Compares scalar/device and fallback batch results by arm identity and
// observable semantics. Execution path, timings and temporary references are
// intentionally ignored so this can be used as a correctness gate before a
// backend is allowed to claim device batching.
bool common_flydelta_arm_batch_result_replay_equivalent(
        const common_flydelta_arm_batch_result & expected,
        const common_flydelta_arm_batch_result & actual,
        float absolute_tolerance,
        std::string & error);

struct common_flydelta_evaluator_config;
struct common_flydelta_evaluator_callbacks;

// Host/runtime capabilities for one concrete model-facing FlyDelta adapter.
// These flags describe what the adapter can execute; they do not grant search
// permission, evidence rank, learning credit or promotion authority.
struct common_flydelta_model_capabilities {
    // Primitive capabilities are the facts a runtime registration can prove.
    // Algorithm capabilities below are derived from these facts, not merely
    // copied from a caller's requested configuration.
    bool bounded_arm_batch = false;
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

// Backend-neutral execution capacity. Search algorithms may propose a larger
// independent arm set, but the model host can partition it into bounded waves
// that fit its resident slots/device batch. A zero arm limit means that the
// registered batch callback owns the limit itself.
struct common_flydelta_model_batch_capacity {
    size_t max_arms_per_batch = 0;
    size_t max_inflight_batches = 1;
};

// Runtime-owned registration point for model-facing FlyDelta execution. The
// host owns model contexts, opaque reference resolution, fresh inference and
// host verification; the common worker only receives the resulting callback.
struct common_flydelta_model_host {
    common_flydelta_model_capabilities capabilities;
    common_flydelta_model_batch_capacity batch_capacity;
    // Executes one generic bounded model-facing arm. Search policy remains in
    // FlyDelta; the host owns contexts, reference resolution and verification.
    std::function<bool(
            const common_flydelta_arm_request & request,
            common_flydelta_arm_result & result,
            std::string & error)> run_bounded_arm;
    // Optional backend optimization. When absent, the generic helper below
    // falls back to run_bounded_arm and preserves per-arm isolation.
    std::function<bool(
            const common_flydelta_arm_batch_request & request,
            common_flydelta_arm_batch_result & result,
            std::string & error)> run_bounded_arm_batch;
    std::function<bool(
            common_flydelta_evaluator_config & config,
            common_flydelta_evaluator_callbacks & callbacks,
            std::string & error)> register_evaluator;
};

bool common_flydelta_run_bounded_arm_batch(
        const common_flydelta_model_host & host,
        const common_flydelta_arm_batch_request & request,
        common_flydelta_arm_batch_result & result,
        std::string & error);

// Materializes a bounded layer/profile wave into the generic ArmRequest
// contract and executes it through the existing scalar-or-device batch seam.
// The returned arms preserve proposal order and per-arm isolation.
bool common_flydelta_run_layer_profile_batch(
        const common_flydelta_model_host & host,
        const std::string & job_id,
        const std::string & context_ref,
        const std::string & fixture_ref,
        const std::string & intervention_ref,
        const std::vector<common_flydelta_layer_profile_arm> & proposals,
        bool request_capture,
        bool request_teacher_forced_margin,
        bool request_generation,
        bool request_host_verification,
        size_t max_capture_bytes,
        size_t max_generated_tokens,
        common_flydelta_arm_batch_result & result,
        std::string & error);

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

// Batch counterpart for independent Whirlpool/region probes. It uses the
// same generic ArmRequest/ArmResult contract as the scalar factory and falls
// back through common_flydelta_run_bounded_arm_batch when the backend offers
// a real batch implementation.
common_flydelta_search_pipeline_batch_runner
common_flydelta_search_pipeline_batch_runner_from_model_host(
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
