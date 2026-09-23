#pragma once

#include "../runtime/agent-runtime-assembly.h"

#include "common.h"
#include "log.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct server_context;
class common_agent_server_context_host;

struct common_agent_server_context_load_key {
    std::string model;
    int n_gpu_layers = 0;
    bool fit_params = true;
    std::string mmproj;
};

struct common_agent_server_context_context_key {
    common_agent_server_context_load_key load_key;
    int n_parallel = 1;
    int n_sequences = 1;
    int n_ctx = 0;
    int n_threads = 2;
};

struct common_agent_server_context_host_config {
    common_agent_server_context_context_key context_key;
    int verbosity = LOG_LEVEL_WARN;
    bool per_sequence_cvec_batch = false;
    bool agent_trace = false;
};

struct common_agent_server_context_running_instance {
    std::unique_ptr<server_context> server;
    common_params params;
    std::unique_ptr<std::thread> loop;
    bool running = false;
};

// Production host-side translation for one generic FlyDelta arm. The server
// context owns model execution; the callbacks own reference resolution,
// semantic request construction and host verification. No search policy is
// implemented here.
struct common_agent_server_flydelta_binding {
    common_flydelta_model_capabilities primitives;
    // Optional shared host-owned material index. If the explicit readiness
    // callback is absent, the model-host seam derives it from this object.
    // Successful concept-capture refs are also observed here, so the
    // readiness state seen by the scheduler is advanced by the same
    // production callback that executed the bounded capture slice.
    std::shared_ptr<common_flydelta_teaching_material_runtime> teaching_material_runtime;
    std::function<bool(
            const common_flydelta_arm_request &,
            common_agent_generation_request &,
            std::string &)> prepare_arm;
    std::function<bool(
            const common_flydelta_arm_request &,
            const common_agent_generation_result &,
            common_flydelta_arm_result &,
            std::string &)> finalize_arm;
    // Runs after the generation wave is complete and before final arm
    // materialization.  It keeps decision-margin scoring outside
    // finalize_arm, so one response reader owns each server task wave.
    std::function<bool(
            const std::vector<common_flydelta_arm_request> &,
            const std::vector<common_agent_generation_request> &,
            common_agent_inference &,
            std::vector<common_flydelta_decision_margin> &,
            std::string &)> score_teacher_forced_margin_batch;
    std::function<bool(
            common_flydelta_evaluator_config &,
            common_flydelta_evaluator_callbacks &,
            std::string &)> register_evaluator;
    std::function<bool(
            const std::string & group_ref,
            bool & relation_set_ready,
            bool & trajectory_material_ready,
            std::string &)> inspect_teaching_material_group;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<std::string> &,
            std::string &)> run_concept_capture;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_capture_manifest> &,
            std::string &)> run_donor_capture;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_counterfactual_report> &,
            std::string &)> run_counterfactual;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_concept_candidate> &,
            std::string &)> run_concept_synthesis;
    std::function<bool(
            const common_flydelta_direction_candidate &,
            std::string &,
            std::string &)> persist_experimental_direction;
    std::function<bool(
            const common_flydelta_experiment_job &,
            const common_flydelta_representation_augmentation_state *,
            common_flydelta_search_pipeline_result &,
            common_flydelta_representation_augmentation_state &,
            std::string &)> run_representation_augmentation_with_state;
};

// Host-owned semantic callbacks used to compose a production binding.  This
// keeps the startup factory small and makes the same binding shape reusable by
// the daemon, embedders and model-backed smokes.  The callbacks still own
// opaque-reference resolution and semantic verification; this struct does not
// manufacture either from configuration.
struct common_agent_server_flydelta_binding_callbacks {
    common_flydelta_model_capabilities primitives;
    std::shared_ptr<common_flydelta_teaching_material_runtime>
        teaching_material_runtime;
    std::function<bool(
            const common_flydelta_arm_request &,
            common_agent_generation_request &,
            std::string &)> prepare_arm;
    std::function<bool(
            const common_flydelta_arm_request &,
            const common_agent_generation_result &,
            common_flydelta_arm_result &,
            std::string &)> finalize_arm;
    std::function<bool(
            const std::vector<common_flydelta_arm_request> &,
            const std::vector<common_agent_generation_request> &,
            common_agent_inference &,
            std::vector<common_flydelta_decision_margin> &,
            std::string &)> score_teacher_forced_margin_batch;
    std::function<bool(
            common_flydelta_evaluator_config &,
            common_flydelta_evaluator_callbacks &,
            std::string &)> register_evaluator;
    std::function<bool(
            const std::string & group_ref,
            bool & relation_set_ready,
            bool & trajectory_material_ready,
            std::string &)> inspect_teaching_material_group;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<std::string> &,
            std::string &)> run_concept_capture;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_capture_manifest> &,
            std::string &)> run_donor_capture;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_counterfactual_report> &,
            std::string &)> run_counterfactual;
    std::function<bool(
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_concept_candidate> &,
            std::string &)> run_concept_synthesis;
    std::function<bool(
            const common_flydelta_direction_candidate &,
            std::string &,
            std::string &)> persist_experimental_direction;
    std::function<bool(
            const common_flydelta_experiment_job &,
            const common_flydelta_representation_augmentation_state *,
            common_flydelta_search_pipeline_result &,
            common_flydelta_representation_augmentation_state &,
            std::string &)> run_representation_augmentation_with_state;
};

// Canonical composition helper for the existing daemon startup seam.  It is
// intentionally a pure field transfer: the host integration supplies the
// semantic callbacks and this helper does not create a fallback verifier.
common_agent_server_flydelta_binding
common_agent_server_flydelta_binding_from_callbacks(
        common_agent_server_flydelta_binding_callbacks callbacks);

// Produces the daemon's startup factory from the host-owned callback bundle.
// The factory validates the callbacks needed by the model-host seam and fails
// closed before workers can consume a job.  The resident host is still passed
// to the factory so a provider may capture host-specific state in its
// callbacks before returning the bundle.
std::function<bool(
        const std::shared_ptr<common_agent_server_context_host> &,
        common_agent_server_flydelta_binding &,
        std::string &)>
common_agent_server_flydelta_binding_factory_from_callbacks(
        common_agent_server_flydelta_binding_callbacks callbacks);

common_agent_server_context_load_key make_agent_server_context_load_key(
    const common_agent_inference_options & options);

common_agent_server_context_context_key make_agent_server_context_context_key(
    const common_agent_inference_options & options);

common_agent_server_context_host_config make_agent_server_context_host_config(
    const common_agent_inference_options & options);

common_params make_agent_server_context_params(
    const common_agent_server_context_host_config & config);

common_params make_agent_server_context_params(
    const common_agent_inference_options & options);

class common_agent_server_context_host {
public:
    common_agent_server_context_host();
    ~common_agent_server_context_host();

    bool start(const common_agent_server_context_host_config & config, std::string & error);
    bool build_inference_session(common_agent_inference_session & session, std::string & error) const;
    void stop();

    const common_agent_server_context_load_key & load_key() const { return current_load_key; }
    const common_agent_server_context_context_key & context_key() const { return current_context_key; }
    const common_agent_server_context_host_config & config() const { return current_config; }
    const common_params & params() const { return instance->params; }
    server_context & server();
    const server_context & server() const;

private:
    common_agent_server_context_load_key current_load_key;
    common_agent_server_context_context_key current_context_key;
    common_agent_server_context_host_config current_config;
    std::unique_ptr<common_agent_server_context_running_instance> instance;
};

// Executes one already host-prepared bounded FlyDelta arm wave against the
// resident server context.  This is the shared production execution primitive
// used by the model host and by host-owned evaluator callbacks; it does not
// resolve semantic references or decide search policy.
bool common_agent_server_context_host_run_flydelta_arm_batch(
        const std::shared_ptr<common_agent_server_context_host> & host,
        const common_agent_server_flydelta_binding & binding,
        bool native_batch,
        const common_flydelta_arm_batch_request & request,
        common_flydelta_arm_batch_result & result,
        std::string & error);

// Creates the existing generic FlyDelta model-host registration around a
// resident server context. The returned host advertises bounded-arm batch
// capability only when the server was started with the explicit per-sequence
// cvec opt-in and at least two parallel sequences are available.
std::shared_ptr<const common_flydelta_model_host>
common_agent_server_context_host_make_flydelta_model_host(
        std::shared_ptr<common_agent_server_context_host> host,
        common_agent_server_flydelta_binding binding,
        std::string & error);
