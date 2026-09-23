#include "tools/agent/runtime/agent-server-context-host.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <string>
#include <utility>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    common_agent_inference_options options;
    options.model = "model.gguf";
    options.n_parallel = 2;
    options.n_sequences = 2;
    options.per_sequence_cvec_batch = true;
    const auto host_config = make_agent_server_context_host_config(options);
    CHECK(host_config.context_key.n_parallel == 2);
    CHECK(host_config.context_key.n_sequences == 2);
    CHECK(host_config.per_sequence_cvec_batch);

    common_agent_server_flydelta_binding_callbacks callbacks;
    callbacks.primitives.capture = true;
    callbacks.primitives.generation = true;
    callbacks.primitives.host_verification = true;
    callbacks.prepare_arm = [](
            const common_flydelta_arm_request &,
            common_agent_generation_request &,
            std::string &) { return true; };
    callbacks.finalize_arm = [](
            const common_flydelta_arm_request &,
            const common_agent_generation_result &,
            common_flydelta_arm_result &,
            std::string &) { return true; };
    callbacks.register_evaluator = [](
            common_flydelta_evaluator_config &,
            common_flydelta_evaluator_callbacks &,
            std::string &) { return true; };
    callbacks.run_donor_capture = [](
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_capture_manifest> &,
            std::string &) { return true; };
    callbacks.run_counterfactual = [](
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_counterfactual_report> &,
            std::string &) { return true; };
    callbacks.run_concept_capture = [](
            const common_flydelta_experiment_job &,
            std::vector<std::string> &,
            std::string &) { return true; };
    callbacks.run_concept_synthesis = [](
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_concept_candidate> &,
            std::string &) { return true; };
    callbacks.run_representation_augmentation_with_state = [](
            const common_flydelta_experiment_job &,
            const common_flydelta_representation_augmentation_state *,
            common_flydelta_search_pipeline_result &,
            common_flydelta_representation_augmentation_state &,
            std::string &) { return true; };

    auto binding = common_agent_server_flydelta_binding_from_callbacks(
        std::move(callbacks));
    CHECK(binding.primitives.capture);
    CHECK(binding.primitives.generation);
    CHECK(binding.primitives.host_verification);
    CHECK(static_cast<bool>(binding.prepare_arm));
    CHECK(static_cast<bool>(binding.finalize_arm));
    CHECK(static_cast<bool>(binding.register_evaluator));
    CHECK(static_cast<bool>(binding.run_donor_capture));
    CHECK(static_cast<bool>(binding.run_counterfactual));
    CHECK(static_cast<bool>(binding.run_concept_capture));
    CHECK(static_cast<bool>(binding.run_concept_synthesis));
    CHECK(static_cast<bool>(binding.run_representation_augmentation_with_state));

    common_flydelta_arm_request arm;
    common_agent_generation_request request;
    std::string error;
    CHECK(binding.prepare_arm(arm, request, error));
    common_flydelta_arm_result result;
    common_agent_generation_result generation;
    CHECK(binding.finalize_arm(arm, generation, result, error));

    common_flydelta_evaluator_config evaluator_config;
    common_flydelta_evaluator_callbacks evaluator_callbacks;
    CHECK(binding.register_evaluator(
        evaluator_config, evaluator_callbacks, error));

    common_agent_server_flydelta_binding_callbacks factory_callbacks;
    factory_callbacks.primitives = binding.primitives;
    factory_callbacks.prepare_arm = binding.prepare_arm;
    factory_callbacks.finalize_arm = binding.finalize_arm;
    factory_callbacks.register_evaluator = binding.register_evaluator;
    factory_callbacks.run_donor_capture = binding.run_donor_capture;
    factory_callbacks.run_counterfactual = binding.run_counterfactual;
    factory_callbacks.run_concept_capture = binding.run_concept_capture;
    factory_callbacks.run_concept_synthesis = binding.run_concept_synthesis;
    factory_callbacks.run_representation_augmentation_with_state =
        binding.run_representation_augmentation_with_state;
    auto factory = common_agent_server_flydelta_binding_factory_from_callbacks(
        std::move(factory_callbacks));
    common_agent_server_flydelta_binding produced;
    auto resident_host = std::make_shared<common_agent_server_context_host>();
    CHECK(factory(resident_host, produced, error));
    CHECK(error.empty());
    CHECK(static_cast<bool>(produced.prepare_arm));

    auto incomplete_factory =
        common_agent_server_flydelta_binding_factory_from_callbacks({});
    CHECK(!incomplete_factory(resident_host, produced, error));
    CHECK(error.find("requires prepare, finalize and evaluator callbacks") !=
        std::string::npos);

    return 0;
}
