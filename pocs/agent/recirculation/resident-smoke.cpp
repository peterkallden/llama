#include "agent/agent-recirculation.h"
#include "agent/adaptation/lifecycle-store.h"
#include "agent-server-context-host.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: llama-recirculation-resident-smoke MODEL.gguf [ALPHA [PROMPT]]\n";
        return 2;
    }
    common_agent_server_context_host host;
    common_agent_server_context_host_config config;
    config.context_key.load_key.model = argv[1];
    config.context_key.load_key.n_gpu_layers = 0;
    config.context_key.load_key.fit_params = false;
    config.context_key.n_parallel = 1;
    config.context_key.n_sequences = 1;
    config.context_key.n_ctx = 1024;
    config.context_key.n_threads = 2;
    config.experimental_recirculation_auto_reuse = true;
    config.recirculation_journal = std::make_shared<common_learning_in_memory_lifecycle_store>();
    config.recirculation_model_profile_fingerprint = std::string("smoke:") + argv[1];
    std::string error;
    if (!host.start(config, error)) throw std::runtime_error("resident start failed: " + error);
    common_agent_inference_session session;
    if (!host.build_inference_session(session, error) || !session.inference)
        throw std::runtime_error("resident inference unavailable: " + error);
    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::conversation;
    request.messages = {{"user", argc >= 4 ? argv[3] : "Write the numbers from 1 to 20, one per line."}};
    request.options.n_predict = 20;
    common_agent_scope scope;
    scope.namespace_id = "recirculation-smoke";
    scope.project_id = "poc";
    scope.session_id = "resident";
    request.scope = scope;
    request.recirculation_experiment_mode = true;
    request.recirculation_context_key = "smoke:generated-window";
    auto rec = std::make_shared<common_agent_recirculation_request>();
    rec->source_layer = 7;
    rec->destination_layer = 2;
    rec->alpha = argc >= 3 ? std::stof(argv[2]) : 0.1f;
    rec->rounds = 2;
    request.recirculation = rec;
    auto baseline_request = request;
    baseline_request.recirculation.reset();
    common_agent_generation_result baseline;
    if (!session.inference->generate(baseline_request, baseline))
        throw std::runtime_error("resident baseline generation failed: " + baseline.error_message);
    common_agent_generation_result result;
    if (!session.inference->generate(request, result))
        throw std::runtime_error("resident generation failed: " + result.error_message);
    if (!result.recirculation_observation || !result.recirculation_observation->attempted ||
            !result.recirculation_observation->failure_reason.empty())
        throw std::runtime_error("resident recirculation was not exercised: " +
            (result.recirculation_observation ? result.recirculation_observation->failure_reason : "missing observation"));
    bool recalled = false;
    if (result.recirculation_observation->committed) {
        common_agent_generation_result repeated;
        if (!session.inference->generate(baseline_request, repeated) ||
                !repeated.recirculation_observation ||
                !repeated.recirculation_observation->attempted ||
                !repeated.recirculation_observation->recalled_profile ||
                !repeated.recirculation_observation->memory_error.empty())
            throw std::runtime_error("committed profile was not recalled on the next matching context");
        recalled = true;
    }
    std::cout << "PASS: resident recirculation; tokens=" << result.decoded_tokens
              << " rounds=" << result.recirculation_observation->completed_rounds
              << " committed=" << result.recirculation_observation->committed
              << " baseline_entropy=" << result.recirculation_observation->baseline_entropy
              << " refined_entropy=" << result.recirculation_observation->refined_entropy
              << " outputs_equal=" << (baseline.content == result.content)
              << " recalled=" << recalled << '\n';
    host.stop();
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
