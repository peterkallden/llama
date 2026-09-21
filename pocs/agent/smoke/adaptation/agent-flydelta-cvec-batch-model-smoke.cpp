#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"

#include <algorithm>
#include <cstdlib>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct options {
    std::string model;
    int n_predict = 8;
    int n_threads = 3;
    int n_gpu_layers = 0;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&](const char * name) -> const char * {
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++index];
        };
        if (argument == "--model") {
            const char * value_arg = next("--model");
            if (!value_arg) return false;
            value.model = value_arg;
        } else if (argument == "--n-predict") {
            const char * value_arg = next("--n-predict");
            if (!value_arg) return false;
            value.n_predict = std::stoi(value_arg);
        } else if (argument == "--threads") {
            const char * value_arg = next("--threads");
            if (!value_arg) return false;
            value.n_threads = std::stoi(value_arg);
        } else if (argument == "--n-gpu-layers") {
            const char * value_arg = next("--n-gpu-layers");
            if (!value_arg) return false;
            value.n_gpu_layers = std::stoi(value_arg);
        } else if (argument == "--help" || argument == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    return true;
}

std::shared_ptr<const common_flydelta_activation_result> make_activation(
        const llama_model * model, float value, const std::string & artifact_id, std::string & error) {
    common_flydelta_gate_config gate_config;
    gate_config.enabled = true;
    gate_config.max_scale = 1.0f;

    common_flydelta_activation_request request;
    request.candidate_id = "flydelta://candidate/cvec-batch-smoke";
    request.artifact_id = artifact_id;
    request.model_profile_fingerprint = "sha256:flydelta-cvec-batch-smoke-model";
    request.capture_layout_revision = "layer-input:v1";
    request.model_n_embd = static_cast<size_t>(llama_model_n_embd(model));
    request.model_n_layers = static_cast<size_t>(llama_model_n_layer(model));
    request.il_start = 1;
    request.il_end = static_cast<int32_t>(request.model_n_layers - 1);

    common_flydelta_basis_direction direction;
    direction.layer_index = 1;
    direction.values.assign(request.model_n_embd, 0.0f);
    direction.values.front() = value;
    request.directions.push_back(std::move(direction));
    request.coefficients = {1.0f};
    request.gate_request.explicit_opt_in = true;
    request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
    request.gate_request.basis_available = true;
    request.gate_request.familiarity = 1.0f;
    request.gate_request.novelty = 0.0f;
    request.gate_request.requested_scale = 1.0f;

    common_flydelta_activation_result activation;
    if (!common_flydelta_prepare_activation(
            gate_config, request, 64U * 1024U * 1024U, activation, error)) {
        return {};
    }
    return std::make_shared<const common_flydelta_activation_result>(std::move(activation));
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL [--n-predict N] [--threads N] [--n-gpu-layers N]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "FlyDelta cvec batch model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0) return 2;

    const char * opt_in = std::getenv("LLAMA_SERVER_PER_SEQUENCE_CVEC");
    if (opt_in == nullptr || (std::string(opt_in) != "1" && std::string(opt_in) != "true")) {
        std::cerr << "FlyDelta cvec batch model smoke requires LLAMA_SERVER_PER_SEQUENCE_CVEC=1\n";
        return 77;
    }

    common_agent_server_context_host host;
    common_agent_server_context_host_config config;
    config.context_key.load_key.model = value.model;
    config.context_key.load_key.n_gpu_layers = value.n_gpu_layers;
    config.context_key.load_key.fit_params = true;
    config.context_key.n_parallel = 2;
    config.context_key.n_sequences = 2;
    config.context_key.n_ctx = 4096;
    config.context_key.n_threads = value.n_threads;
    config.verbosity = LOG_LEVEL_INFO;

    std::string error;
    if (!host.start(config, error)) {
        std::cerr << "could not start two-slot server host: " << error << '\n';
        return 1;
    }

    std::vector<common_agent_inference_session> sessions(2);
    for (auto & session : sessions) {
        if (!host.build_inference_session(session, error) || !session.inference) {
            std::cerr << "could not build two-slot inference session: " << error << '\n';
            return 1;
        }
    }
    if (!sessions[0].inference || !sessions[1].inference) {
        std::cerr << "two-slot inference sessions were not initialized\n";
        return 1;
    }
    auto * context = host.server().get_llama_context();
    if (context == nullptr || llama_get_model(context) == nullptr) {
        std::cerr << "two-slot server host did not expose a loaded model\n";
        return 1;
    }
    const auto * model = llama_get_model(context);
    if (llama_model_n_embd(model) <= 0 || llama_model_n_layer(model) <= 2) return 1;

    auto activation_a = make_activation(model, 0.0001f,
        "flydelta://artifact/cvec-batch-smoke/a", error);
    auto activation_b = make_activation(model, 0.0002f,
        "flydelta://artifact/cvec-batch-smoke/b", error);
    if (!activation_a || !activation_b || !activation_a->sparse_overlay.enabled ||
            !activation_b->sparse_overlay.enabled ||
            activation_a->sparse_overlay.layer_indices.empty() ||
            activation_b->sparse_overlay.layer_indices.empty() ||
            activation_a->overlay.artifact_id == activation_b->overlay.artifact_id ||
            activation_a->overlay.data == activation_b->overlay.data) {
        std::cerr << "could not prepare distinct cvec smoke overlays: " << error << '\n';
        return 1;
    }

    std::mutex mutex;
    std::condition_variable condition;
    bool start = false;
    std::vector<common_agent_generation_result> results(2);
    std::vector<bool> executed(2, false);
    std::vector<std::thread> workers;
    workers.reserve(2);
    for (size_t index = 0; index < 2; ++index) {
        workers.emplace_back([&, index]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&]() { return start; });
            }
            common_agent_generation_request request;
            request.purpose = common_agent_generation_purpose::conversation;
            request.options.n_predict = value.n_predict;
            request.options.n_threads = value.n_threads;
            request.messages = {
                {"system", "Reply with exactly PASS."},
                {"user", index == 0 ? "Run the first isolated check." : "Run the second isolated check."},
            };
            request.flydelta_activation = index == 0 ? activation_a : activation_b;
            executed[index] = sessions[index].inference->generate(request, results[index]);
        });
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        start = true;
    }
    condition.notify_all();
    for (auto & worker : workers) worker.join();

    for (size_t index = 0; index < results.size(); ++index) {
        if (!executed[index] || !common_agent_generation_succeeded(results[index])) {
            std::cerr << "cvec batch arm " << index << " failed: " << results[index].error_message << '\n';
            return 1;
        }
    }

    std::cout << "flydelta_cvec_batch_model_smoke=passed\n"
              << "slots=2\n"
              << "distinct_overlays=yes\n"
              << "sparse_overlays=yes\n"
              << "active_layers=" << activation_a->sparse_overlay.layer_indices.size() << "\n"
              << "results=2\n";
    return 0;
}
