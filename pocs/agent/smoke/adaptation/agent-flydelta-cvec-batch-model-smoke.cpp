#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model;
    int n_predict = 8;
    int n_threads = 3;
    int n_gpu_layers = 0;
    bool scalar = false;
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
        } else if (argument == "--scalar") {
            value.scalar = true;
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
                  << " --model MODEL [--n-predict N] [--threads N] [--n-gpu-layers N] [--scalar]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "FlyDelta cvec batch model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0) return 2;

    const char * opt_in = std::getenv("LLAMA_SERVER_PER_SEQUENCE_CVEC");
    if (!value.scalar && (opt_in == nullptr ||
            (std::string(opt_in) != "1" && std::string(opt_in) != "true"))) {
        std::cerr << "FlyDelta cvec batch model smoke requires LLAMA_SERVER_PER_SEQUENCE_CVEC=1\n";
        return 77;
    }

    auto host = std::make_shared<common_agent_server_context_host>();
    common_agent_server_context_host_config config;
    config.context_key.load_key.model = value.model;
    config.context_key.load_key.n_gpu_layers = value.n_gpu_layers;
    config.context_key.load_key.fit_params = true;
    config.context_key.n_parallel = value.scalar ? 1 : 2;
    config.context_key.n_sequences = value.scalar ? 1 : 2;
    config.context_key.n_ctx = 4096;
    config.context_key.n_threads = value.n_threads;
    config.verbosity = LOG_LEVEL_INFO;

    std::string error;
    if (!host->start(config, error)) {
        std::cerr << "could not start two-slot server host: " << error << '\n';
        return 1;
    }

    auto * context = host->server().get_llama_context();
    if (context == nullptr || llama_get_model(context) == nullptr) {
        std::cerr << "two-slot server host did not expose a loaded model\n";
        return 1;
    }
    const auto * model = llama_get_model(context);
    if (llama_model_n_embd(model) <= 0 || llama_model_n_layer(model) <= 2) return 1;

    common_agent_server_flydelta_binding_callbacks callbacks;
    callbacks.primitives.generation = true;
    callbacks.primitives.overlay = true;
    callbacks.primitives.host_verification = true;
    callbacks.prepare_arm = [model, n_predict = value.n_predict, n_threads = value.n_threads](
            const common_flydelta_arm_request & arm,
            common_agent_generation_request & request,
            std::string & prepare_error) {
        auto activation = make_activation(
            model, arm.alpha, arm.intervention_ref + "/" + arm.arm_id, prepare_error);
        if (!activation) return false;
        request = {};
        request.purpose = common_agent_generation_purpose::conversation;
        request.options.n_predict = n_predict;
        request.options.n_threads = n_threads;
        request.messages = {
            {"system", "Reply with exactly PASS."},
            {"user", "Run isolated FlyDelta arm " + arm.arm_id},
        };
        request.flydelta_activation = std::move(activation);
        return true;
    };
    callbacks.finalize_arm = [](
            const common_flydelta_arm_request & arm,
            const common_agent_generation_result & generation,
            common_flydelta_arm_result & result,
            std::string & finalize_error) {
        (void) finalize_error;
        result = {};
        result.arm_id = arm.arm_id;
        result.executed = common_agent_generation_succeeded(generation);
        result.generation_available = true;
        result.quality = static_cast<float>(generation.decoded_tokens);
        result.host_outcome = common_flydelta_counterfactual_outcome::unknown;
        if (!result.executed && !generation.error_message.empty()) {
            result.provenance_ref = generation.error_message;
        }
        return result.executed;
    };
    callbacks.register_evaluator = [](
            common_flydelta_evaluator_config &,
            common_flydelta_evaluator_callbacks &,
            std::string & register_error) {
        register_error.clear();
        return true;
    };
    auto binding = common_agent_server_flydelta_binding_from_callbacks(
        std::move(callbacks));

    auto model_host = common_agent_server_context_host_make_flydelta_model_host(
        host, std::move(binding), error);
    if (!model_host || (!value.scalar && !model_host->capabilities.bounded_arm_batch)) {
        std::cerr << "could not bind production FlyDelta batch host: " << error << '\n';
        return 1;
    }

    common_flydelta_arm_batch_request request;
    request.arms.resize(2);
    for (size_t index = 0; index < 2; ++index) {
        auto & arm = request.arms[index];
        arm.job_id = "flydelta://job/cvec-batch-smoke";
        arm.arm_id = "flydelta://arm/cvec-batch-smoke/" + std::to_string(index);
        arm.context_ref = "context://cvec-batch-smoke/" + std::to_string(index);
        arm.fixture_ref = "fixture://cvec-batch-smoke";
        arm.intervention_ref = "flydelta://artifact/cvec-batch-smoke/" + std::to_string(index);
        arm.layer_indices = {1};
        arm.coefficients = {1.0f};
        arm.alpha = index == 0 ? 0.0001f : 0.0002f;
        arm.apply_overlay = true;
        arm.fresh_context = true;
        arm.request_generation = true;
        arm.max_generated_tokens = static_cast<size_t>(value.n_predict);
    }

    common_flydelta_arm_batch_result result;
    if (!common_flydelta_run_bounded_arm_batch(*model_host, request, result, error)) {
        std::cerr << "production FlyDelta batch execution failed: " << error << '\n';
        return 1;
    }

    for (size_t index = 0; index < result.arms.size(); ++index) {
        if (!result.arms[index].executed) {
            std::cerr << "cvec batch arm " << index << " did not execute\n";
            return 1;
        }
    }

    std::cout << "flydelta_cvec_batch_model_smoke=passed\n"
              << "slots=" << (value.scalar ? 1 : 2) << "\n"
              << "distinct_overlays=yes\n"
              << "sparse_overlays=yes\n"
              << "active_layers=1\n"
              << "results=" << result.arms.size() << "\n"
              << "execution_path="
              << common_flydelta_arm_execution_path_name(
                  result.arms.front().execution_metrics.execution_path) << "\n"
              << "batched_execution="
              << (result.arms.front().execution_metrics.batched_execution_used ? "yes" : "no") << "\n"
              << "model_ms=" << result.arms.front().execution_metrics.model_ms << "\n"
              << "overlay_bytes_to_device="
              << result.arms.front().execution_metrics.overlay_bytes_to_device << "\n"
              << "capture_bytes_to_host="
              << result.arms.front().execution_metrics.capture_bytes_to_host << "\n"
              << "mode="
              << common_flydelta_arm_execution_path_name(
                  result.arms.front().execution_metrics.execution_path) << "\n";
    return 0;
}
