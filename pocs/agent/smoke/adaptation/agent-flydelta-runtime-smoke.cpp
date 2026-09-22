#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "tools/agent/runtime/agent-model-loaders.h"
#include "tools/agent/runtime/agent-runtime-session-host.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string backend = "server-context";
    int n_predict = 16;
    int n_threads = 3;
    int n_gpu_layers = 0;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * backend = std::getenv("LLAMA_AGENT_BACKEND")) value.backend = backend;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++i];
        };
        if (argument == "--model") {
            const char * path = next("--model"); if (!path) return false; value.model = path;
        } else if (argument == "--backend") {
            const char * backend = next("--backend"); if (!backend) return false; value.backend = backend;
        } else if (argument == "--n-predict") {
            const char * count = next("--n-predict"); if (!count) return false; value.n_predict = std::stoi(count);
        } else if (argument == "--threads") {
            const char * count = next("--threads"); if (!count) return false; value.n_threads = std::stoi(count);
        } else if (argument == "--n-gpu-layers") {
            const char * count = next("--n-gpu-layers"); if (!count) return false; value.n_gpu_layers = std::stoi(count);
        } else if (argument == "--help" || argument == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    return true;
}

bool regular_file(const std::string & path) {
    std::error_code error;
    return !path.empty() && std::filesystem::is_regular_file(path, error);
}

bool response_passed(const common_agent_runtime_session_host_turn_result & result) {
    if (!result.ok || result.response_generation_status != common_agent_generation_status::completed) {
        return false;
    }
    std::string response = result.response;
    std::transform(response.begin(), response.end(), response.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return response.find("PASS") != std::string::npos;
}

common_agent_model_catalog make_catalog(
        const options & value,
        const std::string & profile_id) {
    common_agent_model_catalog catalog;
    catalog.directory = std::filesystem::path(value.model).parent_path().string();
    catalog.max_loaded_generation_models = 1;
    catalog.default_profile = profile_id;
    catalog.bases.emplace("generation", common_agent_model_base_spec{
        "generation", value.backend,
        std::filesystem::path(value.model).filename().string(), {}, "resident"});
    catalog.profiles.emplace(profile_id, common_agent_model_profile_spec{
        "generation", {}, {}, 2048, 1, 1, "resident"});
    return catalog;
}

std::shared_ptr<common_agent_runtime_model_residency> make_residency(
        const options & value,
        const std::string & profile_id) {
    const common_agent_runtime_model_loader_config loader_config{
        value.n_gpu_layers,
        value.n_threads,
        true,
    };
    std::unordered_map<std::string,
        std::shared_ptr<common_agent_runtime_model_loader>> loaders;
    loaders.emplace("cli", std::make_shared<common_agent_runtime_cli_model_loader>(loader_config));
#ifndef LLAMA_AGENT_ANDROID_CLI_ONLY
    loaders.emplace("server-context",
        std::make_shared<common_agent_runtime_server_context_model_loader>(loader_config));
#endif
    return std::make_shared<common_agent_runtime_model_residency>(
        make_catalog(value, profile_id), std::move(loaders));
}

bool prepare_activation(
        size_t model_n_embd,
        size_t model_n_layers,
        float scale,
        common_flydelta_activation_result & result,
        std::string & error) {
    common_flydelta_gate_config gate_config;
    gate_config.enabled = true;

    common_flydelta_activation_request request;
    request.candidate_id = "flydelta://candidate/runtime-three-arm";
    request.artifact_id = "flydelta://sideband/runtime-smoke-v1";
    request.model_profile_fingerprint = "sha256:flydelta-runtime-smoke-model";
    request.capture_layout_revision = "layout:cvec-v1";
    request.model_n_embd = model_n_embd;
    request.model_n_layers = model_n_layers;
    request.il_end = static_cast<int32_t>(model_n_layers - 1);
    common_flydelta_basis_direction direction;
    direction.layer_index = 1;
    direction.values.assign(model_n_embd, 0.0f);
    // Keep this an intentionally tiny, deterministic probe. A passing smoke
    // proves propagation and safe execution, not learned behavioral lift.
    direction.values.front() = 0.0001f;
    request.directions.push_back(std::move(direction));
    request.coefficients = {1.0f};
    request.gate_request.explicit_opt_in = true;
    request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
    request.gate_request.basis_available = true;
    request.gate_request.familiarity = 1.0f;
    request.gate_request.novelty = 0.0f;
    request.gate_request.requested_scale = scale;
    return common_flydelta_prepare_activation(
        gate_config, request, 64U * 1024U * 1024U, result, error);
}

bool run_arm(
        const options & value,
        const std::shared_ptr<common_agent_runtime_model_residency> & residency,
        const std::string & profile_id,
        size_t model_n_embd,
        size_t model_n_layers,
        int arm_index,
        float scale,
        std::string & error) {
    common_memory_in_memory_store memory_store;
    common_plan_in_memory_store plan_store;
    if (!memory_store.open("", error) || !plan_store.open("", error)) return false;

    common_agent_runtime_policy policy;
    policy.agent_inference_backend = value.backend;
    policy.enable_reflection = false;
    policy.max_iterations = 1;
    policy.max_reflection_rounds = 0;
    policy.max_tool_rounds = 0;

    common_agent_runtime_config runtime_config;
    runtime_config.generation_config.n_predict = value.n_predict;
    runtime_config.generation_config.n_threads = value.n_threads;
    runtime_config.generation_config.context_size_tokens = 2048;
    runtime_config.max_continuations = 0;

    common_flydelta_activation_result activation;
    if (scale > 0.0f && !prepare_activation(
            model_n_embd, model_n_layers, scale, activation, error)) return false;
    const auto activation_ptr = scale > 0.0f
        ? std::make_shared<const common_flydelta_activation_result>(std::move(activation))
        : std::shared_ptr<const common_flydelta_activation_result>();

    common_agent_runtime_session_host runtime(
        make_agent_runtime_session_host_config({
            memory_store,
            plan_store,
            {
                "Reply with exactly PASS.",
                "flydelta-runtime-smoke-session-" + std::to_string(arm_index),
                "flydelta-runtime-smoke",
                {},
                std::nullopt,
                value.model,
                value.n_predict,
                value.n_gpu_layers,
                true,
                value.backend,
                common_memory_scope::session,
                common_plan_scope::turn,
                value.n_threads,
                2048,
                {},
            },
            std::move(policy),
            std::move(runtime_config),
            {},
            common_memory_scope::session,
            false,
            {},
            {},
            {},
            residency,
        }));

    common_agent_runtime_session_host_turn_result result;
    const std::string session_id = "flydelta-runtime-smoke-session-" + std::to_string(arm_index);
    const std::string turn_id = "flydelta-runtime-smoke-turn-" + std::to_string(arm_index);
    common_agent_runtime_session_host_turn_request turn_request;
    turn_request.mode = common_agent_runtime_host_mode::chat;
    turn_request.prompt = "Reply with exactly PASS.";
    turn_request.session_id = session_id;
    turn_request.namespace_id = "flydelta-runtime-smoke";
    turn_request.turn_id = turn_id;
    turn_request.memory_scope = common_memory_scope::session;
    turn_request.plan_scope = common_plan_scope::turn;
    turn_request.n_predict = value.n_predict;
    turn_request.model_profile_id = profile_id;
    turn_request.flydelta_activation = activation_ptr;
    const auto started = std::chrono::steady_clock::now();
    const bool executed = runtime.run_turn(turn_request, result, error);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "arm=" << arm_index
              << " scale=" << scale
              << " executed=" << (executed ? "yes" : "no")
              << " host_verified=" << (response_passed(result) ? "yes" : "no")
              << " activation=" << (activation_ptr ? "yes" : "no")
              << " elapsed_ms=" << elapsed << '\n';
    if (!executed && error.empty()) error = result.error;
    return executed && response_passed(result);
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL [--backend cli|server-context]"
                  << " [--n-predict N] [--threads N] [--n-gpu-layers N]\n";
        return 2;
    }
    if (!regular_file(value.model)) {
        std::cerr << "FlyDelta runtime smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.backend != "cli" && value.backend != "server-context") {
        std::cerr << "--backend must be cli or server-context\n";
        return 2;
    }
    if (value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0) {
        std::cerr << "threads must be in range 1..3 and n-predict must be positive\n";
        return 2;
    }
#ifdef LLAMA_AGENT_ANDROID_CLI_ONLY
    if (value.backend == "server-context") {
        std::cerr << "FlyDelta runtime smoke skipped: server-context is unavailable on Android\n";
        return 77;
    }
#endif

    // Inspect the model once to build a dimension-safe activation. The
    // inspection model is released before the resident runtime is created.
    common_agent_model_selection inspect_selection;
    inspect_selection.profile_id = "flydelta-runtime-inspect";
    inspect_selection.backend = "cli";
    inspect_selection.path = value.model;
    inspect_selection.context_size_tokens = 2048;
    common_agent_runtime_cli_model_loader inspect_loader({value.n_gpu_layers, value.n_threads, true});
    std::shared_ptr<common_agent_runtime_resident_model> inspected;
    std::string error;
    if (!inspect_loader.load(inspect_selection, inspected, error)) {
        std::cerr << "FlyDelta runtime smoke could not inspect model: " << error << '\n';
        return 1;
    }
    const auto loaded = common_agent_runtime_loaded_model_cast(inspected);
    if (!loaded || !loaded->model) {
        std::cerr << "FlyDelta runtime smoke received an incomplete inspection model\n";
        return 1;
    }
    const size_t model_n_embd = static_cast<size_t>(llama_model_n_embd(loaded->model));
    const size_t model_n_layers = static_cast<size_t>(llama_model_n_layer(loaded->model));
    inspected.reset();
    if (model_n_embd == 0 || model_n_layers <= 1) {
        std::cerr << "FlyDelta runtime smoke received unsupported model dimensions\n";
        return 1;
    }

    const std::string profile_id = "flydelta-runtime-profile";
    const auto residency = make_residency(value, profile_id);
    const float scales[] = {0.0f, 0.01f, 0.02f};
    for (int arm = 0; arm < 3; ++arm) {
        error.clear();
        if (!run_arm(value, residency, profile_id, model_n_embd, model_n_layers,
                arm, scales[arm], error)) {
            std::cerr << "FlyDelta runtime arm failed: " << error << '\n';
            return 1;
        }
    }
    std::cout << "flydelta_runtime_three_arm=passed\n"
              << "backend=" << value.backend << '\n'
              << "arm_count=3\n"
              << "model_n_embd=" << model_n_embd << '\n'
              << "model_n_layers=" << model_n_layers << '\n'
              << "verdict=execution_verified_neutral\n";
    return 0;
}
