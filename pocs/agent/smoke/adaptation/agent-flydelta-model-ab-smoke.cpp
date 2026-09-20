#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-sideband-registry.h"
#include "tools/agent/cli/agent-cli-inference.h"
#include "tools/agent/runtime/agent-model-loaders.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model;
    int n_predict = 16;
    int n_threads = 3;
    int n_gpu_layers = 0;
    bool multi_arm = false;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << '\n'; return nullptr; }
            return argv[++i];
        };
        if (arg == "--model") {
            const char * path = next("--model"); if (!path) return false; value.model = path;
        } else if (arg == "--n-predict") {
            const char * count = next("--n-predict"); if (!count) return false; value.n_predict = std::stoi(count);
        } else if (arg == "--threads") {
            const char * count = next("--threads"); if (!count) return false; value.n_threads = std::stoi(count);
        } else if (arg == "--n-gpu-layers") {
            const char * count = next("--n-gpu-layers"); if (!count) return false; value.n_gpu_layers = std::stoi(count);
        } else if (arg == "--multi-arm") {
            value.multi_arm = true;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << arg << '\n'; return false;
        }
    }
    return true;
}

bool run_arm(
        common_agent_inference & inference,
        int n_predict,
        int n_threads,
        const std::shared_ptr<const common_flydelta_activation_result> & activation,
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture,
        common_agent_generation_result & result) {
    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::conversation;
    request.options.n_predict = n_predict;
    request.options.n_threads = n_threads;
    request.messages = {
        {"system", "Reply with exactly the single word PASS."},
        {"user", "Run the requested check."},
    };
    request.flydelta_activation = activation;
    request.flydelta_capture = capture;
    return inference.generate(request, result);
}

bool host_verifies(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return false;
    std::string content = result.content;
    std::transform(content.begin(), content.end(), content.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return content.find("PASS") != std::string::npos;
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL [--n-predict N] [--threads N] [--n-gpu-layers N] [--multi-arm]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "FlyDelta model A/B smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0) {
        std::cerr << "threads must be in range 1..3 and n-predict must be positive\n";
        return 2;
    }

    common_agent_model_selection selection;
    selection.profile_id = "flydelta-ab-base";
    selection.base_model_id = "generation-base";
    selection.backend = "cli";
    selection.path = value.model;
    selection.context_size_tokens = 2048;
    selection.load_policy = "resident";

    common_agent_runtime_cli_model_loader loader({value.n_gpu_layers, value.n_threads, true});
    std::shared_ptr<common_agent_runtime_resident_model> resident;
    std::string error;
    if (!loader.load(selection, resident, error)) {
        std::cerr << "FlyDelta model A/B smoke could not load model: " << error << '\n';
        return 1;
    }
    const auto loaded = common_agent_runtime_loaded_model_cast(resident);
    if (!loaded || !loaded->model || !loaded->chat_templates) {
        std::cerr << "FlyDelta model A/B smoke received an incomplete CLI model\n";
        return 1;
    }
    auto inference = make_llama_cli_agent_inference(
        loaded->model, loaded->chat_templates.get());

    common_agent_model_profile profile;
    profile.id = "flydelta-ab-base";
    profile.base_model_id = std::filesystem::path(value.model).filename().string();
    profile.base_model_fingerprint = "sha256:flydelta-smoke-base";
    profile.tokenizer_fingerprint = "sha256:flydelta-smoke-tokenizer";
    profile.chat_template_fingerprint = "sha256:flydelta-smoke-template";
    profile.context_size_tokens = 2048;
    const std::string sideband_id = "flydelta://sideband/smoke-v1";
    profile.sidebands.push_back({sideband_id, 1.0});

    common_flydelta_sideband_manifest manifest;
    manifest.id = sideband_id;
    manifest.artifact_path = "sidebands/smoke-v1.json";
    manifest.artifact_hash = "sha256:flydelta-smoke-artifact";
    manifest.compatibility.base_model_id = profile.base_model_id;
    manifest.compatibility.base_model_fingerprint = profile.base_model_fingerprint;
    manifest.compatibility.tokenizer_fingerprint = profile.tokenizer_fingerprint;
    manifest.compatibility.template_fingerprint = profile.chat_template_fingerprint;
    manifest.compatibility.architecture = "runtime-model";
    manifest.compatibility.inference_layout_revision = "layout:cvec-v1";
    manifest.model_n_embd = static_cast<size_t>(llama_model_n_embd(loaded->model));
    manifest.model_n_layers = static_cast<size_t>(llama_model_n_layer(loaded->model));
    if (manifest.model_n_layers <= 1 || manifest.model_n_layers > static_cast<size_t>(INT32_MAX)) {
        std::cerr << "FlyDelta model A/B smoke received unsupported model layer count\n";
        return 1;
    }
    manifest.il_end = static_cast<int32_t>(manifest.model_n_layers - 1);
    manifest.compatibility.architecture = "runtime-model";

    common_flydelta_sideband_registry registry;
    if (!registry.admit(manifest, error)) {
        std::cerr << "FlyDelta model A/B registry setup failed: " << error
                  << " (id=" << manifest.id
                  << ", artifact_path=" << manifest.artifact_path
                  << ", artifact_hash=" << manifest.artifact_hash
                  << ", model_n_embd=" << manifest.model_n_embd
                  << ", model_n_layers=" << manifest.model_n_layers
                  << ", il_start=" << manifest.il_start
                  << ", il_end=" << manifest.il_end << ")\n";
        return 1;
    }
    if (!registry.stage_canary(sideband_id, "eval:flydelta-ab-smoke", error) ||
            !registry.activate(sideband_id, error)) {
        std::cerr << "FlyDelta model A/B registry setup failed: " << error << '\n';
        return 1;
    }
    common_flydelta_compatibility expected = manifest.compatibility;
    common_flydelta_sideband_manifest resolved;
    double profile_scale = 0.0;
    if (!registry.resolve(profile, sideband_id, expected, manifest.model_n_embd,
            manifest.model_n_layers, resolved, profile_scale, error)) {
        std::cerr << "FlyDelta model A/B registry resolution failed: " << error << '\n';
        return 1;
    }

    common_flydelta_gate_config gate_config;
    gate_config.enabled = true;
    const auto prepare_activation = [&](float scale,
            common_flydelta_activation_result & activation) {
        common_flydelta_activation_request activation_request;
        activation_request.candidate_id = "flydelta://candidate/ab-smoke";
        activation_request.artifact_id = resolved.id;
        activation_request.model_profile_fingerprint = profile.base_model_fingerprint;
        activation_request.capture_layout_revision = resolved.compatibility.inference_layout_revision;
        activation_request.model_n_embd = manifest.model_n_embd;
        activation_request.model_n_layers = manifest.model_n_layers;
        activation_request.il_end = manifest.il_end;
        common_flydelta_basis_direction direction;
        direction.layer_index = 1;
        direction.values.assign(manifest.model_n_embd, 0.0f);
        // This deliberately tiny static direction tests cvec isolation and
        // timing only. It is not a learned repair basis and must not be used
        // to claim a behavior improvement from a neutral smoke result.
        direction.values.front() = 0.0001f;
        activation_request.directions.push_back(std::move(direction));
        activation_request.coefficients = {1.0f};
        activation_request.gate_request.explicit_opt_in = true;
        activation_request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
        activation_request.gate_request.basis_available = true;
        activation_request.gate_request.familiarity = 1.0f;
        activation_request.gate_request.novelty = 0.0f;
        activation_request.gate_request.requested_scale = scale;
        return common_flydelta_prepare_activation(gate_config, activation_request,
            64U * 1024U * 1024U, activation, error);
    };
    common_flydelta_activation_result activation;
    if (!prepare_activation(0.01f, activation)) {
        std::cerr << "FlyDelta model A/B activation preparation failed: " << error << '\n';
        return 1;
    }

    auto capture_request = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture_request->enabled = true;
    capture_request->layer_indices = {1};
    capture_request->max_bytes = 4U * 1024U * 1024U;
    capture_request->model_profile_fingerprint = profile.base_model_fingerprint;
    capture_request->capture_layout_revision = "layer-input:v1";

    common_agent_generation_result capture_seed;
    const bool capture_executed = run_arm(
        *inference, value.n_predict, value.n_threads, {}, capture_request, capture_seed);
    const bool capture_passed = capture_executed && host_verifies(capture_seed) &&
        capture_seed.flydelta_capture != nullptr && capture_seed.flydelta_capture->captured &&
        common_flydelta_hidden_state_capture_validate(
            *capture_seed.flydelta_capture, 64U * 1024U * 1024U, error);
    const auto activation_ptr = std::make_shared<const common_flydelta_activation_result>(std::move(activation));
    common_flydelta_experiment_fixture fixture;
    fixture.id = "flydelta://fixture/model-ab-smoke";
    fixture.task_fingerprint = "sha256:flydelta-model-ab-task";
    fixture.model_profile_fingerprint = profile.base_model_fingerprint;
    fixture.tokenizer_fingerprint = profile.tokenizer_fingerprint;
    fixture.template_fingerprint = profile.chat_template_fingerprint;
    fixture.execution_context_fingerprint = "sha256:flydelta-runtime-context";
    fixture.verifier_revision = "flydelta-model-ab-smoke:v1";
    if (value.multi_arm) {
        // Each arm deliberately creates a fresh inference context. A cvec is
        // applied during prefill, so sharing a baseline KV cache would not be
        // a valid same-fixture counterfactual.
        common_flydelta_alpha_search_config config;
        config.candidates = {0.0025f, 0.005f, 0.01f, 0.02f};
        config.magnitude_penalty = 0.01f;
        struct arm_metric {
            float alpha = 0.0f;
            long long elapsed_ms = 0;
            bool passed = false;
        };
        std::vector<arm_metric> metrics;
        std::vector<common_flydelta_alpha_trial> trials;
        common_flydelta_alpha_selection selection;
        const auto started = std::chrono::steady_clock::now();
        const bool multi_arm_executed = capture_passed && common_flydelta_run_alpha_search(
            fixture, config,
            [&](const common_flydelta_experiment_fixture &, float alpha, bool apply_overlay,
                    common_flydelta_counterfactual_trial & trial, std::string & runner_error) {
                common_flydelta_activation_result arm_activation;
                std::shared_ptr<const common_flydelta_activation_result> arm_activation_ptr;
                if (apply_overlay) {
                    if (!prepare_activation(alpha, arm_activation)) {
                        runner_error = error;
                        return false;
                    }
                    arm_activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                        std::move(arm_activation));
                }
                common_agent_generation_result arm;
                const auto arm_started = std::chrono::steady_clock::now();
                const bool executed = run_arm(
                    *inference, value.n_predict, value.n_threads, arm_activation_ptr, {}, arm);
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - arm_started).count();
                const bool passed = executed && host_verifies(arm);
                metrics.push_back({alpha, elapsed, passed});
                trial.executed = executed;
                trial.verifier_known = executed;
                trial.passed = passed;
                trial.quality = passed ? 1.0f : 0.0f;
                trial.overlay_applied = apply_overlay;
                trial.intervention_count = apply_overlay ? 1 : 0;
                trial.evidence_ref = apply_overlay
                    ? "evidence:flydelta-multi-arm-candidate"
                    : "evidence:flydelta-multi-arm-baseline";
                if (!executed && !arm.error_message.empty()) runner_error = arm.error_message;
                return executed;
            }, trials, selection, error);
        if (!multi_arm_executed) {
            std::cerr << "FlyDelta model multi-arm host verification failed"
                      << " capture=" << (capture_passed ? "pass" : "fail")
                      << " error=" << error << '\n';
            return 1;
        }
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "flydelta_model_multi_arm=passed\n"
                  << "capture_host_verified=" << (capture_passed ? "yes" : "no") << '\n'
                  << "arm_count=" << metrics.size() << '\n'
                  << "total_elapsed_ms=" << total_ms << '\n'
                  << "selected_alpha=" << (selection.selected ? std::to_string(selection.alpha) : "none") << '\n'
                  << "verdict=" << (selection.selected ? "helped" : "neutral") << '\n';
        for (const auto & metric : metrics) {
            std::cout << "arm alpha=" << metric.alpha
                      << " elapsed_ms=" << metric.elapsed_ms
                      << " host_verified=" << (metric.passed ? "yes" : "no") << '\n';
        }
        return 0;
    }
    common_flydelta_counterfactual_report report;
    const bool counterfactual_executed = capture_passed && common_flydelta_run_counterfactual(
        "flydelta://experiment/model-ab-smoke",
        "flydelta://candidate/ab-smoke",
        "flydelta://profile/baseline",
        "flydelta://profile/static-overlay",
        fixture,
        [&](const common_flydelta_experiment_fixture &, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial, std::string & runner_error) {
            common_agent_generation_result arm;
            const bool executed = run_arm(
                *inference, value.n_predict, value.n_threads,
                apply_overlay ? activation_ptr : std::shared_ptr<const common_flydelta_activation_result>{},
                {}, arm);
            trial.executed = executed;
            trial.verifier_known = executed;
            trial.passed = executed && host_verifies(arm);
            trial.quality = trial.passed ? 1.0f : 0.0f;
            trial.overlay_applied = apply_overlay;
            trial.intervention_count = apply_overlay ? 1 : 0;
            trial.evidence_ref = apply_overlay
                ? "evidence:flydelta-candidate-host-verifier"
                : "evidence:flydelta-baseline-host-verifier";
            if (!executed && !arm.error_message.empty()) runner_error = arm.error_message;
            return executed;
        }, report, error);
    if (!counterfactual_executed || !common_flydelta_counterfactual_report_validate(report, error)) {
        std::cerr << "FlyDelta model A/B host verification failed"
                  << " capture=" << (capture_passed ? "pass" : "fail")
                  << " counterfactual=" << (counterfactual_executed ? "pass" : "fail")
                  << " capture_reason=" << (capture_seed.flydelta_capture
                      ? capture_seed.flydelta_capture->failure_reason : "missing")
                  << " error=" << error << '\n';
        return 1;
    }
    std::cout << "flydelta_model_ab=passed\n"
              << "capture_host_verified=yes\n"
              << "baseline_host_verified=" << (report.baseline.passed ? "yes" : "no") << '\n'
              << "candidate_host_verified=" << (report.candidate.passed ? "yes" : "no") << '\n'
              << "candidate_overlay_applied=yes\n"
              << "outcome=" << common_flydelta_counterfactual_outcome_name(report.outcome) << '\n'
              << "quality_delta=" << report.quality_delta << '\n';
    return 0;
}
