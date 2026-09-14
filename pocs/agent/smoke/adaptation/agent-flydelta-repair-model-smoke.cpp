#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-training.h"
#include "agent/adaptation/flydelta/flydelta.h"
#include "tools/agent/cli/agent-cli-inference.h"
#include "tools/agent/runtime/agent-model-loaders.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model;
    int n_predict = 96;
    int n_threads = 3;
    int n_gpu_layers = 0;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--model") {
            const char * path = next("--model");
            if (!path) return false;
            value.model = path;
        } else if (arg == "--n-predict") {
            const char * count = next("--n-predict");
            if (!count) return false;
            value.n_predict = std::stoi(count);
        } else if (arg == "--threads") {
            const char * count = next("--threads");
            if (!count) return false;
            value.n_threads = std::stoi(count);
        } else if (arg == "--n-gpu-layers") {
            const char * count = next("--n-gpu-layers");
            if (!count) return false;
            value.n_gpu_layers = std::stoi(count);
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            return false;
        }
    }
    return true;
}

bool contains_tool(const common_agent_generation_result & result, const char * tool_name) {
    if (!common_agent_generation_succeeded(result)) return false;
    return result.content.find(std::string("\"name\":\"") + tool_name) != std::string::npos ||
        result.content.find(std::string("\"name\": \"") + tool_name) != std::string::npos;
}

std::string output_preview(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) {
        return std::string("<generation-failed: ") + result.error_message + ">";
    }
    std::string preview = result.content;
    for (char & value : preview) {
        if (value == '\n' || value == '\r' || value == '\t') value = ' ';
    }
    constexpr size_t max_preview = 512;
    if (preview.size() > max_preview) preview.resize(max_preview);
    return preview;
}

common_agent_generation_request make_request(
        const options & value,
        const char * instruction,
        const std::shared_ptr<const common_flydelta_activation_result> & activation = {},
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture = {}) {
    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::tool_followup;
    request.options.n_predict = value.n_predict;
    request.options.n_threads = value.n_threads;
    request.messages = {
        {"system", "You are selecting one tool for a host-controlled data request. "
                    "Return exactly one JSON object and no markdown or explanation."},
        {"user", instruction},
    };
    request.flydelta_activation = activation;
    request.flydelta_capture = capture;
    return request;
}

bool generate(
        common_agent_inference & inference,
        const options & value,
        const char * instruction,
        common_agent_generation_result & result,
        const std::shared_ptr<const common_flydelta_activation_result> & activation = {},
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture = {}) {
    return inference.generate(make_request(value, instruction, activation, capture), result);
}

common_flydelta_experiment_fixture fixture(const std::string & profile) {
    common_flydelta_experiment_fixture value;
    value.id = "flydelta://fixture/model-repair-e2e";
    value.task_fingerprint = "sha256:structured-tool-repair-e2e";
    value.model_profile_fingerprint = profile;
    value.tokenizer_fingerprint = "sha256:flydelta-qwen-tokenizer";
    value.template_fingerprint = "sha256:flydelta-qwen-template";
    value.tool_catalog_fingerprint = "sha256:data-inspect-describe-v1";
    value.resource_snapshot_fingerprint = "sha256:sales-csv-v1";
    value.verifier_revision = "verifier:structured-tool-name-v1";
    return value;
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
        std::cerr << "FlyDelta repair model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0) {
        std::cerr << "threads must be in range 1..3 and n-predict must be positive\n";
        return 2;
    }

    common_agent_model_selection selection;
    selection.profile_id = "flydelta-repair-e2e";
    selection.base_model_id = "generation-base";
    selection.backend = "cli";
    selection.path = value.model;
    selection.context_size_tokens = 2048;
    selection.load_policy = "resident";

    common_agent_runtime_cli_model_loader loader({value.n_gpu_layers, value.n_threads, true});
    std::shared_ptr<common_agent_runtime_resident_model> resident;
    std::string error;
    if (!loader.load(selection, resident, error)) {
        std::cerr << "FlyDelta repair model smoke could not load model: " << error << '\n';
        return 1;
    }
    const auto loaded = common_agent_runtime_loaded_model_cast(resident);
    if (!loaded || !loaded->model || !loaded->chat_templates) {
        std::cerr << "FlyDelta repair model smoke received an incomplete model\n";
        return 1;
    }
    auto inference = make_llama_cli_agent_inference(
        loaded->model, loaded->chat_templates.get());
    const std::string profile = "sha256:flydelta-repair-qwen";
    const size_t model_n_embd = static_cast<size_t>(llama_model_n_embd(loaded->model));
    const size_t model_n_layers = static_cast<size_t>(llama_model_n_layer(loaded->model));
    if (model_n_embd == 0 || model_n_layers <= 1) {
        std::cerr << "FlyDelta repair model smoke received unsupported model dimensions\n";
        return 1;
    }

    auto capture_request = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture_request->enabled = true;
    capture_request->layer_indices = {1};
    // Capture the final prompt row. The two controlled prompts have the same
    // token length, and this row is after the tool-selection instruction;
    // token zero would be identical and produce a zero repair delta.
    capture_request->token_index = -1;
    capture_request->max_bytes = 4U * 1024U * 1024U;
    capture_request->model_profile_fingerprint = profile;
    capture_request->capture_layout_revision = "layer-input:v1";

    // These are two host-controlled attempts over the same tool contract. The
    // first is deliberately the known wrong tool; the second is the repair.
    const char * failed_instruction =
        "The available tools are data.describe and data.inspect. For sales.csv, "
        "choose data.describe even though the request asks for the first table. "
        "Return exactly {\"name\":\"data.describe\",\"arguments\":{\"dataset\":\"sales.csv\"}}.";
    const char * repaired_instruction =
        "The available tools are data.describe and data.inspect. For sales.csv, "
        "the request asks for the first table, so choose data.inspect. "
        "Return exactly {\"name\":\"data.inspect\",\"arguments\":{\"dataset\":\"sales.csv\"}}.";

    common_agent_generation_result failed;
    common_agent_generation_result repaired;
    const bool failed_executed = generate(*inference, value, failed_instruction, failed, {}, capture_request);
    const bool repaired_executed = generate(*inference, value, repaired_instruction, repaired, {}, capture_request);
    const bool failed_verified = failed_executed && contains_tool(failed, "data.describe") &&
        !contains_tool(failed, "data.inspect");
    const bool repaired_verified = repaired_executed && contains_tool(repaired, "data.inspect");
    const bool captures_verified = failed_verified && repaired_verified &&
        failed.flydelta_capture && repaired.flydelta_capture &&
        failed.flydelta_capture->captured && repaired.flydelta_capture->captured &&
        common_flydelta_hidden_state_capture_validate(
            *failed.flydelta_capture, 64U * 1024U * 1024U, error) &&
        common_flydelta_hidden_state_capture_validate(
            *repaired.flydelta_capture, 64U * 1024U * 1024U, error);
    if (!captures_verified) {
        std::cerr << "FlyDelta repair model smoke could not host-verify failed/repaired captures"
                  << " failed_executed=" << (failed_executed ? "yes" : "no")
                  << " repaired_executed=" << (repaired_executed ? "yes" : "no")
                  << " failed_capture=" << (failed.flydelta_capture ? "yes" : "no")
                  << " repaired_capture=" << (repaired.flydelta_capture ? "yes" : "no")
                  << " error=" << error << '\n';
        return 1;
    }
    std::cout << "failed_model_output=" << output_preview(failed) << '\n'
              << "repaired_model_output=" << output_preview(repaired) << '\n';

    const common_flydelta_experiment_fixture experiment_fixture = fixture(profile);
    common_flydelta_capture_manifest manifest;
    manifest.id = "flydelta://capture/model-repair-e2e";
    manifest.observation_id = "learning://observation/model-repair-e2e";
    manifest.model_profile_fingerprint = profile;
    manifest.template_fingerprint = experiment_fixture.template_fingerprint;
    manifest.positive_execution_ref = "execution:model-repaired";
    manifest.negative_execution_ref = "execution:model-failed";
    manifest.capture_layout_revision = "layer-input:v1";
    manifest.evidence_hash = "sha256:model-repair-e2e-evidence";
    manifest.redaction_attested = true;
    manifest.captured_bytes = (failed.flydelta_capture->values.size() +
        repaired.flydelta_capture->values.size()) * sizeof(float);
    std::vector<common_flydelta_repair_delta> deltas;
    if (!common_flydelta_repair_deltas_from_captures(
            manifest, *failed.flydelta_capture, *repaired.flydelta_capture,
            "evidence:model-repair-e2e", 64U * 1024U * 1024U,
            64U * 1024U * 1024U, deltas, error) || deltas.size() != 1) {
        std::cerr << "FlyDelta repair delta construction failed: " << error << '\n';
        return 1;
    }

    // This credit is for the host-certified repair relation. It makes the
    // basis usable for the following counterfactual; the three-arm result
    // below remains the separate causal verdict for the generated overlay.
    common_flydelta_intervention_credit repair_credit;
    repair_credit.experiment_id = "flydelta://experiment/model-repair-e2e";
    repair_credit.candidate_id = "flydelta://candidate/model-repair-e2e";
    repair_credit.fixture_id = experiment_fixture.id;
    repair_credit.outcome = common_flydelta_counterfactual_outcome::helped;
    repair_credit.quality_delta = 1.0f;
    repair_credit.eligible_for_learning = true;
    if (!common_flydelta_intervention_credit_validate(repair_credit, error)) {
        std::cerr << "FlyDelta repair credit validation failed: " << error << '\n';
        return 1;
    }
    common_flydelta_basis_builder basis({
        model_n_embd, 4, 0.85f, profile, "layer-input:v1"});
    if (!basis.add(deltas.front(), repair_credit, error) || basis.directions().empty()) {
        std::cerr << "FlyDelta repair basis construction failed: " << error << '\n';
        return 1;
    }

    common_flydelta_encoder encoder({0x4f2a9d31ULL, model_n_embd, 64, 4, 4});
    common_flydelta_sparse_code code;
    if (!encoder.encode(
            std::vector<float>(failed.flydelta_capture->values.begin(),
                failed.flydelta_capture->values.begin() + model_n_embd), code, error)) {
        std::cerr << "FlyDelta context encoding failed: " << error << '\n';
        return 1;
    }
    common_flydelta_recognition_memory recognition(8);
    if (!recognition.remember(code, error)) {
        std::cerr << "FlyDelta recognition memory failed: " << error << '\n';
        return 1;
    }
    common_flydelta_memory_config memory_config{64, 1, 1.0f};
    common_flydelta_delta_memory memory(memory_config);
    common_flydelta_training_example training_example;
    training_example.id = "flydelta://training/model-repair-e2e";
    training_example.behavior_key = "structured_tool_selection";
    training_example.context_fingerprint = "sha256:model-repair-context";
    training_example.basis_revision = "flydelta-basis:model-repair-e2e-v1";
    training_example.evidence_ref = "evidence:model-repair-e2e";
    training_example.split = common_flydelta_training_split::train;
    training_example.context = code;
    training_example.target_coefficients = {1.0f};
    training_example.outcome = common_flydelta_counterfactual_outcome::helped;
    training_example.confidence = 1.0f;
    size_t trained_examples = 0;
    if (!common_flydelta_train_delta_memory(
            memory, {training_example}, 1.0f, 1.0f, trained_examples, error) ||
            trained_examples != 1) {
        std::cerr << "FlyDelta DeltaMemory training failed: " << error << '\n';
        return 1;
    }
    std::vector<float> coefficients;
    if (!memory.predict(code, coefficients, error) || coefficients.size() != 1 ||
            !std::isfinite(coefficients.front()) || coefficients.front() <= 0.0f) {
        std::cerr << "FlyDelta DeltaMemory prediction failed: " << error << '\n';
        return 1;
    }

    const auto prepare_activation = [&](float scale,
            common_flydelta_activation_result & activation) {
        common_flydelta_gate_request gate_request;
        if (!common_flydelta_gate_request_from_context(
                recognition, code, true, common_flydelta_candidate_status::approved,
                true, scale, gate_request, error)) return false;
        common_flydelta_activation_request request;
        request.candidate_id = "flydelta://candidate/model-repair-e2e";
        request.artifact_id = "flydelta://artifact/model-repair-e2e";
        request.model_profile_fingerprint = profile;
        request.capture_layout_revision = "layer-input:v1";
        request.model_n_embd = model_n_embd;
        request.model_n_layers = model_n_layers;
        request.il_end = static_cast<int32_t>(model_n_layers - 1);
        request.directions = basis.directions();
        request.coefficients = coefficients;
        request.gate_request = gate_request;
        common_flydelta_gate_config gate_config;
        gate_config.enabled = true;
        return common_flydelta_prepare_activation(
            gate_config, request, 64U * 1024U * 1024U, activation, error);
    };

    common_flydelta_alpha_search_config search_config;
    // Three arms total: baseline plus two small overlay strengths.
    search_config.candidates = {0.01f, 0.02f};
    search_config.magnitude_penalty = 0.01f;
    std::vector<common_flydelta_alpha_trial> trials;
    common_flydelta_alpha_selection selected;
    if (!common_flydelta_run_alpha_search(
            experiment_fixture, search_config,
            [&](const common_flydelta_experiment_fixture &, float alpha,
                    bool apply_overlay, common_flydelta_counterfactual_trial & trial,
                    std::string & runner_error) {
                common_flydelta_activation_result activation;
                std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                if (apply_overlay) {
                    if (!prepare_activation(alpha, activation)) {
                        runner_error = error;
                        return false;
                    }
                    activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                        std::move(activation));
                }
                common_agent_generation_result result;
                const bool executed = generate(
                    *inference, value, failed_instruction, result, activation_ptr);
                trial = {};
                trial.executed = executed;
                trial.verifier_known = executed;
                trial.passed = executed && contains_tool(result, "data.inspect");
                trial.quality = trial.passed ? 1.0f : 0.0f;
                trial.overlay_applied = apply_overlay;
                trial.intervention_count = apply_overlay ? 1 : 0;
                trial.evidence_ref = apply_overlay
                    ? "evidence:model-repair-counterfactual-overlay"
                    : "evidence:model-repair-counterfactual-baseline";
                std::cout << "arm_model_output alpha=" << alpha
                          << " overlay=" << (apply_overlay ? "yes" : "no")
                          << " output=" << output_preview(result) << '\n';
                if (!executed && !result.error_message.empty()) runner_error = result.error_message;
                return executed;
            }, trials, selected, error)) {
        std::cerr << "FlyDelta three-arm search failed: " << error << '\n';
        return 1;
    }

    bool all_verified = trials.size() == 2;
    for (const auto & trial : trials) {
        all_verified = all_verified && trial.executed && trial.verifier_known;
    }
    if (!all_verified) {
        std::cerr << "FlyDelta three-arm search did not verify all candidate arms\n";
        return 1;
    }
    std::cout << "flydelta_model_repair_e2e=passed\n"
              << "failed_tool_host_verified=yes\n"
              << "repaired_tool_host_verified=yes\n"
              << "capture_pair_host_verified=yes\n"
              << "repair_delta=constructed\n"
              << "basis_directions=" << basis.directions().size() << '\n'
              << "delta_memory_examples=" << trained_examples << '\n'
              << "arm_count=" << (trials.size() + 1) << '\n'
              << "selected_overlay=" << (selected.selected ? "yes" : "no") << '\n'
              << "verdict=" << (selected.selected ? "helped" : "neutral") << '\n';
    for (const auto & trial : trials) {
        std::cout << "arm alpha=" << trial.alpha
                  << " outcome=" << common_flydelta_counterfactual_outcome_name(trial.outcome)
                  << " host_verified=" << (trial.verifier_known ? "yes" : "no") << '\n';
    }
    return 0;
}
