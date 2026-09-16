#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-artifact-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"
#include "agent/adaptation/flydelta/flydelta-layer-search.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-scale-search.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "agent/adaptation/flydelta/flydelta-training.h"
#include "agent/adaptation/flydelta/flydelta.h"
#include "tools/agent/cli/agent-cli-generation.h"
#include "tools/agent/cli/agent-cli-inference.h"
#include "tools/agent/runtime/agent-model-loaders.h"

#include <algorithm>
#include <cctype>
#include <chrono>
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
    bool region_scan = false;
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
        } else if (arg == "--region-scan") {
            value.region_scan = true;
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
    value.execution_context_fingerprint = "sha256:data-inspect-sales-context-v1";
    value.verifier_revision = "verifier:structured-tool-name-v1";
    return value;
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL [--n-predict N] [--threads N] [--n-gpu-layers N]"
                  << " [--region-scan]\n";
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
    if (model_n_embd == 0 || model_n_layers <= 2) {
        std::cerr << "FlyDelta repair model smoke received unsupported model dimensions\n";
        return 1;
    }

    auto capture_request = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture_request->enabled = true;
    // layer-input is sampled before that layer's cvec addition. Capture a
    // small contiguous window so the host can rank actual layer locations and
    // test adjacent neighborhoods without inventing uncaptured layers.
    for (uint32_t layer = 1; layer < model_n_layers && layer <= 5; ++layer) {
        capture_request->layer_indices.push_back(layer);
    }
    // Capture the final prompt row. The two controlled prompts have the same
    // token length, and this row is after the tool-selection instruction;
    // token zero would be identical and produce a zero behavior delta.
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
    common_adaptation_evidence evidence;
    evidence.id = "evidence://flydelta/model-repair-e2e";
    evidence.source = common_adaptation_evidence_source::tool_repair;
    evidence.scope.namespace_id = "local";
    evidence.scope.project_id = "flydelta-model-smoke";
    evidence.scope.session_id = "flydelta-model-repair-e2e";
    evidence.behavior_key = "structured_tool_selection";
    evidence.task_fingerprint = experiment_fixture.task_fingerprint;
    evidence.baseline_ref = "execution:model-failed";
    evidence.candidate_ref = "execution:model-repaired";
    evidence.verifier_ref = experiment_fixture.verifier_revision;
    evidence.transaction_ids = {
        "transaction:model-failed", "transaction:model-repaired"};
    evidence.cause = common_learning_cause::model_behavior;
    evidence.host_verified = true;
    common_flydelta_behavior_transition transition;
    if (!common_flydelta_behavior_transition_from_evidence(
            evidence, evidence.transaction_ids.front(), evidence.transaction_ids.back(),
            transition, error)) {
        std::cerr << "FlyDelta model repair transition construction failed: " << error << '\n';
        return 1;
    }
    common_flydelta_capture_candidate capture_candidate;
    if (!common_flydelta_capture_candidate_from_transition(
            transition, evidence, profile, "layer-input:v1", capture_candidate, error)) {
        std::cerr << "FlyDelta model repair capture candidate construction failed: " << error << '\n';
        return 1;
    }
    common_flydelta_capture_manifest manifest;
    if (!common_flydelta_capture_manifest_from_candidate(
            capture_candidate, evidence, experiment_fixture.template_fingerprint,
            experiment_fixture.execution_context_fingerprint,
            "sha256:model-repair-e2e-evidence",
            (failed.flydelta_capture->values.size() + repaired.flydelta_capture->values.size()) *
                sizeof(float), true, manifest, error)) {
        std::cerr << "FlyDelta model repair capture manifest construction failed: " << error << '\n';
        return 1;
    }
    std::vector<common_flydelta_behavior_delta> deltas;
    if (!common_flydelta_behavior_deltas_from_verified_transition(
            transition, evidence, manifest, *failed.flydelta_capture,
            *repaired.flydelta_capture, 64U * 1024U * 1024U,
            64U * 1024U * 1024U, deltas, error) || deltas.size() < 2) {
        std::cerr << "FlyDelta behavior delta construction failed: " << error << '\n';
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
    common_flydelta_basis_config basis_config;
    basis_config.dimension = model_n_embd;
    basis_config.max_directions = 4;
    basis_config.cluster_similarity = 0.85f;
    basis_config.source = common_adaptation_evidence_source::tool_repair;
    basis_config.behavior_key = "structured_tool_selection";
    basis_config.model_profile_fingerprint = profile;
    basis_config.execution_context_fingerprint = experiment_fixture.execution_context_fingerprint;
    basis_config.capture_layout_revision = "layer-input:v1";
    common_flydelta_basis_builder basis(basis_config);
    for (const auto & delta : deltas) {
        // Layer 5 is captured only as a downstream measurement point for a
        // layer-4 intervention. It is not part of the searched basis window.
        if (delta.layer_index > 4) continue;
        if (!basis.add(delta, repair_credit, error)) {
            std::cerr << "FlyDelta repair basis construction failed: " << error << '\n';
            return 1;
        }
    }
    if (basis.directions().empty()) {
        std::cerr << "FlyDelta repair basis construction produced no directions\n";
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

    std::vector<common_flydelta_artifact_direction> artifact_basis;
    const auto artifact_direction = std::find_if(basis.directions().begin(),
        basis.directions().end(), [](const auto & direction) {
            return direction.layer_index == 2;
        });
    if (artifact_direction == basis.directions().end()) {
        std::cerr << "FlyDelta experimental artifact has no selected layer direction\n";
        return 1;
    }
    // A v2 artifact stores one steering vector per DeltaMemory target
    // coefficient. This candidate is deliberately the selected L2 arm;
    // the other captured layers remain experiment diagnostics, not active
    // artifact basis entries.
    artifact_basis.push_back({artifact_direction->layer_index, artifact_direction->values});
    common_flydelta_compatibility compatibility;
    compatibility.base_model_fingerprint = profile;
    compatibility.tokenizer_fingerprint = experiment_fixture.tokenizer_fingerprint;
    compatibility.template_fingerprint = experiment_fixture.template_fingerprint;
    compatibility.architecture = "qwen2";
    compatibility.inference_layout_revision = "layer-input:v1";
    common_flydelta_artifact experimental_artifact;
    const auto artifact_root = std::filesystem::temp_directory_path() /
        "llama-agent-flydelta-model-smoke-artifacts";
    std::error_code artifact_cleanup_error;
    std::filesystem::remove_all(artifact_root, artifact_cleanup_error);
    common_flydelta_artifact_store artifact_store(artifact_root);
    common_flydelta_sideband_registry artifact_registry;
    common_flydelta_sideband_manifest experimental_manifest;
    if (!common_flydelta_build_experimental_artifact(
            "flydelta://artifact/model-repair-e2e", 1, encoder.config(),
            memory.config(), compatibility, memory.weights(), model_n_embd,
            model_n_layers, 1, static_cast<int32_t>(model_n_layers - 1),
            artifact_basis, experimental_artifact, error) ||
            !common_flydelta_persist_experimental_artifact(
                artifact_store, "experiments/model-repair-e2e.flyd", artifact_registry,
                "local", "flydelta-model-smoke", 0, experimental_artifact,
                experimental_manifest, error) ||
            experimental_manifest.status != common_flydelta_sideband_status::experimental) {
        std::cerr << "FlyDelta experimental artifact persistence failed: " << error << '\n';
        return 1;
    }
    std::cout << "experimental_artifact_hash=" << experimental_artifact.content_hash << '\n'
              << "experimental_artifact_status=experimental\n";

    // Each captured layer gets its own diagnostic. Layer-search later uses
    // the most informative unknown arm to rank locations; diagnostics never
    // create learning evidence by themselves.
    std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_arm_capture;
    struct arm_diagnostic {
        float alpha = 0.0f;
        std::vector<common_flydelta_representation_diagnostics> values;
        bool available = false;
        std::string error;
    };
    std::vector<arm_diagnostic> arm_diagnostics;

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
        request.directions = {basis.directions().front()};
        request.coefficients = {coefficients.front()};
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
                    *inference, value, failed_instruction, result, activation_ptr,
                    capture_request);
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
                if (!apply_overlay && result.flydelta_capture) {
                    baseline_arm_capture = result.flydelta_capture;
                } else if (apply_overlay && !trial.passed && !baseline_arm_capture) {
                    runner_error = "FlyDelta unknown arm has no baseline capture";
                    return false;
                } else if (apply_overlay && !trial.passed && baseline_arm_capture &&
                        result.flydelta_capture) {
                    arm_diagnostic diagnostic;
                    diagnostic.alpha = alpha;
                    diagnostic.available = true;
                    for (const auto & delta : deltas) {
                        common_flydelta_representation_diagnostics values;
                        if (!common_flydelta_representation_diagnostics_from_captures(
                                *baseline_arm_capture, *result.flydelta_capture, delta,
                                64U * 1024U * 1024U, values, diagnostic.error)) {
                            diagnostic.available = false;
                            break;
                        }
                        diagnostic.values.push_back(std::move(values));
                    }
                    arm_diagnostics.push_back(std::move(diagnostic));
                }
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
        if (trial.outcome == common_flydelta_counterfactual_outcome::unknown) {
            const auto diagnostic = std::find_if(arm_diagnostics.begin(), arm_diagnostics.end(),
                [&](const auto & value) { return value.alpha == trial.alpha; });
            if (diagnostic != arm_diagnostics.end() && diagnostic->available) {
                for (const auto & values : diagnostic->values) {
                    std::cout << "unknown_representation_diagnostics alpha=" << trial.alpha
                              << " layer=" << values.layer_index
                              << " cosine=" << values.cosine
                              << " progress=" << values.progress
                              << " leakage=" << values.leakage
                              << " shift_norm=" << values.shift_norm
                              << " promotion=no next_action=layer_search\n";
                }
            } else {
                std::cout << "unknown_representation_diagnostics alpha=" << trial.alpha
                          << " available=no promotion=no next_action=extended_tuning\n";
            }
        }
    }

    if (value.region_scan && !selected.selected) {
        const auto anchor = std::find_if(basis.directions().begin(), basis.directions().end(),
            [](const auto & direction) { return direction.layer_index == 2; });
        if (anchor == basis.directions().end()) {
            std::cerr << "FlyDelta region pipeline has no layer-2 direction\n";
            return 1;
        }

        common_flydelta_search_pipeline_config pipeline_config;
        pipeline_config.dimension = model_n_embd;
        pipeline_config.max_directions = 1;
        pipeline_config.scale.initial_scale = 0.02f;
        pipeline_config.scale.growth_factor = 2.0f;
        pipeline_config.scale.max_scale = 0.16f;
        pipeline_config.scale.max_geometric_trials = 4;
        pipeline_config.scale.max_refinement_trials = 0;
        pipeline_config.scale.min_cosine = 0.3f;
        pipeline_config.scale.max_leakage = 1.0f;
        pipeline_config.scale.max_shift_norm = 1.0f;
        pipeline_config.region_max_singleton_layers = 4;
        pipeline_config.region_max_neighborhoods = 4;
        pipeline_config.region_max_trials = 32;
        pipeline_config.region_max_stalled_scales = 2;

        common_flydelta_search_pipeline_direction pipeline_direction;
        pipeline_direction.direction = {
            1, common_flydelta_direction_kind::raw_repair, anchor->layer_index,
            anchor->values, 1, 1, 1.0f, false};
        for (const auto & delta : deltas) {
            if (delta.layer_index > 0 && delta.layer_index <= 4) {
                pipeline_direction.available_layers.push_back(
                    static_cast<uint32_t>(delta.layer_index));
            }
        }
        std::sort(pipeline_direction.available_layers.begin(),
            pipeline_direction.available_layers.end());
        pipeline_direction.available_layers.erase(std::unique(
            pipeline_direction.available_layers.begin(),
            pipeline_direction.available_layers.end()), pipeline_direction.available_layers.end());

        std::shared_ptr<const common_flydelta_hidden_state_capture> region_baseline_capture;
        const auto region_started = std::chrono::steady_clock::now();
        common_flydelta_search_pipeline_result pipeline_result;
        if (!common_flydelta_run_search_pipeline(
                experiment_fixture, pipeline_config, {pipeline_direction},
                [&](const common_flydelta_experiment_fixture &,
                        const common_flydelta_direction_candidate &,
                        const common_flydelta_layer_candidate * candidate,
                        float, bool apply_overlay,
                        common_flydelta_counterfactual_trial & trial,
                        common_flydelta_decision_margin & margin,
                        common_flydelta_scale_geometry & geometry,
                        std::string & runner_error) {
                    common_agent_generation_result result;
                    geometry = {};
                    std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                    if (apply_overlay) {
                        if (!candidate) {
                            runner_error = "region pipeline arm is missing candidate";
                            return false;
                        }
                        common_flydelta_gate_request gate_request;
                        if (!common_flydelta_gate_request_from_context(
                                recognition, code, true,
                                common_flydelta_candidate_status::approved, true,
                                candidate->per_layer_scale, gate_request, runner_error)) {
                            return false;
                        }
                        common_flydelta_activation_request request;
                        request.candidate_id = "flydelta://candidate/model-repair-region";
                        request.artifact_id = "flydelta://artifact/model-repair-e2e";
                        request.model_profile_fingerprint = profile;
                        request.capture_layout_revision = "layer-input:v1";
                        request.model_n_embd = model_n_embd;
                        request.model_n_layers = model_n_layers;
                        request.il_end = static_cast<int32_t>(model_n_layers - 1);
                        for (const uint32_t layer : candidate->layer_indices) {
                            const auto direction = std::find_if(basis.directions().begin(),
                                basis.directions().end(), [&](const auto & value) {
                                    return value.layer_index == static_cast<int32_t>(layer);
                                });
                            if (direction == basis.directions().end()) {
                                runner_error = "region pipeline arm has no layer-compatible direction";
                                return false;
                            }
                            request.directions.push_back(*direction);
                            request.coefficients.push_back(coefficients.front());
                        }
                        request.gate_request = gate_request;
                        common_flydelta_gate_config gate_config;
                        gate_config.enabled = true;
                        gate_config.max_scale = 1.0f;
                        common_flydelta_activation_result activation;
                        if (!common_flydelta_prepare_activation(
                                gate_config, request, 64U * 1024U * 1024U,
                                activation, runner_error)) return false;
                        activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                            std::move(activation));
                    }
                    const bool executed = generate(*inference, value, failed_instruction, result,
                        activation_ptr, capture_request);
                    trial = {};
                    trial.executed = executed;
                    trial.verifier_known = executed;
                    trial.passed = executed && contains_tool(result, "data.inspect");
                    trial.quality = trial.passed ? 1.0f : 0.0f;
                    trial.overlay_applied = apply_overlay;
                    trial.intervention_count = apply_overlay && candidate
                        ? candidate->layer_indices.size() : 0;
                    trial.evidence_ref = apply_overlay
                        ? "evidence:model-repair-intervention-region"
                        : "evidence:model-repair-intervention-region-baseline";
                    const auto scoring_request = make_request(value, failed_instruction);
                    if (!score_chat_choice_margin(
                            loaded->model, loaded->chat_templates.get(), scoring_request.messages,
                            scoring_request.tools, scoring_request.tool_choice, scoring_request.options,
                            "{\"name\":\"", "data.inspect", "data.describe", margin,
                            nullptr, scoring_request.json_schema, {}, {},
                            apply_overlay && activation_ptr ? activation_ptr->overlay
                                : common_flydelta_static_overlay{}, &runner_error)) {
                        return false;
                    }
                    if (!apply_overlay && result.flydelta_capture) {
                        region_baseline_capture = result.flydelta_capture;
                    }
                    if (apply_overlay && candidate && result.flydelta_capture &&
                            region_baseline_capture) {
                        const auto measurement = std::find_if(deltas.begin(), deltas.end(),
                            [&](const auto & delta) {
                                return delta.layer_index == static_cast<int>(candidate->anchor_layer_index);
                            });
                        if (measurement != deltas.end()) {
                            common_flydelta_representation_diagnostics diagnostics;
                            if (!common_flydelta_representation_diagnostics_from_captures(
                                    *region_baseline_capture, *result.flydelta_capture, *measurement,
                                    64U * 1024U * 1024U, diagnostics, runner_error)) return false;
                            geometry.available = true;
                            geometry.cosine = diagnostics.cosine;
                            geometry.progress = diagnostics.progress;
                            geometry.leakage = diagnostics.leakage;
                            geometry.shift_norm = diagnostics.shift_norm;
                        }
                    }
                    std::cout << "region_model_output layers=";
                    if (candidate) {
                        for (size_t index = 0; index < candidate->layer_indices.size(); ++index) {
                            if (index != 0) std::cout << ',';
                            std::cout << candidate->layer_indices[index];
                        }
                    } else {
                        std::cout << "baseline";
                    }
                    std::cout << " scale=" << (candidate ? candidate->total_scale : 0.0f)
                              << " output=" << output_preview(result) << '\n';
                    if (!executed && !result.error_message.empty()) runner_error = result.error_message;
                    return executed;
                }, pipeline_result, error)) {
            std::cerr << "FlyDelta search pipeline failed: " << error << '\n';
            return 1;
        }
        if (pipeline_result.directions.empty()) {
            std::cerr << "FlyDelta search pipeline produced no direction result\n";
            return 1;
        }
        const auto & region_trials = pipeline_result.directions.front().region_trials;
        const auto & region_selection = pipeline_result.directions.front().region_selection;
        std::cout << "intervention_region_search=completed"
                  << " pipeline=default"
                  << " trials=" << region_trials.size()
                  << " model_calls=" << (region_trials.size() + 1)
                  << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - region_started).count()
                  << " selected=" << (region_selection.selected ? "yes" : "no") << '\n';
        for (const auto & trial : region_trials) {
            std::cout << "region_trial layers=";
            for (const uint32_t layer : trial.candidate.layer_indices) std::cout << layer << ',';
            std::cout << " scale=" << trial.candidate.total_scale
                      << " outcome=" << common_flydelta_counterfactual_outcome_name(trial.outcome)
                      << " promising=" << (trial.promising ? "yes" : "no")
                      << " safe_to_continue=" << (trial.safe_to_continue ? "yes" : "no");
            if (trial.geometry_available) {
                std::cout << " cosine=" << trial.geometry.cosine
                          << " progress=" << trial.geometry.progress
                          << " leakage=" << trial.geometry.leakage
                          << " shift_norm=" << trial.geometry.shift_norm;
            }
            std::cout << '\n';
        }
    }

    common_flydelta_layer_search_selection layer_selection;
    std::vector<common_flydelta_layer_search_trial> layer_trials;
    if (!value.region_scan && !selected.selected) {
        const auto best_arm = std::max_element(arm_diagnostics.begin(), arm_diagnostics.end(),
            [](const auto & left, const auto & right) {
                const auto best_score = [](const auto & arm) {
                    float score = 0.0f;
                    for (const auto & value : arm.values) {
                        if (value.cosine >= 0.3f && value.progress > 0.0f) {
                            score = std::max(score, value.progress / (1.0f + value.leakage));
                        }
                    }
                    return score;
                };
                return best_score(left) < best_score(right);
            });
        if (best_arm != arm_diagnostics.end() && best_arm->available) {
            std::vector<common_flydelta_layer_diagnostic> diagnostics;
            for (const auto & value : best_arm->values) {
                diagnostics.push_back({value.layer_index, value.cosine, value.progress,
                    value.leakage, value.shift_norm});
            }
            common_flydelta_layer_search_config layer_config;
            layer_config.max_regions = 2;
            layer_config.max_singletons = 4;
            layer_config.max_neighborhoods = 4;
            layer_config.max_candidates = 8;
            layer_config.min_region_separation = 2;
            layer_config.min_cosine = 0.3f;
            layer_config.total_scale = 0.02f;
            std::vector<uint32_t> available_layers;
            for (const auto & delta : deltas) {
                available_layers.push_back(static_cast<uint32_t>(delta.layer_index));
            }
            common_flydelta_layer_search_plan layer_plan;
            if (!common_flydelta_build_layer_search_plan(
                    diagnostics, available_layers, layer_config, layer_plan, error)) {
                std::cerr << "FlyDelta layer search plan failed: " << error << '\n';
                return 1;
            }
            const auto prepare_layer_activation = [&](const common_flydelta_layer_candidate & candidate,
                    common_flydelta_activation_result & activation,
                    float gate_max_scale = 0.25f,
                    const common_flydelta_basis_direction * direction_override = nullptr) {
                common_flydelta_gate_request gate_request;
                if (!common_flydelta_gate_request_from_context(
                        recognition, code, true, common_flydelta_candidate_status::approved,
                        true, candidate.per_layer_scale, gate_request, error)) return false;
                common_flydelta_activation_request request;
                request.candidate_id = "flydelta://candidate/model-repair-layer-search";
                request.artifact_id = "flydelta://artifact/model-repair-e2e";
                request.model_profile_fingerprint = profile;
                request.capture_layout_revision = "layer-input:v1";
                request.model_n_embd = model_n_embd;
                request.model_n_layers = model_n_layers;
                request.il_end = static_cast<int32_t>(model_n_layers - 1);
                for (const uint32_t layer : candidate.layer_indices) {
                    if (direction_override && candidate.layer_indices.size() == 1 && layer == 2) {
                        request.directions.push_back(*direction_override);
                    } else {
                        const auto direction = std::find_if(basis.directions().begin(), basis.directions().end(),
                            [&](const auto & value) { return value.layer_index == static_cast<int32_t>(layer); });
                        if (direction == basis.directions().end()) {
                            error = "FlyDelta layer search candidate has no compatible basis direction";
                            return false;
                        }
                        request.directions.push_back(*direction);
                    }
                    request.coefficients.push_back(coefficients.front());
                }
                request.gate_request = gate_request;
                common_flydelta_gate_config gate_config;
                gate_config.enabled = true;
                gate_config.max_scale = gate_max_scale;
                return common_flydelta_prepare_activation(
                    gate_config, request, 64U * 1024U * 1024U, activation, error);
            };
            if (!common_flydelta_run_layer_search(
                    experiment_fixture, layer_plan,
                    [&](const common_flydelta_experiment_fixture &,
                            const common_flydelta_layer_candidate * candidate,
                            bool apply_overlay,
                            common_flydelta_counterfactual_trial & trial,
                            std::string & runner_error) {
                        common_flydelta_activation_result activation;
                        std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                        if (apply_overlay) {
                            if (!candidate || !prepare_layer_activation(*candidate, activation)) {
                                runner_error = error;
                                return false;
                            }
                            activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                                std::move(activation));
                        }
                        common_agent_generation_result result;
                        const bool executed = generate(*inference, value, failed_instruction, result,
                            activation_ptr, capture_request);
                        trial = {};
                        trial.executed = executed;
                        trial.verifier_known = executed;
                        trial.passed = executed && contains_tool(result, "data.inspect");
                        trial.quality = trial.passed ? 1.0f : 0.0f;
                        trial.overlay_applied = apply_overlay;
                        trial.intervention_count = apply_overlay ? candidate->layer_indices.size() : 0;
                        trial.evidence_ref = apply_overlay
                            ? "evidence:model-repair-layer-search"
                            : "evidence:model-repair-layer-baseline";
                        std::cout << "layer_search_model_output layers="
                                  << (candidate ? common_flydelta_layer_search_candidate_source_name(candidate->source)
                                      : "baseline")
                                  << " overlay=" << (apply_overlay ? "yes" : "no")
                                  << " output=" << output_preview(result) << '\n';
                        if (!executed && !result.error_message.empty()) runner_error = result.error_message;
                        return executed;
                    }, layer_trials, layer_selection, error)) {
                std::cerr << "FlyDelta layer search failed: " << error << '\n';
                return 1;
            }
            std::cout << "layer_search_singletons=" << layer_plan.singleton_candidates.size()
                      << " layer_search_neighborhoods=" << layer_plan.neighborhood_candidates.size()
                      << " layer_search_trials=" << layer_trials.size()
                      << " layer_search_selected=" << (layer_selection.selected ? "yes" : "no") << '\n';

            // L2 was the only layer that passed the diagnostic cosine gate in
            // the previous smoke. Keep the next experiment narrow and search
            // scale there before widening the layer mask again.
            // Captures are layer-input samples, before that layer's own cvec
            // addition. Measure an L2 injection at the next captured layer so
            // a same-layer zero is not mistaken for saturation.
            const auto layer2_injection = std::find_if(deltas.begin(), deltas.end(),
                [](const auto & delta) { return delta.layer_index == 2; });
            const auto layer2_effect = std::find_if(deltas.begin(), deltas.end(),
                [](const auto & delta) { return delta.layer_index > 2; });
            const auto layer2_direction = std::find_if(basis.directions().begin(), basis.directions().end(),
                [](const auto & direction) { return direction.layer_index == 2; });
            if (layer2_injection != deltas.end() && layer2_effect != deltas.end() &&
                    layer2_direction != basis.directions().end()) {
                common_flydelta_direction_search_config direction_config;
                direction_config.dimension = model_n_embd;
                direction_config.layer_index = 2;
                // Fewer samples remain useful for CPU diagnostics, but the
                // model-facing aggregate search is not ready until six
                // natural host-certified repair pairs exist.
                direction_config.min_samples = 6;
                direction_config.max_samples = 32;
                direction_config.min_median_alignment = 0.25f;
                direction_config.trim_fraction = 0.20f;
                direction_config.variance_ridge = 0.001f;
                direction_config.source = common_adaptation_evidence_source::tool_repair;
                direction_config.behavior_key = "structured_tool_selection";
    direction_config.model_profile_fingerprint = profile;
    direction_config.execution_context_fingerprint = experiment_fixture.execution_context_fingerprint;
                direction_config.capture_layout_revision = "layer-input:v1";
                std::vector<common_flydelta_direction_candidate> direction_candidates;
                if (!common_flydelta_build_direction_candidates(
                        direction_config, {{*layer2_injection, repair_credit}},
                        direction_candidates, error) || direction_candidates.empty()) {
                    std::cerr << "FlyDelta direction search failed: " << error << '\n';
                    return 1;
                }
                // The current model smoke has one natural host-certified pair,
                // so only the raw control is eligible here. Aggregate
                // candidates become available when more certified pairs are
                // supplied; they are covered by the CPU contract test.
                common_flydelta_basis_direction scale_direction;
                scale_direction.layer_index = direction_candidates.front().layer_index;
                scale_direction.values = direction_candidates.front().values;
                scale_direction.helped_observations = 1;
                std::cout << "l2_direction_search_candidates=" << direction_candidates.size()
                          << " raw_control=" << common_flydelta_direction_kind_name(
                              direction_candidates.front().kind)
                          << " deep_ready=" << (direction_candidates.front().source_samples >= direction_config.min_samples ? "yes" : "no")
                          << " sample_count=" << direction_candidates.front().source_samples << '\n';
                common_flydelta_scale_search_config scale_config;
                scale_config.initial_scale = 0.02f;
                scale_config.growth_factor = 2.0f;
                scale_config.max_scale = 1.0f;
                scale_config.max_geometric_trials = 6;
                scale_config.max_refinement_trials = 1;
                scale_config.min_cosine = 0.3f;
                scale_config.max_leakage = 1.0f;
                scale_config.max_shift_norm = 1.0f;
                std::vector<common_flydelta_scale_trial> scale_trials;
                common_flydelta_scale_selection scale_selection;
                std::shared_ptr<const common_flydelta_hidden_state_capture> scale_baseline_capture;
                const auto scale_search_started = std::chrono::steady_clock::now();
                if (!common_flydelta_run_scale_search(
                        experiment_fixture, scale_config,
                        [&](const common_flydelta_experiment_fixture &, float scale,
                                bool apply_overlay,
                                common_flydelta_counterfactual_trial & trial,
                                common_flydelta_scale_geometry & geometry,
                                std::string & runner_error) {
                            common_agent_generation_result result;
                            const auto generation_started = std::chrono::steady_clock::now();
                            std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                            common_flydelta_layer_candidate candidate;
                            if (apply_overlay) {
                                candidate.layer_indices = {2};
                                candidate.anchor_layer_index = 2;
                                candidate.diagnostic_score = 1.0f;
                                candidate.total_scale = scale;
                                candidate.per_layer_scale = scale;
                                candidate.source = common_flydelta_layer_search_candidate_source::diagnostic_singleton;
                                common_flydelta_activation_result activation;
                                if (!prepare_layer_activation(candidate, activation, 1.0f, &scale_direction)) {
                                    runner_error = error;
                                    return false;
                                }
                                activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                                    std::move(activation));
                            }
                            const bool executed = generate(*inference, value, failed_instruction, result,
                                activation_ptr, capture_request);
                            trial = {};
                            trial.executed = executed;
                            trial.verifier_known = executed;
                            trial.passed = executed && contains_tool(result, "data.inspect");
                            trial.quality = trial.passed ? 1.0f : 0.0f;
                            trial.overlay_applied = apply_overlay;
                            trial.intervention_count = apply_overlay ? 1 : 0;
                            trial.evidence_ref = apply_overlay
                                ? "evidence:model-repair-l2-scale-search"
                                : "evidence:model-repair-l2-scale-baseline";
                            if (!apply_overlay && result.flydelta_capture) {
                                scale_baseline_capture = result.flydelta_capture;
                            }
                            if (apply_overlay && result.flydelta_capture) {
                                common_flydelta_representation_diagnostics values;
                                if (!common_flydelta_representation_diagnostics_from_captures(
                                        *scale_baseline_capture, *result.flydelta_capture, *layer2_effect,
                                        64U * 1024U * 1024U, values, runner_error)) {
                                    return false;
                                }
                                geometry.available = true;
                                geometry.cosine = values.cosine;
                                geometry.progress = values.progress;
                                geometry.leakage = values.leakage;
                                geometry.shift_norm = values.shift_norm;
                            }
                            std::cout << "l2_scale_model_output scale=" << scale
                                      << " overlay=" << (apply_overlay ? "yes" : "no")
                                      << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now() - generation_started).count()
                                      << " output=" << output_preview(result) << '\n';
                            if (!executed && !result.error_message.empty()) runner_error = result.error_message;
                            return executed;
                        }, scale_trials, scale_selection, error)) {
                    std::cerr << "FlyDelta L2 scale search failed: " << error << '\n';
                    return 1;
                }
                const auto scale_search_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - scale_search_started).count();
                std::cout << "l2_scale_search_injection_layer=2"
                          << " l2_scale_search_measurement_layer=" << layer2_effect->layer_index << '\n'
                          << "l2_scale_search_trials=" << scale_trials.size()
                          << " l2_scale_search_model_calls=" << (scale_trials.size() + 1)
                          << " l2_scale_search_elapsed_ms=" << scale_search_elapsed_ms
                          << " l2_scale_search_selected=" << (scale_selection.selected ? "yes" : "no") << '\n';
                for (const auto & trial : scale_trials) {
                    std::cout << "l2_scale_trial scale=" << trial.scale
                              << " outcome=" << common_flydelta_counterfactual_outcome_name(trial.outcome)
                              << " safe_to_escalate=" << (trial.safe_to_escalate ? "yes" : "no")
                              << " refinement=" << (trial.refinement ? "yes" : "no");
                    if (trial.geometry_available) {
                        std::cout << " cosine=" << trial.geometry.cosine
                                  << " progress=" << trial.geometry.progress
                                  << " leakage=" << trial.geometry.leakage
                                  << " shift_norm=" << trial.geometry.shift_norm;
                    }
                    std::cout << '\n';
                }

                // Exercise the optional TFO-lite coefficient strategy through
                // the same model-facing seam. The layer and WHAT direction
                // are fixed by the preceding searches; only WHEN/HOW MUCH is
                // explored here. This remains experimental and host verdicts
                // are the only source of HELPED evidence.
                common_flydelta_low_rank_basis coefficient_basis;
                if (!common_flydelta_build_low_rank_basis(
                        model_n_embd, 4, direction_candidates,
                        coefficient_basis, error)) {
                    std::cerr << "FlyDelta TFO-lite basis construction failed: " << error << '\n';
                    return 1;
                }
                common_flydelta_coefficient_search_config tfo_config;
                tfo_config.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
                tfo_config.step = 0.04f;
                tfo_config.max_candidates = 6;
                tfo_config.max_l2_norm = 0.32f;
                tfo_config.seed = 0x464c5944454c5441ULL;
                tfo_config.population_size = 3;
                tfo_config.iterations = 2;
                tfo_config.exploration_scale = 1.0f;
                tfo_config.norm_penalty = 0.05f;
                tfo_config.leakage_penalty = 0.10f;
                std::vector<common_flydelta_coefficient_trial> tfo_trials;
                common_flydelta_coefficient_selection tfo_selection;
                std::shared_ptr<const common_flydelta_hidden_state_capture> coefficient_baseline_capture;
                const auto tfo_started = std::chrono::steady_clock::now();
                if (!common_flydelta_run_low_rank_coefficient_search(
                        experiment_fixture, coefficient_basis, tfo_config,
                        [&](const common_flydelta_experiment_fixture &,
                                const common_flydelta_low_rank_basis &,
                                const std::vector<float> & tfo_coefficients,
                                bool apply_overlay,
                                common_flydelta_counterfactual_trial & trial,
                                common_flydelta_decision_margin & margin,
                                common_flydelta_representation_diagnostics & geometry,
                                bool & geometry_available,
                                std::string & runner_error) {
                            common_agent_generation_result result;
                            geometry = {};
                            geometry_available = false;
                            std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                            if (apply_overlay) {
                                common_flydelta_activation_result activation;
                                common_flydelta_gate_request gate_request;
                                if (!common_flydelta_gate_request_from_context(
                                        recognition, code, true,
                                        common_flydelta_candidate_status::approved,
                                        true, 1.0f, gate_request, error)) {
                                    runner_error = error;
                                    return false;
                                }
                                common_flydelta_activation_request request;
                                request.candidate_id = "flydelta://candidate/model-repair-tfo-lite";
                                request.artifact_id = "flydelta://artifact/model-repair-e2e";
                                request.model_profile_fingerprint = profile;
                                request.capture_layout_revision = "layer-input:v1";
                                request.model_n_embd = model_n_embd;
                                request.model_n_layers = model_n_layers;
                                request.il_end = static_cast<int32_t>(model_n_layers - 1);
                                for (const auto & vector : coefficient_basis.vectors) {
                                    request.directions.push_back({coefficient_basis.layer_index, vector});
                                }
                                request.coefficients = tfo_coefficients;
                                request.gate_request = gate_request;
                                common_flydelta_gate_config gate_config;
                                gate_config.enabled = true;
                                gate_config.max_scale = 1.0f;
                                if (!common_flydelta_prepare_activation(
                                        gate_config, request, 64U * 1024U * 1024U,
                                        activation, error)) {
                                    runner_error = error;
                                    return false;
                                }
                                activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                                    std::move(activation));
                            }
                            const bool executed = generate(
                                *inference, value, failed_instruction, result,
                                activation_ptr, capture_request);
                            trial = {};
                            trial.executed = executed;
                            trial.verifier_known = executed;
                            trial.passed = executed && contains_tool(result, "data.inspect");
                            trial.quality = trial.passed ? 1.0f : 0.0f;
                            trial.overlay_applied = apply_overlay;
                            trial.intervention_count = apply_overlay
                                ? coefficient_basis.vectors.size() : 0;
                            trial.evidence_ref = apply_overlay
                                ? "evidence:model-repair-tfo-lite-overlay"
                                : "evidence:model-repair-tfo-lite-baseline";
                            margin = {};
                            std::cout << "tfo_lite_model_output coefficients=";
                            for (size_t index = 0; index < tfo_coefficients.size(); ++index) {
                                if (index != 0) std::cout << ',';
                                std::cout << tfo_coefficients[index];
                            }
                            std::cout << " overlay=" << (apply_overlay ? "yes" : "no")
                                      << " output=" << output_preview(result) << '\n';
                            if (!executed && !result.error_message.empty()) {
                                runner_error = result.error_message;
                            }
                            if (!apply_overlay && result.flydelta_capture) {
                                coefficient_baseline_capture = result.flydelta_capture;
                            }
                            if (apply_overlay && result.flydelta_capture && coefficient_baseline_capture &&
                                    layer2_effect != deltas.end()) {
                                if (!common_flydelta_representation_diagnostics_from_captures(
                                        *coefficient_baseline_capture, *result.flydelta_capture,
                                        *layer2_effect, 64U * 1024U * 1024U, geometry, runner_error)) {
                                    return false;
                                }
                                geometry_available = true;
                            }
                            return executed;
                        }, tfo_trials, tfo_selection, error)) {
                    std::cerr << "FlyDelta TFO-lite model search failed: " << error << '\n';
                    return 1;
                }
                std::cout << "tfo_lite_search_strategy=tfo_lite"
                          << " tfo_lite_basis_rank=" << coefficient_basis.vectors.size()
                          << " tfo_lite_trials=" << tfo_trials.size()
                          << " tfo_lite_model_calls=" << (tfo_trials.size() + 1)
                          << " tfo_lite_elapsed_ms="
                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - tfo_started).count()
                          << " tfo_lite_selected=" << (tfo_selection.selected ? "yes" : "no")
                          << '\n';
                for (const auto & trial : tfo_trials) {
                    std::cout << "tfo_lite_trial iteration=" << trial.iteration
                              << " mutation=" << trial.mutation_kind
                              << " search_fitness=" << trial.search_fitness
                              << " outcome=" << common_flydelta_counterfactual_outcome_name(
                                  trial.outcome)
                              << " host_verified=" << (trial.verifier_known ? "yes" : "no");
                    if (trial.geometry_available) {
                        std::cout << " cosine=" << trial.geometry.cosine
                                  << " progress=" << trial.geometry.progress
                                  << " leakage=" << trial.geometry.leakage
                                  << " shift_norm=" << trial.geometry.shift_norm;
                    }
                    std::cout << '\n';
                }
            }
        }
    }
    std::filesystem::remove_all(artifact_root, artifact_cleanup_error);
    return 0;
}
