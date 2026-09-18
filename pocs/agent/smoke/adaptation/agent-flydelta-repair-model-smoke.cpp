#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-artifact-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"
#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"
#include "agent/adaptation/flydelta/flydelta-layer-discovery.h"
#include "agent/adaptation/flydelta/flydelta-layer-search.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"
#include "agent/adaptation/flydelta/flydelta-scale-search.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "agent/adaptation/flydelta/flydelta-training.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"
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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct options {
    std::string model;
    int n_predict = 96;
    int n_threads = 3;
    int n_gpu_layers = 0;
    // The model-backed smoke exercises the current Whirlpool/worker path by
    // default; the explicit flag remains accepted for compatibility.
    bool region_scan = true;
    // Test-only escape hatch: exercise the plateau -> orthogonal -> rank-two
    // surface path with the existing model-backed runner. It never changes
    // the production worker policy or evidence rank.
    bool force_plateau_escape = false;
    // Test-only model-facing bridge for the representation-augmentation seam.
    bool force_representation_augmentation = false;
    std::vector<uint32_t> region_layers;
};

bool parse_layer_list(const std::string & text, std::vector<uint32_t> & layers) {
    layers.clear();
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) return false;
        size_t consumed = 0;
        unsigned long value = 0;
        try {
            value = std::stoul(item, &consumed);
        } catch (...) {
            return false;
        }
        if (consumed != item.size() || value == 0 || value > 64) return false;
        layers.push_back(static_cast<uint32_t>(value));
    }
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    return !layers.empty();
}

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    if (const char * layers = std::getenv("LLAMA_AGENT_REGION_LAYERS")) {
        if (!parse_layer_list(layers, value.region_layers)) {
            std::cerr << "invalid LLAMA_AGENT_REGION_LAYERS\n";
            return false;
        }
    }
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
        } else if (arg == "--force-plateau-escape") {
            value.force_plateau_escape = true;
        } else if (arg == "--force-representation-augmentation") {
            value.force_representation_augmentation = true;
        } else if (arg == "--region-layers") {
            const char * layers = next("--region-layers");
            if (!layers || !parse_layer_list(layers, value.region_layers)) {
                std::cerr << "invalid --region-layers list\n";
                return false;
            }
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

void print_margin(const char * prefix,
        const common_flydelta_decision_margin & margin,
        const common_flydelta_decision_margin * baseline = nullptr) {
    std::cout << ' ' << prefix << "_available=" << (margin.available ? "yes" : "no");
    if (!margin.available) return;
    std::cout << ' ' << prefix << "_total_delta=" << margin.total_delta()
              << ' ' << prefix << "_normalized_delta=" << margin.normalized_delta()
              << ' ' << prefix << "_positive_logprob=" << margin.positive_total_logprob
              << ' ' << prefix << "_negative_logprob=" << margin.negative_total_logprob
              << ' ' << prefix << "_positive_tokens=" << margin.positive_token_count
              << ' ' << prefix << "_negative_tokens=" << margin.negative_token_count;
    if (baseline && baseline->available) {
        std::cout << ' ' << prefix << "_delta_from_baseline="
                  << margin.normalized_delta() - baseline->normalized_delta();
    }
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
                  << " [--region-scan] [--region-layers L1,L2,...]"
                  << " [--force-plateau-escape]"
                  << " [--force-representation-augmentation]\n";
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
    // bounded dense profile so host-side discovery can find the useful region
    // from activation separation instead of assuming that early layers win.
    for (uint32_t layer = 1; layer < model_n_layers && layer <= 64; ++layer) {
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

    common_flydelta_layer_discovery_config discovery_config;
    discovery_config.max_layers = 64;
    discovery_config.max_anchors = 6;
    discovery_config.min_anchor_separation = 2;
    discovery_config.max_capture_bytes = 64U * 1024U * 1024U;
    common_flydelta_layer_discovery_result discovery;
    if (!common_flydelta_discover_layer_regions(
            {*repaired.flydelta_capture}, {*failed.flydelta_capture},
            discovery_config, discovery, error)) {
        std::cerr << "FlyDelta model layer discovery failed: " << error << '\n';
        return 1;
    }
    std::cout << "layer_discovery_layers=" << discovery.scores.size()
              << " layer_discovery_anchors=";
    for (size_t index = 0; index < discovery.anchor_layers.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << discovery.anchor_layers[index];
    }
    std::cout << '\n';

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
    basis_config.max_directions = 64;
    basis_config.cluster_similarity = 0.85f;
    basis_config.source = common_adaptation_evidence_source::tool_repair;
    basis_config.behavior_key = "structured_tool_selection";
    basis_config.model_profile_fingerprint = profile;
    basis_config.execution_context_fingerprint = experiment_fixture.execution_context_fingerprint;
    basis_config.capture_layout_revision = "layer-input:v1";
    common_flydelta_basis_builder basis(basis_config);
    for (const auto & delta : deltas) {
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
                if (!apply_overlay && result.flydelta_capture && result.flydelta_capture->captured) {
                    baseline_arm_capture = result.flydelta_capture;
                } else if (apply_overlay && !trial.passed && !baseline_arm_capture) {
                    runner_error = "FlyDelta unknown arm has no baseline capture";
                    return false;
                } else if (apply_overlay && !trial.passed && baseline_arm_capture &&
                        result.flydelta_capture && result.flydelta_capture->captured) {
                    arm_diagnostic diagnostic;
                    diagnostic.alpha = alpha;
                    diagnostic.available = true;
                    for (const auto & delta : deltas) {
                        // layer-input captures are sampled before the cvec is
                        // injected at the selected basis layer. The same
                        // layer is therefore a pre-injection control, not a
                        // propagated intervention measurement. Keep only the
                        // first downstream layers for arm geometry so the
                        // layer search cannot rank an artificial zero.
                        if (delta.layer_index <= basis.directions().front().layer_index) {
                            continue;
                        }
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
        pipeline_config.region_max_singleton_layers = 6;
        pipeline_config.region_max_neighborhoods = 4;
        pipeline_config.region_max_trials = 32;
        pipeline_config.region_max_stalled_scales = 2;

        common_flydelta_search_pipeline_direction pipeline_direction;
        pipeline_direction.direction = {
            1, common_flydelta_direction_kind::raw_repair, anchor->layer_index,
            anchor->values, 1, 1, 1.0f, false};
        for (const auto & delta : deltas) {
            if (delta.layer_index > 0) pipeline_direction.available_layers.push_back(
                static_cast<uint32_t>(delta.layer_index));
        }
        pipeline_direction.layer_anchors = discovery.anchor_layers;
        std::sort(pipeline_direction.available_layers.begin(),
            pipeline_direction.available_layers.end());
        pipeline_direction.available_layers.erase(std::unique(
            pipeline_direction.available_layers.begin(),
            pipeline_direction.available_layers.end()), pipeline_direction.available_layers.end());
        if (!value.region_layers.empty()) {
            for (const uint32_t layer : value.region_layers) {
                if (!std::binary_search(pipeline_direction.available_layers.begin(),
                            pipeline_direction.available_layers.end(), layer)) {
                    std::cerr << "FlyDelta region pipeline layer is unavailable: "
                              << layer << '\n';
                    return 1;
                }
            }
            pipeline_direction.available_layers = value.region_layers;
            pipeline_direction.layer_anchors.clear();
            for (const uint32_t layer : value.region_layers) {
                if (std::binary_search(discovery.anchor_layers.begin(),
                            discovery.anchor_layers.end(), layer)) {
                    pipeline_direction.layer_anchors.push_back(layer);
                }
            }
            if (pipeline_direction.layer_anchors.empty()) {
                pipeline_direction.layer_anchors = value.region_layers;
            }
            std::cout << "region_layer_override=";
            for (size_t index = 0; index < value.region_layers.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << value.region_layers[index];
            }
            std::cout << '\n';
        }

        std::shared_ptr<const common_flydelta_hidden_state_capture> region_baseline_capture;
        common_flydelta_decision_margin region_baseline_margin;
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
                    if (!apply_overlay && result.flydelta_capture &&
                            result.flydelta_capture->captured) {
                        region_baseline_capture = result.flydelta_capture;
                        region_baseline_margin = margin;
                    }
                    if (apply_overlay && candidate && result.flydelta_capture &&
                            result.flydelta_capture->captured &&
                            region_baseline_capture) {
                        const auto measurement = std::find_if(deltas.begin(), deltas.end(),
                            [&](const auto & delta) {
                                // layer-input captures are taken before the
                                // candidate layer's own injection. Measure
                                // the first captured downstream layer so a
                                // valid overlay is not reported as zero shift.
                                return delta.layer_index > static_cast<int>(
                                    candidate->layer_indices.back());
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
        const auto & whirlpool = pipeline_result.directions.front().whirlpool_trace;
        std::cout << "intervention_region_search=completed"
                  << " pipeline=default"
                  << " trials=" << region_trials.size()
                  << " model_calls=" << (region_trials.size() + 1)
                  << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - region_started).count()
                  << " selected=" << (region_selection.selected ? "yes" : "no") << '\n';
        std::cout << "whirlpool_search=completed"
                  << " model_evaluations=" << whirlpool.model_evaluations
                  << " best_trial_index=" << whirlpool.best_trial_index
                  << " best_search_score=" << whirlpool.best_search_score
                  << " final_centre=" << whirlpool.final_centre
                  << " final_radius=" << whirlpool.final_radius << '\n';
        for (const auto & round : whirlpool.rounds) {
            std::cout << "whirlpool_round round=" << round.round
                      << " centre_before=" << round.centre_before
                      << " radius_before=" << round.radius_before
                      << " probed_layers=";
            for (size_t index = 0; index < round.probed_layers.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << round.probed_layers[index];
            }
            std::cout << " best_probe_layer=" << round.best_probe_layer
                      << " best_probe_score=" << round.best_probe_score
                      << " centre_after=" << round.centre_after
                      << " radius_after=" << round.radius_after << '\n';
        }
        for (const auto & trial : region_trials) {
            std::cout << "region_trial layers=";
            for (const uint32_t layer : trial.candidate.layer_indices) std::cout << layer << ',';
            std::cout << " scale=" << trial.candidate.total_scale
                      << " outcome=" << common_flydelta_counterfactual_outcome_name(trial.outcome)
                      << " promising=" << (trial.promising ? "yes" : "no")
                      << " safe_to_continue=" << (trial.safe_to_continue ? "yes" : "no");
            print_margin("margin", trial.margin);
            if (trial.geometry_available) {
                std::cout << " cosine=" << trial.geometry.cosine
                          << " progress=" << trial.geometry.progress
                          << " leakage=" << trial.geometry.leakage
                          << " shift_norm=" << trial.geometry.shift_norm;
            }
            std::cout << '\n';
        }

        // Adapt the legacy model smoke to the current host orchestration. The
        // model smoke has one natural repair pair, so this deliberately ends
        // at Bootstrap; it must not manufacture a rank-two basis. The
        // dataset-question smoke supplies the real two-sample continuation.
        common_flydelta_search_continuation continuation;
        common_flydelta_evidence_depth_result evidence_depth;
        common_flydelta_experiment_plan continuation_plan;
        if (!common_flydelta_select_search_continuation(
                pipeline_result, continuation, error)) {
            std::cerr << "FlyDelta continuation selection failed: " << error << '\n';
            return 1;
        }
        const auto continuation_delta = std::find_if(deltas.begin(), deltas.end(),
            [&](const auto & delta) {
                return delta.layer_index == static_cast<int32_t>(
                    continuation.region.anchor_layer_index);
            });
        if (continuation_delta == deltas.end()) {
            std::cerr << "FlyDelta continuation has no compatible anchor delta\n";
            return 1;
        }
        common_flydelta_direction_search_config evidence_identity;
        evidence_identity.dimension = model_n_embd;
        evidence_identity.layer_index = continuation_delta->layer_index;
        evidence_identity.min_samples = 2;
        evidence_identity.max_samples = 32;
        evidence_identity.min_median_alignment = 0.25f;
        evidence_identity.source = common_adaptation_evidence_source::tool_repair;
        evidence_identity.behavior_key = "structured_tool_selection";
        evidence_identity.model_profile_fingerprint = profile;
        evidence_identity.execution_context_fingerprint =
            experiment_fixture.execution_context_fingerprint;
        evidence_identity.capture_layout_revision = "layer-input:v1";
        common_flydelta_evidence_depth_config evidence_config;
        if (!common_flydelta_assess_evidence_depth(
                evidence_identity, evidence_config,
                {{*continuation_delta, repair_credit}}, evidence_depth, error) ||
                !common_flydelta_plan_search_continuation(
                    continuation, evidence_depth, continuation_plan, error)) {
            std::cerr << "FlyDelta continuation planning failed: " << error << '\n';
            return 1;
        }
        std::cout << "flydelta_orchestration continuation=yes"
                  << " layer=" << continuation.region.anchor_layer_index
                  << " search_score=" << continuation.search_score
                  << " host_helped=" << (continuation.host_helped ? "yes" : "no")
                  << " evidence_depth=" << common_flydelta_search_depth_name(
                      evidence_depth.depth)
                  << " compatible_samples=" << evidence_depth.compatible_samples
                  << " effective_rank=" << evidence_depth.effective_rank
                  << " phase=" << common_flydelta_experiment_phase_name(
                      continuation_plan.phase)
                  << " shallow_allowed=" << (evidence_depth.shallow_ready ? "yes" : "no")
                  << " deep_allowed=" << (evidence_depth.deep_ready ? "yes" : "no")
                  << " tfo_allowed_by_evidence="
                  << (continuation_plan.tfo_lite_permitted_by_evidence ? "yes" : "no")
                  << '\n';

        const auto selected_region_trial = continuation.region_trial_index < region_trials.size()
            ? &region_trials[continuation.region_trial_index] : nullptr;
        common_flydelta_subspace_utility_observation utility_observation;
        if (selected_region_trial != nullptr) {
            utility_observation.safe_to_continue = selected_region_trial->safe_to_continue;
            utility_observation.decision_margin_available =
                selected_region_trial->margin.available && region_baseline_margin.available;
            utility_observation.decision_margin_delta =
                utility_observation.decision_margin_available
                ? selected_region_trial->margin.normalized_delta() -
                    region_baseline_margin.normalized_delta() : 0.0f;
            utility_observation.geometry_available = selected_region_trial->geometry_available;
            if (utility_observation.geometry_available) {
                utility_observation.geometry.cosine = selected_region_trial->geometry.cosine;
                utility_observation.geometry.progress = selected_region_trial->geometry.progress;
                utility_observation.geometry.leakage = selected_region_trial->geometry.leakage;
                utility_observation.geometry.shift_norm = selected_region_trial->geometry.shift_norm;
            }
        }
        common_flydelta_utility_gate_config utility_config;
        common_flydelta_utility_gate_decision utility_decision;
        if (!common_flydelta_decide_subspace_utility(
                utility_config, continuation_plan.depth, continuation_plan.phase,
                {utility_observation}, {}, utility_decision, error)) {
            std::cerr << "FlyDelta continuation utility decision failed: " << error << '\n';
            return 1;
        }
        std::cout << "flydelta_utility_gate phase="
                  << common_flydelta_experiment_phase_name(continuation_plan.phase)
                  << " qualified=" << (utility_decision.utility_qualified ? "yes" : "no")
                  << " action=" << common_flydelta_utility_gate_action_name(
                      utility_decision.action)
                  << " margin_delta=" << utility_observation.decision_margin_delta
                  << " qualifying_streak=" << utility_decision.history.qualifying_streak
                  << " nonqualifying_streak=" << utility_decision.history.nonqualifying_streak
                  << "\n";

        std::vector<common_flydelta_bootstrap_zoom_trial> zoom_trials;
        common_flydelta_bootstrap_zoom_selection zoom_selection;

        // A Bootstrap-only evidence set may still have enough decision utility
        // to spend a small, strictly rank-one local budget. This uses each
        // layer's own repair direction; a profile never transports L24's
        // representation to L25 and therefore does not accidentally become a
        // different WHAT candidate.
        if (utility_decision.action ==
                common_flydelta_utility_gate_action::refine_bootstrap ||
                value.force_plateau_escape) {
            common_flydelta_bootstrap_zoom_config zoom_config;
            std::vector<common_flydelta_bootstrap_zoom_candidate> alpha_candidates;
            if (!common_flydelta_propose_bootstrap_alpha_zoom(
                    continuation.region.anchor_layer_index,
                    continuation.region.total_scale, zoom_config,
                    alpha_candidates, error)) {
                std::cerr << "FlyDelta BootstrapZoom alpha planning failed: " << error << '\n';
                return 1;
            }

            const auto run_zoom = [&](const common_flydelta_bootstrap_zoom_candidate & candidate,
                    common_flydelta_counterfactual_outcome & outcome,
                    common_flydelta_decision_margin & margin,
                    common_flydelta_representation_diagnostics & diagnostics,
                    bool & diagnostics_available) {
                common_flydelta_gate_request gate_request;
                if (!common_flydelta_gate_request_from_context(
                        recognition, code, true,
                        common_flydelta_candidate_status::approved, true,
                        candidate.total_scale, gate_request, error)) return false;
                common_flydelta_activation_request request;
                request.candidate_id = "flydelta://candidate/model-repair-bootstrap-zoom";
                request.artifact_id = "flydelta://artifact/model-repair-e2e";
                request.model_profile_fingerprint = profile;
                request.capture_layout_revision = "layer-input:v1";
                request.model_n_embd = model_n_embd;
                request.model_n_layers = model_n_layers;
                request.il_end = static_cast<int32_t>(model_n_layers - 1);
                for (size_t index = 0; index < candidate.layer_indices.size(); ++index) {
                    const auto direction = std::find_if(basis.directions().begin(),
                        basis.directions().end(), [&](const auto & value) {
                            return value.layer_index == static_cast<int32_t>(
                                candidate.layer_indices[index]);
                        });
                    if (direction == basis.directions().end()) {
                        error = "FlyDelta BootstrapZoom profile has no layer-compatible direction";
                        return false;
                    }
                    request.directions.push_back(*direction);
                    request.coefficients.push_back(candidate.layer_weights[index]);
                }
                request.gate_request = gate_request;
                common_flydelta_gate_config gate_config;
                gate_config.enabled = true;
                gate_config.max_scale = 1.0f;
                common_flydelta_activation_result activation;
                if (!common_flydelta_prepare_activation(
                        gate_config, request, 64U * 1024U * 1024U, activation, error)) return false;
                const auto activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                    std::move(activation));
                common_agent_generation_result result;
                const bool executed = generate(*inference, value, failed_instruction, result,
                    activation_ptr, capture_request);
                outcome = executed && contains_tool(result, "data.inspect")
                    ? common_flydelta_counterfactual_outcome::helped
                    : common_flydelta_counterfactual_outcome::unknown;
                const auto scoring_request = make_request(value, failed_instruction);
                if (!score_chat_choice_margin(
                        loaded->model, loaded->chat_templates.get(), scoring_request.messages,
                        scoring_request.tools, scoring_request.tool_choice, scoring_request.options,
                        "{\"name\":\"", "data.inspect", "data.describe", margin,
                        nullptr, scoring_request.json_schema, {}, {}, activation_ptr->overlay, &error)) {
                    return false;
                }
                diagnostics = {};
                diagnostics_available = false;
                if (result.flydelta_capture && result.flydelta_capture->captured &&
                        region_baseline_capture) {
                    const uint32_t measured_after = *std::max_element(
                        candidate.layer_indices.begin(), candidate.layer_indices.end());
                    const auto measurement = std::find_if(deltas.begin(), deltas.end(),
                        [&](const auto & delta) {
                            return delta.layer_index > static_cast<int>(measured_after);
                        });
                    if (measurement != deltas.end()) {
                        if (!common_flydelta_representation_diagnostics_from_captures(
                                *region_baseline_capture, *result.flydelta_capture, *measurement,
                                64U * 1024U * 1024U, diagnostics, error)) return false;
                        diagnostics_available = true;
                    }
                }
                std::cout << "bootstrap_zoom_model_output phase="
                          << common_flydelta_bootstrap_zoom_phase_name(candidate.phase)
                          << " layers=";
                for (size_t index = 0; index < candidate.layer_indices.size(); ++index) {
                    if (index != 0) std::cout << ',';
                    std::cout << candidate.layer_indices[index] << ':' << candidate.layer_weights[index];
                }
                std::cout << " scale=" << candidate.total_scale
                          << " output=" << output_preview(result) << '\n';
                if (!executed && !result.error_message.empty()) error = result.error_message;
                return executed;
            };

            const auto evaluate_zoom = [&](const common_flydelta_bootstrap_zoom_candidate & candidate,
                    common_flydelta_counterfactual_outcome & outcome,
                    common_flydelta_decision_margin & margin,
                    common_flydelta_representation_diagnostics & diagnostics,
                    bool & diagnostics_available) {
                if (!run_zoom(candidate, outcome, margin, diagnostics, diagnostics_available)) return false;
                std::cout << "bootstrap_zoom_trial phase="
                          << common_flydelta_bootstrap_zoom_phase_name(candidate.phase)
                          << " scale=" << candidate.total_scale
                          << " opposite_sign_control="
                          << (candidate.opposite_sign_control ? "yes" : "no")
                          << " outcome=" << common_flydelta_counterfactual_outcome_name(outcome);
                print_margin("margin", margin, &region_baseline_margin);
                if (diagnostics_available) {
                    std::cout << " cosine=" << diagnostics.cosine
                              << " progress=" << diagnostics.progress
                              << " leakage=" << diagnostics.leakage
                              << " shift_norm=" << diagnostics.shift_norm;
                }
                std::cout << '\n';
                return true;
            };

            const auto retain_zoom_trial = [&](const common_flydelta_bootstrap_zoom_candidate & candidate,
                    common_flydelta_counterfactual_outcome outcome,
                    const common_flydelta_decision_margin & margin,
                    const common_flydelta_representation_diagnostics & diagnostics,
                    bool diagnostics_available) {
                common_flydelta_bootstrap_zoom_trial trial;
                trial.candidate = candidate;
                // The smoke host executed and classified every arm; UNKNOWN
                // remains experimental search evidence rather than learning
                // credit.
                trial.outcome = outcome;
                trial.host_evaluated = true;
                trial.verifier_known = true;
                trial.margin_available = margin.available && region_baseline_margin.available;
                trial.margin_delta = trial.margin_available
                    ? margin.normalized_delta() - region_baseline_margin.normalized_delta()
                    : 0.0f;
                trial.diagnostics_available = diagnostics_available;
                if (diagnostics_available) trial.diagnostics = diagnostics;
                zoom_trials.push_back(std::move(trial));
            };

            std::cout << "bootstrap_zoom=started max_extra_model_trials="
                      << zoom_config.max_extra_model_trials << " stage=alpha\n";
            float selected_zoom_scale = continuation.region.total_scale;
            float best_zoom_margin = utility_observation.decision_margin_delta;
            for (const auto & candidate : alpha_candidates) {
                common_flydelta_counterfactual_outcome outcome;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics diagnostics;
                bool diagnostics_available = false;
                if (!evaluate_zoom(candidate, outcome, margin, diagnostics, diagnostics_available)) {
                    std::cerr << "FlyDelta BootstrapZoom alpha execution failed: " << error << '\n';
                    return 1;
                }
                retain_zoom_trial(candidate, outcome, margin, diagnostics, diagnostics_available);
                const float margin_delta = margin.available && region_baseline_margin.available
                    ? margin.normalized_delta() - region_baseline_margin.normalized_delta() : 0.0f;
                const bool geometry_safe = !diagnostics_available ||
                    (diagnostics.cosine >= utility_config.min_cosine && diagnostics.progress > 0.0f &&
                     diagnostics.leakage <= utility_config.max_leakage &&
                     diagnostics.shift_norm <= utility_config.max_shift_norm);
                if (geometry_safe && margin_delta > best_zoom_margin +
                        zoom_config.min_margin_improvement) {
                    selected_zoom_scale = candidate.total_scale;
                    best_zoom_margin = margin_delta;
                }
            }
            std::vector<uint32_t> local_layers;
            for (const int offset : {-1, 0, 1}) {
                const int layer = static_cast<int>(continuation.region.anchor_layer_index) + offset;
                if (layer > 0 && std::binary_search(pipeline_direction.available_layers.begin(),
                        pipeline_direction.available_layers.end(), static_cast<uint32_t>(layer))) {
                    local_layers.push_back(static_cast<uint32_t>(layer));
                }
            }
            if (local_layers.empty()) local_layers.push_back(continuation.region.anchor_layer_index);
            std::vector<common_flydelta_bootstrap_zoom_candidate> profile_candidates;
            if (!common_flydelta_propose_bootstrap_profile_zoom(
                    local_layers, continuation.region.anchor_layer_index,
                    selected_zoom_scale, zoom_config, profile_candidates, error)) {
                std::cerr << "FlyDelta BootstrapZoom profile planning failed: " << error << '\n';
                return 1;
            }
            std::cout << "bootstrap_zoom=started max_extra_model_trials="
                      << zoom_config.max_extra_model_trials << " stage=profile\n";
            std::cout << "bootstrap_zoom_alpha_selection scale=" << selected_zoom_scale
                      << " margin_delta=" << best_zoom_margin << '\n';
            for (const auto & candidate : profile_candidates) {
                common_flydelta_counterfactual_outcome outcome;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics diagnostics;
                bool diagnostics_available = false;
                if (!evaluate_zoom(candidate, outcome, margin, diagnostics, diagnostics_available)) {
                    std::cerr << "FlyDelta BootstrapZoom profile execution failed: " << error << '\n';
                    return 1;
                }
                retain_zoom_trial(candidate, outcome, margin, diagnostics, diagnostics_available);
            }
            if (!common_flydelta_select_bootstrap_zoom_trial(
                    zoom_trials, zoom_selection, error)) {
                std::cerr << "FlyDelta BootstrapZoom candidate selection failed: " << error << '\n';
                return 1;
            }
            const auto & retained = zoom_trials[zoom_selection.trial_index];
            std::cout << "bootstrap_zoom_selected trial=" << zoom_selection.trial_index
                      << " phase=" << common_flydelta_bootstrap_zoom_phase_name(
                          retained.candidate.phase)
                      << " scale=" << retained.candidate.total_scale
                      << " layers=";
            for (size_t index = 0; index < retained.candidate.layer_indices.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << retained.candidate.layer_indices[index] << ':'
                          << retained.candidate.layer_weights[index];
            }
            std::cout << " outcome=" << common_flydelta_counterfactual_outcome_name(
                              retained.outcome)
                      << " margin_delta=" << retained.margin_delta
                      << " search_score=" << zoom_selection.search_score << '\n';
        }

        // BootstrapZoom is followed by the rank-one plateau gate. This is a
        // search transition only: it may request the bounded orthogonal
        // escape, but it must not manufacture Shallow/Deep evidence. Build
        // the rounds from the actual region/zoom observations so the model
        // smoke exercises the same gate as the worker-facing pipeline.
        common_flydelta_rank1_plateau_config plateau_config;
        std::vector<common_flydelta_rank1_plateau_round> plateau_rounds;
        const auto make_region_round = [&]() {
            common_flydelta_rank1_plateau_round round;
            round.anchor_layer = continuation.region.anchor_layer_index;
            round.safe_to_continue = !region_trials.empty();
            for (const auto & trial : region_trials) {
                if (!trial.safe_to_continue) round.safe_to_continue = false;
                if (trial.safe_to_continue && trial.margin.available) {
                    round.best_margin_delta = std::max(
                        round.best_margin_delta, trial.margin.normalized_delta() -
                            region_baseline_margin.normalized_delta());
                    ++round.evaluated_arms;
                }
            }
            return round;
        };
        const auto make_zoom_round = [&](common_flydelta_bootstrap_zoom_phase phase) {
            common_flydelta_rank1_plateau_round round;
            round.anchor_layer = continuation.region.anchor_layer_index;
            for (const auto & trial : zoom_trials) {
                if (trial.candidate.phase != phase) continue;
                if (!trial.verifier_known ||
                        trial.outcome == common_flydelta_counterfactual_outcome::harmed) {
                    round.safe_to_continue = false;
                    continue;
                }
                round.safe_to_continue = true;
                ++round.evaluated_arms;
                if (trial.margin_available) {
                    round.best_margin_delta = std::max(round.best_margin_delta,
                        trial.margin_delta);
                }
            }
            return round;
        };
        plateau_rounds.push_back(make_region_round());
        const auto alpha_round = make_zoom_round(common_flydelta_bootstrap_zoom_phase::alpha_zoom);
        if (alpha_round.evaluated_arms != 0) plateau_rounds.push_back(alpha_round);
        const auto profile_round = make_zoom_round(common_flydelta_bootstrap_zoom_phase::profile_zoom);
        if (profile_round.evaluated_arms != 0) plateau_rounds.push_back(profile_round);
        common_flydelta_rank1_plateau_result plateau_result;
        common_flydelta_utility_gate_decision plateau_decision;
        common_flydelta_experiment_plan plateau_plan;
        bool plateau_advanced = false;
        if (!common_flydelta_evaluate_rank1_plateau(
                plateau_config, evidence_depth.effective_rank, plateau_rounds,
                plateau_result, error) ||
                !common_flydelta_decide_rank1_plateau_utility(
                    plateau_result, plateau_decision, error) ||
                !common_flydelta_advance_experiment_plan(
                    continuation_plan, plateau_decision, plateau_plan,
                    plateau_advanced, error)) {
            std::cerr << "FlyDelta plateau transition failed: " << error << '\n';
            return 1;
        }
        std::cout << "flydelta_plateau_gate eligible="
                  << (plateau_result.eligible ? "yes" : "no")
                  << " plateau=" << (plateau_result.plateau ? "yes" : "no")
                  << " action=" << common_flydelta_utility_gate_action_name(
                      plateau_decision.action)
                  << " safe_arms=" << plateau_result.safe_arm_count
                  << " plateau_streak=" << plateau_result.plateau_streak
                  << " best_margin_delta=" << plateau_result.best_margin_delta
                  << " recent_gain_ratio=" << plateau_result.recent_gain_ratio
                  << " advanced=" << (plateau_advanced ? "yes" : "no")
                  << " run_orthogonal_search="
                  << (plateau_plan.run_orthogonal_search ? "yes" : "no")
                  << " evidence_rank=" << evidence_depth.effective_rank << '\n';

        // An orthogonal escape is a new experimental search surface, not an
        // evidence-depth transition. The explicit smoke flag exists only to
        // exercise this path with the single natural repair sample; normal
        // orchestration enters it only when the plateau gate says so.
        const bool run_orthogonal_surface = plateau_plan.run_orthogonal_search ||
            value.force_plateau_escape;
        std::vector<common_flydelta_bootstrap_zoom_trial> surface_trials;
        common_flydelta_orthogonal_search_result orthogonal_surface;
        common_flydelta_bootstrap_zoom_state surface_state;
        if (run_orthogonal_surface && !zoom_trials.empty()) {
            std::vector<uint32_t> surface_layers;
            for (const auto & trial : zoom_trials) {
                for (const uint32_t layer : trial.candidate.layer_indices) {
                    if (std::find(surface_layers.begin(), surface_layers.end(), layer) ==
                            surface_layers.end()) surface_layers.push_back(layer);
                }
            }
            std::sort(surface_layers.begin(), surface_layers.end());
            if (surface_layers.size() > 3) surface_layers.resize(3);
            if (surface_layers.empty()) {
                surface_layers = continuation.region.layer_indices;
                if (surface_layers.size() > 3) surface_layers.resize(3);
            }
            common_flydelta_orthogonal_search_config orthogonal_config;
            // The common helper consumes only persisted search state and
            // diagnostics. The model host below owns the fresh validation
            // generation; this keeps the same preparation path usable by a
            // production post-Bootstrap callback.
            surface_state = {};
            surface_state.behavior_key = evidence.behavior_key;
            surface_state.model_profile_fingerprint = profile;
            surface_state.capture_layout_revision = "layer-input:v1";
            surface_state.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
            surface_state.anchor_layer = continuation.region.anchor_layer_index;
            surface_state.selected_scale = std::max(0.0001f,
                zoom_trials[zoom_selection.selected ? zoom_selection.trial_index : 0].candidate.total_scale);
            surface_state.best_margin_delta = plateau_result.best_margin_delta;
            surface_state.best_search_score = continuation.search_score;
            surface_state.extra_model_trials = zoom_trials.size();
            surface_state.next_candidate_index = zoom_trials.size();
            surface_state.evidence_rank = evidence_depth.effective_rank;
            surface_state.local_layers = surface_layers;
            surface_state.completed_trials = zoom_trials;
            surface_state.selection = zoom_selection;
            common_flydelta_orthogonal_search_input orthogonal_input;
            if (!common_flydelta_prepare_orthogonal_search_input(
                    orthogonal_config, surface_state, orthogonal_input, error)) {
                std::cerr << "FlyDelta orthogonal input preparation failed: " << error << '\n';
                return 1;
            }
            surface_layers = orthogonal_input.local_layers;
            const auto & rank1_trial = zoom_trials[zoom_selection.selected
                ? zoom_selection.trial_index : 0];
            const std::vector<float> rank1_profile = orthogonal_input.rank1_intervention;
            if (!common_flydelta_build_orthogonal_search_direction(
                    orthogonal_config, rank1_profile, orthogonal_input.arms,
                    orthogonal_surface, error)) {
                std::cerr << "FlyDelta orthogonal surface construction failed: " << error << '\n';
                return 1;
            }
            std::cout << "flydelta_orthogonal_surface available="
                      << (orthogonal_surface.available ? "yes" : "no")
                      << " experimental_only="
                      << (orthogonal_surface.experimental_only ? "yes" : "no")
                      << " source_arms=" << orthogonal_surface.source_arm_count
                      << " residual_norm=" << orthogonal_surface.residual_norm
                      << " fit_quality=" << orthogonal_surface.fit_quality
                      << " response_signal="
                      << common_flydelta_orthogonal_response_signal_name(
                             orthogonal_surface.response_signal)
                      << " evidence_rank=" << evidence_depth.effective_rank << '\n';

            if (orthogonal_surface.available) {
                const auto normalize_profile = [](std::vector<float> profile) {
                    float energy = 0.0f;
                    for (const float value : profile) energy += value * value;
                    const float scale = std::sqrt(energy);
                    if (scale > 0.000001f) for (float & value : profile) value /= scale;
                    return profile;
                };
                const auto make_surface_candidate = [&](const std::vector<float> & profile,
                        bool opposite_sign) {
                    common_flydelta_bootstrap_zoom_candidate candidate;
                    candidate.phase = opposite_sign
                        ? common_flydelta_bootstrap_zoom_phase::sign_control
                        : common_flydelta_bootstrap_zoom_phase::profile_zoom;
                    candidate.total_scale = std::max(0.0001f,
                        zoom_trials[zoom_selection.trial_index].candidate.total_scale);
                    candidate.opposite_sign_control = opposite_sign;
                    for (size_t index = 0; index < profile.size(); ++index) {
                        if (std::fabs(profile[index]) <= 0.000001f) continue;
                        candidate.layer_indices.push_back(surface_layers[index]);
                        candidate.layer_weights.push_back(profile[index]);
                    }
                    candidate.layer_weights = normalize_profile(candidate.layer_weights);
                    return candidate;
                };
                const auto run_surface_candidate =
                    [&](const common_flydelta_bootstrap_zoom_candidate & candidate,
                        common_flydelta_counterfactual_trial & counterfactual,
                        common_flydelta_decision_margin & margin,
                        common_flydelta_representation_diagnostics & diagnostics,
                        bool & diagnostics_available) {
                    common_flydelta_gate_request gate_request;
                    if (!common_flydelta_gate_request_from_context(
                            recognition, code, true,
                            common_flydelta_candidate_status::approved, true,
                            candidate.total_scale, gate_request, error)) return false;
                    common_flydelta_activation_request request;
                    request.candidate_id = "flydelta://candidate/model-repair-orthogonal-surface";
                    request.artifact_id = "flydelta://artifact/model-repair-e2e";
                    request.model_profile_fingerprint = profile;
                    request.capture_layout_revision = "layer-input:v1";
                    request.model_n_embd = model_n_embd;
                    request.model_n_layers = model_n_layers;
                    request.il_end = static_cast<int32_t>(model_n_layers - 1);
                    for (size_t index = 0; index < candidate.layer_indices.size(); ++index) {
                        const auto direction = std::find_if(basis.directions().begin(),
                            basis.directions().end(), [&](const auto & value) {
                                return value.layer_index == static_cast<int32_t>(
                                    candidate.layer_indices[index]);
                            });
                        if (direction == basis.directions().end()) {
                            error = "FlyDelta orthogonal surface has no layer-compatible direction";
                            return false;
                        }
                        request.directions.push_back(*direction);
                        request.coefficients.push_back(candidate.layer_weights[index]);
                    }
                    request.gate_request = gate_request;
                    common_flydelta_gate_config gate_config;
                    gate_config.enabled = true;
                    gate_config.max_scale = 1.0f;
                    common_flydelta_activation_result activation;
                    if (!common_flydelta_prepare_activation(
                            gate_config, request, 64U * 1024U * 1024U, activation, error)) return false;
                    const auto activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                        std::move(activation));
                    common_agent_generation_result generated;
                    const bool executed = generate(*inference, value, failed_instruction,
                        generated, activation_ptr, capture_request);
                    counterfactual = {};
                    counterfactual.executed = executed;
                    counterfactual.verifier_known = executed;
                    counterfactual.passed = executed && contains_tool(generated, "data.inspect");
                    counterfactual.quality = counterfactual.passed ? 1.0f : 0.0f;
                    counterfactual.overlay_applied = true;
                    counterfactual.intervention_count = candidate.layer_indices.size();
                    counterfactual.evidence_ref = "evidence:model-repair-orthogonal-surface";
                    const auto scoring_request = make_request(value, failed_instruction);
                    if (!score_chat_choice_margin(
                            loaded->model, loaded->chat_templates.get(), scoring_request.messages,
                            scoring_request.tools, scoring_request.tool_choice, scoring_request.options,
                            "{\"name\":\"", "data.inspect", "data.describe", margin,
                            nullptr, scoring_request.json_schema, {}, {}, activation_ptr->overlay, &error)) {
                        return false;
                    }
                    diagnostics = {};
                    diagnostics_available = false;
                    if (generated.flydelta_capture && generated.flydelta_capture->captured &&
                            region_baseline_capture) {
                        const uint32_t measured_after = *std::max_element(
                            candidate.layer_indices.begin(), candidate.layer_indices.end());
                        const auto measurement = std::find_if(deltas.begin(), deltas.end(),
                            [&](const auto & delta) { return delta.layer_index >
                                static_cast<int>(measured_after); });
                        if (measurement != deltas.end()) {
                            if (!common_flydelta_representation_diagnostics_from_captures(
                                    *region_baseline_capture, *generated.flydelta_capture,
                                    *measurement, 64U * 1024U * 1024U, diagnostics, error)) return false;
                            diagnostics_available = true;
                        }
                    }
                    std::cout << "flydelta_rank2_control_model_output layers=";
                    for (size_t index = 0; index < candidate.layer_indices.size(); ++index) {
                        if (index != 0) std::cout << ',';
                        std::cout << candidate.layer_indices[index] << ':' <<
                            candidate.layer_weights[index];
                    }
                    std::cout << " output=" << output_preview(generated) << '\n';
                    if (!executed && !generated.error_message.empty()) error = generated.error_message;
                    return executed;
                };
                const std::vector<std::pair<const char *, std::vector<float>>> controls = {
                    {"rank1", {1.0f, 0.0f}}, {"orthogonal", {0.0f, 1.0f}},
                    {"sum", {0.70710678f, 0.70710678f}},
                    {"difference", {0.70710678f, -0.70710678f}},
                };
                for (const auto & control : controls) {
                    std::vector<float> profile_mix(rank1_profile.size(), 0.0f);
                    for (size_t index = 0; index < profile_mix.size(); ++index) {
                        profile_mix[index] = control.second[0] * rank1_profile[index] +
                            control.second[1] * orthogonal_surface.direction[index];
                    }
                    const bool opposite = std::string(control.first) == "difference";
                    auto candidate = make_surface_candidate(profile_mix, opposite);
                    std::string candidate_error;
                    if (!common_flydelta_bootstrap_zoom_candidate_validate(candidate, candidate_error)) {
                        std::cerr << "FlyDelta rank2 control skipped label=" << control.first
                                  << " reason=" << candidate_error << '\n';
                        continue;
                    }
                    common_flydelta_counterfactual_trial counterfactual;
                    common_flydelta_decision_margin margin;
                    common_flydelta_representation_diagnostics diagnostics;
                    bool diagnostics_available = false;
                    if (!run_surface_candidate(candidate, counterfactual, margin,
                            diagnostics, diagnostics_available)) {
                        std::cerr << "FlyDelta rank2 control failed label=" << control.first
                                  << ": " << error << '\n';
                        return 1;
                    }
                    common_flydelta_bootstrap_zoom_trial trial;
                    trial.candidate = std::move(candidate);
                    trial.outcome = counterfactual.passed
                        ? common_flydelta_counterfactual_outcome::helped
                        : common_flydelta_counterfactual_outcome::unknown;
                    trial.host_evaluated = true;
                    trial.verifier_known = true;
                    trial.margin_available = margin.available && region_baseline_margin.available;
                    trial.margin_delta = trial.margin_available
                        ? margin.normalized_delta() - region_baseline_margin.normalized_delta() : 0.0f;
                    trial.diagnostics_available = diagnostics_available;
                    if (diagnostics_available) trial.diagnostics = diagnostics;
                    surface_trials.push_back(std::move(trial));
                    std::cout << "flydelta_rank2_control label=" << control.first
                              << " outcome=" << common_flydelta_counterfactual_outcome_name(
                                  surface_trials.back().outcome);
                    print_margin("margin", margin, &region_baseline_margin);
                    std::cout << " margin_delta=" << surface_trials.back().margin_delta << '\n';
                }
                common_flydelta_subspace_utility_observation best_surface_utility;
                for (const auto & trial : surface_trials) {
                    if (!trial.margin_available) continue;
                    if (!best_surface_utility.decision_margin_available ||
                            trial.margin_delta > best_surface_utility.decision_margin_delta) {
                        best_surface_utility.safe_to_continue = trial.verifier_known &&
                            trial.outcome != common_flydelta_counterfactual_outcome::harmed;
                        best_surface_utility.decision_margin_available = true;
                        best_surface_utility.decision_margin_delta = trial.margin_delta;
                        best_surface_utility.geometry_available = trial.diagnostics_available;
                        if (trial.diagnostics_available) best_surface_utility.geometry = trial.diagnostics;
                    }
                }
                common_flydelta_utility_gate_decision surface_utility_decision;
                if (!surface_trials.empty() &&
                        !common_flydelta_decide_subspace_utility(
                            utility_config, evidence_depth.depth,
                            common_flydelta_experiment_phase::bootstrap,
                            {best_surface_utility}, {}, surface_utility_decision, error)) {
                    std::cerr << "FlyDelta rank2 surface utility decision failed: " << error << '\n';
                    return 1;
                }
                std::cout << "flydelta_rank2_surface_gate controls=" << surface_trials.size()
                          << " action=" << common_flydelta_utility_gate_action_name(
                              surface_utility_decision.action)
                          << " utility_qualified=" <<
                              (surface_utility_decision.utility_qualified ? "yes" : "no")
                          << " evidence_rank=" << evidence_depth.effective_rank << '\n';

                // Persist the new surface even when it remains UNKNOWN. This
                // is the resume point for later samples; it is not a learned
                // sideband and cannot promote itself.
                surface_state = {};
                surface_state.behavior_key = evidence.behavior_key;
                surface_state.model_profile_fingerprint = profile;
                surface_state.capture_layout_revision = "layer-input:v1";
                surface_state.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
                surface_state.anchor_layer = continuation.region.anchor_layer_index;
                surface_state.selected_scale = rank1_trial.candidate.total_scale;
                surface_state.best_margin_delta = plateau_result.best_margin_delta;
                surface_state.best_search_score = continuation.search_score;
                surface_state.extra_model_trials = zoom_trials.size();
                surface_state.next_candidate_index = zoom_trials.size();
                surface_state.surface_revision = 2;
                surface_state.parent_surface_revision = 1;
                surface_state.search_rank = 2;
                surface_state.evidence_rank = evidence_depth.effective_rank;
                surface_state.surface_origin = "orthogonal_search";
                surface_state.parent_surface_ref = "flydelta://state/model-repair/bootstrap-zoom";
                surface_state.local_layers = surface_layers;
                surface_state.completed_trials = zoom_trials;
                surface_state.surface_trials = surface_trials;
                surface_state.selection = zoom_selection;
                if (!common_flydelta_bootstrap_zoom_state_validate(surface_state, error)) {
                    std::cerr << "FlyDelta orthogonal surface state invalid: " << error << '\n';
                    return 1;
                }
                std::cout << "flydelta_surface_state revision=" << surface_state.surface_revision
                          << " parent_revision=" << surface_state.parent_surface_revision
                          << " search_rank=" << surface_state.search_rank
                          << " evidence_rank=" << surface_state.evidence_rank
                          << " origin=" << surface_state.surface_origin
                          << " controls=" << surface_state.surface_trials.size() << '\n';

                // A useful rank-two surface earns a small local WHERE
                // recenter, still with fresh contexts and the same model
                // runner. The probe is intentionally singleton: the stored
                // rank-two profile supplies the per-layer coefficient while
                // Whirlpool moves only the layer anchor.
                if (surface_utility_decision.utility_qualified) {
                    common_flydelta_whirlpool_search_config recenter_config;
                    recenter_config.available_layers = surface_layers;
                    recenter_config.seed_layers = {continuation.region.anchor_layer_index};
                    recenter_config.total_scale = rank1_trial.candidate.total_scale;
                    recenter_config.max_rounds = 1;
                    recenter_config.probes_per_round = std::min<size_t>(3, surface_layers.size());
                    recenter_config.max_trials = recenter_config.probes_per_round;
                    recenter_config.initial_radius = 1;
                    recenter_config.shrink_factor = 0.5f;
                    for (const auto & region_trial : region_trials) {
                        if (!region_trial.geometry_available) continue;
                        if (std::find(surface_layers.begin(), surface_layers.end(),
                                region_trial.candidate.anchor_layer_index) == surface_layers.end()) continue;
                        recenter_config.layer_diagnostics.push_back({
                            region_trial.candidate.anchor_layer_index,
                            region_trial.geometry.cosine,
                            region_trial.geometry.progress,
                            region_trial.geometry.leakage,
                            region_trial.geometry.shift_norm});
                    }
                    const auto run_recenter_probe =
                        [&](const common_flydelta_experiment_fixture &,
                            const common_flydelta_intervention_region_candidate * candidate,
                            common_flydelta_counterfactual_trial & counterfactual,
                            common_flydelta_decision_margin & margin,
                            common_flydelta_representation_diagnostics & diagnostics,
                            bool & diagnostics_available, std::string & runner_error) {
                        std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                        if (candidate != nullptr) {
                            const auto layer_it = std::find(surface_layers.begin(), surface_layers.end(),
                                candidate->anchor_layer_index);
                            if (layer_it == surface_layers.end()) {
                                runner_error = "Whirlpool recenter probe is outside the new surface";
                                return false;
                            }
                            const size_t layer_offset = static_cast<size_t>(
                                layer_it - surface_layers.begin());
                            float surface_weight = rank1_profile[layer_offset] +
                                0.5f * orthogonal_surface.direction[layer_offset];
                            if (std::fabs(surface_weight) <= 0.000001f) surface_weight = 1.0f;
                            const auto direction = std::find_if(basis.directions().begin(),
                                basis.directions().end(), [&](const auto & value) {
                                    return value.layer_index == static_cast<int32_t>(
                                        candidate->anchor_layer_index);
                                });
                            if (direction == basis.directions().end()) {
                                runner_error = "Whirlpool recenter has no layer-compatible direction";
                                return false;
                            }
                            common_flydelta_gate_request gate_request;
                            if (!common_flydelta_gate_request_from_context(
                                    recognition, code, true,
                                    common_flydelta_candidate_status::approved, true,
                                    candidate->per_layer_scale, gate_request, runner_error)) return false;
                            common_flydelta_activation_request request;
                            request.candidate_id = "flydelta://candidate/model-repair-surface-recenter";
                            request.artifact_id = "flydelta://artifact/model-repair-e2e";
                            request.model_profile_fingerprint = profile;
                            request.capture_layout_revision = "layer-input:v1";
                            request.model_n_embd = model_n_embd;
                            request.model_n_layers = model_n_layers;
                            request.il_end = static_cast<int32_t>(model_n_layers - 1);
                            request.directions.push_back(*direction);
                            request.coefficients.push_back(surface_weight);
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
                        common_agent_generation_result generated;
                        const bool executed = generate(*inference, value, failed_instruction,
                            generated, activation_ptr, capture_request);
                        counterfactual = {};
                        counterfactual.executed = executed;
                        counterfactual.verifier_known = executed;
                        counterfactual.passed = executed && contains_tool(generated, "data.inspect");
                        counterfactual.quality = counterfactual.passed ? 1.0f : 0.0f;
                        counterfactual.overlay_applied = candidate != nullptr;
                        counterfactual.intervention_count = candidate != nullptr ? 1 : 0;
                        counterfactual.evidence_ref = candidate != nullptr
                            ? "evidence:model-repair-surface-recenter"
                            : "evidence:model-repair-surface-recenter-baseline";
                        const auto scoring_request = make_request(value, failed_instruction);
                        if (!score_chat_choice_margin(
                                loaded->model, loaded->chat_templates.get(), scoring_request.messages,
                                scoring_request.tools, scoring_request.tool_choice, scoring_request.options,
                                "{\"name\":\"", "data.inspect", "data.describe", margin,
                                nullptr, scoring_request.json_schema, {}, {},
                                activation_ptr ? activation_ptr->overlay : common_flydelta_static_overlay{},
                                &runner_error)) return false;
                        diagnostics = {};
                        diagnostics_available = false;
                        if (candidate != nullptr && generated.flydelta_capture &&
                                generated.flydelta_capture->captured && region_baseline_capture) {
                            const auto measurement = std::find_if(deltas.begin(), deltas.end(),
                                [&](const auto & delta) { return delta.layer_index > static_cast<int>(
                                    candidate->anchor_layer_index); });
                            if (measurement != deltas.end()) {
                                if (!common_flydelta_representation_diagnostics_from_captures(
                                        *region_baseline_capture, *generated.flydelta_capture, *measurement,
                                        64U * 1024U * 1024U, diagnostics, runner_error)) return false;
                                diagnostics_available = true;
                            }
                        }
                        if (!executed && !generated.error_message.empty()) runner_error = generated.error_message;
                        return executed;
                    };
                    std::vector<common_flydelta_intervention_region_trial> recenter_trials;
                    common_flydelta_intervention_region_selection recenter_selection;
                    common_flydelta_whirlpool_trace recenter_trace;
                    if (!common_flydelta_run_whirlpool_search(
                            experiment_fixture, recenter_config, run_recenter_probe,
                            recenter_trials, recenter_selection, recenter_trace, error)) {
                        std::cerr << "FlyDelta surface Whirlpool recenter failed: " << error << '\n';
                        return 1;
                    }
                    std::cout << "flydelta_surface_recenter trials=" << recenter_trials.size()
                              << " model_calls=" << recenter_trace.model_evaluations
                              << " final_centre=" << recenter_trace.final_centre
                              << " final_radius=" << recenter_trace.final_radius
                              << " best_trial=" << recenter_trace.best_trial_index
                              << " best_search_score=" << recenter_trace.best_search_score
                              << " selected_helped=" << (recenter_selection.selected ? "yes" : "no")
                              << " surface_revision=" << surface_state.surface_revision << '\n';
                }
            }
        }

        // Model-facing representation augmentation. This is deliberately an
        // explicit smoke mode: the normal worker remains governed by natural
        // evidence depth, while this mode verifies the real capture/donor /
        // residualization seam against the same resident model.
        if (value.force_representation_augmentation) {
            const uint32_t augmentation_layer = continuation.region.anchor_layer_index;
            const auto base_delta = std::find_if(deltas.begin(), deltas.end(),
                [&](const auto & delta) {
                    return delta.layer_index == static_cast<int32_t>(augmentation_layer);
                });
            if (base_delta == deltas.end()) {
                std::cerr << "FlyDelta augmentation has no anchor-layer surface direction\n";
                return 1;
            }

            const auto layer_values = [&](const common_flydelta_hidden_state_capture & capture,
                    uint32_t layer, std::vector<float> & values) {
                const auto layer_it = std::find(capture.layer_indices.begin(),
                    capture.layer_indices.end(), layer);
                if (layer_it == capture.layer_indices.end()) return false;
                const size_t offset = static_cast<size_t>(layer_it - capture.layer_indices.begin()) *
                    capture.n_embd;
                if (offset + capture.n_embd > capture.values.size()) return false;
                values.assign(capture.values.begin() + offset,
                    capture.values.begin() + offset + capture.n_embd);
                return true;
            };

            // This is a second natural host-certified context in the same
            // behavior family. It is intentionally different from the first
            // repair pair, so residualization has a chance to reveal a new
            // experimental axis instead of reproducing the parent delta.
            const char * donor_instruction =
                "The available tools are data.describe and data.inspect. For inventory.csv, "
                "the request asks to inspect the first table, so choose data.inspect. "
                "Return exactly {\"name\":\"data.inspect\",\"arguments\":"
                "{\"dataset\":\"inventory.csv\"}}.";
            common_agent_generation_result donor;
            const bool donor_executed = generate(*inference, value, donor_instruction, donor,
                {}, capture_request);
            const bool donor_verified = donor_executed && contains_tool(donor, "data.inspect") &&
                donor.flydelta_capture && donor.flydelta_capture->captured &&
                common_flydelta_hidden_state_capture_validate(
                    *donor.flydelta_capture, 64U * 1024U * 1024U, error);
            std::cout << "augmentation_donor_model_output=" << output_preview(donor) << '\n'
                      << "augmentation_donor_host_verified="
                      << (donor_verified ? "yes" : "no") << '\n';
            if (!donor_verified) {
                std::cerr << "FlyDelta augmentation donor was not host-certified: " << error << '\n';
                return 1;
            }

            common_flydelta_representation_donor_candidate donor_candidate;
            donor_candidate.donor_id = "flydelta://donor/model-repair-augmentation";
            donor_candidate.source_type = "host_certified_repair";
            donor_candidate.source_ref = "evidence://flydelta/model-repair-augmentation";
            donor_candidate.behavior_key = evidence.behavior_key;
            donor_candidate.context_payload_ref = "context://host/model-repair-augmentation";
            donor_candidate.qualification_policy = "host_verified_or_margin_guided";
            donor_candidate.provenance = "model-facing smoke; opaque host context reference";
            if (!common_flydelta_representation_donor_candidate_validate(
                    donor_candidate, error)) {
                std::cerr << "FlyDelta augmentation donor candidate is invalid: " << error << '\n';
                return 1;
            }

            common_flydelta_representation_donor_qualification observation;
            observation.donor_id = donor_candidate.donor_id;
            observation.host_evaluated = true;
            observation.verifier_known = true;
            observation.safe_to_continue = true;
            observation.host_outcome = common_flydelta_counterfactual_outcome::helped;
            const auto donor_request = make_request(value, donor_instruction);
            common_flydelta_decision_margin donor_margin;
            if (!score_chat_choice_margin(
                    loaded->model, loaded->chat_templates.get(), donor_request.messages,
                    donor_request.tools, donor_request.tool_choice, donor_request.options,
                    "{\"name\":\"", "data.inspect", "data.describe", donor_margin,
                    nullptr, donor_request.json_schema, {}, {}, {}, &error)) {
                // Margin is optional for a host-certified donor. Keep the
                // donor usable, but make the missing diagnostic visible.
                observation.decision_margin_available = false;
            } else {
                observation.decision_margin_available = true;
                observation.margin_gain = donor_margin.normalized_delta();
            }
            // The diagnostic helper needs a behavior delta. Its values are
            // useful here only as geometry for donor search qualification.
            common_flydelta_representation_diagnostics donor_geometry;
            if (common_flydelta_representation_diagnostics_from_captures(
                    *failed.flydelta_capture, *donor.flydelta_capture, *base_delta,
                    64U * 1024U * 1024U, donor_geometry, error)) {
                observation.geometry_available = true;
                observation.geometry = donor_geometry;
                observation.margin_gain = std::max(observation.margin_gain, 0.0f);
            }
            common_flydelta_representation_donor_qualification qualified;
            common_flydelta_representation_augmentation_config augmentation_config;
            if (!common_flydelta_qualify_representation_donor(
                    donor_candidate, observation, augmentation_config.minimum_margin_gain,
                    augmentation_config.max_leakage, augmentation_config.max_shift_norm,
                    qualified, error) || !qualified.search_qualified) {
                std::cerr << "FlyDelta augmentation donor was not search-qualified: " << error
                          << " reason=" << qualified.reason << '\n';
                return 1;
            }

            std::vector<float> donor_values;
            std::vector<float> target_values;
            if (!layer_values(*donor.flydelta_capture, augmentation_layer, donor_values) ||
                    !layer_values(*failed.flydelta_capture, augmentation_layer, target_values)) {
                std::cerr << "FlyDelta augmentation donor/target layer capture is unavailable\n";
                return 1;
            }
            common_flydelta_representation_latent_delta latent;
            if (!common_flydelta_build_residualized_latent_delta(
                    donor_candidate.donor_id, static_cast<int32_t>(augmentation_layer),
                    donor_values, target_values, {base_delta->values},
                    augmentation_config.minimum_residual_norm, latent, error)) {
                std::cerr << "FlyDelta augmentation residualization failed: " << error << '\n';
                return 1;
            }
            std::cout << "augmentation_latent_delta available="
                      << (latent.available ? "yes" : "no")
                      << " layer=" << latent.layer_index
                      << " raw_norm=" << latent.raw_norm
                      << " residual_norm=" << latent.residual_norm
                      << " removed_norm=" << latent.removed_norm << '\n';

            common_flydelta_representation_augmentation_state augmentation_state;
            augmentation_state.state_ref =
                "flydelta://state/representation-augmentation/model-repair-e2e";
            augmentation_state.model_fingerprint = profile;
            augmentation_state.behavior_key = evidence.behavior_key;
            augmentation_state.direction_family_id = "structured_tool_selection";
            augmentation_state.parent_surface_revision = surface_state.surface_revision == 0
                ? 1 : surface_state.surface_revision;
            augmentation_state.parent_search_state_ref =
                "flydelta://state/model-repair/bootstrap-zoom";
            augmentation_state.parent_evidence_rank = std::max(0.1f,
                static_cast<float>(evidence_depth.effective_rank));
            augmentation_state.evidence_rank = augmentation_state.parent_evidence_rank;
            augmentation_state.search_rank = 1;
            augmentation_state.selected_region = {augmentation_layer};
            augmentation_state.target_fixture_ref = experiment_fixture.id;
            augmentation_state.donor_candidate_refs = {donor_candidate.donor_id};
            augmentation_state.qualified_donor_refs = {donor_candidate.donor_id};
            augmentation_state.phase =
                common_flydelta_representation_augmentation_phase::build_latent_delta;
            augmentation_state.best_donor_ref = donor_candidate.donor_id;
            augmentation_state.best_margin_gain = qualified.margin_gain;
            augmentation_state.remaining_budget = augmentation_config.max_controls +
                augmentation_config.max_local_whirlpool_probes +
                augmentation_config.max_full_generation;
            augmentation_state.surface_revision = augmentation_state.parent_surface_revision;
            if (!common_flydelta_apply_representation_augmentation(
                    augmentation_state, latent, error)) {
                if (!latent.available) {
                    std::cout << "augmentation_search=retain reason=residual_below_threshold\n";
                } else {
                    std::cerr << "FlyDelta augmentation state transition failed: " << error << '\n';
                    return 1;
                }
            } else {
                std::vector<std::vector<float>> controls;
                if (!common_flydelta_propose_representation_augmentation_controls(
                        augmentation_config, controls, error)) {
                    std::cerr << "FlyDelta augmentation controls failed: " << error << '\n';
                    return 1;
                }
                const auto base_direction = common_flydelta_basis_direction{
                    base_delta->layer_index, base_delta->values, 1, 0, 0};
                float best_control_margin = 0.0f;
                size_t control_index = 0;
                for (const auto & control : controls) {
                    common_flydelta_activation_request request;
                    request.candidate_id = "flydelta://candidate/model-repair-augmentation";
                    request.artifact_id = "flydelta://artifact/model-repair-e2e";
                    request.model_profile_fingerprint = profile;
                    request.capture_layout_revision = "layer-input:v1";
                    request.model_n_embd = model_n_embd;
                    request.model_n_layers = model_n_layers;
                    request.il_end = static_cast<int32_t>(model_n_layers - 1);
                    request.directions = {base_direction,
                        {static_cast<int32_t>(augmentation_layer), latent.values}};
                    request.coefficients = control;
                    if (!common_flydelta_gate_request_from_context(
                            recognition, code, true,
                            common_flydelta_candidate_status::approved, true,
                            continuation.region.total_scale, request.gate_request, error)) {
                        std::cerr << "FlyDelta augmentation control gate failed: " << error << '\n';
                        return 1;
                    }
                    common_flydelta_gate_config gate_config;
                    gate_config.enabled = true;
                    gate_config.max_scale = 1.0f;
                    common_flydelta_activation_result activation;
                    if (!common_flydelta_prepare_activation(
                            gate_config, request, 64U * 1024U * 1024U,
                            activation, error)) {
                        std::cerr << "FlyDelta augmentation control activation failed: " << error << '\n';
                        return 1;
                    }
                    const auto activation_ptr =
                        std::make_shared<const common_flydelta_activation_result>(
                            std::move(activation));
                    common_agent_generation_result generated;
                    if (!generate(*inference, value, failed_instruction, generated,
                            activation_ptr, capture_request)) {
                        std::cerr << "FlyDelta augmentation control generation failed: "
                                  << generated.error_message << '\n';
                        return 1;
                    }
                    common_flydelta_decision_margin margin;
                    const auto scoring_request = make_request(value, failed_instruction);
                    if (!score_chat_choice_margin(
                            loaded->model, loaded->chat_templates.get(),
                            scoring_request.messages, scoring_request.tools,
                            scoring_request.tool_choice, scoring_request.options,
                            "{\"name\":\"", "data.inspect", "data.describe", margin,
                            nullptr, scoring_request.json_schema, {}, {},
                            activation_ptr->overlay, &error)) {
                        std::cerr << "FlyDelta augmentation control margin failed: " << error << '\n';
                        return 1;
                    }
                    const float margin_delta = region_baseline_margin.available && margin.available
                        ? margin.normalized_delta() - region_baseline_margin.normalized_delta() : 0.0f;
                    best_control_margin = std::max(best_control_margin, margin_delta);
                    std::cout << "augmentation_control index=" << control_index++
                              << " c0=" << control[0] << " c1=" << control[1]
                              << " host_outcome=" << (contains_tool(generated, "data.inspect")
                                  ? "helped" : "unknown")
                              << " margin_delta=" << margin_delta
                              << " output=" << output_preview(generated) << '\n';
                }
                augmentation_state.phase =
                    common_flydelta_representation_augmentation_phase::run_controls;
                common_flydelta_representation_augmentation_action action;
                if (!common_flydelta_decide_representation_augmentation(
                        augmentation_config, augmentation_state, {qualified},
                        best_control_margin > 0.0f, false, action, error)) {
                    std::cerr << "FlyDelta augmentation UtilityGate failed: " << error << '\n';
                    return 1;
                }
                augmentation_state.next_action =
                    common_flydelta_representation_augmentation_action_name(action);
                std::cout << "augmentation_utility_gate best_margin_delta="
                          << best_control_margin << " action="
                          << augmentation_state.next_action
                          << " search_rank=" << augmentation_state.search_rank
                          << " evidence_rank=" << augmentation_state.evidence_rank
                          << " surface_revision=" << augmentation_state.surface_revision
                          << " promotion=no\n";
            }
        }

    // Exercise the actual evaluator/worker handoff. The first queue slice
    // owns the model-backed Bootstrap/BootstrapZoom result and persists typed
    // zoom state. The second slice carries a separate opaque post-Bootstrap
    // state reference and must use the generic state-aware callback. With one
    // natural repair sample this remains rank-one: the smoke must not
    // fabricate Shallow/Deep/TFO evidence.
        if (value.region_scan && !selected.selected && !zoom_trials.empty()) {
        common_flydelta_experiment_seed state_seed;
        if (!common_flydelta_experiment_seed_from_evidence(
                evidence, evidence.behavior_key, profile,
                experiment_fixture.tokenizer_fingerprint,
                experiment_fixture.template_fingerprint,
                experiment_fixture.execution_context_fingerprint,
                common_flydelta_training_split::train, state_seed, error)) {
            std::cerr << "FlyDelta state smoke seed construction failed: " << error << '\n';
            return 1;
        }
        common_flydelta_experiment_job state_job;
        state_job.id = "flydelta://job/model-repair-state/bootstrap";
        state_job.kind = common_flydelta_experiment_job_kind::search_pipeline;
        state_job.seed = state_seed;
        state_job.capture_manifest_ids = {"manifest:model-repair-e2e"};
        state_job.behavior_delta_ids = {"delta:model-repair-e2e"};
        state_job.alpha_search.candidates = {0.02f};
        state_job.alpha_search.max_candidates = 1;
        state_job.code_revision = "flydelta-model-repair-state-smoke:v1";

        common_flydelta_evaluator_config state_evaluator_config;
        state_evaluator_config.pipeline = pipeline_config;
        state_evaluator_config.max_references = 128;
        common_flydelta_evaluator_callbacks state_callbacks;
        state_callbacks.run_search_pipeline_with_state = [&](const auto & job,
                const auto * resume_state, auto & output, auto & next_state,
                std::string & runner_error) {
            if (resume_state != nullptr) {
                runner_error = "initial Qwen state slice unexpectedly received a resume state";
                return false;
            }
            output = pipeline_result;
            next_state = {};
            next_state.behavior_key = job.seed.behavior_key;
            next_state.model_profile_fingerprint = job.seed.model_profile_fingerprint;
            next_state.capture_layout_revision = "layer-input:v1";
            next_state.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
            next_state.anchor_layer = continuation.region.anchor_layer_index;
            next_state.selected_scale = std::max(0.0001f,
                continuation.region.total_scale);
            next_state.best_margin_delta = utility_observation.decision_margin_delta;
            next_state.best_search_score = continuation.search_score;
            next_state.extra_model_trials = zoom_trials.size();
            next_state.next_candidate_index = zoom_trials.size();
            next_state.local_layers = {continuation.region.anchor_layer_index};
            next_state.completed_trials = zoom_trials;
            next_state.selection = zoom_selection;
            if (surface_state.surface_revision > 1) {
                next_state.surface_revision = surface_state.surface_revision;
                next_state.parent_surface_revision = surface_state.parent_surface_revision;
                next_state.search_rank = surface_state.search_rank;
                next_state.evidence_rank = surface_state.evidence_rank;
                next_state.surface_origin = surface_state.surface_origin;
                next_state.parent_surface_ref = surface_state.parent_surface_ref;
                next_state.local_layers = surface_state.local_layers;
                next_state.surface_trials = surface_state.surface_trials;
            }
            return true;
        };
        state_callbacks.persist_bootstrap_zoom_state = [&](const auto & state,
                std::string & state_ref, std::string & persist_error) {
            if (!common_flydelta_bootstrap_zoom_state_validate(state, persist_error)) {
                return false;
            }
            state_ref = "flydelta://state/model-repair/bootstrap-zoom";
            return true;
        };

        const std::string post_bootstrap_state_ref =
            "flydelta://state/model-repair/post-bootstrap";
        state_callbacks.run_search_pipeline_with_search_state =
            [&](const auto & job, const std::string & resume_state_ref,
                    auto & output, std::string & next_state_ref,
                    std::string & runner_error) {
                if (resume_state_ref != post_bootstrap_state_ref ||
                        job.bootstrap_zoom_state_ref.empty()) {
                    runner_error = "post-Bootstrap Qwen state reference was not carried correctly";
                    return false;
                }
                common_flydelta_gate_request gate_request;
                if (!common_flydelta_gate_request_from_context(
                        recognition, code, true,
                        common_flydelta_candidate_status::approved, true,
                        continuation.region.per_layer_scale, gate_request, runner_error)) {
                    return false;
                }
                common_flydelta_activation_request request;
                request.candidate_id = "flydelta://candidate/model-repair-post-bootstrap";
                request.artifact_id = "flydelta://artifact/model-repair-e2e";
                request.model_profile_fingerprint = profile;
                request.capture_layout_revision = "layer-input:v1";
                request.model_n_embd = model_n_embd;
                request.model_n_layers = model_n_layers;
                request.il_end = static_cast<int32_t>(model_n_layers - 1);
                for (const uint32_t layer : continuation.region.layer_indices) {
                    const auto direction = std::find_if(basis.directions().begin(),
                        basis.directions().end(), [&](const auto & value) {
                            return value.layer_index == static_cast<int32_t>(layer);
                        });
                    if (direction == basis.directions().end()) {
                        runner_error = "post-Bootstrap state has no layer-compatible direction";
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
                const auto activation_ptr =
                    std::make_shared<const common_flydelta_activation_result>(
                        std::move(activation));
                common_agent_generation_result generated;
                if (!generate(*inference, value, failed_instruction, generated,
                        activation_ptr, capture_request)) {
                    runner_error = generated.error_message.empty()
                        ? "post-Bootstrap Qwen state generation failed"
                        : generated.error_message;
                    return false;
                }
                std::cout << "flydelta_post_bootstrap_state_model_output="
                          << output_preview(generated) << '\n';
                output = pipeline_result;
                next_state_ref = "flydelta://state/model-repair/post-bootstrap-next";
                return true;
            };

        const auto state_root = std::filesystem::temp_directory_path() /
            "llama-agent-flydelta-model-state-smoke";
        std::error_code state_cleanup_error;
        std::filesystem::remove_all(state_root, state_cleanup_error);
        common_flydelta_experiment_worker_report initial_state_report;
        if (!common_flydelta_experiment_queue_enqueue(
                state_root, state_job, {}, error) ||
                !common_flydelta_experiment_worker_run_evaluator_once(
                    state_root, {}, state_evaluator_config, state_callbacks,
                    initial_state_report, error) ||
                initial_state_report.state !=
                    common_flydelta_experiment_queue_state::succeeded ||
                initial_state_report.bootstrap_zoom_state_ref.empty()) {
            std::filesystem::remove_all(state_root, state_cleanup_error);
            std::cerr << "FlyDelta initial state-aware worker slice failed: " << error << '\n';
            return 1;
        }
        std::cout << "flydelta_stateful_worker_initial state="
                  << common_flydelta_experiment_queue_state_name(initial_state_report.state)
                  << " bootstrap_zoom_state_ref=" << initial_state_report.bootstrap_zoom_state_ref
                  << " search_state_ref=" << initial_state_report.search_state_ref << '\n';

        common_flydelta_experiment_job post_state_job = state_job;
        post_state_job.id = "flydelta://job/model-repair-state/post-bootstrap";
        post_state_job.bootstrap_zoom_state_ref = initial_state_report.bootstrap_zoom_state_ref;
        post_state_job.search_state_ref = post_bootstrap_state_ref;
        common_flydelta_experiment_worker_report post_state_report;
        if (!common_flydelta_experiment_queue_enqueue(
                state_root, post_state_job, {}, error) ||
                !common_flydelta_experiment_worker_run_evaluator_once(
                    state_root, {}, state_evaluator_config, state_callbacks,
                    post_state_report, error) ||
                post_state_report.state !=
                    common_flydelta_experiment_queue_state::succeeded ||
                post_state_report.search_state_ref.empty()) {
            std::filesystem::remove_all(state_root, state_cleanup_error);
            std::cerr << "FlyDelta post-Bootstrap state-aware worker slice failed: " << error << '\n';
            return 1;
        }
        std::cout << "flydelta_stateful_worker_post state="
                  << common_flydelta_experiment_queue_state_name(post_state_report.state)
                  << " bootstrap_zoom_state_ref=" << post_state_report.bootstrap_zoom_state_ref
                  << " search_state_ref=" << post_state_report.search_state_ref << '\n';
        std::filesystem::remove_all(state_root, state_cleanup_error);
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
                const auto layer2_separation = std::find_if(discovery.scores.begin(),
                    discovery.scores.end(), [](const auto & score) {
                        return score.layer_index == 2;
                    });
                if (layer2_separation != discovery.scores.end() &&
                        layer2_separation->separation > 0.0f) {
                    scale_config.separation_calibrated = true;
                    scale_config.reference_separation = layer2_separation->separation;
                    scale_config.max_resolved_scale = 1.0f;
                }
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
                            if (!apply_overlay && result.flydelta_capture &&
                                    result.flydelta_capture->captured) {
                                scale_baseline_capture = result.flydelta_capture;
                            }
                            if (apply_overlay && result.flydelta_capture &&
                                    result.flydelta_capture->captured && scale_baseline_capture) {
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
                              << " requested_scale=" << trial.requested_scale
                              << " separation_calibrated="
                              << (trial.separation_calibrated ? "yes" : "no")
                              << " scale_clamped=" << (trial.scale_clamped ? "yes" : "no")
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
                            if (!apply_overlay && result.flydelta_capture &&
                                    result.flydelta_capture->captured) {
                                coefficient_baseline_capture = result.flydelta_capture;
                            }
                            if (apply_overlay && result.flydelta_capture &&
                                    result.flydelta_capture->captured && coefficient_baseline_capture &&
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
                    print_margin("margin", trial.margin);
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
