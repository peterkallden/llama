#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"
#include "hash/hash.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model;
    int n_predict = 8;
    int n_threads = 3;
    int n_gpu_layers = 0;
    int n_parallel = 2;
    int n_ctx_per_sequence = 6144;
    int arm_count = 2;
    int warmup_count = 0;
    int repeat_count = 1;
    int layer_index = -1;
    int factorized_rank = 1;
    bool scalar = false;
    bool factorized = false;
    bool single_sparse_device = false;
    bool device_diagnostics = false;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    if (const char * parallel = std::getenv("LLAMA_AGENT_N_PARALLEL")) value.n_parallel = std::stoi(parallel);
    if (const char * per_sequence_ctx = std::getenv("LLAMA_AGENT_N_CTX_PER_SEQUENCE")) {
        value.n_ctx_per_sequence = std::stoi(per_sequence_ctx);
    }
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
        } else if (argument == "--n-parallel") {
            const char * value_arg = next("--n-parallel");
            if (!value_arg) return false;
            value.n_parallel = std::stoi(value_arg);
        } else if (argument == "--n-ctx-per-sequence") {
            const char * value_arg = next("--n-ctx-per-sequence");
            if (!value_arg) return false;
            value.n_ctx_per_sequence = std::stoi(value_arg);
        } else if (argument == "--arm-count") {
            const char * value_arg = next("--arm-count");
            if (!value_arg) return false;
            value.arm_count = std::stoi(value_arg);
        } else if (argument == "--warmup-count") {
            const char * value_arg = next("--warmup-count");
            if (!value_arg) return false;
            value.warmup_count = std::stoi(value_arg);
        } else if (argument == "--repeat-count") {
            const char * value_arg = next("--repeat-count");
            if (!value_arg) return false;
            value.repeat_count = std::stoi(value_arg);
        } else if (argument == "--n-gpu-layers") {
            const char * value_arg = next("--n-gpu-layers");
            if (!value_arg) return false;
            value.n_gpu_layers = std::stoi(value_arg);
        } else if (argument == "--layer-index") {
            const char * value_arg = next("--layer-index");
            if (!value_arg) return false;
            value.layer_index = std::stoi(value_arg);
        } else if (argument == "--rank") {
            const char * value_arg = next("--rank");
            if (!value_arg) return false;
            value.factorized_rank = std::stoi(value_arg);
        } else if (argument == "--scalar") {
            value.scalar = true;
        } else if (argument == "--factorized") {
            value.factorized = true;
        } else if (argument == "--single-sparse-device") {
            value.single_sparse_device = true;
        } else if (argument == "--device-diagnostics") {
            value.device_diagnostics = true;
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
        const llama_model * model, float value, const std::string & artifact_id,
        const bool factorized, const int32_t layer_index,
        std::shared_ptr<const std::vector<float>> shared_basis,
        std::string & error) {
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
    direction.layer_index = layer_index;
    const int32_t rank = shared_basis && request.model_n_embd > 0
        ? static_cast<int32_t>(shared_basis->size() / request.model_n_embd) : 1;
    direction.values.assign(request.model_n_embd, 0.0f);
    const float component_scale = 1.0f / std::sqrt(static_cast<float>(rank));
    for (int32_t component = 0; component < rank; ++component) {
        direction.values[static_cast<size_t>(component)] = component_scale;
    }
    request.directions.push_back(std::move(direction));
    request.coefficients = {value};
    request.allow_factorized_overlay = factorized;
    if (factorized) {
        request.factorized_basis = std::move(shared_basis);
        request.factorized_rank = static_cast<int32_t>(request.factorized_basis->size() /
            request.model_n_embd);
        request.coefficients.assign(static_cast<size_t>(request.factorized_rank),
            value * component_scale);
        request.factorized_layer_index = layer_index;
        std::string basis_identity = "cvec-batch-smoke:f32:rank=" +
            std::to_string(request.factorized_rank) + ":d=" +
            std::to_string(request.model_n_embd) + ":layer=" + std::to_string(layer_index);
        basis_identity.append(reinterpret_cast<const char *>(request.factorized_basis->data()),
            request.factorized_basis->size() * sizeof(float));
        request.factorized_basis_fingerprint = "sha256:" +
            hash_sha256_hex(basis_identity.data(), basis_identity.size());
    }
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
                  << " --model MODEL [--n-predict N] [--threads N] [--n-gpu-layers N]"
                  << " [--scalar] [--factorized] [--rank 1|2|4] [--layer-index N]"
                  << " [--n-parallel N] [--n-ctx-per-sequence N] [--arm-count N]"
                  << " [--warmup-count N] [--repeat-count N]"
                  << " [--single-sparse-device] [--device-diagnostics]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "FlyDelta cvec batch model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0 ||
            value.n_parallel <= 0 || value.n_parallel > 16 ||
            value.n_ctx_per_sequence <= 0 || value.arm_count <= 0 || value.arm_count > 64 ||
            value.warmup_count < 0 || value.repeat_count <= 0 ||
            static_cast<int64_t>(value.n_parallel) * value.n_ctx_per_sequence >
                std::numeric_limits<int32_t>::max()) return 2;
    if (value.factorized_rank != 1 && value.factorized_rank != 2 && value.factorized_rank != 4) return 2;

    auto host = std::make_shared<common_agent_server_context_host>();
    common_agent_server_context_host_config config;
    config.context_key.load_key.model = value.model;
    config.context_key.load_key.n_gpu_layers = value.n_gpu_layers;
    config.context_key.load_key.fit_params = true;
    // A single logical arm still runs in the ordinary two-slot resident
    // context. Sparse scalar binding is an overlay representation choice,
    // not a reason to impose a separate n_parallel=1 runtime topology.
    if (value.device_diagnostics && value.scalar) {
        std::cerr << "device diagnostics require the paired two-slot context\n";
        return 2;
    }
    const bool one_slot = value.scalar;
    config.context_key.n_parallel = one_slot ? 1 : value.n_parallel;
    config.context_key.n_sequences = one_slot ? 1 : value.n_parallel;
    // Keep the direct model smoke on the same bounded resident context as the
    // production runtime smoke.  The batch contract is what this fixture
    // exercises; a larger standalone context is not part of that contract.
    config.context_key.n_ctx = one_slot ? value.n_ctx_per_sequence
        : value.n_parallel * value.n_ctx_per_sequence;
    config.context_key.n_threads = value.n_threads;
    config.verbosity = LOG_LEVEL_INFO;
    config.per_sequence_cvec_batch = !value.scalar;
    config.per_sequence_factorized_cvec_batch = !value.scalar && value.factorized;

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
    std::cout << "cvec_smoke_context requested_parallel=" << config.context_key.n_parallel
              << " requested_ctx_per_sequence=" << value.n_ctx_per_sequence
              << " requested_total_ctx=" << config.context_key.n_ctx
              << " actual_total_ctx=" << llama_n_ctx(context)
              << " actual_ctx_per_sequence=" << llama_n_ctx(context) / config.context_key.n_parallel
              << " actual_parallel=" << config.context_key.n_parallel << '\n';
    if (llama_model_n_embd(model) <= 0 || llama_model_n_layer(model) <= 2) return 1;

    common_agent_server_flydelta_binding_callbacks callbacks;
    callbacks.primitives.generation = true;
    callbacks.primitives.overlay = true;
    callbacks.primitives.host_verification = true;
    auto captured_rows = std::make_shared<std::map<std::string, std::vector<float>>>();
    // Layer inputs precede the overlay at that layer. Leave a downstream
    // layer available when measuring the intervention against a zero arm.
    const int32_t default_layer = llama_model_n_layer(model) - (value.device_diagnostics ? 2 : 1);
    const int32_t cvec_layer = value.layer_index >= 0 ? value.layer_index : default_layer;
    if (cvec_layer < 1 || cvec_layer >= llama_model_n_layer(model)) {
        std::cerr << "cvec layer index is outside the model's non-embedding layers\n";
        return 2;
    }
    const int32_t factorized_rank = value.factorized_rank;
    const size_t model_embd = static_cast<size_t>(llama_model_n_embd(model));
    auto shared_basis_values = std::make_shared<std::vector<float>>(
        model_embd * static_cast<size_t>(factorized_rank), 0.0f);
    for (int32_t component = 0; component < factorized_rank; ++component) {
        (*shared_basis_values)[static_cast<size_t>(component) * model_embd + component] = 1.0f;
    }
    std::shared_ptr<const std::vector<float>> shared_basis = shared_basis_values;
    callbacks.prepare_arm = [model, n_predict = value.n_predict, n_threads = value.n_threads,
            factorized = value.factorized, factorized_batch = !value.scalar && !value.single_sparse_device,
            cvec_layer, shared_basis](
            const common_flydelta_arm_request & arm,
            common_agent_generation_request & request,
            std::string & prepare_error) {
        request = {};
        request.purpose = common_agent_generation_purpose::conversation;
        request.options.n_predict = n_predict;
        request.options.n_threads = n_threads;
        request.options.generation_trace = std::getenv("LLAMA_AGENT_RESIDENT_TRACE") != nullptr;
        request.trace_id = "flydelta-cvec-batch:" + arm.arm_id;
        request.messages = {
            {"system", "Reply with exactly PASS."},
            {"user", "Run the isolated FlyDelta arm."},
        };
        if (arm.apply_overlay || (arm.request_device_diagnostics && !arm.apply_overlay)) {
            auto activation = make_activation(
                model, arm.apply_overlay ? arm.alpha : 0.0f,
                arm.intervention_ref + "/" + arm.arm_id,
                factorized && factorized_batch, cvec_layer, shared_basis, prepare_error);
            if (!activation) return false;
            request.flydelta_activation = std::move(activation);
        }
        if (arm.device_diagnostics && arm.device_diagnostics->enabled) {
            request.flydelta_device_diagnostics = arm.device_diagnostics;
        } else if (arm.request_device_diagnostics) {
            auto reduction = std::make_shared<common_flydelta_device_diagnostics_request>();
            reduction->enabled = true;
            reduction->group_id = arm.arm_id;
            reduction->baseline_group_id = arm.diagnostics_baseline_arm_id;
            reduction->layer = arm.layer_indices.front() + 1;
            reduction->absolute_position = arm.diagnostics_position;
            reduction->direction.assign(llama_model_n_embd(model), 0.0f);
            reduction->direction.front() = 1.0f;
            request.flydelta_device_diagnostics = std::move(reduction);
        }
        if (arm.request_device_diagnostics) {
            auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
            capture->enabled = true;
            capture->layer_indices = {static_cast<uint32_t>(cvec_layer + 1)};
            capture->position = common_flydelta_capture_position::generation_boundary;
            capture->model_profile_fingerprint = "sha256:flydelta-cvec-batch-smoke-model";
            capture->capture_layout_revision = "layer-input:v1";
            request.flydelta_capture = std::move(capture);
        }
        if (!arm.request_generation) request.options.n_predict = 0;
        return true;
    };
    callbacks.finalize_arm = [captured_rows](
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
        if (arm.request_device_diagnostics) {
            if (!generation.flydelta_capture || !generation.flydelta_capture->captured) {
                finalize_error = "sequence-specific downstream capture was not available";
                return false;
            }
            (*captured_rows)[arm.arm_id] = generation.flydelta_capture->values;
        }
        if (arm.apply_overlay && !generation.flydelta_runtime.applied) {
            finalize_error = "server did not confirm that the overlay reached a successful decode graph";
            return false;
        }
        if (arm.request_generation) {
            std::cout << "cvec_smoke_arm_output arm=" << std::quoted(arm.arm_id)
                      << " content=" << std::quoted(generation.content) << '\n';
        }
        if (arm.request_device_diagnostics && !arm.diagnostics_baseline_arm_id.empty() &&
                (!generation.flydelta_device_diagnostics ||
                 !generation.flydelta_device_diagnostics->available)) {
            finalize_error = generation.flydelta_device_diagnostics &&
                    !generation.flydelta_device_diagnostics->failure_reason.empty()
                ? generation.flydelta_device_diagnostics->failure_reason
                : "device diagnostics did not return a non-CPU reduction";
            return false;
        }
        if (generation.flydelta_device_diagnostics &&
                generation.flydelta_device_diagnostics->available) {
            const auto & reduction = *generation.flydelta_device_diagnostics;
            std::cout << "cvec_smoke_device_reduction arm=" << std::quoted(arm.arm_id)
                      << " layer=" << reduction.layer
                      << " dot=" << reduction.dot_shift_delta
                      << " shift_squared=" << reduction.shift_squared
                      << " delta_squared=" << reduction.delta_squared
                      << " residual_squared=" << reduction.residual_squared << '\n';
            common_flydelta_representation_diagnostics diagnostics;
            if (!common_flydelta_representation_diagnostics_from_reductions(
                    generation.flydelta_device_diagnostics->layer,
                    generation.flydelta_device_diagnostics->dot_shift_delta,
                    generation.flydelta_device_diagnostics->shift_squared,
                    generation.flydelta_device_diagnostics->delta_squared,
                    generation.flydelta_device_diagnostics->residual_squared,
                    diagnostics, finalize_error)) return false;
            result.geometry_available = true;
            result.cosine = diagnostics.cosine;
            result.progress = diagnostics.progress;
            result.leakage = diagnostics.leakage;
            result.shift_norm = diagnostics.shift_norm;
            result.execution_metrics.available = true;
            result.execution_metrics.device_reduction_used = true;
            result.execution_metrics.diagnostics_bytes_to_host =
                common_flydelta_compact_geometry_bytes;
        }
        if (!result.executed && !generation.error_message.empty()) {
            result.provenance_ref = generation.error_message;
        }
        return result.executed;
    };
    callbacks.score_teacher_forced_margin_batch = [model, n_predict = value.n_predict,
            n_threads = value.n_threads, factorized = value.factorized,
            factorized_batch = !value.scalar && !value.single_sparse_device,
            cvec_layer, shared_basis](
            const std::vector<common_flydelta_arm_request> & arms,
            const std::vector<common_agent_generation_request> &,
            common_agent_inference & scorer,
            std::vector<common_flydelta_decision_margin> & margins,
            std::string & score_error) {
        score_error.clear();
        margins.assign(arms.size(), {});
        common_agent_teacher_forced_choice_batch_request score_batch;
        score_batch.choices.reserve(arms.size());
        for (const auto & arm : arms) {
            common_agent_teacher_forced_choice_request choice;
            choice.sequence_id = arm.arm_id;
            choice.context = {};
            choice.context.purpose = common_agent_generation_purpose::conversation;
            choice.context.options.n_predict = n_predict;
            choice.context.options.n_threads = n_threads;
            choice.context.messages = {
                {"system", "Reply with exactly PASS."},
                {"user", "Run the isolated FlyDelta arm."},
            };
            if (arm.apply_overlay) {
                auto activation = make_activation(model, arm.alpha,
                    arm.intervention_ref + "/" + arm.arm_id,
                    factorized && factorized_batch, cvec_layer, shared_basis, score_error);
                if (!activation) return false;
                choice.context.flydelta_activation = std::move(activation);
            }
            choice.context.flydelta_capture.reset();
            choice.choice_prefix = "\n";
            choice.positive_choice = "PASS";
            choice.negative_choice = "FAIL";
            choice.positive_continuation = "PASS";
            choice.negative_continuation = "FAIL";
            score_batch.choices.push_back(std::move(choice));
        }
        common_agent_teacher_forced_choice_batch_result scored;
        if (!scorer.score_teacher_forced_choice_batch(score_batch, scored)) {
            score_error = scored.error_message.empty()
                ? "cvec model smoke teacher-forced margin failed" : scored.error_message;
            return false;
        }
        if (scored.choices.size() != margins.size()) {
            score_error = "cvec model smoke teacher-forced result count mismatch";
            return false;
        }
        for (size_t index = 0; index < margins.size(); ++index) {
            const auto & choice = scored.choices[index];
            margins[index].available = choice.available;
            margins[index].positive_total_logprob = choice.positive_total_logprob;
            margins[index].negative_total_logprob = choice.negative_total_logprob;
            margins[index].positive_token_count = choice.positive_token_count;
            margins[index].negative_token_count = choice.negative_token_count;
        }
        return true;
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
    if (!model_host ||
            (!value.scalar && !value.single_sparse_device && !model_host->capabilities.bounded_arm_batch)) {
        std::cerr << "could not bind production FlyDelta batch host: " << error << '\n';
        return 1;
    }

    common_flydelta_arm_batch_request request;
    request.arms.resize(value.device_diagnostics ? 2 :
        value.single_sparse_device ? 1 : static_cast<size_t>(value.arm_count));
    for (size_t index = 0; index < request.arms.size(); ++index) {
        auto & arm = request.arms[index];
        arm.job_id = "flydelta://job/cvec-batch-smoke";
        arm.arm_id = "flydelta://arm/cvec-batch-smoke/" + std::to_string(index);
        // Both arms intentionally share the same immutable prompt/context
        // identity.  The overlays are the varying dimension; distinct
        // context refs would correctly partition the logical wave into scalar
        // batches under the host compatibility contract.
        arm.context_ref = "context://cvec-batch-smoke";
        arm.fixture_ref = "fixture://cvec-batch-smoke";
        arm.intervention_ref = "flydelta://artifact/cvec-batch-smoke/" + std::to_string(index);
        arm.layer_indices = {static_cast<uint32_t>(cvec_layer)};
        arm.coefficients = {1.0f};
        arm.alpha = value.device_diagnostics && index == 0
            ? 0.0f : static_cast<float>(index) * 0.0002f;
        arm.apply_overlay = !value.device_diagnostics || index != 0;
        arm.fresh_context = true;
        arm.request_generation = !value.device_diagnostics;
        arm.request_teacher_forced_margin = !value.device_diagnostics;
        arm.max_generated_tokens = static_cast<size_t>(value.n_predict);
        if (value.device_diagnostics) {
            arm.request_device_diagnostics = true;
            arm.diagnostics_position = -1;
            if (index == 1) {
                arm.diagnostics_baseline_arm_id =
                    "flydelta://arm/cvec-batch-smoke/0";
            }
        }
    }

    common_flydelta_arm_batch_result result;
    for (int warmup = 0; warmup < value.warmup_count; ++warmup) {
        common_flydelta_arm_batch_result warm_result;
        if (!common_flydelta_run_bounded_arm_batch(*model_host, request, warm_result, error)) {
            std::cerr << "production FlyDelta warmup failed: " << error << '\n';
            return 1;
        }
    }
    double measured_wall_ms = 0.0;
    double measured_generation_ms = 0.0;
    double measured_teacher_forced_ms = 0.0;
    double measured_finalize_ms = 0.0;
    for (int repeat = 0; repeat < value.repeat_count; ++repeat) {
        if (!common_flydelta_run_bounded_arm_batch(*model_host, request, result, error)) {
            std::cerr << "production FlyDelta batch execution failed: " << error << '\n';
            return 1;
        }
        measured_wall_ms += result.execution_stats.wall_ms;
        measured_generation_ms += result.execution_stats.model_ms;
        measured_teacher_forced_ms += result.execution_stats.teacher_forced_ms;
        measured_finalize_ms += result.execution_stats.finalize_ms;
        std::cout << "cvec_smoke_repeat index=" << repeat
                  << " wall_ms=" << result.execution_stats.wall_ms
                  << " generation_ms=" << result.execution_stats.model_ms
                  << " teacher_forced_ms=" << result.execution_stats.teacher_forced_ms
                  << " finalize_ms=" << result.execution_stats.finalize_ms
                  << " physical_batches=" << result.execution_stats.physical_batch_count
                  << " largest_physical_batch=" << result.execution_stats.largest_physical_batch
                  << '\n';
    }
    if (value.device_diagnostics) {
        const auto & baseline = captured_rows->at(request.arms[0].arm_id);
        const auto & overlay = captured_rows->at(request.arms[1].arm_id);
        if (baseline.size() != overlay.size() || baseline.empty()) {
            std::cerr << "paired sequence captures have incompatible shapes\n";
            return 1;
        }
        double squared = 0;
        for (size_t i = 0; i < baseline.size(); ++i) {
            const double shift = overlay[i] - baseline[i];
            squared += shift * shift;
        }
        const double norm = std::sqrt(squared);
        std::cout << "cvec_smoke_host_capture_shift norm=" << norm << '\n';
        if (std::abs(norm - 0.0002) > 1e-6) {
            std::cerr << "sequence-specific host captures disagree with the applied overlay\n";
            return 1;
        }
    }

    for (size_t index = 0; index < result.arms.size(); ++index) {
        if (!result.arms[index].executed) {
            std::cerr << "cvec batch arm " << index << " did not execute\n";
            return 1;
        }
        if (!value.device_diagnostics && !result.arms[index].margin_available) {
            std::cerr << "cvec batch arm " << index << " has no teacher-forced margin\n";
            return 1;
        }
        if (!value.device_diagnostics) {
            std::cout << "cvec_smoke_teacher_margin arm="
                      << std::quoted(result.arms[index].arm_id)
                      << " total=" << std::setprecision(9)
                      << result.arms[index].margin_total
                      << " normalized=" << result.arms[index].margin_normalized << '\n';
        }
    }
    if (value.device_diagnostics &&
            (result.arms.size() != 2 || !result.arms[1].geometry_available ||
             !result.arms[1].execution_metrics.device_reduction_used)) {
        std::cerr << "device diagnostics smoke did not receive a paired non-CPU reduction\n";
        return 1;
    }
    if (value.factorized && !value.single_sparse_device && !value.device_diagnostics) {
        const auto found = std::find_if(result.arms.begin(), result.arms.end(),
            [](const auto & arm) {
                return arm.execution_metrics.factorized_binding_used &&
                    arm.execution_metrics.overlay_representation == "factorized";
            });
        if (found == result.arms.end()) {
            std::cerr << "factorized cvec was requested but no arm used the factorized graph binding\n";
            return 1;
        }
    }

    const auto & reported_arm = value.device_diagnostics && result.arms.size() > 1
        ? result.arms[1] : result.arms.front();
    if (value.device_diagnostics &&
            (!reported_arm.geometry_available || reported_arm.shift_norm <= 0.0f)) {
        std::cerr << "nonzero overlay did not produce a downstream shift against the matched zero arm\n";
        return 1;
    }
    if (value.device_diagnostics) {
        std::cout << "cvec_smoke_device_geometry shift_norm=" << reported_arm.shift_norm
                  << " cosine=" << reported_arm.cosine
                  << " progress=" << reported_arm.progress
                  << " leakage=" << reported_arm.leakage << '\n';
    }
    std::cout << "flydelta_cvec_batch_model_smoke=passed\n"
              << "slots=" << config.context_key.n_parallel << "\n"
              << "requested_arms=" << request.arms.size() << "\n"
              << "warmups=" << value.warmup_count << "\n"
              << "repetitions=" << value.repeat_count << "\n"
              << "average_wall_ms=" << measured_wall_ms / value.repeat_count << "\n"
              << "average_generation_ms=" << measured_generation_ms / value.repeat_count << "\n"
              << "average_teacher_forced_ms=" << measured_teacher_forced_ms / value.repeat_count << "\n"
              << "average_finalize_ms=" << measured_finalize_ms / value.repeat_count << "\n"
              << "physical_batches=" << result.execution_stats.physical_batch_count << "\n"
              << "largest_physical_batch=" << result.execution_stats.largest_physical_batch << "\n"
              << "distinct_overlays=" << (value.single_sparse_device ? "not_applicable" : "yes") << "\n"
              << "sparse_overlays=yes\n"
              << "overlay_representation=" << reported_arm.execution_metrics.overlay_representation << "\n"
              << "active_layers=" << cvec_layer << "\n"
              << "results=" << result.arms.size() << "\n"
              << "execution_path="
              << common_flydelta_arm_execution_path_name(
                  result.arms.front().execution_metrics.execution_path) << "\n"
              << "batched_execution="
              << (result.arms.front().execution_metrics.batched_execution_used ? "yes" : "no") << "\n"
              << "sparse_device_binding="
              << (result.arms.front().execution_metrics.sparse_device_binding_used ? "yes" : "no") << "\n"
              << "model_ms=" << result.arms.front().execution_metrics.model_ms << "\n"
              << "overlay_bytes_to_device="
              << result.arms.front().execution_metrics.overlay_bytes_to_device << "\n"
              << "overlay_bytes_bound="
              << reported_arm.execution_metrics.overlay_bytes_bound << "\n"
              << "basis_bytes_bound="
              << reported_arm.execution_metrics.basis_bytes_bound << "\n"
              << "basis_host_copy_bytes="
              << reported_arm.execution_metrics.basis_host_copy_bytes << "\n"
              << "coefficient_bytes_bound="
              << reported_arm.execution_metrics.coefficient_bytes_bound << "\n"
              << "composed_overlay_resident_bytes="
              << reported_arm.execution_metrics.composed_overlay_resident_bytes << "\n"
              << "overlay_composition_us="
              << reported_arm.execution_metrics.overlay_composition_us << "\n"
              << "basis_upload_count="
              << reported_arm.execution_metrics.basis_upload_count << "\n"
              << "basis_cache_hit="
              << (reported_arm.execution_metrics.basis_cache_hit ? "yes" : "no") << "\n"
              << "overlay_cache_hit="
              << (reported_arm.execution_metrics.overlay_cache_hit ? "yes" : "no") << "\n"
              << "overlay_binding_ms="
              << reported_arm.execution_metrics.overlay_binding_ms << "\n"
              << "physical_batch_size="
              << reported_arm.execution_metrics.physical_batch_size << "\n"
              << "device_resident="
              << (reported_arm.execution_metrics.device_resident ? "yes" : "no") << "\n"
              << "factorized_binding_used="
              << (reported_arm.execution_metrics.factorized_binding_used ? "yes" : "no") << "\n"
              << "factorized_rank=" << factorized_rank << "\n"
              << "fallback_reason="
              << (reported_arm.execution_metrics.fallback_reason.empty()
                    ? "none" : reported_arm.execution_metrics.fallback_reason) << "\n"
              << "capture_bytes_to_host="
              << result.arms.front().execution_metrics.capture_bytes_to_host << "\n"
              << "mode="
              << common_flydelta_arm_execution_path_name(
                  reported_arm.execution_metrics.execution_path) << "\n"
              << "device_reduction_used="
              << (reported_arm.execution_metrics.device_reduction_used ? "yes" : "no") << "\n"
              << "geometry_available="
              << (reported_arm.geometry_available ? "yes" : "no") << "\n"
              << "cosine=" << reported_arm.cosine << "\n"
              << "progress=" << reported_arm.progress << "\n"
              << "leakage=" << reported_arm.leakage << "\n"
              << "shift_norm=" << reported_arm.shift_norm << "\n";
    return 0;
}
