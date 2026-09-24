#include "tools/agent/daemon/agent-daemon-adapter.h"
#include "tools/agent/daemon/agent-daemon-dispatcher.h"
#include "tools/agent/daemon/agent-daemon-service.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/runtime/agent-model-loaders.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"

#include "agent/adaptation/flydelta/flydelta-artifact-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-queue.h"
#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

constexpr const char * kDefaultConfig =
    "docs/examples/agent-host-config-flydelta-intel-tiny.json";
constexpr const char * kTokenizerFingerprint = "daemon:tokenizer-unspecified-v1";
constexpr const char * kTemplateFingerprint = "daemon:template-unspecified-v1";
constexpr const char * kExecutionContextFingerprint = "daemon:server-context-v1";

struct smoke_trace_state {
    std::mutex mutex;
    std::string worker_trace_json;
};

struct probe_result {
    std::string baseline;
    std::string candidate;
};

struct candidate_material {
    common_flydelta_artifact artifact;
    common_flydelta_sideband_manifest manifest;
    std::string candidate_id;
    std::string artifact_uri;
    std::string expected_contains;
};

bool put_json_resource(
        agent_resource_store & resources,
        const std::string & name,
        const std::string & text,
        std::string & uri,
        std::string & error) {
    agent_resource_put_request request;
    request.name = name;
    request.description = "model-backed FlyDelta daemon smoke resource";
    request.mime_type = "application/json";
    request.text = text;
    request.scope = common_runtime_resource_scope::session;
    request.namespace_id = "default-namespace";
    request.session_id = "default-session";
    request.project_id = "llama-agent-model-smoke";
    request.source_provider = "llama-agent-smoke";
    request.source_tool = "flydelta.model_backed_smoke";
    agent_resource_descriptor descriptor;
    if (!resources.put_text(request, descriptor, error)) return false;
    uri = descriptor.uri;
    return !uri.empty();
}

std::string first_unique_word(const std::string & candidate, const std::string & baseline) {
    size_t index = 0;
    while (index < candidate.size()) {
        while (index < candidate.size() && !std::isalnum(static_cast<unsigned char>(candidate[index]))) ++index;
        const size_t begin = index;
        while (index < candidate.size() && std::isalnum(static_cast<unsigned char>(candidate[index]))) ++index;
        if (index > begin && index - begin >= 2) {
            const std::string word = candidate.substr(begin, index - begin);
            if (baseline.find(word) == std::string::npos) return word;
        }
    }
    return {};
}

std::shared_ptr<const common_flydelta_activation_result> make_activation(
        const std::vector<float> & values,
        int32_t layer,
        const std::string & candidate_id,
        const std::string & artifact_id,
        const std::string & profile,
        const std::string & layout,
        size_t n_embd,
        size_t n_layers,
        float alpha,
        std::string & error) {
    common_flydelta_activation_request request;
    request.candidate_id = candidate_id;
    request.artifact_id = artifact_id;
    request.model_profile_fingerprint = profile;
    request.capture_layout_revision = layout;
    request.model_n_embd = n_embd;
    request.model_n_layers = n_layers;
    request.il_start = 1;
    request.il_end = static_cast<int32_t>(n_layers - 1);
    request.directions = {{layer, values}};
    request.coefficients = {1.0f};
    request.gate_request.explicit_opt_in = true;
    request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
    request.gate_request.basis_available = true;
    request.gate_request.familiarity = 1.0f;
    request.gate_request.novelty = 0.0f;
    request.gate_request.requested_scale = alpha;
    common_flydelta_gate_config gate;
    gate.enabled = true;
    gate.max_scale = 1.0f;
    common_flydelta_activation_result result;
    if (!common_flydelta_prepare_activation(
            gate, request, 64U * 1024U * 1024U, result, error)) return {};
    return std::make_shared<const common_flydelta_activation_result>(std::move(result));
}

bool probe_candidate(
        const std::shared_ptr<common_agent_server_context_host> & host,
        const std::vector<float> & values,
        int32_t layer,
        const std::string & candidate_id,
        const std::string & profile,
        const std::string & layout,
        size_t n_embd,
        size_t n_layers,
        int n_threads,
        int n_predict,
        probe_result & output,
        std::string & error) {
    output = {};
    common_agent_server_flydelta_binding binding;
    binding.primitives.generation = true;
    binding.primitives.overlay = true;
    std::mutex output_mutex;
    binding.prepare_arm = [=](
            const common_flydelta_arm_request & arm,
            common_agent_generation_request & request,
            std::string & prepare_error) {
        request = {};
        request.purpose = common_agent_generation_purpose::draft;
        request.options.n_predict = n_predict;
        request.options.n_threads = n_threads;
        request.messages = {{"user", "Reply with exactly PASS."}};
        if (arm.apply_overlay) {
            request.flydelta_activation = make_activation(
                values, layer, candidate_id, arm.intervention_ref, profile, layout,
                n_embd, n_layers, arm.alpha, prepare_error);
            if (!request.flydelta_activation) return false;
        }
        return true;
    };
    binding.finalize_arm = [&](
            const common_flydelta_arm_request & arm,
            const common_agent_generation_result & generation,
            common_flydelta_arm_result & result,
            std::string & finalize_error) {
        result = {};
        result.arm_id = arm.arm_id;
        result.executed = common_agent_generation_succeeded(generation);
        result.generation_available = true;
        result.quality = static_cast<float>(generation.decoded_tokens);
        result.generation_ref = arm.arm_id + ":generation";
        if (arm.apply_overlay) {
            std::lock_guard<std::mutex> lock(output_mutex);
            output.candidate = generation.content;
        } else {
            std::lock_guard<std::mutex> lock(output_mutex);
            output.baseline = generation.content;
        }
        if (!result.executed) {
            finalize_error = generation.error_message;
            return false;
        }
        return common_flydelta_arm_result_validate(result, finalize_error);
    };

    common_flydelta_arm_batch_request request;
    request.batch_id = candidate_id + ":probe";
    request.wave_id = "model-backed-probe";
    for (const bool overlay : {false, true}) {
        common_flydelta_arm_request arm;
        arm.job_id = candidate_id + ":probe";
        arm.wave_id = request.wave_id;
        arm.proposal_index = overlay ? 1 : 0;
        arm.arm_id = candidate_id + (overlay ? ":candidate" : ":baseline");
        arm.context_ref = "probe-context";
        arm.fixture_ref = "probe-fixture";
        arm.intervention_ref = candidate_id;
        arm.layer_indices = {static_cast<uint32_t>(layer)};
        arm.coefficients = {1.0f};
        arm.alpha = overlay ? 1.0f : 0.0f;
        arm.apply_overlay = overlay;
        arm.fresh_context = true;
        arm.request_generation = true;
        arm.max_generated_tokens = static_cast<size_t>(n_predict);
        request.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result result;
    return common_agent_server_context_host_run_flydelta_arm_batch(
        host, binding, false, request, result, error);
}

bool persist_worker_report(
        const std::shared_ptr<common_learning_lifecycle_store> & lifecycle,
        const std::shared_ptr<smoke_trace_state> & trace_state,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_report & report,
        std::string & error) {
    error.clear();
    {
        std::lock_guard<std::mutex> lock(trace_state->mutex);
        trace_state->worker_trace_json = report.trace_json;
    }
    if (!lifecycle || !report.has_evaluation_report) return true;
    common_flydelta_lifecycle_event_context context;
    context.event_id = job.id + ":evaluation";
    context.idempotency_key = context.event_id;
    context.source_id = "daemon-flydelta-model-smoke";
    context.scope = job.seed.scope;
    context.content_hash = report.evaluation_report.revision_id + ":" + report.evaluation_report.candidate_id;
    context.created_at = job.id;
    if (!common_flydelta_append_evaluation_lifecycle(
            *lifecycle, context, report.evaluation_report,
            report.evaluation_fixture_results, error)) return false;
    if (!report.counterfactual_reports.empty()) {
        common_flydelta_promotion_summary summary;
        common_flydelta_promotion_policy policy;
        if (!common_flydelta_promotion_summary_from_reports(
                job.id + ":promotion-summary", report.counterfactual_reports,
                policy, summary, error)) return false;
        common_flydelta_lifecycle_event_context summary_context;
        summary_context.event_id = summary.id;
        summary_context.idempotency_key = summary.id;
        summary_context.source_id = "daemon-flydelta-model-smoke";
        summary_context.scope = job.seed.scope;
        summary_context.content_hash = summary.candidate_id + ":" + summary.id;
        summary_context.created_at = job.id;
        if (!common_flydelta_append_promotion_summary_lifecycle(
                *lifecycle, summary_context, summary, error)) return false;
    }
    for (const auto & counterfactual : report.counterfactual_reports) {
        common_flydelta_lifecycle_event_context result_context;
        result_context.event_id = job.id + ":counterfactual:" + counterfactual.experiment_id + ":" + counterfactual.fixture_id;
        result_context.idempotency_key = result_context.event_id;
        result_context.source_id = "daemon-flydelta-model-smoke";
        result_context.scope = job.seed.scope;
        result_context.content_hash = counterfactual.experiment_id;
        result_context.created_at = job.id;
        if (!common_flydelta_append_counterfactual_lifecycle(
                *lifecycle, result_context, counterfactual, error)) return false;
    }
    return true;
}

bool seed_candidate(
        common_flydelta_sideband_review_store & reviews,
        common_flydelta_sideband_registry & registry,
        common_flydelta_sideband_manifest manifest,
        const std::string & evaluation_revision,
        const std::string & candidate_id,
        std::string & error) {
    common_flydelta_sideband_review admit;
    admit.event_id = candidate_id + ":review:admit";
    admit.actor_id = "model-backed-smoke";
    admit.reason = "model-backed daemon smoke candidate";
    admit.source = common_flydelta_review_source::operator_action;
    admit.action = common_flydelta_review_action::admit_experimental;
    admit.manifest = manifest;
    if (!reviews.apply_and_append(registry, admit, true, error)) return false;
    manifest.status = common_flydelta_sideband_status::candidate;
    common_flydelta_sideband_review promote;
    promote.event_id = candidate_id + ":review:candidate";
    promote.actor_id = "model-backed-smoke";
    promote.evaluation_revision = evaluation_revision;
    promote.reason = "model-backed daemon smoke candidate review";
    promote.action = common_flydelta_review_action::promote_to_candidate;
    promote.manifest = manifest;
    return reviews.apply_and_append(registry, promote, true, error);
}

bool load_config(const std::string & path, daemon_options & options, std::string & error) {
    agent_host_config config;
    if (!load_agent_host_config(path, config, error)) return false;
    apply_agent_host_config_to_daemon_options(config, options);
    return true;
}

bool wait_for_report(
        const common_learning_lifecycle_store & lifecycle,
        const std::string & candidate_id,
        common_flydelta_evaluation_report & report,
        std::vector<common_flydelta_evaluation_fixture_result> & fixtures,
        std::string & error) {
    for (size_t attempt = 0; attempt < 600; ++attempt) {
        if (common_flydelta_load_evaluation_report(
                lifecycle, candidate_id, report, &fixtures, error)) return true;
        error.clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    error = "timed out waiting for model-backed FlyDelta evaluation persistence";
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string config_path = argc > 1 ? argv[1] : kDefaultConfig;
    daemon_options options;
    std::string error;
    if (!load_config(config_path, options, error)) {
        std::fprintf(stderr, "model-backed FlyDelta smoke config failed: %s\n", error.c_str());
        return 2;
    }
    if (options.model_profile.empty() || !options.adaptation_flydelta_enabled) {
        std::fprintf(stderr, "model-backed FlyDelta smoke requires a model profile and FlyDelta\n");
        return 2;
    }
    const auto smoke_suffix = "model-admin-smoke-" + std::to_string(static_cast<long long>(getpid()));
    options.n_predict = 4;
    options.n_gpu_layers = std::max(options.n_gpu_layers, 99);
    options.adaptation_flydelta_queue_path = "var/agent/flydelta/" + smoke_suffix + "-queue";
    options.adaptation_flydelta_lifecycle_path = "var/agent/flydelta/" + smoke_suffix + "-lifecycle.cozo";
    options.adaptation_transaction_path = "var/agent/flydelta/" + smoke_suffix + "-adaptation.cozo";
    options.adaptation_flydelta_worker_count = 1;
    options.worker_count = 1;
    options.queue_capacity = 8;
    options.inference_max_active = 1;
    options.agent_trace = true;

    common_agent_daemon_runtime runtime;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        std::fprintf(stderr, "model-backed FlyDelta environment failed: %s\n", error.c_str());
        return 1;
    }
    if (!runtime.flydelta_model_handle || !runtime.flydelta_model_host || !runtime.resource_store) {
        std::fprintf(stderr, "model-backed FlyDelta runtime did not expose resident host/resources\n");
        return 1;
    }
    if (!runtime.flydelta_model_adapter) {
        runtime.flydelta_model_adapter = common_flydelta_model_adapter_from_host(
            *runtime.flydelta_model_host, error);
        if (!runtime.flydelta_model_adapter) {
            std::fprintf(stderr, "model-backed FlyDelta adapter failed: %s\n", error.c_str());
            return 1;
        }
    }
    runtime.flydelta_job_enqueue = [queue_root = std::filesystem::path(options.adaptation_flydelta_queue_path)](
            const common_flydelta_experiment_job & job, std::string & enqueue_error) {
        return common_flydelta_experiment_queue_enqueue(
            queue_root, job, common_flydelta_experiment_queue_limits{}, enqueue_error);
    };

    const auto loaded = common_agent_runtime_loaded_model_cast(runtime.flydelta_model_handle->model);
    if (!loaded || !loaded->server_context_host) {
        std::fprintf(stderr, "model-backed FlyDelta runtime lost resident server-context host\n");
        return 1;
    }
    const auto * context = loaded->server_context_host->server().get_llama_context();
    const auto * model = context == nullptr ? nullptr : llama_get_model(context);
    if (model == nullptr) {
        std::fprintf(stderr, "model-backed FlyDelta resident host did not expose a model\n");
        return 1;
    }
    const size_t n_embd = static_cast<size_t>(llama_model_n_embd(model));
    const size_t n_layers = static_cast<size_t>(llama_model_n_layer(model));
    if (n_embd == 0 || n_layers < 3) return 1;
    const std::string profile = options.adaptation_flydelta_model_profile_fingerprint;
    const std::string layout = options.adaptation_flydelta_capture_layout_revision;

    const std::vector<std::pair<int32_t, float>> variants = {
        {1, 8.0f}, {1, -8.0f}, {1, 32.0f}, {1, -32.0f},
        {std::min<int32_t>(2, static_cast<int32_t>(n_layers - 1)), 8.0f},
        {std::min<int32_t>(2, static_cast<int32_t>(n_layers - 1)), -8.0f},
    };
    candidate_material material;
    probe_result selected_probe;
    for (size_t variant = 0; variant < variants.size(); ++variant) {
        const auto [layer, magnitude] = variants[variant];
        std::vector<float> values(n_embd, 0.0f);
        // The daemon artifact parser normalizes steering directions before
        // composing the static overlay. Probe the same unit direction so the
        // calibration result is evidence for the production artifact path.
        values.front() = magnitude > 0.0f ? 1.0f : -1.0f;
        const std::string candidate_id = "flydelta://sideband/model-admin-smoke-" + std::to_string(static_cast<long long>(getpid())) + "-" + std::to_string(variant);
        probe_result probe;
        if (!probe_candidate(
                loaded->server_context_host, values, layer, candidate_id, profile,
                layout, n_embd, n_layers, options.n_threads, options.n_predict,
                probe, error)) {
            std::fprintf(stderr, "model-backed FlyDelta probe failed: %s\n", error.c_str());
            return 1;
        }
        const std::string expected = first_unique_word(probe.candidate, probe.baseline);
        std::printf("flydelta_model_probe=%s\n", json{
            {"variant", variant}, {"layer", layer}, {"baseline", probe.baseline},
            {"candidate", probe.candidate}, {"candidate_differs", probe.candidate != probe.baseline},
            {"unique_expected", expected},
        }.dump().c_str());
        std::fflush(stdout);
        if (!expected.empty()) {
            material.candidate_id = candidate_id;
            material.expected_contains = expected;
            selected_probe = std::move(probe);
            std::vector<common_flydelta_artifact_direction> basis;
            basis.push_back({layer, std::move(values)});
            common_flydelta_encoder encoder({0x7a11c0deULL, n_embd, 64, 4, 4});
            common_flydelta_memory_config memory{64, 1, 1.0f};
            common_flydelta_compatibility compatibility;
            compatibility.base_model_id = std::filesystem::path(options.model).filename().string();
            compatibility.base_model_fingerprint = profile;
            compatibility.tokenizer_fingerprint = kTokenizerFingerprint;
            compatibility.template_fingerprint = kTemplateFingerprint;
            compatibility.architecture = "model-backed-daemon-smoke";
            compatibility.inference_layout_revision = layout;
            if (!common_flydelta_build_experimental_artifact(
                    material.candidate_id, 1, encoder.config(), memory, compatibility,
                    std::vector<float>(64, 0.0f), n_embd, n_layers, 1,
                    static_cast<int32_t>(n_layers - 1), basis, material.artifact, error)) {
                std::fprintf(stderr, "model-backed FlyDelta artifact build failed: %s\n", error.c_str());
                return 1;
            }
            const std::string artifact_json = common_flydelta_artifact_to_json(material.artifact);
            if (!put_json_resource(
                    *runtime.resource_store, "model-admin-artifact-" + std::to_string(variant) + ".json",
                    artifact_json, material.artifact_uri, error)) {
                std::fprintf(stderr, "model-backed artifact resource failed: %s\n", error.c_str());
                return 1;
            }
            material.manifest.id = material.candidate_id;
            material.manifest.status = common_flydelta_sideband_status::experimental;
            material.manifest.artifact_path = material.artifact_uri;
            material.manifest.artifact_hash = material.artifact.content_hash;
            material.manifest.compatibility = compatibility;
            material.manifest.applicability.behavior_key = "flydelta/pre-canary-evaluation";
            material.manifest.applicability.scope_fingerprint = "default-namespace:default-session";
            material.manifest.applicability.verifier_revision = "verifier:model-admin-smoke-v1";
            material.manifest.model_n_embd = n_embd;
            material.manifest.model_n_layers = n_layers;
            material.manifest.il_end = static_cast<int32_t>(n_layers - 1);
            break;
        }
    }
    if (material.candidate_id.empty()) {
        std::fprintf(stderr, "model-backed FlyDelta probe found no candidate-only verifier token\n");
        return 1;
    }
    const std::string evaluation_revision = "evaluation:model-admin-smoke-v1";
    if (!seed_candidate(
            *runtime.flydelta_sideband_review_store, *runtime.flydelta_sideband_registry,
            material.manifest, evaluation_revision, material.candidate_id, error)) {
        std::fprintf(stderr, "model-backed FlyDelta candidate seed failed: %s\n", error.c_str());
        return 1;
    }
    std::string context_uri;
    if (!put_json_resource(
            *runtime.resource_store, "model-admin-context.json",
            json{{"prompt", "Reply with exactly PASS."}, {"n_predict", options.n_predict}}.dump(),
            context_uri, error)) {
        std::fprintf(stderr, "model-backed context resource failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<std::string> fixture_uris;
    for (size_t index = 0; index < 8; ++index) {
        std::string fixture_uri;
        if (!put_json_resource(
                *runtime.resource_store, "model-admin-fixture-" + std::to_string(index) + ".json",
                json{{"expected_contains", material.expected_contains}}.dump(), fixture_uri, error)) {
            std::fprintf(stderr, "model-backed fixture resource failed: %s\n", error.c_str());
            return 1;
        }
        fixture_uris.push_back(std::move(fixture_uri));
    }
    std::string suite_uri;
    const char * suite_names[] = {"intended", "holdout", "retention", "agent_regression"};
    json suite = json{{"kind", "flydelta_evaluation_suite"}};
    suite["revision"] = evaluation_revision;
    suite["fixtures"] = json::array();
    for (size_t index = 0; index < fixture_uris.size(); ++index) {
        suite["fixtures"].push_back({
            {"suite_kind", suite_names[index % 4]},
            {"fixture_ref", fixture_uris[index]},
            {"context_ref", context_uri},
        });
    }
    if (!put_json_resource(*runtime.resource_store, "model-admin-suite.json", suite.dump(), suite_uri, error)) {
        std::fprintf(stderr, "model-backed suite resource failed: %s\n", error.c_str());
        return 1;
    }
    auto trace_state = std::make_shared<smoke_trace_state>();
    const auto lifecycle_store = runtime.flydelta_review_lifecycle_store;
    const auto review_store = runtime.flydelta_sideband_review_store;
    const auto registry = runtime.flydelta_sideband_registry;
    common_agent_daemon_flydelta_worker_config worker_config;
    worker_config.enabled = true;
    worker_config.worker_count = 1;
    worker_config.queue_root = options.adaptation_flydelta_queue_path;
    worker_config.model_adapter = runtime.flydelta_model_adapter;
    worker_config.poll_interval = std::chrono::milliseconds(5);
    worker_config.persist_completed_report = [lifecycle = runtime.flydelta_review_lifecycle_store, trace_state] (
            const common_flydelta_experiment_job & job,
            const common_flydelta_experiment_worker_report & report,
            std::string & persist_error) {
        return persist_worker_report(lifecycle, trace_state, job, report, persist_error);
    };

    int result_code = 0;
    {
        const size_t dispatcher_worker_count = std::max<size_t>(2, options.worker_count);
        common_agent_daemon_dispatcher dispatcher(
            std::move(runtime), options.queue_capacity, dispatcher_worker_count,
            std::move(worker_config));
        const auto execute_admin = [&](const json & request, const char * expected_event,
                common_agent_daemon_command_result & result) {
            common_agent_daemon_command command;
            std::string command_error;
            if (!parse_agent_daemon_command(
                    request, options, common_agent_runtime_host_mode::chat,
                    command, command_error)) {
                error = command_error.empty() ? result.error : command_error;
                return false;
            }
            if (!dispatcher.execute(command, result, command_error)) {
                error = command_error.empty() ? result.error : command_error;
                return false;
            }
            if (result.event != expected_event) {
                error = "unexpected model-backed FlyDelta event: " + result.event;
                return false;
            }
            std::printf("flydelta_model_admin_trace=%s\n", json{
                {"operation", request.value("command", "")}, {"event", result.event},
                {"target_request_id", result.target_request_id},
                {"payload", result.payload_json.empty() ? json::object() : json::parse(result.payload_json)},
            }.dump().c_str());
            return true;
        };
        common_agent_daemon_command_result result;
        const json evaluate = {
            {"request_id", "model-admin-evaluate"}, {"command", "flydelta.evaluate_candidate"},
            {"candidate_id", material.candidate_id}, {"candidate_manifest_ref", material.artifact_uri},
            {"suite_ref", suite_uri}, {"evaluation_revision", evaluation_revision},
            {"verifier_revision", "verifier:model-admin-smoke-v1"},
            {"model_profile_fingerprint", profile}, {"tokenizer_fingerprint", kTokenizerFingerprint},
            {"template_fingerprint", kTemplateFingerprint}, {"execution_context_fingerprint", kExecutionContextFingerprint},
            {"limits", {{"max_fixtures", 8}, {"max_model_calls", 16}, {"max_generated_tokens", options.n_predict}, {"max_retries", 0}}},
        };
        common_flydelta_evaluation_report report;
        std::vector<common_flydelta_evaluation_fixture_result> fixtures;
        if (!execute_admin(evaluate, "flydelta.evaluation.queued", result) ||
                !wait_for_report(*lifecycle_store, material.candidate_id, report, fixtures, error)) {
            std::fprintf(stderr, "model-backed FlyDelta evaluation failed: %s\n", error.c_str());
            result_code = 1;
        } else {
            common_flydelta_promotion_summary summary;
            if (!common_flydelta_load_promotion_summary(
                    *lifecycle_store, material.candidate_id, summary, error) ||
                    summary.status != common_flydelta_candidate_status::eligible) {
                if (error.empty()) error = "model-backed evaluation did not produce an eligible promotion summary";
                std::fprintf(stderr, "model-backed FlyDelta promotion gate failed: %s\n", error.c_str());
                result_code = 1;
            } else if (!execute_admin(
                    json{{"request_id", "model-admin-get-evaluation"}, {"command", "flydelta.get_evaluation"}, {"candidate_id", material.candidate_id}},
                    "flydelta.evaluation.loaded", result) ||
                !execute_admin(
                    json{{"request_id", "model-admin-get-summary"}, {"command", "flydelta.get_promotion_summary"}, {"candidate_id", material.candidate_id}},
                    "flydelta.promotion_summary.loaded", result) ||
                !execute_admin(
                    json{{"request_id", "model-admin-review"}, {"command", "flydelta.review_candidate"}, {"candidate_id", material.candidate_id}, {"decision", "approve_canary"}, {"evaluation_revision", evaluation_revision}, {"reason", "model-backed daemon smoke approval"}, {"actor_id", "smoke-operator"}},
                    "flydelta.review.recorded", result) ||
                !execute_admin(
                    json{{"request_id", "model-admin-stage"}, {"command", "flydelta.stage_canary"}, {"candidate_id", material.candidate_id}, {"explicit_host_approval", true}, {"reason", "model-backed daemon smoke canary"}, {"actor_id", "smoke-operator"}},
                    "flydelta.canary.staged", result)) {
                std::fprintf(stderr, "model-backed FlyDelta admin chain failed: %s\n", error.c_str());
                result_code = 1;
            }
        }
    }

    if (result_code == 0) {
        const auto current = registry->list().find(material.candidate_id);
        common_flydelta_sideband_registry replayed;
        if (current == registry->list().end() ||
                current->second.status != common_flydelta_sideband_status::canary ||
                !review_store->replay(replayed, error) ||
                replayed.list().at(material.candidate_id).status != common_flydelta_sideband_status::canary) {
            std::fprintf(stderr, "model-backed durable canary replay failed: %s\n", error.c_str());
            result_code = 1;
        } else {
            std::string trace_json;
            {
                std::lock_guard<std::mutex> lock(trace_state->mutex);
                trace_json = trace_state->worker_trace_json;
            }
            const auto trace = json::parse(trace_json);
            if (trace.value("kind", "") != "flydelta_trace" ||
                    trace.value("phase", "") != "evaluation" ||
                    trace.value("model_evaluations", 0U) == 0U) {
                std::fprintf(stderr, "model-backed worker trace is incomplete: %s\n", trace_json.c_str());
                result_code = 1;
            } else {
                std::printf("flydelta_model_worker_trace=%s\n", trace_json.c_str());
                std::printf("flydelta_model_admin_smoke=passed category=model-backed-functional expected=%s\n", material.expected_contains.c_str());
            }
        }
    }
    return result_code;
}
