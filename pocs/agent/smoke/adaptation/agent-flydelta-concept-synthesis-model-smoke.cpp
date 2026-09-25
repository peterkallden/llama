#include "tools/agent/daemon/agent-daemon-adapter.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/runtime/agent-model-loaders.h"
#include "tools/server/server-context.h"

#include "agent/adaptation/flydelta/flydelta-collection.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-teaching-material.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/runtime/model-catalog.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

constexpr const char * kDefaultConfig =
    "docs/examples/agent-host-config-flydelta-full.json";
constexpr const char * kTokenizerFingerprint = "daemon:tokenizer-unspecified-v1";
constexpr const char * kTemplateFingerprint = "daemon:template-unspecified-v1";
constexpr const char * kExecutionContextFingerprint = "daemon:server-context-v1";

struct smoke_args {
    std::string config_path = kDefaultConfig;
    std::string model_path;
    int threads = 4;
    int n_predict = 8;
    int n_gpu_layers = 99;
};

int fail(const std::string & message) {
    std::cerr << "flydelta_concept_synthesis_model_smoke=failed reason=" << message << '\n';
    return 1;
}

bool parse_args(int argc, char ** argv, smoke_args & args, std::string & error) {
    for (int i = 1; i < argc; ++i) {
        const std::string value = argv[i];
        const auto next = [&](const char * name, std::string & output) {
            if (i + 1 >= argc) {
                error = std::string("missing value for ") + name;
                return false;
            }
            output = argv[++i];
            return true;
        };
        if (value == "--config") {
            if (!next("--config", args.config_path)) return false;
        } else if (value == "--model") {
            if (!next("--model", args.model_path)) return false;
        } else if (value == "--threads") {
            std::string text;
            if (!next("--threads", text)) return false;
            args.threads = std::stoi(text);
        } else if (value == "--n-predict") {
            std::string text;
            if (!next("--n-predict", text)) return false;
            args.n_predict = std::stoi(text);
        } else if (value == "--n-gpu-layers") {
            std::string text;
            if (!next("--n-gpu-layers", text)) return false;
            args.n_gpu_layers = std::stoi(text);
        } else if (value == "--help" || value == "-h") {
            std::cout << "usage: " << argv[0]
                      << " --model PATH [--config PATH] [--threads N]"
                      << " [--n-predict N] [--n-gpu-layers N]\n";
            std::exit(0);
        } else {
            error = "unknown argument: " + value;
            return false;
        }
    }
    if (args.model_path.empty()) {
        const char * environment_model = std::getenv("LLAMA_AGENT_MODEL");
        if (environment_model != nullptr) args.model_path = environment_model;
    }
    if (args.model_path.empty()) {
        error = "a model is required through --model or LLAMA_AGENT_MODEL";
        return false;
    }
    if (args.threads < 1 || args.n_predict < 1 || args.n_gpu_layers < 0) {
        error = "threads, n-predict and n-gpu-layers have invalid values";
        return false;
    }
    return true;
}

bool load_config(const std::string & path, daemon_options & options, std::string & error) {
    agent_host_config config;
    if (!load_agent_host_config(path, config, error)) return false;
    apply_agent_host_config_to_daemon_options(config, options);
    return true;
}

void configure_model_catalog(daemon_options & options, const smoke_args & args) {
    options.model = args.model_path;
    options.model_profile = "agent-default";
    options.n_threads = args.threads;
    options.n_predict = args.n_predict;
    options.n_gpu_layers = args.n_gpu_layers;
    options.context_size = std::max(options.context_size, 2048);

    options.model_catalog = {};
    options.model_catalog.directory = std::filesystem::path(args.model_path).parent_path().string();
    options.model_catalog.default_profile = options.model_profile;
    options.model_catalog.max_loaded_generation_models = 1;
    common_agent_model_base_spec base;
    base.kind = "generation";
    base.backend = "server-context";
    base.path = std::filesystem::path(args.model_path).filename().string();
    base.load_policy = "resident";
    options.model_catalog.bases.emplace("smoke-model", std::move(base));
    common_agent_model_profile_spec profile;
    profile.base_model_id = "smoke-model";
    profile.context_size_tokens = static_cast<size_t>(options.context_size);
    profile.n_parallel = 2;
    profile.n_sequences = 2;
    profile.load_policy = "resident";
    options.model_catalog.profiles.emplace(options.model_profile, std::move(profile));
    options.adaptation_flydelta_model_profile_fingerprint = "profile:" + options.model_profile;
}

common_agent_scope smoke_scope() {
    common_agent_scope scope;
    scope.namespace_id = "concept-synthesis-smoke";
    scope.project_id = "flydelta-model-smoke";
    scope.session_id = "qwen-concept-synthesis";
    scope.turn_id = "turn-1";
    return scope;
}

bool put_json_resource(
        agent_resource_store & resources,
        const common_agent_scope & scope,
        const std::string & name,
        const json & value,
        std::string & uri,
        std::string & error) {
    agent_resource_put_request request;
    request.name = name;
    request.description = "portable FlyDelta concept synthesis smoke resource";
    request.mime_type = "application/json";
    request.text = value.dump();
    request.scope = common_runtime_resource_scope::session;
    request.namespace_id = scope.namespace_id;
    request.session_id = scope.session_id;
    request.project_id = scope.project_id;
    request.turn_id = scope.turn_id;
    request.source_provider = "llama-agent-smoke";
    request.source_tool = "flydelta.concept_synthesis_model_smoke";
    agent_resource_descriptor descriptor;
    if (!resources.put_text(request, descriptor, error)) return false;
    uri = descriptor.uri;
    if (uri.empty()) {
        error = "resource store returned an empty URI";
        return false;
    }
    agent_resource_read_authority authority;
    authority.namespace_id = scope.namespace_id;
    authority.session_id = scope.session_id;
    authority.project_id = scope.project_id;
    authority.turn_id = scope.turn_id;
    authority.now = static_cast<int64_t>(std::time(nullptr));
    std::string readback;
    if (!resources.read_text(uri, authority, 4U * 1024U * 1024U, readback, error)) {
        error = "resource read-back failed for " + name + ": " + error;
        return false;
    }
    return true;
}

common_flydelta_teaching_relation make_relation(
        const common_agent_scope & scope,
        const std::string & id,
        const std::string & baseline,
        const std::string & conditioned,
        const std::string & control,
        const std::string & verifier) {
    common_flydelta_teaching_relation relation;
    relation.id = "teaching://concept-synthesis/" + id;
    relation.teaching_key = "qwen.instruction-following.v1";
    relation.source = common_adaptation_evidence_source::procedure_blueprint;
    relation.behavior_key = "concept/qwen/instruction-following";
    relation.scope = scope;
    relation.task_fingerprint = "task:concept-synthesis:" + id;
    relation.baseline_ref = baseline;
    relation.conditioned_ref = conditioned;
    relation.control_ref = control;
    relation.verifier_ref = verifier;
    relation.evidence_ref = "evidence://concept-synthesis/" + id;
    relation.contrast_ref = "contrast://concept-synthesis/" + id;
    relation.procedure_ref = "procedure://qwen-instruction-following/v1";
    relation.blueprint_ref = "blueprint://qwen-instruction-following/v1";
    relation.status = common_flydelta_teaching_relation_status::resolved;
    relation.baseline_origin = common_flydelta_teaching_origin::host_derived;
    relation.conditioned_origin = common_flydelta_teaching_origin::host_derived;
    relation.control_origin = common_flydelta_teaching_origin::host_counterfactual;
    relation.confidence = 1.0f;
    relation.host_approved = true;
    return relation;
}

common_flydelta_experiment_collection_request make_search_request(
        const common_agent_scope & scope,
        const std::string & baseline_ref,
        const std::string & direction_ref,
        const std::string & group_ref,
        const std::string & profile,
        const std::vector<std::string> & delta_refs,
        const std::string & verifier_ref) {
    common_flydelta_experiment_collection_request request;
    request.enabled = true;
    request.kind = common_flydelta_experiment_job_kind::search_pipeline;
    request.evidence.id = "evidence://concept-synthesis/search";
    request.evidence.source = common_adaptation_evidence_source::procedure_blueprint;
    request.evidence.scope = scope;
    request.evidence.behavior_key = "concept/qwen/instruction-following";
    request.evidence.task_fingerprint = "task:concept-synthesis/search";
    request.evidence.baseline_ref = baseline_ref;
    request.evidence.candidate_ref = direction_ref;
    request.evidence.verifier_ref = verifier_ref;
    request.evidence.transaction_ids = {"transaction://concept-synthesis/search"};
    request.evidence.cause = common_learning_cause::host_contract;
    request.evidence.host_verified = true;
    request.behavior_key = request.evidence.behavior_key;
    request.model_profile_fingerprint = profile;
    request.tokenizer_fingerprint = kTokenizerFingerprint;
    request.template_fingerprint = kTemplateFingerprint;
    request.execution_context_fingerprint = kExecutionContextFingerprint;
    request.capture_manifest_ids = {"capture://concept-synthesis/search"};
    request.behavior_delta_ids = delta_refs;
    request.alpha_search.candidates = {0.01f, 0.02f};
    request.alpha_search.max_candidates = 2;
    request.code_revision = "concept-synthesis-model-smoke:v1";
    request.teaching_material_group_ref = group_ref;
    return request;
}

bool report_succeeded(
        const common_flydelta_experiment_worker_report & report,
        common_flydelta_experiment_job_kind kind,
        std::string & error) {
    if (report.state != common_flydelta_experiment_queue_state::succeeded) {
        const std::string detail = error.empty() ? report.safe_summary : error;
        error = std::string("worker did not succeed for ") +
            common_flydelta_experiment_job_kind_name(kind) +
            (detail.empty() ? std::string() : ": " + detail);
        return false;
    }
    if (report.completed_job.kind != kind) {
        error = "worker completed an unexpected FlyDelta job kind";
        return false;
    }
    return true;
}

void print_search_summary(
        const common_flydelta_experiment_worker_report & report,
        const char * phase) {
    json whirlpool = json::array();
    for (const auto & trace : report.trace.whirlpool) {
        json rounds = json::array();
        for (const auto & round : trace.rounds) {
            rounds.push_back({
                {"round", round.round},
                {"centre_before", round.centre_before},
                {"centre_after", round.centre_after},
                {"radius_before", round.radius_before},
                {"radius_after", round.radius_after},
                {"probed_layers", round.probed_layers},
                {"best_probe_layer", round.best_probe_layer},
                {"best_probe_score", round.best_probe_score},
            });
        }
        whirlpool.push_back({
            {"model_evaluations", trace.model_evaluations},
            {"best_trial_index", trace.best_trial_index},
            {"best_search_score", trace.best_search_score},
            {"final_centre", trace.final_centre},
            {"final_radius", trace.final_radius},
            {"rounds", std::move(rounds)},
        });
    }
    std::cout << "flydelta_concept_synthesis_search=" << json{
        {"phase", phase},
        {"state", common_flydelta_experiment_queue_state_name(report.state)},
        {"algorithm", "Whirlpool/BootstrapZoom via production search_pipeline"},
        {"arms", report.trace.arms.size()},
        {"model_evaluations", report.trace.model_evaluations},
        {"search_status", report.trace.search_status},
        {"next_action", common_flydelta_next_action_name(report.next_action)},
        {"bootstrap_state_ref", report.bootstrap_zoom_state_ref},
        {"search_state_ref", report.search_state_ref},
        {"whirlpool", std::move(whirlpool)},
    }.dump() << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    std::string error;
    if (!parse_args(argc, argv, args, error)) {
        if (error == "a model is required through --model or LLAMA_AGENT_MODEL") return 77;
        return fail(error);
    }

    daemon_options options;
    if (!load_config(args.config_path, options, error)) return fail("config: " + error);
    configure_model_catalog(options, args);

    const std::string suffix = "concept-synthesis-model-smoke-" +
        std::to_string(static_cast<long long>(getpid()));
    options.adaptation_capture = true;
    options.adaptation_collection_allowed = false;
    options.adaptation_flydelta_enabled = true;
    options.adaptation_flydelta_capture_candidates = true;
    options.adaptation_flydelta_worker_count = 2;
    options.worker_count = 4;
    options.inference_max_active = 1;
    options.queue_capacity = 8;
    options.agent_trace = true;
    options.adaptation_flydelta_queue_path = "var/agent/flydelta/" + suffix + "-queue";
    options.adaptation_flydelta_lifecycle_path = "var/agent/flydelta/" + suffix + "-lifecycle.cozo";
    options.adaptation_transaction_path = "var/agent/flydelta/" + suffix + "-adaptation.cozo";
    options.adaptation_transaction_backend = "cozo";
    options.adaptation_flydelta_lifecycle_backend = "cozo";

    common_agent_daemon_runtime runtime;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        return fail("daemon environment: " + error);
    }
    if (!runtime.flydelta_model_handle || !runtime.flydelta_model_host ||
            !runtime.flydelta_model_adapter || !runtime.resource_store ||
            !runtime.flydelta_teaching_material_runtime) {
        return fail("production daemon did not expose resident FlyDelta/server-context seams");
    }
    const auto loaded = common_agent_runtime_loaded_model_cast(runtime.flydelta_model_handle->model);
    if (!loaded || !loaded->server_context_host) {
        return fail("resident server-context host is unavailable");
    }
    const auto * context = loaded->server_context_host->server().get_llama_context();
    const auto * model = context == nullptr ? nullptr : llama_get_model(context);
    if (model == nullptr) return fail("resident llama model is unavailable");
    const size_t n_embd = static_cast<size_t>(llama_model_n_embd(model));
    const size_t n_layers = static_cast<size_t>(llama_model_n_layer(model));
    if (n_embd == 0 || n_layers < 3) return fail("resident model dimensions are incompatible");

    common_flydelta_evaluator_config evaluator_config;
    common_flydelta_evaluator_callbacks callbacks;
    if (!runtime.flydelta_model_host->register_evaluator(
            evaluator_config, callbacks, error)) {
        return fail("production evaluator registration: " + error);
    }
    if (!callbacks.run_concept_capture || !callbacks.run_concept_synthesis ||
            !callbacks.persist_experimental_direction ||
            !callbacks.run_search_pipeline_with_state) {
        return fail("production evaluator did not register concept and search callbacks");
    }

    const common_agent_scope scope = smoke_scope();
    std::string baseline_ref;
    if (!put_json_resource(
            *runtime.resource_store, scope, "concept-synthesis-baseline.json",
            json{{"prompt", "Follow the instruction and answer with one short sentence."},
                 {"n_predict", 1}}, baseline_ref, error)) {
        return fail("baseline resource: " + error);
    }
    std::string verifier_ref;
    if (!put_json_resource(
            *runtime.resource_store, scope, "concept-synthesis-fixture.json",
            json{{"choice_prefix", "Answer:"},
                 {"positive_continuation", " yes"},
                 {"negative_continuation", " no"}}, verifier_ref, error)) {
        return fail("fixture resource: " + error);
    }
    std::string direction_ref;
    json direction = json{{"directions", json::array()}};
    json values = json::array();
    for (size_t index = 0; index < n_embd; ++index) values.push_back(index == 0 ? 1.0f : 0.0f);
    direction["directions"].push_back({{"layer_index", 1}, {"values", std::move(values)}});
    if (!put_json_resource(
            *runtime.resource_store, scope, "concept-synthesis-direction.json",
            direction, direction_ref, error)) {
        return fail("direction resource: " + error);
    }

    std::vector<std::string> delta_refs;
    for (size_t index = 0; index < 2; ++index) {
        json delta_values = json::array();
        for (size_t value_index = 0; value_index < n_embd; ++value_index) {
            delta_values.push_back(value_index == 0 ? 1.0f : 0.0f);
        }
        std::string delta_ref;
        if (!put_json_resource(
                *runtime.resource_store, scope,
                "concept-synthesis-delta-" + std::to_string(index) + ".json",
                json{
                    {"kind", "flydelta_behavior_delta"},
                    {"id", "delta://concept-synthesis/search/" + std::to_string(index)},
                    {"source", "procedure_blueprint"},
                    {"behavior_key", "concept/qwen/instruction-following"},
                    {"capture_manifest_id", baseline_ref},
                    {"host_evidence_ref", "evidence://concept-synthesis/search"},
                    {"scope_fingerprint", "scope:concept-synthesis-smoke"},
                    {"tokenizer_fingerprint", kTokenizerFingerprint},
                    {"template_fingerprint", kTemplateFingerprint},
                    {"generation_semantics_fingerprint", "generation:bounded-qwen-v1"},
                    {"model_profile_fingerprint", options.adaptation_flydelta_model_profile_fingerprint},
                    {"execution_context_fingerprint", kExecutionContextFingerprint},
                    {"capture_layout_revision", options.adaptation_flydelta_capture_layout_revision},
                    {"layer_index", 1},
                    {"values", std::move(delta_values)},
                    {"credit", {
                        {"experiment_id", "experiment://concept-synthesis/search"},
                        {"candidate_id", direction_ref},
                        {"fixture_id", verifier_ref},
                        {"outcome", "unknown"},
                        {"quality_delta", 0.0f},
                        {"eligible_for_learning", false},
                    }},
                }, delta_ref, error)) {
            return fail("behavior delta resource: " + error);
        }
        delta_refs.push_back(std::move(delta_ref));
    }

    std::vector<common_flydelta_teaching_relation> relations;
    for (size_t index = 0; index < 2; ++index) {
        std::string conditioned_ref;
        if (!put_json_resource(
                *runtime.resource_store, scope,
                "concept-synthesis-conditioned-" + std::to_string(index) + ".json",
                json{{"prompt", index == 0
                    ? "Follow the instruction and answer with one short sentence about a cat."
                    : "Follow the instruction and answer with one short sentence about a river."},
                     {"n_predict", 1}}, conditioned_ref, error)) {
            return fail("conditioned resource: " + error);
        }
        std::string control_ref;
        if (!put_json_resource(
                *runtime.resource_store, scope,
                "concept-synthesis-control-" + std::to_string(index) + ".json",
                json{{"prompt", "Ignore the subject and answer with one short sentence."},
                     {"n_predict", 1}}, control_ref, error)) {
            return fail("control resource: " + error);
        }
        relations.push_back(make_relation(
            scope, std::to_string(index), baseline_ref, conditioned_ref, control_ref,
            verifier_ref));
        if (!runtime.flydelta_teaching_material_observer(
                relations.back(), error)) {
            return fail("teaching relation observation: " + error);
        }
    }

    common_flydelta_teaching_material_identity material_identity;
    material_identity.model_profile_fingerprint = options.adaptation_flydelta_model_profile_fingerprint;
    material_identity.tokenizer_fingerprint = kTokenizerFingerprint;
    material_identity.template_fingerprint = kTemplateFingerprint;
    material_identity.capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
    material_identity.execution_context_fingerprint = kExecutionContextFingerprint;
    material_identity.scope_fingerprint = "daemon:teaching-material-v1";
    material_identity.verifier_revision = "daemon:host-verifier-v1";
    common_flydelta_teaching_material_group group;
    if (!runtime.flydelta_teaching_material_runtime->store().group_ready(
            "qwen.instruction-following.v1", "concept/qwen/instruction-following",
            material_identity, 2, group, error) || !group.relation_set_ready) {
        return fail("teaching material group: " + error);
    }

    const auto request = make_search_request(
        scope, baseline_ref, direction_ref, group.group_ref,
        options.adaptation_flydelta_model_profile_fingerprint, delta_refs, verifier_ref);
    common_flydelta_experiment_collection_result collection_result;
    const std::filesystem::path queue_root = options.adaptation_flydelta_queue_path;
    if (!common_flydelta_collect_experiment_job(
            queue_root, {}, request, collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("initial search queue: " + error);
    }

    common_flydelta_experiment_worker_report search_report;
    if (!common_flydelta_experiment_worker_run_evaluator_once(
            queue_root, {}, evaluator_config, callbacks, search_report, error) ||
            !report_succeeded(
                search_report, common_flydelta_experiment_job_kind::search_pipeline, error)) {
        return fail("initial production search: " + error);
    }
    print_search_summary(search_report, "initial");
    if (search_report.bootstrap_zoom_state_ref.empty() && search_report.search_state_ref.empty()) {
        return fail("initial search produced no persisted localized anchor state");
    }

    if (!common_flydelta_collect_next_action_job(
            queue_root, {}, search_report.completed_job,
            common_flydelta_next_action::prepare_concept_material,
            search_report.bootstrap_zoom_state_ref, search_report.search_state_ref,
            search_report.representation_augmentation_state_ref, {},
            collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("concept capture queue: " + error);
    }
    common_flydelta_experiment_worker_report capture_report;
    if (!common_flydelta_experiment_worker_run_evaluator_once(
            queue_root, {}, evaluator_config, callbacks, capture_report, error) ||
            !report_succeeded(
                capture_report, common_flydelta_experiment_job_kind::concept_capture, error) ||
            capture_report.concept_trajectory_refs.size() != 2) {
        return fail("production concept capture: " + error);
    }
    std::cout << "flydelta_concept_capture_model=" << json{
        {"state", common_flydelta_experiment_queue_state_name(capture_report.state)},
        {"relations", group.relation_refs.size()},
        {"arms_per_relation", 3},
        {"capture_arms", group.relation_refs.size() * 3},
        {"trajectory_refs", capture_report.concept_trajectory_refs.size()},
        {"server_context", true},
        {"batch_boundary", "production common_flydelta_arm_batch"},
        {"learning_credit", "none"},
    }.dump() << '\n';

    if (!common_flydelta_collect_next_action_job(
            queue_root, {}, capture_report.completed_job,
            common_flydelta_next_action::run_concept_synthesis,
            capture_report.bootstrap_zoom_state_ref, capture_report.search_state_ref,
            capture_report.representation_augmentation_state_ref, {},
            collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("concept synthesis queue: " + error);
    }
    common_flydelta_experiment_worker_report synthesis_report;
    if (!common_flydelta_experiment_worker_run_evaluator_once(
            queue_root, {}, evaluator_config, callbacks, synthesis_report, error) ||
            !report_succeeded(
                synthesis_report, common_flydelta_experiment_job_kind::concept_synthesis, error) ||
            synthesis_report.concept_candidates.empty() ||
            synthesis_report.graft_direction_ref.empty()) {
        return fail("production concept synthesis: " + error);
    }
    json candidates = json::array();
    for (const auto & candidate : synthesis_report.concept_candidates) {
        candidates.push_back({
            {"kind", common_flydelta_concept_candidate_kind_name(candidate.kind)},
            {"concept_key", candidate.concept_key},
            {"layer_index", candidate.layer_index},
            {"source_trajectories", candidate.source_trajectories},
            {"retained_trajectories", candidate.retained_trajectories},
            {"median_alignment", candidate.median_alignment},
            {"control_residualized", candidate.control_residualized},
            {"experimental_only", candidate.experimental_only},
            {"learning_eligible", candidate.learning_eligible},
        });
    }
    std::cout << "flydelta_concept_synthesis_model=" << json{
        {"state", common_flydelta_experiment_queue_state_name(synthesis_report.state)},
        {"candidate_count", synthesis_report.concept_candidates.size()},
        {"candidates", std::move(candidates)},
        {"graft_direction_ref", synthesis_report.graft_direction_ref},
        {"next_action", common_flydelta_next_action_name(synthesis_report.next_action)},
        {"promotion", false},
    }.dump() << '\n';

    if (!common_flydelta_collect_next_action_job(
            queue_root, {}, synthesis_report.completed_job,
            common_flydelta_next_action::run_bootstrap,
            synthesis_report.bootstrap_zoom_state_ref, synthesis_report.search_state_ref,
            synthesis_report.representation_augmentation_state_ref,
            synthesis_report.graft_direction_ref, collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("grafted search queue: " + error);
    }
    common_flydelta_experiment_worker_report graft_report;
    if (!common_flydelta_experiment_worker_run_evaluator_once(
            queue_root, {}, evaluator_config, callbacks, graft_report, error) ||
            !report_succeeded(
                graft_report, common_flydelta_experiment_job_kind::search_pipeline, error)) {
        return fail("grafted production search: " + error);
    }
    print_search_summary(graft_report, "concept-graft");
    if (graft_report.completed_job.seed.candidate_ref != synthesis_report.graft_direction_ref) {
        return fail("grafted search did not consume the persisted concept direction");
    }

    std::cout << "flydelta_concept_synthesis_model_smoke=completed"
              << " model=" << args.model_path
              << " agent_workers=" << options.worker_count
              << " flydelta_workers=" << options.adaptation_flydelta_worker_count
              << " inference_max_active=" << options.inference_max_active
              << " relations=" << group.relation_refs.size()
              << " capture_arms=" << group.relation_refs.size() * 3
              << " trajectories=" << capture_report.concept_trajectory_refs.size()
              << " candidates=" << synthesis_report.concept_candidates.size()
              << " graft_to_search=yes"
              << " learning_credit=none"
              << " promotion=false\n";
    return 0;
}
