#include "tools/agent/daemon/agent-daemon-flydelta-internal.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "agent/adaptation/flydelta/flydelta-worker-commit.h"
#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"
#include "agent/adaptation/flydelta/flydelta-sideband-registry.h"

#include <chrono>
#include <filesystem>
#include <iostream>

using namespace agent_daemon_flydelta_internal;
using ordered_json = nlohmann::ordered_json;
#define CHECK(condition) do { if (!(condition)) { std::cerr << __LINE__ << ": " << error << '\n'; return __LINE__; } } while (false)

// All orchestration, resource codecs, journal writes and queue transitions
// below are production components. Only the bounded model transport is fake.
struct transport_observations {
    size_t captures = 0;
    size_t interventions = 0;
    std::vector<std::string> intervention_refs;
};

static common_flydelta_teaching_material_identity execution_identity() {
    common_flydelta_teaching_material_identity id;
    id.model_profile_fingerprint = "model:deterministic-host:v1";
    id.tokenizer_fingerprint = "tokenizer:deterministic-host:v1";
    id.template_fingerprint = "template:deterministic-host:v1";
    id.execution_context_fingerprint = "context:deterministic-host:v1";
    id.capture_layout_revision = "layer-input:v1";
    id.scope_fingerprint = "scope:positive-integration";
    id.verifier_revision = "verifier:positive-integration:v1";
    return id;
}

struct durable_host {
    std::unique_ptr<agent_resource_store> resources;
    std::shared_ptr<common_learning_jsonl_lifecycle_store> journal;
    std::shared_ptr<common_flydelta_teaching_material_runtime> material;
    std::shared_ptr<daemon_flydelta_resource_provider> provider;
    common_flydelta_evaluator_config config;
    common_flydelta_evaluator_callbacks callbacks;

    void close() {
        callbacks = {};
        provider.reset();
        material.reset();
        journal.reset();
        resources.reset();
    }

    bool open(const std::filesystem::path & root, transport_observations & observations, std::string & error) {
        close();
        agent_resource_store_config storage;
        storage.blob_backend = "fs";
        storage.blob_root = (root / "blobs").string();
        storage.metadata_backend = "cozo";
        storage.metadata_db = (root / "resources.cozo").string();
        resources = make_agent_resource_store(storage, error);
        if (!resources) return false;
        journal = std::make_shared<common_learning_jsonl_lifecycle_store>();
        if (!journal->open(root / "lifecycle.jsonl", error)) return false;
        material = make_agent_flydelta_teaching_material_runtime("cozo", (root / "material.cozo").string(),
            execution_identity(), error);
        if (!material) return false;
        material->bind_assessment_journal(journal);
        provider = std::make_shared<daemon_flydelta_resource_provider>();
        provider->resources = resources.get();
        provider->lifecycle_store = journal;
        provider->teaching_material_runtime = material;
        provider->model_profile_fingerprint = execution_identity().model_profile_fingerprint;
        provider->capture_layout_revision = execution_identity().capture_layout_revision;
        provider->model_n_embd = 2;
        provider->model_n_layers = 6;
        provider->n_predict = 64;
        provider->n_threads = 1;
        provider->authority.namespace_id = "integration";
        provider->authority.project_id = "project";
        provider->authority.session_id = "session";
        provider->oracle_registry = common_flydelta_make_default_oracle_registry();
        provider->run_prepared_batch = [&observations](const common_agent_server_flydelta_binding & binding,
                const common_flydelta_arm_batch_request & batch, common_flydelta_arm_batch_result & result,
                std::string & error) {
            result = {};
            for (const auto & arm : batch.arms) {
                common_agent_generation_request request;
                if (!binding.prepare_arm(arm, request, error)) return false;
                common_agent_generation_result generation;
                generation.status = common_agent_generation_status::completed;
                generation.stop_reason = common_agent_generation_stop_reason::eos;
                // Neither side passes the Oracle. Diagnostic geometry is not
                // rewarded with a fabricated HELPED observation.
                generation.content = arm.request_generation ? "open! missing.tool" : "";
                generation.decoded_tokens = arm.request_generation ? 1 : 0;
                generation.flydelta_runtime.applied = arm.apply_overlay;
                generation.flydelta_runtime.artifact_id = arm.apply_overlay ? arm.arm_id : "";
                if (request.flydelta_capture) {
                    auto capture = std::make_shared<common_flydelta_hidden_state_capture>();
                    capture->captured = true;
                    capture->model_profile_fingerprint = request.flydelta_capture->model_profile_fingerprint;
                    capture->capture_layout_revision = request.flydelta_capture->capture_layout_revision;
                    capture->position = common_flydelta_capture_position::generation_boundary;
                    capture->token_index = 0;
                    capture->layer_indices = request.flydelta_capture->layer_indices;
                    capture->n_embd = 2;
                    for (size_t i = 0; i < capture->layer_indices.size(); ++i) {
                        capture->values.push_back(arm.apply_overlay ? 1.01f : 1.0f);
                        capture->values.push_back(0.2f);
                    }
                    generation.flydelta_capture = std::move(capture);
                }
                common_flydelta_arm_result observed;
                if (!binding.finalize_arm(arm, generation, observed, error)) return false;
                if (arm.wave_id == "concept-capture") {
                    if (arm.apply_overlay || arm.request_host_verification || observed.host_evaluated ||
                            observed.verifier_known || observed.overlay_applied) {
                        error = "positive capture crossed the behavioral evidence boundary";
                        return false;
                    }
                    ++observations.captures;
                }
                if (arm.apply_overlay) {
                    if (!observed.executed || !observed.overlay_applied) {
                        error = "experimental intervention was not executed";
                        return false;
                    }
                    ++observations.interventions;
                    observations.intervention_refs.push_back(arm.intervention_ref);
                }
                // A deterministic teacher-score observation makes diagnostic
                // frontier selection reproducible. It is not an Oracle result.
                observed.margin.available = true;
                observed.margin.positive_token_count = observed.margin.negative_token_count = 1;
                observed.margin.positive_total_logprob = arm.apply_overlay ? -0.1f : -0.2f;
                observed.margin.negative_total_logprob = -0.5f;
                result.arms.push_back(std::move(observed));
            }
            result.execution_stats.logical_arm_count = batch.arms.size();
            result.execution_stats.physical_batch_count = batch.arms.size();
            result.execution_stats.largest_physical_batch = batch.arms.empty() ? 0 : 1;
            result.execution_stats.scalar_fallback_arm_count = batch.arms.size();
            return common_flydelta_arm_batch_result_validate(result, batch, error);
        };
        return daemon_flydelta_register_evaluator(provider, config, callbacks, error);
    }

    bool put(const ordered_json & value, std::string & ref, std::string & error) {
        return daemon_flydelta_put_json_resource(provider, "fixture.ordered_json", "host-fixture", {}, value.dump(), ref, error);
    }

    bool slice(const std::filesystem::path & queue, common_flydelta_experiment_job & executed_job,
            common_flydelta_experiment_worker_report & report, std::string & error) {
        return common_flydelta_experiment_worker_run_once(queue, {},
            [&](const common_flydelta_experiment_job & job, common_flydelta_experiment_worker_result & output,
                    std::string & error) {
                executed_job = job;
                common_flydelta_evaluator_result evaluated;
                return common_flydelta_evaluate_job(job, config, callbacks, evaluated, error) &&
                    common_flydelta_worker_result_from_evaluator(evaluated, output, error);
            }, report, error,
            [&](const auto & job, const auto & completed, std::string & error) {
                return common_flydelta_append_worker_commit_intent(*journal, job, completed, error);
            });
    }

    bool schedule(const std::filesystem::path & queue, const common_flydelta_experiment_job & executed_job,
            const common_flydelta_experiment_worker_report & report, std::string & error) {
        common_flydelta_experiment_job parent;
        if (!common_flydelta_resolve_worker_commit_parent(*journal, executed_job, parent, error)) return false;
        if (report.has_next_action) {
            common_flydelta_experiment_collection_result collected;
            if (!common_flydelta_collect_next_action_job(queue, {}, parent, report.next_action,
                    report.bootstrap_zoom_state_ref, report.search_state_ref,
                    report.representation_augmentation_state_ref, report.graft_direction_ref, collected, error)) return false;
        }
        return common_flydelta_acknowledge_worker_commit(*journal, executed_job, error);
    }
};

int main() {
    std::string error;
    const auto root = std::filesystem::temp_directory_path() /
        ("flydelta-positive-integration-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const auto queue = root / "queue";
    transport_observations observations;
    durable_host host;
    CHECK(host.open(root, observations, error));
    common_agent_scope scope;
    scope.namespace_id = "integration";
    scope.project_id = "project";
    scope.session_id = "session";
    scope.turn_id = "evaluation-turn";
    std::string baseline_ref, verifier_ref;
    CHECK(host.put(ordered_json{{"prompt", "Host evaluation task"}}, baseline_ref, error));
    const ordered_json expected_arguments = {{"dataset", "dataset:host"}, {"group_by", {"region"}},
        {"measures", {{{"function", "sum"}, {"column", "amount"}}}}};
    const ordered_json expected_call = {{"name", "data.aggregate"}, {"arguments", expected_arguments}};
    CHECK(host.put(ordered_json{{"verification_mode", "normalized_call"}, {"expected_decision", expected_call}}, verifier_ref, error));
    for (int i = 0; i < 3; ++i) {
        std::string context_ref;
        CHECK(host.put(ordered_json{{"prompt", "Independent authorized concept example " + std::to_string(i)}}, context_ref, error));
        common_flydelta_teaching_relation source;
        source.schema_version = 2;
        source.positive_only = true;
        source.id = "host-concept:" + std::to_string(i);
        source.teaching_key = "concept:integration";
        source.behavior_key = "behavior:integration";
        source.source = common_adaptation_evidence_source::user_taught_concept;
        source.scope = scope;
        source.task_fingerprint = "task:independent:" + std::to_string(i);
        source.conditioned_ref = context_ref;
        source.conditioned_origin = common_flydelta_teaching_origin::user_supplied;
        source.verifier_ref = verifier_ref;
        source.evidence_ref = context_ref; // Input provenance, not a behavioral verdict.
        source.host_approved = true;
        source.status = common_flydelta_teaching_relation_status::resolved;
        CHECK(common_flydelta_teaching_relation_validate(source, error));
        common_flydelta_teaching_relation stored;
        CHECK(daemon_flydelta_persist_teaching_relation(host.resources.get(), host.provider->authority, source, stored, error));
        CHECK(host.material->observe_relation(stored, error));
    }
    common_flydelta_experiment_job context;
    context.seed.scope = scope;
    context.seed.teaching_key = "concept:integration";
    context.seed.behavior_key = "behavior:integration";
    context.seed.task_fingerprint = "task:evaluation";
    context.seed.baseline_ref = baseline_ref;
    context.seed.verifier_ref = verifier_ref;
    context.seed.verifier_contract_revision = "verifier:integration:v1";
    context.alpha_search.candidates = {0.05f};
    context.alpha_search.max_candidates = 1;
    // A persisted host-selected WHERE origin carries no completed trial,
    // positive behavior or learned evidence rank.
    common_flydelta_bootstrap_zoom_state origin;
    origin.behavior_key = context.seed.behavior_key;
    origin.model_profile_fingerprint = execution_identity().model_profile_fingerprint;
    origin.capture_layout_revision = execution_identity().capture_layout_revision;
    origin.anchor_layer = 2;
    origin.local_layers = {2};
    origin.selected_scale = 0.05f;
    origin.evidence_rank = 0;
    CHECK(host.callbacks.persist_bootstrap_zoom_state_for_job(context, origin, context.bootstrap_zoom_state_ref, error));
    {
        common_agent_daemon_runtime runtime;
        runtime.flydelta_review_lifecycle_store = host.journal;
        runtime.flydelta_sideband_registry = std::make_shared<common_flydelta_sideband_registry>();
        runtime.flydelta_sideband_review_store = std::make_shared<common_flydelta_sideband_review_store>(*host.journal);
        runtime.flydelta_concept_admit = [provider = host.provider, queue](const common_flydelta_experiment_job & context,
                common_flydelta_teaching_material_mode mode, common_flydelta_experiment_collection_result & collected,
                std::string & error) {
            return daemon_flydelta_admit_concept_experiment(provider, queue, {}, context, mode, collected, error);
        };
        common_agent_daemon_service daemon(std::move(runtime));
        const ordered_json request = {{"command", "flydelta.admit_concept"}, {"request_id", "positive-integration"},
            {"namespace_id", scope.namespace_id}, {"project_id", scope.project_id}, {"session_id", scope.session_id},
            {"turn_id", scope.turn_id},
            {"teaching_key", context.seed.teaching_key}, {"behavior_key", context.seed.behavior_key},
            {"task_fingerprint", context.seed.task_fingerprint}, {"baseline_ref", context.seed.baseline_ref},
            {"verifier_ref", context.seed.verifier_ref}, {"verifier_revision", context.seed.verifier_contract_revision},
            {"where_origin_ref", context.bootstrap_zoom_state_ref}, {"synthesis_mode", "auto"}};
        agent_daemon_foreground_request parsed;
        daemon_options options;
        CHECK(parse_agent_daemon_foreground_request(request, options, common_agent_runtime_host_mode::chat, parsed, error));
        common_agent_daemon_command_outcome outcome;
        std::vector<common_agent_daemon_event> events;
        CHECK(daemon.execute_outcome(parsed.command, outcome, events, error));
        CHECK(outcome.ok && outcome.event == "flydelta.concept_experiment.queued");
        CHECK(daemon.execute_outcome(parsed.command, outcome, events, error));
        CHECK(ordered_json::parse(outcome.payload_json)["already_present"] == true);

    }
    common_flydelta_experiment_job capture_job;
    common_flydelta_experiment_worker_report capture_report;
    CHECK(host.slice(queue, capture_job, capture_report, error));
    CHECK(capture_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(capture_job.kind == common_flydelta_experiment_job_kind::concept_capture);
    CHECK(capture_job.seed.schema_version == 2 && capture_job.seed.transaction_ids.empty());
    CHECK(capture_job.capture_manifest_ids.empty() && capture_job.behavior_delta_ids.empty());
    CHECK(observations.captures == 3 && observations.interventions == 0);
    CHECK(capture_report.next_action == common_flydelta_next_action::run_concept_synthesis);
    CHECK(host.schedule(queue, capture_job, capture_report, error));
    host.close();

    // Real close/reopen: filesystem blobs, Cozo resource catalog/material and
    // JSONL lifecycle. No in-process captures or direction caches survive.
    CHECK(host.open(root, observations, error));
    common_flydelta_experiment_job synthesis_job;
    common_flydelta_experiment_worker_report synthesis_report;
    CHECK(host.slice(queue, synthesis_job, synthesis_report, error));
    CHECK(synthesis_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(synthesis_job.kind == common_flydelta_experiment_job_kind::concept_synthesis);
    CHECK(synthesis_report.concept_candidates.size() == 1);
    CHECK(synthesis_report.concept_candidates.front().experimental_only &&
        !synthesis_report.concept_candidates.front().learning_eligible);
    CHECK(!synthesis_report.graft_direction_ref.empty());
    CHECK(host.schedule(queue, synthesis_job, synthesis_report, error));
    host.close();

    CHECK(host.open(root, observations, error));
    common_flydelta_experiment_job search_job;
    common_flydelta_experiment_worker_report search_report;
    CHECK(host.slice(queue, search_job, search_report, error));
    CHECK(search_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(search_job.kind == common_flydelta_experiment_job_kind::search_pipeline);
    CHECK(search_job.seed.admission_ref == capture_job.seed.admission_ref);
    CHECK(search_job.seed.scope.namespace_id == scope.namespace_id && search_job.seed.scope.project_id == scope.project_id &&
        search_job.seed.scope.session_id == scope.session_id && search_job.seed.scope.turn_id == scope.turn_id);
    CHECK(search_job.seed.model_profile_fingerprint == capture_job.seed.model_profile_fingerprint);
    CHECK(search_job.seed.tokenizer_fingerprint == capture_job.seed.tokenizer_fingerprint);
    CHECK(search_job.seed.template_fingerprint == capture_job.seed.template_fingerprint);
    CHECK(search_job.seed.execution_context_fingerprint == capture_job.seed.execution_context_fingerprint);
    CHECK(search_job.teaching_material_group_ref == capture_job.teaching_material_group_ref);
    CHECK(search_job.concept_synthesis_ref == synthesis_report.graft_direction_ref);
    CHECK(observations.interventions > 0);
    bool guidance_seen = false;
    for (const auto & record : host.journal->list(error)) {
        CHECK(record.status != common_learning_lifecycle_status::active &&
            record.status != common_learning_lifecycle_status::canary);
        CHECK(record.kind != common_learning_lifecycle_kind::adapter &&
            record.kind != common_learning_lifecycle_kind::training_job);
        const auto value = ordered_json::parse(record.payload_json);
        if (!value.contains("synthesis_diagnostics")) continue;
        const auto & diagnostics = value["synthesis_diagnostics"];
        CHECK(record.namespace_id == scope.namespace_id && record.project_id == scope.project_id && record.session_id == scope.session_id);
        CHECK(diagnostics["source_ref"] == synthesis_report.graft_direction_ref);
        CHECK(diagnostics["concept_ref"] == capture_job.teaching_material_group_ref);
        CHECK(diagnostics["sample_count"] == 3 && diagnostics["synthesis_semantics"] == "positive_only");
        CHECK(!diagnostics["direction_fingerprint"].get<std::string>().empty());
        guidance_seen = true;
    }
    CHECK(error.empty() && guidance_seen);
    CHECK(search_report.evidence_depth.effective_rank == 0);
    for (const auto & arm : search_report.trace.arms)
        CHECK(arm.host_outcome != common_flydelta_counterfactual_outcome::helped);
    std::string synthesis_text;
    CHECK(host.resources->read_text(search_job.concept_synthesis_ref, host.provider->authority,
        1024U * 1024U, synthesis_text, error));
    auto mismatched_artifact = ordered_json::parse(synthesis_text);
    mismatched_artifact["synthesis"]["model_profile_fingerprint"] = "other-model";
    std::string mismatched_ref;
    CHECK(host.put(mismatched_artifact, mismatched_ref, error));
    auto mismatched_job = search_job;
    mismatched_job.concept_synthesis_ref = mismatched_ref;
    const auto executions_before_rejection = observations.interventions;
    common_flydelta_search_pipeline_result rejected_search;
    CHECK(!daemon_flydelta_run_search_pipeline(host.provider, mismatched_job, host.config.pipeline, rejected_search, error));
    CHECK(observations.interventions == executions_before_rejection);
    // A scoped admission cannot be transplanted to another session/model.
    auto wrong_job = search_job;
    wrong_job.seed.scope.session_id = "other-session";
    auto wrong_provider = daemon_flydelta_provider_for_scope(host.provider, wrong_job.seed.scope);
    CHECK(!daemon_flydelta_validate_concept_admission(*wrong_provider, wrong_job, error));
    wrong_job = search_job;
    wrong_job.seed.model_profile_fingerprint = "other-model";
    CHECK(!daemon_flydelta_validate_concept_admission(*host.provider, wrong_job, error));
    // Read back the actual completed queue envelope and persisted worker result.
    common_flydelta_experiment_queue_state queue_state;
    CHECK(common_flydelta_experiment_queue_job_state(queue, search_job, {}, queue_state, error));
    CHECK(queue_state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(host.schedule(queue, search_job, search_report, error));
    host.close();
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    std::cout << "positive_only_daemon_integration=passed transport=deterministic model_quality_claim=no\n";
    return 0;
}
