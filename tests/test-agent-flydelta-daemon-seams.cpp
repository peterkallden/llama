#include "tools/agent/daemon/agent-daemon-flydelta-internal.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <cmath>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { std::cerr << __LINE__ << ": " << error << '\n'; return __LINE__; } } while (false)

using namespace agent_daemon_flydelta_internal;
using flydelta_json = agent_daemon_flydelta_internal::json;

int main() {
    std::string error;
    agent_catalogued_resource_store resources(std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    auto provider = std::make_shared<daemon_flydelta_resource_provider>();
    CHECK(!daemon_flydelta_factorized_wave_eligible(common_flydelta_experiment_phase::bootstrap, 4));
    CHECK(!daemon_flydelta_factorized_wave_eligible(common_flydelta_experiment_phase::shallow_controls, 4));
    CHECK(!daemon_flydelta_factorized_wave_eligible(common_flydelta_experiment_phase::deep_controls, 1));
    CHECK(daemon_flydelta_factorized_wave_eligible(common_flydelta_experiment_phase::deep_controls, 2));
    provider->resources = &resources;
    provider->host = std::make_shared<common_agent_server_context_host>();
    provider->n_predict = 512;
    provider->n_threads = 1;
    provider->model_n_embd = 2;
    provider->model_n_layers = 6;
    provider->model_profile_fingerprint = "model:seams";
    provider->capture_layout_revision = "layer-input:v1";
    provider->authority.namespace_id = "seams";
    provider->authority.session_id = "test";
    common_flydelta_teaching_relation relation;
    relation.id = "teaching://seams/relation";
    relation.teaching_key = "teaching:seams";
    relation.behavior_key = "behavior:seams";
    relation.evidence_ref = "evidence://seams/semantic-reference";
    common_flydelta_teaching_relation persisted_relation;
    CHECK(daemon_flydelta_persist_teaching_relation(
        &resources, provider->authority, relation, persisted_relation, error));
    std::string relation_text;
    CHECK(resources.read_text(persisted_relation.id, provider->authority,
        1024 * 1024, relation_text, error));
    CHECK(flydelta_json::parse(relation_text).value("evidence_ref", "") == relation.evidence_ref);
    provider->oracle_registry = common_flydelta_make_default_oracle_registry();
    common_agent_scope job_scope;
    job_scope.namespace_id = "seams";
    job_scope.session_id = "test";
    const auto scoped_provider = daemon_flydelta_provider_for_scope(provider, job_scope);
    CHECK(scoped_provider && scoped_provider->outcome_projection_cache == provider->outcome_projection_cache);
    common_flydelta_search_pipeline_result incomplete_result;
    CHECK(!daemon_flydelta_run_search_pipeline(provider, {}, {}, incomplete_result, error));
    CHECK(error.find("source evidence reference") != std::string::npos);
    const auto put = [&](const flydelta_json & value, std::string & uri) {
        agent_resource_put_request request;
        request.name = "fixture.json";
        request.mime_type = "application/json";
        request.text = value.dump();
        request.namespace_id = "seams";
        request.session_id = "test";
        request.scope = common_runtime_resource_scope::session;
        agent_resource_descriptor descriptor;
        if (!resources.put_text(request, descriptor, error)) return false;
        uri = descriptor.uri;
        return true;
    };
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    CHECK(catalog.bootstrap("analysis", bootstrap, error));
    const auto * definition = catalog.find_definition("data.aggregate");
    CHECK(definition != nullptr);
    const std::string schema = common_tool_model_input_schema(*definition);
    const flydelta_json arguments = {{"dataset", "dataset://local/sales"}, {"group_by", {"region"}},
        {"measures", {{{"function", "sum"}, {"column", "amount"}}}}};
    const auto preferred = common_render_flat_tool_call("data.aggregate", schema, arguments.dump(), error);
    CHECK(!preferred.empty());
    common_flydelta_arm_request arm;
    arm.arm_id = "arm:seams";
    arm.fresh_context = true;
    arm.layer_indices = {2};
    arm.coefficients = {1};
    arm.alpha = 0.05f;
    arm.request_generation = true;
    arm.request_host_verification = true;
    CHECK(put({{"messages", {{{"role", "user"}, {"content", "Sum amount by region"}}}},
        {"model_tools", {{{"name", "data.aggregate"}, {"parameters", flydelta_json::parse(schema)}}}}}, arm.context_ref));
    CHECK(put({{"verification_mode", "normalized_call"},
        {"expected_decision", {{"name", "data.aggregate"}, {"arguments", arguments}}}}, arm.fixture_ref));
    provider->composed_directions["direction:seams"] = {{2, {1, 0}}};
    arm.intervention_ref = "direction:seams";
    common_agent_server_flydelta_binding binding;
    daemon_flydelta_bind_execution_callbacks(provider, {}, binding);
    common_agent_generation_request request;
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(request.options.n_predict == 512 && request.tools.empty());
    std::vector<common_chat_tool> schemas;
    CHECK(daemon_flydelta_parse_context(*provider, arm.context_ref, request, error, &schemas));
    CHECK(schemas.size() == 1 && request.tools.empty());
    const auto text_context_ref = arm.context_ref;
    CHECK(put({{"messages", {{{"role", "user"}, {"content", "Sum amount by region"}}}},
        {"tools", {{{"name", "data.aggregate"}, {"parameters", flydelta_json::parse(schema)}}}},
        {"tool_choice", "required"}}, arm.context_ref));
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(request.tools.size() == 1 && request.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);
    CHECK(put({{"prompt", "Sum amount by region"}, {"tools", flydelta_json::array()},
        {"model_tools", flydelta_json::array()}}, arm.context_ref));
    CHECK(!binding.prepare_arm(arm, request, error));
    arm.context_ref = text_context_ref;
    arm.request_generation = arm.request_host_verification = false;
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(request.options.n_predict == 0);
    arm.request_generation = arm.request_host_verification = true;
    arm.max_generated_tokens = 100;
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(request.options.n_predict == 100);
    arm.max_generated_tokens = 0;
    arm.apply_overlay = true;
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(request.flydelta_activation != nullptr);
    arm.coefficients = {-1};
    CHECK(common_flydelta_arm_request_validate(arm, error));
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(request.flydelta_activation->sparse_overlay.data.front() < 0);
    arm.coefficients = {1};

    // Controlled backend observations exercise real daemon callbacks, not a
    // claim about model quality or real GPU application.
    common_agent_generation_result generated;
    generated.status = common_agent_generation_status::completed;
    generated.stop_reason = common_agent_generation_stop_reason::eos;
    generated.content = preferred;
    common_flydelta_arm_result result;
    CHECK(!binding.finalize_arm(arm, generated, result, error));
    generated.flydelta_runtime.applied = true;
    CHECK(!binding.finalize_arm(arm, generated, result, error));
    generated.flydelta_runtime.artifact_id = arm.arm_id;
    CHECK(binding.finalize_arm(arm, generated, result, error));
    CHECK(result.executed && result.host_evaluated && result.verifier_known && result.verifier_passed);
    const auto good = result;
    auto bad = good;
    bad.verifier_passed = false;
    CHECK(daemon_flydelta_classify_arm_pair(bad, good) == common_flydelta_counterfactual_outcome::helped);
    CHECK(daemon_flydelta_classify_arm_pair(good, bad) == common_flydelta_counterfactual_outcome::harmed);
    CHECK(daemon_flydelta_classify_arm_pair(good, good) == common_flydelta_counterfactual_outcome::neutral);
    CHECK(daemon_flydelta_classify_arm_pair(bad, bad) == common_flydelta_counterfactual_outcome::unknown);
    generated.stop_reason = common_agent_generation_stop_reason::limit;
    CHECK(binding.finalize_arm(arm, generated, result, error));
    CHECK(!result.host_evaluated && !result.verifier_known && !result.verifier_passed);
    CHECK(result.verifier_violation_code == "generation_token_budget_exhausted");
    CHECK(daemon_flydelta_classify_arm_pair(result, good) == common_flydelta_counterfactual_outcome::unknown);
    arm.request_generation = false;
    arm.request_host_verification = false;
    CHECK(binding.finalize_arm(arm, generated, result, error));
    CHECK(!result.host_evaluated && result.verifier_violation_code.empty());
    arm.request_generation = true;
    arm.request_host_verification = true;
    generated.stop_reason = common_agent_generation_stop_reason::eos;

    common_flydelta_experiment_fixture fixture;
    fixture.schema_version = 1;
    fixture.id = "fixture:seams";
    fixture.task_fingerprint = "task";
    fixture.model_profile_fingerprint = "model";
    fixture.tokenizer_fingerprint = "tokenizer";
    fixture.template_fingerprint = "template";
    fixture.execution_context_fingerprint = "execution";
    fixture.verifier_revision = "verifier";
    common_flydelta_direction_candidate direction;
    direction.layer_index = 2;
    direction.values = {1, 0};
    direction.source_samples = direction.retained_samples = 6;
    direction.median_alignment = 1;
    direction.experimental_only = true;
    common_flydelta_deep_search_config config;
    config.max_rank = config.max_directions = config.full_generation_top_k = 1;
    config.coefficients.max_candidates = 4;
    const auto runner = [&](bool full) {
        return [&, full](const common_flydelta_experiment_fixture &, const common_flydelta_low_rank_basis &,
                const std::vector<float> & coefficients, bool overlay,
                common_flydelta_counterfactual_trial & trial, common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry, bool & available, std::string & callback_error) {
            auto actual_arm = arm;
            actual_arm.apply_overlay = overlay;
            actual_arm.request_host_verification = full;
            actual_arm.request_generation = full;
            auto observation = generated;
            observation.content = overlay && coefficients.front() > 0 ? preferred : "open! missing.tool";
            common_flydelta_arm_result actual;
            if (!binding.finalize_arm(actual_arm, observation, actual, callback_error)) return false;
            trial = {};
            trial.executed = actual.executed;
            trial.overlay_applied = actual.overlay_applied;
            trial.host_evaluated = actual.host_evaluated;
            trial.verifier_known = actual.verifier_known;
            trial.passed = actual.verifier_passed;
            trial.quality = actual.verifier_passed ? 1 : 0;
            trial.evidence_ref = "evidence:controlled-backend";
            margin = {};
            margin.available = true;
            margin.positive_token_count = margin.negative_token_count = 1;
            margin.positive_total_logprob = coefficients.front();
            geometry = {};
            available = false;
            return true;
        };
    };
    common_flydelta_deep_search_result search;
    CHECK(common_flydelta_run_deep_search(fixture, config, {{direction, true, 1}}, runner(false), runner(true), search, error));
    CHECK(search.coefficient_selection.selected);
    const auto & winner = search.coefficient_trials.at(search.coefficient_selection.trial_index);
    CHECK(winner.outcome == common_flydelta_counterfactual_outcome::helped);
    common_flydelta_intervention_region_trial region;
    CHECK(common_flydelta_coefficient_trial_to_region(winner, search.basis, region, error));
    CHECK(region.host_evaluated && region.verifier_known && region.candidate.total_scale > 0);

    std::string reference;
    CHECK(daemon_flydelta_register_composed_direction(provider, arm.intervention_ref, search.basis,
        {-0.05f}, reference, error));
    provider->composed_directions.clear();
    std::vector<common_flydelta_basis_direction> reloaded;
    CHECK(daemon_flydelta_parse_directions(*provider, reference, reloaded, error));
    CHECK(reloaded.size() == 1 && reloaded.front().values.front() < 0);
    common_flydelta_low_rank_basis rank_two = search.basis;
    rank_two.vectors = {{1, 0}, {0, 1}};
    CHECK(daemon_flydelta_register_composed_direction(provider, reference, rank_two, {0.03f, -0.04f}, reference, error));
    provider->composed_directions.clear();
    CHECK(daemon_flydelta_parse_directions(*provider, reference, reloaded, error));
    CHECK(std::abs(reloaded.front().values[0] - 0.6f) < 1e-6f);
    CHECK(std::abs(reloaded.front().values[1] + 0.8f) < 1e-6f);
    provider->model_n_embd = 4;
    auto rank_four = rank_two;
    rank_four.dimension = 4;
    rank_four.vectors = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    CHECK(daemon_flydelta_register_composed_direction(provider, reference, rank_four,
        {0.01f, -0.02f, 0.02f, -0.01f}, reference, error));
    provider->composed_directions.clear();
    CHECK(daemon_flydelta_parse_directions(*provider, reference, reloaded, error));
    const float strength = std::sqrt(0.001f);
    CHECK(std::abs(reloaded.front().values[0] * strength - 0.01f) < 1e-6f);
    CHECK(std::abs(reloaded.front().values[3] * strength + 0.01f) < 1e-6f);
    // Exercise the real resource adapter and synthesis across two layer-local
    // realizations. These captures are controlled fixtures, not model proof.
    provider->model_n_embd = 2;
    provider->lifecycle_store = std::make_shared<common_learning_in_memory_lifecycle_store>();
    common_flydelta_teaching_material_identity material_identity;
    material_identity.model_profile_fingerprint = "model:seams";
    material_identity.tokenizer_fingerprint = "tokenizer:seams";
    material_identity.template_fingerprint = "template:seams";
    material_identity.capture_layout_revision = "layer-input:v1";
    material_identity.execution_context_fingerprint = "context:seams";
    material_identity.scope_fingerprint = "scope:seams";
    material_identity.verifier_revision = "verifier:seams";
    provider->teaching_material_runtime = std::make_shared<common_flydelta_teaching_material_runtime>(material_identity);
    common_flydelta_evaluator_config localization_config;
    common_flydelta_evaluator_callbacks localization_callbacks;
    CHECK(daemon_flydelta_register_evaluator(provider, localization_config, localization_callbacks, error));
    provider->resolve_bootstrap_zoom_state_for_job = [](const auto &, const std::string & ref,
            common_flydelta_bootstrap_zoom_state & state, std::string & callback_error) {
        callback_error.clear();
        if (ref != "state:seams-origin") return false;
        state.behavior_key = "behavior:seams";
        state.model_profile_fingerprint = "model:seams";
        state.capture_layout_revision = "layer-input:v1";
        state.anchor_layer = 2;
        state.local_layers = {2};
        state.surface_revision = 1;
        return true;
    };
    std::vector<std::string> relation_refs;
    for (size_t index = 0; index < 2; ++index) {
        common_flydelta_teaching_relation input;
        input.id = "semantic:relation:" + std::to_string(index);
        input.teaching_key = "concept:seams";
        input.behavior_key = "behavior:seams";
        input.source = common_adaptation_evidence_source::user_taught_concept;
        input.scope = job_scope;
        input.task_fingerprint = "task:relation:" + std::to_string(index);
        input.baseline_ref = "baseline:relation";
        input.conditioned_ref = "conditioned:relation";
        input.control_ref = "control:relation";
        input.verifier_ref = "verifier:relation";
        input.evidence_ref = "evidence:relation";
        input.status = common_flydelta_teaching_relation_status::resolved;
        input.baseline_origin = common_flydelta_teaching_origin::host_derived;
        input.conditioned_origin = common_flydelta_teaching_origin::host_derived;
        input.control_origin = common_flydelta_teaching_origin::host_counterfactual;
        input.confidence = 1.0f;
        input.host_approved = true;
        common_flydelta_teaching_relation stored;
        CHECK(daemon_flydelta_persist_teaching_relation(&resources, provider->authority, input, stored, error));
        CHECK(stored.id != input.id);
        CHECK(provider->teaching_material_runtime->observe_relation(stored, error));
        relation_refs.push_back(stored.id);
    }
    const auto material_group = provider->teaching_material_runtime->store().groups().front();
    common_flydelta_experiment_job localization_job;
    localization_job.id = "job:localization";
    localization_job.kind = common_flydelta_experiment_job_kind::concept_synthesis;
    localization_job.seed.scope = job_scope;
    localization_job.seed.model_profile_fingerprint = material_identity.model_profile_fingerprint;
    localization_job.seed.tokenizer_fingerprint = material_identity.tokenizer_fingerprint;
    localization_job.seed.template_fingerprint = material_identity.template_fingerprint;
    localization_job.seed.execution_context_fingerprint = material_identity.execution_context_fingerprint;
    localization_job.seed.behavior_key = "behavior:seams";
    localization_job.seed.baseline_ref = "baseline:experiment";
    localization_job.seed.verifier_ref = "verifier:experiment";
    localization_job.teaching_material_group_ref = material_group.group_ref;
    localization_job.concept_localization.capture_set_ref = "challenge:seams";
    localization_job.concept_localization.origin_state_ref = "state:seams-origin";
    localization_job.concept_localization.origin_direction_ref = "direction:origin";
    localization_job.concept_localization.layers = {2, 4};
    localization_job.concept_localization.max_capture_model_calls = 12;
    const auto material_key = common_flydelta_concept_localization_material_key(localization_job);
    for (int32_t layer : {2, 4}) {
        for (const auto & relation_ref : relation_refs) {
            daemon_flydelta_concept_relation_material material;
            CHECK(daemon_flydelta_parse_teaching_relation(*provider, relation_ref, material, error));
            material.relation_resource_ref = relation_ref;
            material.layer_index = layer;
            material.capture_set_ref = material_key;
            material.capture_identity = material_identity;
            common_flydelta_hidden_state_capture baseline;
            baseline.captured = true;
            baseline.model_profile_fingerprint = provider->model_profile_fingerprint;
            baseline.capture_layout_revision = provider->capture_layout_revision;
            baseline.layer_indices = {static_cast<uint32_t>(layer)};
            baseline.n_embd = 2;
            baseline.token_index = 0;
            baseline.position = common_flydelta_capture_position::generation_boundary;
            baseline.values = {0.1f, 0.2f};
            auto control = baseline;
            auto conditioned = baseline;
            conditioned.values[layer == 2 ? 0 : 1] += 1.0f;
            std::string trajectory_ref;
            CHECK(daemon_flydelta_persist_concept_trajectory(provider, material, material_group.group_ref,
                "job:capture", "capture:baseline", "capture:conditioned", "capture:control",
                baseline, conditioned, control, true, false, "", {}, trajectory_ref, error));
            CHECK(provider->teaching_material_runtime->observe_layer_trajectory(material_group.group_ref,
                layer, relation_ref, trajectory_ref, error, material_key));
        }
    }
    auto snapshot = common_flydelta_teaching_material_store_to_json(provider->teaching_material_runtime->store());
    common_flydelta_teaching_material_store reloaded_material;
    CHECK(common_flydelta_teaching_material_store_from_json(snapshot, reloaded_material, error));
    provider->teaching_material_runtime->store() = std::move(reloaded_material);
    std::vector<common_flydelta_concept_candidate> local_candidates;
    std::string scan_ref;
    CHECK(daemon_flydelta_run_concept_synthesis(provider, localization_job, local_candidates, error, &scan_ref));
    std::vector<common_flydelta_concept_candidate> layer_family;
    for (const auto & candidate : local_candidates) {
        if (candidate.kind != common_flydelta_concept_candidate_kind::raw_mean ||
                candidate.synthesis_semantics != common_flydelta_concept_synthesis_semantics::control_residualized) continue;
        layer_family.push_back(candidate);
    }
    CHECK(layer_family.size() == 2 && layer_family[0].values != layer_family[1].values);
    std::string family_ref;
    CHECK(localization_callbacks.persist_experimental_layer_family_for_job(
        localization_job, layer_family, scan_ref, family_ref, error));
    localization_job.seed.candidate_ref = family_ref;
    std::vector<common_flydelta_basis_direction> family_directions;
    uint32_t origin = 0;
    CHECK(daemon_flydelta_resolve_concept_layer_family(provider, localization_job, family_directions, origin, error));
    CHECK(origin == 2 && family_directions.size() == 2);
    std::string family_json;
    CHECK(resources.read_text(family_ref, provider->authority, 1024 * 1024, family_json, error));
    auto corrupted_family = flydelta_json::parse(family_json);
    corrupted_family["directions"][0]["values"] = {0.0f, 1.0f};
    std::string corrupted_ref;
    CHECK(put(corrupted_family, corrupted_ref));
    auto corrupt_job = localization_job;
    corrupt_job.seed.candidate_ref = corrupted_ref;
    CHECK(!daemon_flydelta_resolve_concept_layer_family(provider, corrupt_job, family_directions, origin, error));
    auto wrong_baseline = localization_job;
    wrong_baseline.seed.baseline_ref = "baseline:other";
    CHECK(!daemon_flydelta_resolve_concept_layer_family(provider, wrong_baseline, family_directions, origin, error));
    arm.intervention_ref = family_ref;
    arm.layer_indices = {2};
    CHECK(binding.prepare_arm(arm, request, error));
    const auto first_overlay = request.flydelta_activation->sparse_overlay.data;
    arm.layer_indices = {4};
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(first_overlay != request.flydelta_activation->sparse_overlay.data);
    arm.layer_indices = {3};
    CHECK(!binding.prepare_arm(arm, request, error));
    std::cout << "daemon_seams=passed controlled_backend=yes model_quality_claim=no\n";
}
