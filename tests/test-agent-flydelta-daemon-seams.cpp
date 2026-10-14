#include "tools/agent/daemon/agent-daemon-flydelta-internal.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <cmath>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { std::cerr << __LINE__ << ": " << error << '\n'; return __LINE__; } } while (false)

using namespace agent_daemon_flydelta_internal;

int main() {
    std::string error;
    agent_catalogued_resource_store resources(std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    auto provider = std::make_shared<daemon_flydelta_resource_provider>();
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
    provider->oracle_registry = common_flydelta_make_default_oracle_registry();
    const auto put = [&](const json & value, std::string & uri) {
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
    const json arguments = {{"dataset", "dataset://local/sales"}, {"group_by", {"region"}},
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
        {"model_tools", {{{"name", "data.aggregate"}, {"parameters", json::parse(schema)}}}}}, arm.context_ref));
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
        {"tools", {{{"name", "data.aggregate"}, {"parameters", json::parse(schema)}}}},
        {"tool_choice", "required"}}, arm.context_ref));
    CHECK(binding.prepare_arm(arm, request, error));
    CHECK(request.tools.size() == 1 && request.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);
    CHECK(put({{"prompt", "Sum amount by region"}, {"tools", json::array()},
        {"model_tools", json::array()}}, arm.context_ref));
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
    CHECK(result.host_evaluated && !result.verifier_known && !result.verifier_passed);
    generated.stop_reason = common_agent_generation_stop_reason::eos;

    common_flydelta_experiment_fixture fixture{1, "fixture:seams", "task", "model", "tokenizer",
        "template", "execution", "verifier"};
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
    std::cout << "daemon_seams=passed controlled_backend=yes model_quality_claim=no\n";
}
