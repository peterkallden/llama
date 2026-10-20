#include "agent-daemon-adapter.h"
#include "../adaptation/semi-oracle-store-factory.h"
#include "agent/adaptation/semi-oracle-tool.h"
#include "agent/adaptation/semi-oracle-target.h"
#include "agent/prediction/outcome-guidance.h"

#include "../cli/agent-cli-host-adapter.h"
#include "../cli/agent-cli-selection.h"
#include "../runtime/agent-inference-capacity-gate.h"
#include "../runtime/agent-model-loaders.h"
#include "../runtime/agent-server-context-host.h"
#include "hash/hash.h"
#include "tools/server/server-context.h"
#include "../mcp/agent-mcp-server-tool-registry.h"
#include "llama.h"
#include "agent/agent-scope.h"
#include "agent/agent-bootstrap.h"
#include "tools/agent/cli/agent-cli-memory-tools.h"
#include "../runtime/agent-runtime-package-io.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "../data/agent-data-store-factory.h"
#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif
#ifdef LLAMA_PLAN_USE_COZO
#include "plan/cozo/plan-cozo.h"
#endif
#ifdef LLAMA_MEMORY_USE_SQLITE
#include "memory/sqlite/memory-sqlite.h"
#endif
#ifdef LLAMA_PLAN_USE_SQLITE
#include "plan/sqlite/plan-sqlite.h"
#endif

#include <cstdio>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "agent-daemon-flydelta.h"

namespace {

// The user-owned turn text is the only automatic source for an initial
// algebra state. The model never supplies state through oracle.action.verify.
// This deliberately recognizes only the small exact-linear surface accepted
// by the V1 evaluator; anything broader remains an explicit host-state need.
bool initial_linear_algebra_state_from_prompt(
        const std::string & prompt, std::string & state_json) {
    static const std::regex equation(
        R"(([0-9xX+\-*/().[:space:]]+)=([0-9xX+\-*/().[:space:]]+))");
    std::smatch match;
    if (!std::regex_search(prompt, match, equation) || match.size() != 3) return false;
    const auto compact = [](std::string value) {
        value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
            return std::isspace(c) != 0;
        }), value.end());
        return value;
    };
    auto lhs = compact(match[1].str());
    auto rhs = compact(match[2].str());
    if (lhs.empty() || rhs.empty()) return false;
    state_json = nlohmann::ordered_json{{"lhs", std::move(lhs)}, {"rhs", std::move(rhs)}}.dump();
    return true;
}

bool open_daemon_model_residency(
        const daemon_options & options,
        std::shared_ptr<common_agent_runtime_model_residency> & residency,
        std::string & error) {
    residency.reset();
    auto catalog = options.model_catalog;
    if (options.adaptation_flydelta_factorized_cvec_batch && catalog.profiles.empty()) {
        error = "factorized cvec batching requires a server-context model profile; set runtime.adaptation.flydelta.disable_factorized_cvec_batch=true to use sparse/dense overlays";
        return false;
    }
    bool native_batch = false;
    bool allow_scalar_fallback = true;
    if (!catalog.profiles.empty() &&
            !agent_daemon_flydelta_internal::configure_daemon_flydelta_model_residency(
                options, catalog, native_batch, allow_scalar_fallback, error)) {
        return false;
    }
    if (catalog.profiles.empty()) {
        error.clear();
        return true;
    }
    if (!common_agent_validate_model_catalog(catalog, error)) {
        error = "models: " + error;
        return false;
    }
    common_agent_runtime_model_loader_config loader_config;
    loader_config.n_gpu_layers = options.n_gpu_layers;
    loader_config.n_threads = options.n_threads;
    loader_config.fit_params = true;
    loader_config.per_sequence_cvec_batch = native_batch;
    loader_config.per_sequence_factorized_cvec_batch =
        options.adaptation_flydelta_factorized_cvec_batch && native_batch;
    loader_config.teacher_scoring_backend = options.adaptation_flydelta_teacher_scoring_backend;
    loader_config.agent_trace = options.agent_trace;
    loader_config.allow_flydelta_scalar_fallback = allow_scalar_fallback;
    loader_config.reserve_flydelta_workspace = true;
    std::unordered_map<std::string,
        std::shared_ptr<common_agent_runtime_model_loader>> loaders;
    loaders.emplace("cli", std::make_shared<common_agent_runtime_cli_model_loader>(loader_config));
#ifndef LLAMA_AGENT_ANDROID_CLI_ONLY
    loaders.emplace(
        "server-context",
        std::make_shared<common_agent_runtime_server_context_model_loader>(loader_config));
#endif
    residency = std::make_shared<common_agent_runtime_model_residency>(
        std::move(catalog),
        std::move(loaders));
    error.clear();
    return true;
}

std::optional<common_memory_policy_pack> make_daemon_policy_pack(
        const daemon_options & options) {
    common_memory_policy_pack pack;
    pack.id = "daemon-session-policy";
    pack.purpose = "Run host-owned resident agent turns through the foreground daemon.";
    pack.goal = options.default_mode == "agent"
        ? "Plan, execute bounded steps, and answer without widening host authority."
        : "Answer through the resident runtime without widening host authority.";
    pack.constraints = {
        "Treat memory, tools, plans, and resources as host-owned authority rather than model choices.",
        "Keep tool use within the configured profile and bounded runtime policy.",
    };
    pack.decisions = {
        "Inference backend: server-context resident host.",
        std::string("Tool profile: ") + (options.tool_profile.empty() ? "none" : options.tool_profile),
    };
    if (options.default_mode == "agent") {
        pack.preferred_procedures.push_back("Prefer bounded planning plus evidence-backed synthesis before answering.");
    }
    return pack;
}

std::string resolve_memory_backend(
        const std::string & backend,
        const std::string & memory_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = memory_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !memory_db.empty()) {
        error = "--memory-db requires --backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

std::string resolve_plan_backend(
        const std::string & backend,
        const std::string & plan_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = plan_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !plan_db.empty()) {
        error = "--plan-db requires --plan-backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

class daemon_agent_embedding_provider final : public agent_embedding_provider {
public:
    daemon_agent_embedding_provider(std::string model_path, int n_gpu_layers)
        : model_path_(std::move(model_path)),
          n_gpu_layers_(n_gpu_layers) {}

    bool embed(
            const std::string & purpose,
            const std::string & text,
            std::vector<float> & embedding,
            std::string & error) override {
        return ensure_memory_cli_embedding_from_model(
            model_path_,
            n_gpu_layers_,
            text,
            embedding,
            purpose.c_str(),
            error);
    }

private:
    std::string model_path_;
    int n_gpu_layers_ = 0;
};

common_agent_runtime_policy make_daemon_runtime_policy(const daemon_options & options) {
    common_agent_runtime_policy policy;
    policy.agent_inference_backend = "server-context";
    policy.tool_profile = options.tool_profile;
    policy.memory_learn = options.memory_learn;
    policy.memory_learn_show_candidate = options.memory_learn_show_candidate;
    policy.plan_show_summary = options.plan_show_summary;
    policy.agent_trace = options.agent_trace;
    policy.enable_reflection = true;
    policy.max_iterations = 2;
    policy.max_reflection_rounds = 1;
    // A zero value in host/bootstrap configuration means "use the safe
    // default". Keep daemon turns consistent with the CLI/runtime policy;
    // callers can still select a smaller positive budget explicitly.
    policy.max_tool_rounds = options.max_tool_rounds > 0
        ? options.max_tool_rounds
        : 16;
    common_tool_profile_snapshot profile_snapshot;
    std::string profile_error;
    if (resolve_common_tool_profile_snapshot(
            options.tool_profile,
            options.tool_capabilities,
            options.tool_profiles,
            profile_snapshot,
            profile_error)) {
        policy.allow_policy_gated_tool_proposals =
            profile_snapshot.allow_policy_gated_writes.value_or(false);
    }
    std::string deliberation_error;
    common_agent_deliberation_policy deliberation_policy;
    if (resolve_common_agent_deliberation_policy(
            options.thinking_mode,
            options.max_reflection_rounds,
            options.max_plan_revisions,
            options.max_research_iterations,
            deliberation_policy,
            deliberation_error)) {
        deliberation_policy.max_tool_rounds = options.max_tool_rounds > 0
            ? static_cast<int>(options.max_tool_rounds)
            : deliberation_policy.max_tool_rounds;
        policy.deliberation_policy = deliberation_policy;
    }
    return policy;
}

common_agent_runtime_config make_daemon_runtime_config(
        const daemon_options & options,
        const common_agent_daemon_runtime & runtime) {
    common_agent_runtime_config config;
    config.generation_config.n_predict = options.n_predict;
    config.generation_config.memory_learning_n_predict = options.memory_learning_n_predict;
    config.generation_config.n_threads = options.n_threads;
    config.generation_config.agent_trace = options.agent_trace;
    config.generation_config.generation_trace = options.generation_trace;
    config.generation_config.context_size_tokens = static_cast<size_t>(std::max(0, options.context_size));
    // Keep daemon turns on the same host-owned family preflight path as the
    // CLI.  The daemon already exposes agent_plan=auto, but previously only
    // stored that value in orchestration config; the runtime generation flag
    // remained disabled and went straight to the full planner.
    config.generation_config.enable_tool_family_routing = options.agent_plan == "auto";
    config.generation_config.context_budgets = options.context_budgets;
    config.context_budgets = options.context_budgets;
    config.context_management = options.context_management;
    config.max_continuations = options.max_continuations;
    config.enable_memory_learning = options.memory_learn == "post-turn";
    config.memory_learning_config.min_confidence = options.memory_learn_min_confidence;
    config.memory_learning_config.min_expected_reuse = options.memory_learn_min_reuse;
    config.enable_adaptation_capture = options.adaptation_capture;
    config.adaptation_transaction_backend = options.adaptation_transaction_backend;
    config.adaptation_transaction_path = options.adaptation_transaction_path;
    config.adaptation_config.collection_allowed = options.adaptation_collection_allowed;
    config.adaptation_config.max_evidence = options.adaptation_max_evidence;
    config.adaptation_config.cause_classifier.stable_model_facing_tools =
        options.adaptation_stable_model_facing_tools;
    config.adaptation_config.domain_policy = options.adaptation_domains;
    config.procedure_teaching_request_provider =
        runtime.procedure_teaching_request_provider;
    config.user_correction_teaching_request_provider =
        runtime.user_correction_teaching_request_provider;
    config.user_taught_concept_relation_provider =
        runtime.user_taught_concept_relation_provider;
    config.semi_oracle_teaching_relation_provider =
        runtime.semi_oracle_teaching_relation_provider;
    config.semantic_concept_hypothesis_provider =
        runtime.semantic_concept_hypothesis_provider;
    config.semantic_concept_hypothesis_batch_provider =
        runtime.semantic_concept_hypothesis_batch_provider;
    if (!config.semantic_concept_hypothesis_batch_provider &&
            !runtime.semantic_concept_hypothesis_provider) {
        config.semantic_concept_hypothesis_batch_provider =
            common_agent_make_research_concept_hypothesis_batch_provider();
    }
    config.semantic_concept_contrast_provider =
        runtime.semantic_concept_contrast_provider;
    config.semantic_concept_grounding_provider =
        runtime.semantic_concept_grounding_provider;
    if (!config.semantic_concept_grounding_provider &&
            config.semantic_concept_contrast_provider) {
        config.semantic_concept_grounding_provider =
            common_agent_make_concept_grounding_provider(
                config.semantic_concept_contrast_provider);
    }
    agent_daemon_flydelta_internal::configure_daemon_flydelta_runtime_config(
        options, runtime, config);
    return config;
}

common_agent_orchestration_config make_daemon_orchestration_config(const daemon_options & options) {
    common_agent_orchestration_config config;
    config.prompt = "";
    config.agent_plan = options.agent_plan;
    config.agent_blueprint = options.agent_blueprint;
    return config;
}

common_agent_runtime_resident_request_config make_resident_request_config(
        const daemon_options & options) {
    common_agent_runtime_resident_request_config config{
        "",
        "",
        "",
        "",
        make_daemon_policy_pack(options),
        options.model,
        options.n_predict,
        options.n_gpu_layers,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
        options.n_threads,
        static_cast<size_t>(std::max(0, options.context_size)),
    };
    config.mmproj = options.mmproj;
    config.tool_argument_bindings = make_agent_host_openapi_argument_bindings(
        options.openapi_providers);
    return config;
}

common_memory_query make_daemon_memory_query(
        const common_agent_runtime_session_host_turn_request & request) {
    common_memory_query query;
    query.text = request.prompt;
    query.limit = 8;
    query.token_budget = 768;

    common_agent_scope scope;
    scope.namespace_id = request.namespace_id;
    scope.session_id = request.session_id;
    scope.project_id = request.project_id;
    scope.turn_id = request.turn_id;
    scope.memory_scope = request.memory_scope;
    scope.plan_scope = request.plan_scope;
    common_agent_scope_apply(scope, query);
    return query;
}

agent_host_tool_selection_request make_daemon_tool_request(
        const daemon_options & options,
        const common_agent_runtime_session_host_turn_request & request,
        std::optional<bool> allow_policy_gated_writes) {
    agent_host_tool_selection_request tool_request;
    tool_request.tool_context.request_id = "daemon";
    tool_request.tool_context.turn_id = request.turn_id;
    tool_request.tool_context.scope.namespace_id = request.namespace_id;
    tool_request.tool_context.scope.session_id = request.session_id;
    tool_request.tool_context.scope.project_id = request.project_id;
    tool_request.tool_context.scope.turn_id = request.turn_id;
    tool_request.tool_context.scope.memory_scope = request.memory_scope;
    tool_request.tool_context.scope.plan_scope = request.plan_scope;
    tool_request.tool_context.memory_scope = request.memory_scope;
    tool_request.tool_context.plan_scope = request.plan_scope;
    tool_request.tool_context.profile_id = options.tool_profile;
    tool_request.tool_context.allowed_exposed_tool_names = request.allowed_exposed_tool_names;
    tool_request.tool_context.repository_root = options.repository_root;
    tool_request.tool_context.allow_network =
        has_enabled_mcp_provider(options.mcp_providers) ||
        !options.mcp_tool_command.empty();
    bool policy_gated_writes = false;
    if (allow_policy_gated_writes.has_value()) {
        policy_gated_writes = *allow_policy_gated_writes;
    } else if (request.allow_policy_gated_writes.has_value()) {
        policy_gated_writes = *request.allow_policy_gated_writes;
    }
    agent_tool_context_apply_policy_gated_writes(
        tool_request.tool_context, policy_gated_writes);
    tool_request.tool_context.max_calls = options.max_tool_rounds > 0 ? options.max_tool_rounds : 1;
    tool_request.tool_context.execution_control = request.execution_control;
    tool_request.tool_context.default_timeout_ms =
        options.tool_timeout_ms > 0
            ? options.tool_timeout_ms
            : tool_request.tool_context.default_timeout_ms;
    tool_request.repository_root = options.repository_root.empty()
        ? std::string()
        : std::filesystem::weakly_canonical(options.repository_root).string();
    tool_request.diagnostics = options.diagnostics;
    tool_request.resource_store_config = {
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    };
    tool_request.data_store_config = {options.data_backend, options.data_db};
    tool_request.tool_capabilities = options.tool_capabilities;
    tool_request.tool_family_descriptions = options.tool_family_descriptions;
    tool_request.tool_profiles = options.tool_profiles;
    tool_request.sandbox = options.sandbox;
    tool_request.resource_processor_policies = options.resource_processor_policies;
    append_configured_stdio_mcp_providers(options.mcp_providers, tool_request.mcp_providers);
    tool_request.openapi_providers = options.openapi_providers;
    tool_request.dynamic_openapi_registry = request.dynamic_openapi_registry;
    if (tool_request.mcp_providers.empty()) {
        append_legacy_stdio_mcp_provider(
            options.mcp_tool_command,
            options.mcp_tool_args,
            options.mcp_tool_server_name,
            options.mcp_tool_prefix,
            tool_request.mcp_providers);
    }
    return tool_request;
}

bool daemon_semi_oracle_parse_placement_coordinates(
        const std::string & action_dsl, int & row, int & column) {
    row = -1;
    column = -1;
    std::istringstream lines(action_dsl);
    std::string line;
    while (std::getline(lines, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const auto key = line.substr(0, colon);
        try {
            if (key == "row") row = std::stoi(line.substr(colon + 1));
            if (key == "column") column = std::stoi(line.substr(colon + 1));
        } catch (...) {
            return false;
        }
    }
    return row >= 0 && column >= 0;
}

nlohmann::ordered_json daemon_semi_oracle_result_json(
        const common_semi_oracle_result & result) {
    return {{"verdict", common_semi_oracle_verdict_name(result.verdict)},
        {"ruleset_id", result.ruleset_id}, {"revision", result.ruleset_revision},
        {"evaluator", result.evaluator_ref}, {"evaluator_revision", result.evaluator_revision},
        {"level", common_semi_oracle_level_name(result.level)},
        {"proof", nlohmann::ordered_json::parse(result.proof_json.empty() ? "null" : result.proof_json, nullptr, false)},
        {"counterexample", nlohmann::ordered_json::parse(result.counterexample_json.empty() ? "null" : result.counterexample_json, nullptr, false)}};
}

bool daemon_semi_oracle_result_from_json(const std::string & text,
        common_semi_oracle_result & result) {
    using json = nlohmann::ordered_json;
    const auto value = json::parse(text, nullptr, false);
    if (!value.is_object()) return false;
    result = {};
    result.ruleset_id = value.value("ruleset_id", "");
    result.ruleset_revision = value.value("revision", uint64_t(0));
    result.evaluator_ref = value.value("evaluator", "");
    result.evaluator_revision = value.value("evaluator_revision", "");
    const auto verdict = value.value("verdict", "");
    if (verdict == "SATISFIED") result.verdict = common_semi_oracle_verdict::satisfied;
    else if (verdict == "VIOLATED") result.verdict = common_semi_oracle_verdict::violated;
    else return false;
    const auto proof = value.value("proof", json(nullptr));
    const auto counterexample = value.value("counterexample", json(nullptr));
    result.proof_json = proof.is_object() ? proof.dump() : std::string{};
    result.counterexample_json = counterexample.is_object() ? counterexample.dump() : std::string{};
    return true;
}

bool daemon_semi_oracle_render_algebra_rewrite(const std::string & before,
        const std::string & next, std::string & action_dsl) {
    using json = nlohmann::ordered_json;
    const auto old_state = json::parse(before, nullptr, false);
    const auto new_state = json::parse(next, nullptr, false);
    if (!old_state.is_object() || !new_state.is_object() ||
            !old_state.contains("lhs") || !old_state.contains("rhs") ||
            !new_state.contains("lhs") || !new_state.contains("rhs") ||
            !old_state["lhs"].is_string() || !old_state["rhs"].is_string() ||
            !new_state["lhs"].is_string() || !new_state["rhs"].is_string()) return false;
    action_dsl = "open! algebra.rewrite_equation";
    if (old_state["lhs"] != new_state["lhs"]) action_dsl += "\nlhs: " + new_state["lhs"].get<std::string>();
    if (old_state["rhs"] != new_state["rhs"]) action_dsl += "\nrhs: " + new_state["rhs"].get<std::string>();
    return action_dsl != "open! algebra.rewrite_equation";
}

bool daemon_semi_oracle_make_algebra_repair(common_semi_oracle_tool_service & service,
        const std::string & before, std::string & preferred_dsl,
        common_semi_oracle_result & preferred, std::string & error) {
    common_semi_oracle_result search;
    if (!service.search_active_algebra_goal(search, error) ||
            search.verdict != common_semi_oracle_verdict::satisfied) return false;
    const auto proof = nlohmann::ordered_json::parse(search.proof_json, nullptr, false);
    if (!proof.is_object() || !proof.contains("path") || !proof["path"].is_array() ||
            proof["path"].size() < 2 || !proof["path"][1].is_object() ||
            !proof["path"][1].contains("state")) { error = "algebra A* proof has no first transition"; return false; }
    if (!daemon_semi_oracle_render_algebra_rewrite(before, proof["path"][1]["state"].dump(), preferred_dsl)) {
        error = "algebra A* first transition cannot be rendered as rewrite DSL"; return false;
    }
    return service.verify_action(before, preferred_dsl, common_semi_oracle_level::local,
        {}, {1, 1}, {}, preferred, error) && preferred.verdict == common_semi_oracle_verdict::satisfied;
}

bool daemon_semi_oracle_put_forced_context(
        agent_resource_store & resources,
        const common_agent_request & request,
        const common_semi_oracle_interaction & interaction,
        const std::string & state_json,
        const std::string & action_dsl,
        const char * arm,
        std::string & reference,
        std::string & error) {
    using json = nlohmann::ordered_json;
    const json document{{"messages", json::array({
        {{"role", "system"}, {"content", "Use the active RuleSet action contract."}},
        {{"role", "user"}, {"content", "RuleSet: " + interaction.ruleset_id +
            " revision " + std::to_string(interaction.ruleset_revision) +
            "\nState:\n" + state_json + "\nReturn one action."}},
        {{"role", "assistant"}, {"content", action_dsl}},
    })}, {"capture_mode", "host.forced_context.v1"},
        {"semi_oracle_interaction", interaction.id}, {"arm", arm}};
    const std::string identity = interaction.id + "\n" + arm + "\n" + state_json + "\n" + action_dsl;
    agent_resource_put_request put;
    put.name = "semi-oracle-forced-context-" +
        hash_sha256_hex(identity.data(), identity.size()).substr(0, 24) + ".json";
    put.description = "Host-owned forced-context capture material for a verified SemiOracle contrast";
    put.mime_type = "application/json";
    put.text = document.dump();
    put.scope = common_runtime_resource_scope::session;
    put.namespace_id = request.namespace_id;
    put.project_id = request.project_id;
    put.session_id = request.session_id;
    put.turn_id = request.turn_id;
    put.source_provider = "semi-oracle";
    put.source_tool = "forced-context";
    put.created_at = std::time(nullptr);
    put.metadata.purpose = "semi_oracle_forced_context";
    put.metadata.content_summary = "Teacher-forced SemiOracle action continuation.";
    put.metadata.processing_cache_key = identity;
    put.lineage.parent_uri = interaction.evidence_ref;
    put.lineage.derivation = "host-forced-context";
    agent_resource_descriptor descriptor;
    if (!resources.put_text(put, descriptor, error) || descriptor.uri.empty()) {
        if (error.empty()) error = "SemiOracle forced context has no resource URI";
        return false;
    }
    reference = descriptor.uri;
    return true;
}

common_agent_semi_oracle_teaching_relation_provider
make_daemon_semi_oracle_relation_provider(
        common_semi_oracle_store & store, agent_resource_store & resources) {
    return [&store, &resources](const common_agent_request & request,
            const common_plan_state &, const common_agent_result & result,
            const common_learning_transaction &, std::vector<common_flydelta_teaching_relation> & relations,
            std::string & error) {
        error.clear();
        for (const auto & signal : result.learning_signals) {
            if (signal.type != common_learning_signal_type::semi_oracle_preference ||
                    signal.repair_context_json.empty()) continue;
            const auto context = nlohmann::ordered_json::parse(signal.repair_context_json, nullptr, false);
            const std::string ref = context.is_object()
                ? context.value("semi_oracle_interaction_ref", std::string{}) : std::string{};
            if (ref.empty()) continue; // A verified move without a matched control is diagnostic only.
            common_semi_oracle_interaction interaction;
            bool found = false;
            if (!store.get_interaction(ref, interaction, found, error)) return false;
            if (!found || interaction.session_id != request.session_id) continue;
            common_semi_oracle_ruleset ruleset;
            if (!store.get_revision(interaction.ruleset_id, interaction.ruleset_revision,
                    ruleset, found, error)) return false;
            if (!found || ruleset.scope != common_semi_oracle_scope::session ||
                    ruleset.scope_id != request.session_id) continue;
            common_semi_oracle_tool_service service(store, request.namespace_id, request.session_id);
            if (interaction.domain == "linear_algebra" &&
                    interaction.contrast_kind == "correctness_repair") {
                common_semi_oracle_result rejected, accepted;
                if (!daemon_semi_oracle_result_from_json(interaction.baseline_result_json, rejected) ||
                        !daemon_semi_oracle_result_from_json(interaction.preferred_result_json, accepted)) continue;
                std::string baseline_ref, preferred_ref, control_ref;
                if (!daemon_semi_oracle_put_forced_context(resources, request, interaction,
                        interaction.state_json, interaction.baseline_action_dsl, "baseline", baseline_ref, error) ||
                        !daemon_semi_oracle_put_forced_context(resources, request, interaction,
                        interaction.state_json, interaction.preferred_action_dsl, "preferred", preferred_ref, error) ||
                        !daemon_semi_oracle_put_forced_context(resources, request, interaction,
                        interaction.control_state_json, interaction.control_action_dsl, "control", control_ref, error)) return false;
                common_agent_scope scope;
                scope.namespace_id = request.namespace_id;
                scope.project_id = request.project_id;
                scope.session_id = request.session_id;
                scope.turn_id = request.turn_id;
                common_flydelta_teaching_relation relation;
                if (!common_semi_oracle_make_controlled_teaching_relation(ruleset,
                        rejected, accepted, scope, interaction.id, interaction.task_fingerprint,
                        baseline_ref, preferred_ref, control_ref, interaction.evidence_ref,
                        relation, error)) return false;
                relation.model_profile_fingerprint = "daemon-resident-model-v1";
                relation.tokenizer_fingerprint = "daemon-tokenizer-v1";
                relation.template_fingerprint = "semi-oracle-forced-context-v1";
                relation.capture_layout_revision = "forced-context-boundary-v1";
                relations.push_back(std::move(relation));
                continue;
            }
            if (interaction.domain != "placement_game") continue;
            std::vector<common_semi_oracle_placement_action_value> ranking;
            if (!service.rank_placements(interaction.state_json, {100000, 16}, ranking, error)) return false;
            int baseline_row, baseline_column, preferred_row, preferred_column;
            if (!daemon_semi_oracle_parse_placement_coordinates(interaction.baseline_action_dsl, baseline_row, baseline_column) ||
                    !daemon_semi_oracle_parse_placement_coordinates(interaction.preferred_action_dsl, preferred_row, preferred_column)) continue;
            const auto locate = [&](int row, int column) {
                return std::find_if(ranking.begin(), ranking.end(), [&](const auto & value) {
                    return value.row == row && value.column == column;
                });
            };
            const auto baseline = locate(baseline_row, baseline_column);
            const auto preferred = locate(preferred_row, preferred_column);
            if (baseline == ranking.end() || preferred == ranking.end()) continue;
            std::string baseline_ref, preferred_ref, control_ref;
            if (!daemon_semi_oracle_put_forced_context(resources, request, interaction,
                    interaction.state_json, interaction.baseline_action_dsl, "baseline", baseline_ref, error) ||
                    !daemon_semi_oracle_put_forced_context(resources, request, interaction,
                    interaction.state_json, interaction.preferred_action_dsl, "preferred", preferred_ref, error) ||
                    !daemon_semi_oracle_put_forced_context(resources, request, interaction,
                    interaction.control_state_json, interaction.control_action_dsl, "control", control_ref, error)) return false;
            common_flydelta_teaching_relation relation;
            if (!service.make_preference_relation(*baseline, *preferred, interaction.id,
                    baseline_ref, preferred_ref, control_ref, interaction.evidence_ref, relation, error)) return false;
            relation.model_profile_fingerprint = "daemon-resident-model-v1";
            relation.tokenizer_fingerprint = "daemon-tokenizer-v1";
            relation.template_fingerprint = "semi-oracle-forced-context-v1";
            relation.capture_layout_revision = "forced-context-boundary-v1";
            relations.push_back(std::move(relation));
        }
        return true;
    };
}

common_tool_execution_result daemon_prediction_predict_target_hunt(
        common_semi_oracle_store & store,
        const std::string & namespace_id,
        const std::string & project_id,
        const std::string & session_id,
        const std::string & input) {
    using json = nlohmann::ordered_json;
    const auto args = json::parse(input, nullptr, false);
    if (!args.is_object()) return common_tool_execution_result::failure(
        "tool.prediction.invalid_arguments", common_tool_failure_class::validation, false,
        "Prediction arguments must be an object.", "arguments are not a JSON object");
    if (args.contains("max_candidates") && (!args["max_candidates"].is_number_integer() ||
            args["max_candidates"].get<int64_t>() < 1 || args["max_candidates"].get<int64_t>() > 16))
        return common_tool_execution_result::failure("tool.prediction.invalid_limit",
            common_tool_failure_class::validation, false,
            "max_candidates must be between 1 and 16.", "max_candidates outside supported range");
    const size_t limit = args.value("max_candidates", size_t{8});
    if (!limit || limit > 16) return common_tool_execution_result::failure(
        "tool.prediction.invalid_limit", common_tool_failure_class::validation, false,
        "max_candidates must be between 1 and 16.", "max_candidates outside supported range");
    std::string ruleset_id, error;
    uint64_t revision = 0;
    bool bound = false;
    if (!store.get_session_binding(session_id, ruleset_id, revision, bound, error))
        return common_tool_execution_result::failure("tool.prediction.history_unavailable",
            common_tool_failure_class::execution, true,
            "Target Hunt history is temporarily unavailable.", std::move(error));
    if (!bound) return common_tool_execution_result::failure("tool.prediction.no_active_target_hunt",
        common_tool_failure_class::policy, false,
        "There is no active Target Hunt RuleSet in this session.", "missing session RuleSet binding");
    common_semi_oracle_ruleset ruleset;
    bool found = false;
    if (!store.get_revision(ruleset_id, revision, ruleset, found, error))
        return common_tool_execution_result::failure("tool.prediction.history_unavailable",
            common_tool_failure_class::execution, true,
            "Target Hunt history is temporarily unavailable.", std::move(error));
    if (!found || ruleset.domain != common_semi_oracle_domain::target_hunt)
        return common_tool_execution_result::failure("tool.prediction.wrong_family",
            common_tool_failure_class::policy, false,
            "The active Oracle problem is not Target Hunt.", "active RuleSet is not target_hunt");

    std::vector<common_semi_oracle_search_history_record> stored;
    uint64_t watermark = 0;
    bool watermark_supported = false;
    if (!store.read_search_history_snapshot(ruleset_id, revision, 1024, stored,
            watermark, watermark_supported, error))
        return common_tool_execution_result::failure("tool.prediction.history_unavailable",
            common_tool_failure_class::execution, true,
            "Target Hunt history is temporarily unavailable.", std::move(error));

    const std::string scope_material = json::array({namespace_id, project_id, session_id}).dump();
    const std::string scope = "sha256:" + hash_sha256_hex(scope_material.data(), scope_material.size());
    struct target_row {
        std::string ref;
        std::string evaluation_id;
        std::string fixture;
        std::string fine;
        std::string coarse;
        std::string outcome;
        std::string identity;
    };
    std::vector<target_row> candidates;
    std::vector<common_outcome_guidance_projection_observation> fine_rows, coarse_rows;
    json canonical_rows = json::array();
    std::map<std::string, std::string> signatures;
    std::set<std::string> conflicted;
    for (const auto & entry : stored) {
        const auto guidance = json::parse(entry.guidance_json, nullptr, false);
        if (!guidance.is_object() || !guidance.contains("context") || !guidance["context"].is_object() ||
                guidance["context"].value("scope", std::string{}) != scope ||
                !guidance.contains("branch_quality") || !guidance["branch_quality"].is_object()) continue;
        const auto & branch = guidance["branch_quality"];
        if (branch.value("family", std::string{}) != "target_hunt_branch_quality_v1" ||
                branch.value("ruleset_id", std::string{}) != ruleset_id ||
                branch.value("ruleset_revision", uint64_t{0}) != revision ||
                branch.value("ruleset_fingerprint", std::string{}) != ruleset.semantic_fingerprint ||
                branch.value("fixture_group", std::string{}).empty() ||
                branch.value("feature_version", std::string{}) != "target_hunt_features_v3" ||
                branch.value("truth_pass_version", std::string{}) != "reverse_bfs_regret_v1" ||
                !branch.contains("observations") || !branch["observations"].is_array()) continue;
        const auto fixture = branch["fixture_group"].get<std::string>();
        for (const auto & row : branch["observations"]) {
            if (!row.is_object() || !row.contains("fine_category") || !row["fine_category"].is_string() ||
                    !row.contains("coarse_category") || !row["coarse_category"].is_string() ||
                    !row.contains("outcome") || !row["outcome"].is_string() ||
                    !row.contains("state_before") || !row["state_before"].is_string() ||
                    !row.contains("action") || !row["action"].is_string()) continue;
            const auto outcome = row["outcome"].get<std::string>();
            const size_t outcome_index = outcome == "optimal" ? 0 : outcome == "detour" ? 1 :
                outcome == "unreachable" ? 2 : 3;
            if (outcome_index > 2) continue;
            const std::string identity = json::array({entry.evaluation_id,
                row["state_before"], row["action"]}).dump();
            const std::string signature = json::array({fixture, row["fine_category"],
                row["coarse_category"], outcome}).dump();
            const auto inserted = signatures.emplace(identity, signature);
            if (!inserted.second) {
                if (inserted.first->second != signature) conflicted.insert(identity);
                continue;
            }
            const std::string ref_material = json::array({namespace_id, project_id, session_id,
                ruleset_id, revision, entry.evaluation_id, row["state_before"], row["action"]}).dump();
            target_row candidate;
            candidate.ref = "outcome:target-hunt:sha256:" +
                hash_sha256_hex(ref_material.data(), ref_material.size());
            candidate.evaluation_id = entry.evaluation_id;
            candidate.fixture = fixture;
            candidate.fine = row["fine_category"].get<std::string>();
            candidate.coarse = row["coarse_category"].get<std::string>();
            candidate.outcome = outcome;
            candidate.identity = identity;
            candidates.push_back(std::move(candidate));
        }
    }
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [&](const auto & candidate) { return conflicted.count(candidate.identity) != 0; }),
        candidates.end());
    for (const auto & candidate : candidates) {
        canonical_rows.push_back({{"evaluation_id",candidate.evaluation_id},
            {"fixture_group",candidate.fixture},{"fine_category",candidate.fine},
            {"coarse_category",candidate.coarse},{"outcome",candidate.outcome},
            {"identity",candidate.identity}});
        std::vector<double> weights(3, 0.0);
        weights[candidate.outcome == "optimal" ? 0 : candidate.outcome == "detour" ? 1 : 2] = 1.0;
        fine_rows.push_back({candidate.fixture, candidate.fine, weights});
        coarse_rows.push_back({candidate.fixture, candidate.coarse, weights});
    }
    const std::string fingerprint_material = json{{"family","target_hunt_branch_quality_v1"},
        {"ruleset",ruleset.semantic_fingerprint},{"scope",scope},{"watermark",watermark},
        {"rows",canonical_rows}}.dump();
    const std::string history_fingerprint = "sha256:" + hash_sha256_hex(
        fingerprint_material.data(), fingerprint_material.size());
    std::vector<std::string> requested;
    if (args.contains("candidate_refs")) {
        const auto & refs = args["candidate_refs"];
        if (!refs.is_array() || refs.size() > 16) return common_tool_execution_result::failure(
            "tool.prediction.invalid_refs", common_tool_failure_class::validation, false,
            "candidate_refs must contain at most 16 opaque references.", "invalid candidate_refs array");
        for (const auto & ref : refs) {
            if (!ref.is_string() || ref.get<std::string>().empty() || ref.get<std::string>().size() > 256)
                return common_tool_execution_result::failure(
                "tool.prediction.invalid_refs", common_tool_failure_class::validation, false,
                "candidate_refs must contain opaque string references.", "non-string candidate ref");
            requested.push_back(ref.get<std::string>());
        }
    } else {
        for (auto it = candidates.rbegin(); it != candidates.rend() && requested.size() < limit; ++it)
            requested.push_back(it->ref);
    }
    std::set<std::string> handled;
    json predictions = json::array(), unresolved = json::array();
    size_t evidence_groups = 0;
    const auto project = [&](const auto & rows, common_outcome_guidance_projection & out,
            std::string & projection_error) {
        return common_outcome_guidance_build_projection(rows,
            "target-hunt-retrospective-v1", history_fingerprint, watermark, out, projection_error);
    };
    common_outcome_guidance_projection fine_projection, coarse_projection;
    if (!project(fine_rows, fine_projection, error) ||
            !project(coarse_rows, coarse_projection, error))
        return common_tool_execution_result::failure("tool.prediction.history_invalid",
            common_tool_failure_class::execution, false,
            "Target Hunt history did not satisfy its prediction contract.", std::move(error));
    for (const auto & ref : requested) {
        if (!handled.insert(ref).second) continue;
        const auto target = std::find_if(candidates.begin(), candidates.end(),
            [&](const auto & candidate) { return candidate.ref == ref; });
        if (target == candidates.end()) { unresolved.push_back(ref); continue; }
        std::map<std::string, std::vector<double>> fine_weights, coarse_weights;
        std::map<std::string, size_t> fine_groups, coarse_groups;
        fine_projection.aggregate_excluding(target->fixture, fine_weights, fine_groups);
        coarse_projection.aggregate_excluding(target->fixture, coarse_weights, coarse_groups);
        const bool use_fine = fine_groups[target->fine] >= 8;
        const auto & selected = use_fine ? fine_weights : coarse_weights;
        const auto & selected_groups = use_fine ? fine_groups : coarse_groups;
        const auto key = use_fine ? target->fine : target->coarse;
        const auto group_count = selected_groups.find(key);
        const size_t candidate_groups = group_count == selected_groups.end() ? 0 : group_count->second;
        evidence_groups += candidate_groups;
        const auto found_weights = selected.find(key);
        std::vector<double> weights = found_weights == selected.end()
            ? std::vector<double>{0.0,0.0,0.0} : found_weights->second;
        const double known = weights[0] + weights[1] + weights[2];
        common_outcome_guidance_posterior posterior;
        if (!common_outcome_guidance_make_posterior(known, 0.0, 1.0,
                weights, {1.0/3.0,1.0/3.0,1.0/3.0}, 0.0, 0.0, posterior)) {
            unresolved.push_back(ref); continue;
        }
        predictions.push_back({{"candidate_ref",ref},{"family","target_hunt_branch_quality_v1"},
            {"prediction_kind","retrospective_leave_fixture_group_out"},
            {"p_known",posterior.p_known},{"p_optimal_given_known",posterior.category_probabilities[0]},
            {"p_detour_given_known",posterior.category_probabilities[1]},
            {"p_unreachable_given_known",posterior.category_probabilities[2]},
            {"expected_utility",posterior.category_probabilities[0]-posterior.category_probabilities[2]},
            {"evidence_groups",candidate_groups},{"backoff_level",use_fine?"fine_feature":"coarse_feature"},
            {"history_fingerprint",history_fingerprint},{"observed_outcome",target->outcome}});
    }
    return common_tool_execution_result::success(json{{"family","target_hunt_branch_quality_v1"},
        {"status",evidence_groups?"available":"insufficient_history"},
        {"scope_mode","session"},{"scope_fingerprint",scope},{"history_watermark",watermark},
        {"history_fingerprint",history_fingerprint},{"predictions",std::move(predictions)},
        {"unresolved_refs",std::move(unresolved)}}.dump());
}

common_tool_execution_result daemon_prediction_predict(
        common_learning_lifecycle_store * store,
        common_semi_oracle_store * semi_oracle_store,
        common_flydelta_outcome_guidance_config config,
        const common_outcome_guidance_generalization_config & generalization,
        const std::string & namespace_id,
        const std::string & project_id,
        const std::string & session_id,
        const std::string & input) {
    using json = nlohmann::ordered_json;
    const auto args = json::parse(input, nullptr, false);
    if (!args.is_object()) return common_tool_execution_result::failure(
        "tool.prediction.invalid_arguments", common_tool_failure_class::validation, false,
        "Prediction arguments must be an object.", "arguments are not a JSON object");
    const auto family = args.value("family", std::string("flydelta"));
    if (family == "target_hunt") {
        if (semi_oracle_store == nullptr) return common_tool_execution_result::failure(
            "tool.prediction.family_unavailable", common_tool_failure_class::policy, false,
            "Target Hunt outcome history is unavailable in this runtime.", "no SemiOracle store is bound");
        return daemon_prediction_predict_target_hunt(*semi_oracle_store, namespace_id,
            project_id, session_id, input);
    }
    if (family != "flydelta") return common_tool_execution_result::failure(
        "tool.prediction.invalid_family", common_tool_failure_class::validation, false,
            "family must be flydelta or target_hunt.", "unsupported prediction family");
    if (store == nullptr) return common_tool_execution_result::failure(
        "tool.prediction.family_unavailable", common_tool_failure_class::policy, false,
        "FlyDelta outcome history is unavailable in this runtime.", "no lifecycle store is bound");
    // This is an explicit, policy-authorized read through the ordinary tool
    // view. FlyDelta's operator mode controls its automatic search adviser;
    // it does not disable a caller's read-only prediction tool. Reuse the
    // statistical settings with shadow semantics so this call has no search
    // or lifecycle side effects.
    config.mode = common_flydelta_outcome_guidance_mode::shadow;
    if (args.contains("max_candidates") && !args["max_candidates"].is_number_unsigned() &&
            !args["max_candidates"].is_number_integer()) return common_tool_execution_result::failure(
        "tool.prediction.invalid_limit", common_tool_failure_class::validation, false,
        "max_candidates must be an integer.", "max_candidates is not an integer");
    const size_t limit = args.value("max_candidates", size_t{8});
    if (limit == 0 || limit > 16) return common_tool_execution_result::failure(
        "tool.prediction.invalid_limit", common_tool_failure_class::validation, false,
        "max_candidates must be between 1 and 16.", "max_candidates outside supported range");
    std::vector<common_learning_lifecycle_record> records;
    uint64_t watermark = 0;
    std::string error;
    if (generalization.project_history && project_id.empty()) return common_tool_execution_result::failure(
        "tool.prediction.project_scope_required", common_tool_failure_class::policy, false,
        "Project history is enabled but this request has no project scope.", "project_history requires project_id");
    if (!(generalization.project_history
            ? store->list_project_with_watermark(namespace_id, project_id, records, watermark, error)
            : store->list_scope_with_watermark(namespace_id, project_id, session_id, records, watermark, error)))
        return common_tool_execution_result::failure(
        "tool.prediction.history_unavailable", common_tool_failure_class::execution, true,
        "Historical outcome guidance is temporarily unavailable.", std::move(error));

    struct row_entry {
        common_flydelta_outcome_guidance_observation observation;
        std::string opaque_ref;
        uint64_t registration_order = 0;
        std::string source_session;
    };
    std::vector<row_entry> rows;
    for (const auto & record : records) {
        if (record.kind != common_learning_lifecycle_kind::flydelta_result ||
                (record.status != common_learning_lifecycle_status::observed &&
                 record.status != common_learning_lifecycle_status::rejected) ||
                record.registration_order == 0) continue;
        try {
            const auto payload = json::parse(record.payload_json, nullptr, false);
            if (!payload.is_object() || payload.value("record_type", std::string{}) != "candidate_search_decision" ||
                    !payload.value("host_evaluated", false) || !payload.contains("outcome_guidance") ||
                    !payload["outcome_guidance"].is_object()) continue;
            const auto & g = payload["outcome_guidance"];
            if (!g.contains("schema_version") || !g["schema_version"].is_number_integer() ||
                    g["schema_version"].get<int>() != 1 ||
                    !g.value("comparison_completed", false) || !g.value("overlay_applied", false)) continue;
            if (!g.contains("rank") || (!g["rank"].is_number_unsigned() && !g["rank"].is_number_integer()) ||
                    !g.contains("relative_layer") || !g["relative_layer"].is_number() ||
                    !g.contains("relative_dose") || !g["relative_dose"].is_number() ||
                    !g.contains("mixed_coefficients") || !g["mixed_coefficients"].is_boolean() ||
                    !g.contains("diagnostic_margin_sign") || !g["diagnostic_margin_sign"].is_number_integer() ||
                    !g.contains("outcome_known") || !g["outcome_known"].is_boolean() ||
                    !payload.contains("verifier_known") || !payload["verifier_known"].is_boolean()) continue;
            row_entry entry;
            entry.registration_order = record.registration_order;
            entry.source_session = record.session_id;
            auto & o = entry.observation;
            o.stable_observation_id = g.value("observation_id", std::string{});
            o.fixture_group_id = g.value("fixture_group_id", std::string{});
            o.context.scope_fingerprint = g.value("scope_fingerprint", std::string{});
            // The lifecycle envelope, not payload text, authorizes the
            // namespace/project/session read. Older payloads do not persist
            // all scope-policy bits needed to reconstruct their scope hash.
            if (generalization.project_history) {
                const std::string project_scope_material = json::array({namespace_id, project_id}).dump();
                o.context.scope_fingerprint = "sha256:" + hash_sha256_hex(
                    project_scope_material.data(), project_scope_material.size());
            }
            o.context.model_profile_fingerprint = g.value("model_profile_fingerprint", std::string{});
            o.context.tokenizer_fingerprint = g.value("tokenizer_fingerprint", std::string{});
            o.context.template_fingerprint = g.value("template_fingerprint", std::string{});
            o.context.baseline_deployment_fingerprint = g.value("baseline_deployment_fingerprint", std::string{});
            o.context.verifier_revision = g.value("verifier_revision", std::string{});
            o.context.ruleset_revision = g.value("ruleset_revision", std::string{});
            o.context.action_contract_fingerprint = g.value("action_contract_fingerprint", std::string{});
            o.context.semantic_domain = g.value("domain", std::string{});
            o.context.strategy = g.value("strategy", std::string{});
            o.context.estimator = g.value("estimator", std::string{});
            o.basis_identity = g.value("basis_id", std::string{});
            o.rank = g.value("rank", size_t{0});
            o.relative_layer = g.value("relative_layer", 0.0);
            o.relative_dose = g.value("relative_dose", 0.0);
            o.mixed_coefficients = g.value("mixed_coefficients", false);
            o.diagnostic_margin_sign = g.value("diagnostic_margin_sign", 0);
            o.comparison_completed = g.value("comparison_completed", false);
            o.overlay_applied = g.value("overlay_applied", false);
            if (o.rank == 0 || o.rank > 16 || !std::isfinite(o.relative_layer) ||
                    !std::isfinite(o.relative_dose) || o.relative_layer < 0.0 || o.relative_layer > 1.0 ||
                    o.relative_dose < 0.0 || o.relative_dose > 1.0 ||
                    o.diagnostic_margin_sign < -1 || o.diagnostic_margin_sign > 1 ||
                    o.stable_observation_id.empty() || o.stable_observation_id.size() > 512) continue;
            const auto outcome = payload.value("outcome", std::string("unknown"));
            if (outcome == "helped") o.outcome = common_flydelta_counterfactual_outcome::helped;
            else if (outcome == "neutral") o.outcome = common_flydelta_counterfactual_outcome::neutral;
            else if (outcome == "harmed") o.outcome = common_flydelta_counterfactual_outcome::harmed;
            else if (outcome == "unknown") o.outcome = common_flydelta_counterfactual_outcome::unknown;
            else continue;
            // Lifecycle rejection is the host's disposition for a verified
            // HARMED candidate, not evidence that the comparison is unusable.
            // Keep it in the posterior while rejecting inconsistent records.
            if ((record.status == common_learning_lifecycle_status::rejected) !=
                    (o.outcome == common_flydelta_counterfactual_outcome::harmed)) continue;
            o.outcome_known = g.value("outcome_known", false);
            o.verifier_known = payload.value("verifier_known", false);
            if (o.outcome_known != (o.outcome != common_flydelta_counterfactual_outcome::unknown) ||
                    (o.outcome_known && !o.verifier_known) || o.rank == 0 ||
                    o.context.scope_fingerprint.empty() || o.context.model_profile_fingerprint.empty() ||
                    o.context.tokenizer_fingerprint.empty() || o.context.template_fingerprint.empty() ||
                    o.context.baseline_deployment_fingerprint.empty() || o.context.verifier_revision.empty() ||
                    o.context.semantic_domain.empty() || o.context.strategy.empty() || o.context.estimator.empty() ||
                    o.fixture_group_id.empty() ||
                    (!o.basis_identity.empty() && o.basis_identity.size() > 512) ||
                    (o.basis_identity.empty() && !(o.context.strategy == "whirlpool" &&
                        o.context.estimator == "layer-region-prior-v2"))) continue;
            const std::string ref_material = json::array({
                record.namespace_id, record.project_id, record.session_id, o.stable_observation_id}).dump();
            entry.opaque_ref = "outcome:sha256:" + hash_sha256_hex(
                ref_material.data(), ref_material.size());
            rows.push_back(std::move(entry));
        } catch (...) {
            // Invalid legacy records are ignored; they are not model-visible evidence.
        }
    }

    std::vector<std::string> requested;
    if (args.contains("candidate_refs")) {
        const auto & refs = args["candidate_refs"];
        if (!refs.is_array() || refs.size() > 16) return common_tool_execution_result::failure(
            "tool.prediction.invalid_refs", common_tool_failure_class::validation, false,
            "candidate_refs must contain at most 16 observation references.", "invalid candidate_refs array");
        for (const auto & ref : refs) {
            if (!ref.is_string() || ref.get<std::string>().empty()) return common_tool_execution_result::failure(
                "tool.prediction.invalid_refs", common_tool_failure_class::validation, false,
                "candidate_refs contains an invalid observation reference.", "reference must be a non-empty string");
            if (ref.get<std::string>().size() > 256) return common_tool_execution_result::failure(
                "tool.prediction.invalid_refs", common_tool_failure_class::validation, false,
                "candidate_refs contains an invalid observation reference.", "reference exceeds 256 characters");
            requested.push_back(ref.get<std::string>());
        }
    } else {
        std::set<std::string> seen;
        for (auto it = rows.rbegin(); it != rows.rend() && requested.size() < limit; ++it)
            if (it->source_session == session_id && seen.insert(it->opaque_ref).second)
                requested.push_back(it->opaque_ref);
    }

    json predictions = json::array();
    std::set<std::string> handled;
    json unresolved_refs = json::array();
    size_t effective_groups = 0;
    std::string fingerprints;
    const auto same_observation = [](const auto & a, const auto & b) {
        return a.stable_observation_id == b.stable_observation_id &&
            a.fixture_group_id == b.fixture_group_id && a.basis_identity == b.basis_identity &&
            a.context.scope_fingerprint == b.context.scope_fingerprint &&
            a.context.model_profile_fingerprint == b.context.model_profile_fingerprint &&
            a.context.tokenizer_fingerprint == b.context.tokenizer_fingerprint &&
            a.context.template_fingerprint == b.context.template_fingerprint &&
            a.context.baseline_deployment_fingerprint == b.context.baseline_deployment_fingerprint &&
            a.context.verifier_revision == b.context.verifier_revision &&
            a.context.ruleset_revision == b.context.ruleset_revision &&
            a.context.action_contract_fingerprint == b.context.action_contract_fingerprint &&
            a.context.semantic_domain == b.context.semantic_domain && a.context.strategy == b.context.strategy &&
            a.context.estimator == b.context.estimator && a.rank == b.rank &&
            a.relative_layer == b.relative_layer && a.relative_dose == b.relative_dose &&
            a.mixed_coefficients == b.mixed_coefficients &&
            a.diagnostic_margin_sign == b.diagnostic_margin_sign &&
            a.comparison_completed == b.comparison_completed && a.overlay_applied == b.overlay_applied &&
            a.outcome_known == b.outcome_known && a.verifier_known == b.verifier_known && a.outcome == b.outcome;
    };
    for (const auto & ref : requested) {
        if (!handled.insert(ref).second) continue;
        const row_entry * target = nullptr;
        bool conflict = false;
        for (const auto & row : rows) if (row.opaque_ref == ref && row.source_session == session_id) {
            if (target && !same_observation(target->observation, row.observation)) {
                conflict = true;
                break;
            }
            target = &row;
        }
        if (!target || conflict) {
            unresolved_refs.push_back(ref);
            continue;
        }
        std::vector<common_flydelta_outcome_guidance_observation> history;
        for (const auto & row : rows) {
            const auto & a = row.observation.context;
            const auto & b = target->observation.context;
            // list_scope_with_watermark() has already restricted this snapshot
            // to the current namespace/project/session. Keep the chronological
            // and leave-one-fixture-group-out checks here, without reconstructing
            // scope from delimiter-joined strings.
            if (row.registration_order < target->registration_order &&
                    row.observation.fixture_group_id != target->observation.fixture_group_id &&
                    a.scope_fingerprint == b.scope_fingerprint && a.model_profile_fingerprint == b.model_profile_fingerprint &&
                    a.tokenizer_fingerprint == b.tokenizer_fingerprint && a.template_fingerprint == b.template_fingerprint &&
                    a.baseline_deployment_fingerprint == b.baseline_deployment_fingerprint && a.verifier_revision == b.verifier_revision &&
                    a.ruleset_revision == b.ruleset_revision && a.action_contract_fingerprint == b.action_contract_fingerprint)
                history.push_back(row.observation);
        }
        common_flydelta_outcome_guidance_candidate candidate;
        candidate.arm_id = ref;
        candidate.rank = target->observation.rank;
        candidate.relative_layer = target->observation.relative_layer;
        candidate.relative_dose = target->observation.relative_dose;
        candidate.mixed_coefficients = target->observation.mixed_coefficients;
        candidate.diagnostic_margin_sign = target->observation.diagnostic_margin_sign;
        candidate.basis_identity = target->observation.basis_identity;
        common_flydelta_outcome_guidance_decision decision;
        std::string predict_error;
        common_flydelta_outcome_guidance_projection_ptr projection;
        if (!common_flydelta_outcome_guidance_build_projection(target->observation.context,
                history, projection, predict_error, target->observation.fixture_group_id) ||
                !common_flydelta_outcome_guidance_decide_from_projection(config,
                target->observation.context, projection, {candidate}, 1, 0, decision, predict_error) ||
                decision.predictions.size() != 1) {
            unresolved_refs.push_back(ref);
            continue;
        }
        auto prediction = decision.predictions.front();
        std::string backoff_level = prediction.level;
        std::string family_id;
        std::vector<std::string> family_domains;
        if (generalization.tool_family_backoff) {
            for (const auto & family : generalization.tool_families) {
                if (std::find(family.second.begin(), family.second.end(),
                        target->observation.context.semantic_domain) != family.second.end()) {
                    family_id = family.first;
                    family_domains = family.second;
                    break;
                }
            }
        }
        if (!family_id.empty() && std::max(prediction.fixture_groups,
                prediction.backoff_fixture_groups) < config.min_backoff_groups) {
            auto family_context = target->observation.context;
            family_context.semantic_domain = "tool-family:" + family_id;
            std::vector<common_flydelta_outcome_guidance_observation> family_history;
            std::set<std::string> other_domain_groups;
            for (auto row : history) {
                if (std::find(family_domains.begin(), family_domains.end(),
                        row.context.semantic_domain) == family_domains.end()) continue;
                if (row.context.semantic_domain != target->observation.context.semantic_domain)
                    other_domain_groups.insert(row.fixture_group_id);
                row.context.semantic_domain = family_context.semantic_domain;
                family_history.push_back(std::move(row));
            }
            common_flydelta_outcome_guidance_projection_ptr family_projection;
            common_flydelta_outcome_guidance_decision family_decision;
            if (!other_domain_groups.empty() &&
                    common_flydelta_outcome_guidance_build_projection(family_context, family_history,
                    family_projection, predict_error, target->observation.fixture_group_id) &&
                    common_flydelta_outcome_guidance_decide_from_projection(config, family_context,
                        family_projection, {candidate}, 1, 0, family_decision, predict_error) &&
                    family_decision.predictions.size() == 1 &&
                    std::max(family_decision.predictions.front().fixture_groups,
                        family_decision.predictions.front().backoff_fixture_groups) >= config.min_backoff_groups) {
                prediction = family_decision.predictions.front();
                decision.history_fingerprint = family_decision.history_fingerprint;
                backoff_level = "tool_family_backoff";
            }
        }
        const auto & p = prediction;
        const size_t reported_groups = std::max(p.fixture_groups,
            (p.backoff_prior_strength > 0 || p.backoff_categorical_prior_strength > 0)
                ? p.backoff_fixture_groups : size_t{0});
        effective_groups += reported_groups;
        fingerprints += ref;
        fingerprints.push_back('\0');
        fingerprints += decision.history_fingerprint;
        fingerprints.push_back('\n');
        predictions.push_back({
            {"observation_ref", ref}, {"family", "flydelta_intervention_v1"},
            {"p_known", p.p_known},
            {"p_known_interval", {p.p_known_low, p.p_known_high}},
            {"p_helped_given_known", p.p_helped_given_known},
            {"p_neutral_given_known", p.p_neutral_given_known},
            {"p_harmed_given_known", p.p_harmed_given_known},
            {"expected_utility", p.expected_utility},
            {"evidence_groups", reported_groups},
            {"backoff_level", backoff_level}, {"tool_family", family_id},
            {"history_fingerprint", decision.history_fingerprint},
            {"prediction_kind", "chronological_leave_fixture_group_out"},
            {"domain", target->observation.context.semantic_domain},
            {"strategy", target->observation.context.strategy},
            {"estimator", target->observation.context.estimator},
            {"observed_outcome", target->observation.outcome_known ?
                (target->observation.outcome == common_flydelta_counterfactual_outcome::helped ? "helped" :
                 target->observation.outcome == common_flydelta_counterfactual_outcome::neutral ? "neutral" : "harmed") : "unknown"},
        });
    }
    const std::string fingerprint = hash_sha256_hex(fingerprints.data(), fingerprints.size());
    const std::string scope_material = generalization.project_history
        ? json::array({namespace_id, project_id}).dump()
        : json::array({namespace_id, project_id, session_id}).dump();
    const std::string scope_fingerprint = "sha256:" + hash_sha256_hex(
        scope_material.data(), scope_material.size());
    return common_tool_execution_result::success(json{
        {"family", "flydelta_intervention_v1"},
        {"scope_mode", generalization.project_history ? "project" : "session"},
        {"status", effective_groups ? "available" : "insufficient_history"},
        {"scope_fingerprint", scope_fingerprint}, {"history_watermark", watermark},
        {"history_fingerprint", "sha256:" + fingerprint}, {"predictions", std::move(predictions)},
        {"unresolved_refs", std::move(unresolved_refs)},
    }.dump());
}

} // namespace

bool resolve_agent_daemon_tooling(
        const daemon_options & options,
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        agent_resource_store * resource_store,
        common_agent_runtime_tooling & tooling,
        std::string & error,
        common_agent_data_store * data_store,
        std::optional<bool> allow_policy_gated_writes,
        common_semi_oracle_store * semi_oracle_store,
        common_learning_lifecycle_store * learning_lifecycle_store) {
    tooling = {};
    common_memory_query query = make_daemon_memory_query(request);
    auto tool_request = make_daemon_tool_request(
        options, request, allow_policy_gated_writes);
    tool_request.data_store = data_store;
    if (semi_oracle_store != nullptr && !request.session_id.empty()) {
        tool_request.semi_oracle_tool_call = [semi_oracle_store,
                guidance_config = options.adaptation_semi_oracle_outcome_guidance,
                trace_enabled = options.agent_trace,
                project_id = request.project_id,
                host_namespace = request.namespace_id.empty() ? std::string("local") : request.namespace_id,
                session_id = request.session_id,
                source_turn_id = request.turn_id,
                prompt = request.prompt](const std::string & name, const std::string & input) {
            using json = nlohmann::ordered_json;
            const auto arguments = json::parse(input, nullptr, false);
            if (!arguments.is_object()) return common_tool_execution_result::failure(
                "tool.oracle.invalid_arguments", common_tool_failure_class::validation, false,
                "SemiOracle arguments must be an object.", "invalid JSON object");
            common_semi_oracle_tool_service service(*semi_oracle_store, host_namespace, session_id,
                guidance_config, project_id);
            if (name == "oracle.ruleset.define") {
                if (!arguments.contains("ruleset_dsl") || !arguments["ruleset_dsl"].is_string())
                    return common_tool_execution_result::failure("tool.oracle.invalid_ruleset", common_tool_failure_class::validation, false,
                        "A RuleSpec DSL is required.", "ruleset_dsl must be a string");
                common_semi_oracle_ruleset ruleset; std::string error;
                if (!service.define_ruleset(arguments["ruleset_dsl"].get<std::string>(),
                        common_semi_oracle_rule_authority::declared, ruleset, error))
                    return common_tool_execution_result::failure("tool.oracle.ruleset_rejected", common_tool_failure_class::validation, false,
                        "The RuleSet did not pass host validation.", std::move(error));
                std::string contract;
                if (!service.action_contract(contract, error))
                    return common_tool_execution_result::failure("tool.oracle.contract_unavailable", common_tool_failure_class::execution, false,
                        "The active RuleSet contract could not be rendered.", std::move(error));
                return common_tool_execution_result::success(json({
                    {"ruleset_id", ruleset.id}, {"revision", ruleset.revision},
                    {"domain", common_semi_oracle_domain_name(ruleset.domain)},
                    {"scope", common_semi_oracle_scope_name(ruleset.scope)},
                    {"action_contract", contract}, {"active", ruleset.active},
                    {"source_turn_id", source_turn_id},
                }).dump());
            }
            if (name == "oracle.problem.define") {
                std::string state, problem_error;
                if (!arguments.contains("problem_dsl") || !arguments["problem_dsl"].is_string() ||
                        !service.define_problem(arguments["problem_dsl"].get<std::string>(),state,problem_error))
                    return common_tool_execution_result::failure("tool.oracle.problem_rejected",common_tool_failure_class::validation,false,
                        "The Target Hunt problem did not pass host validation.",problem_error);
                const auto problem=json::parse(state);
                std::string contract; if(!service.action_contract(contract,problem_error))
                    return common_tool_execution_result::failure("tool.oracle.contract_unavailable",common_tool_failure_class::execution,false,
                        "The action contract is unavailable.",problem_error);
                return common_tool_execution_result::success(json{{"problem_id",problem["problem_id"]},
                    {"problem_fingerprint",problem["problem_fingerprint"]},{"position",problem["position"]},
                    {"action_contract",contract}}.dump());
            }
            if (name == "oracle.goal.search") {
                common_semi_oracle_result result;
                std::string search_error;
                if (!service.search_active_goal(result, search_error))
                    return common_tool_execution_result::failure("tool.oracle.search_unavailable",
                        common_tool_failure_class::validation, false,
                        "The host could not search the active goal.", std::move(search_error));
                const auto guidance = json::parse(result.search_guidance_json.empty() ? "{}" : result.search_guidance_json,
                    nullptr, false);
                if (trace_enabled && guidance.is_object() && guidance.contains("diagnostics"))
                    std::clog << "[semi-oracle] outcome_guidance " << guidance["diagnostics"].dump() << '\n';
                auto guidance_summary = guidance.is_object() && guidance.contains("diagnostics")
                    ? guidance["diagnostics"] : json::object();
                guidance_summary.erase("decisions");
                guidance_summary.erase("history_diagnostic");
                guidance_summary.erase("history_evaluation_refs");
                const auto proof = json::parse(result.proof_json.empty() ? "null" : result.proof_json,
                    nullptr, false);
                const std::string algorithm = result.applied_search_policy.algorithm ==
                    common_semi_oracle_search_algorithm::focal ? "focal_astar_v1" : "exact_astar_v1";
                return common_tool_execution_result::success(json{
                    {"verdict", common_semi_oracle_verdict_name(result.verdict)},
                    {"ruleset_id", result.ruleset_id}, {"revision", result.ruleset_revision},
                    {"evaluator", result.evaluator_ref}, {"evaluator_revision", result.evaluator_revision},
                    {"level", common_semi_oracle_level_name(result.level)},
                    {"expansions", result.expansions}, {"depth_reached", result.depth_reached},
                    {"proof", proof},
                    {"counterexample", json::parse(result.counterexample_json.empty() ? "null" : result.counterexample_json, nullptr, false)},
                    {"search_policy", {{"algorithm", algorithm},
                        {"suboptimality_bound", result.applied_search_policy.suboptimality_bound},
                        {"optimality_proven", proof.is_object() && proof.value("optimality_proven", false)},
                        {"suboptimality_bound_proven", proof.is_object() && proof.value("suboptimality_bound_proven", false)}}},
                    {"guidance", guidance_summary},
                    {"reason", result.reason}}.dump());
            }
            if (name != "oracle.action.verify" || !arguments.contains("action_dsl") ||
                    !arguments["action_dsl"].is_string())
                return common_tool_execution_result::failure("tool.oracle.invalid_action", common_tool_failure_class::validation, false,
                    "A supported oracle action and action_dsl are required.", "unsupported tool or missing action_dsl");
            common_semi_oracle_ruleset active; bool found = false;
            std::string error;
            if (!service.active_ruleset(active, found, error) || !found) {
                return common_tool_execution_result::failure("tool.oracle.action_unavailable",
                    common_tool_failure_class::validation, false,
                    "No active RuleSet is available for this action.", std::move(error));
            }
            if (active.domain == common_semi_oracle_domain::linear_algebra) {
                std::string state; bool state_found = false;
                if (!semi_oracle_store->get_session_state(session_id, active.id,
                        active.revision, state, state_found, error)) {
                    return common_tool_execution_result::failure("tool.oracle.state_unavailable",
                        common_tool_failure_class::execution, false,
                        "The host could not load algebra state.", std::move(error));
                }
                if (!state_found) {
                    if (!initial_linear_algebra_state_from_prompt(prompt, state) ||
                            !service.initialize_active_state(state, false, error)) {
                        return common_tool_execution_result::failure("tool.oracle.state_required",
                            common_tool_failure_class::validation, false,
                            "The host needs a supported equation from the user before verifying an algebra rewrite.",
                            error.empty() ? "no supported equation was found in the user request" : std::move(error));
                    }
                }
            }
            common_semi_oracle_result result; std::string before, after;
            if (!service.verify_active_action(
                    arguments["action_dsl"].get<std::string>(), result, before, after, error))
                return common_tool_execution_result::failure("tool.oracle.action_unavailable", common_tool_failure_class::validation, false,
                    "The host could not evaluate this action in the active session state.", std::move(error));
            json response{{"verdict", common_semi_oracle_verdict_name(result.verdict)},
                {"ruleset_id", result.ruleset_id}, {"revision", result.ruleset_revision},
                {"evaluator", result.evaluator_ref}, {"evaluator_revision", result.evaluator_revision},
                {"level", common_semi_oracle_level_name(result.level)},
                {"proof", json::parse(result.proof_json.empty() ? "null" : result.proof_json, nullptr, false)},
                {"counterexample", json::parse(result.counterexample_json.empty() ? "null" : result.counterexample_json, nullptr, false)},
                {"reason", result.reason}};
            auto execution = common_tool_execution_result::success(response.dump());
            if (result.verdict == common_semi_oracle_verdict::satisfied) {
                if (service.active_ruleset(active, found, error) && found) {
                    std::vector<common_semi_oracle_placement_action_value> ranking;
                    if (service.rank_placements(before, {100000, 16}, ranking, error) &&
                            !ranking.empty() && ranking.front().ranking_complete &&
                            common_semi_oracle_learning_authorized(active, result)) {
                        int row = -1, column = -1;
                        std::istringstream lines(arguments["action_dsl"].get<std::string>());
                        std::string line;
                        while (std::getline(lines, line)) {
                            const auto colon = line.find(':'); if (colon == std::string::npos) continue;
                            const auto key = line.substr(0, colon);
                            try { if (key == "row") row = std::stoi(line.substr(colon + 1));
                                  if (key == "column") column = std::stoi(line.substr(colon + 1)); } catch (...) {}
                        }
                        const auto chosen = std::find_if(ranking.begin(), ranking.end(), [&](const auto & value) {
                            return value.row == row && value.column == column;
                        });
                        const auto best = std::max_element(ranking.begin(), ranking.end(), [](const auto & a, const auto & b) {
                            return static_cast<int>(a.value) < static_cast<int>(b.value);
                        });
                        if (chosen != ranking.end() && best != ranking.end() &&
                                static_cast<int>(chosen->value) < static_cast<int>(best->value)) {
                            // Keep the model's observed move and the host's
                            // counterfactuals together before the generic
                            // runtime replaces the signal evidence id with
                            // its tool-observation id.  This record contains
                            // no activations and grants no credit on its own.
                            const auto control = std::find_if(ranking.begin(), ranking.end(),
                                [&](const auto & value) {
                                    return value.row != chosen->row || value.column != chosen->column
                                        ? value.value == chosen->value : false;
                                });
                            std::string interaction_ref;
                            if (control != ranking.end()) {
                                const auto render_action = [](const auto & value) {
                                    return std::string("open! game.place_mark\nplayer: ") + value.player +
                                        "\nrow: " + std::to_string(value.row) +
                                        "\ncolumn: " + std::to_string(value.column);
                                };
                                const std::string identity = session_id + "\n" + active.id + "\n" +
                                    std::to_string(active.revision) + "\n" + before + "\n" +
                                    arguments["action_dsl"].get<std::string>() + "\n" +
                                    best->persisted_evaluation_ref;
                                common_semi_oracle_interaction interaction;
                                interaction.id = "semi-oracle://interaction/" +
                                    hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
                                interaction.session_id = session_id;
                                interaction.ruleset_id = active.id;
                                interaction.ruleset_revision = active.revision;
                                interaction.action_contract_fingerprint = active.action_contract_fingerprint;
                                interaction.domain = "placement_game";
                                interaction.contrast_kind = "strategic_preference";
                                interaction.evidence_ref = best->persisted_evaluation_ref;
                                interaction.verifier_ref = "semi-oracle://evaluator/placement_game";
                                interaction.verifier_revision = "minimax-v1";
                                interaction.task_fingerprint = chosen->state_fingerprint;
                                interaction.state_json = before;
                                interaction.control_state_json = before;
                                interaction.baseline_action_dsl = arguments["action_dsl"].get<std::string>();
                                interaction.preferred_action_dsl = render_action(*best);
                                interaction.control_action_dsl = render_action(*control);
                                interaction.baseline_result_json = response.dump();
                                interaction.preferred_result_json = json{{"verdict", "SATISFIED"},
                                    {"ruleset_id", active.id}, {"revision", active.revision},
                                    {"ranking_evaluation_ref", best->persisted_evaluation_ref}}.dump();
                                interaction.contrast_ref = best->persisted_evaluation_ref + ":" +
                                    std::to_string(chosen->row) + "," + std::to_string(chosen->column) + "<" +
                                    std::to_string(best->row) + "," + std::to_string(best->column);
                                std::string interaction_error;
                                if (semi_oracle_store->put_interaction(interaction, interaction_error)) {
                                    interaction_ref = interaction.id;
                                }
                            }
                            response["strategy"] = {{"chosen_value", static_cast<int>(chosen->value)},
                                {"best_value", static_cast<int>(best->value)},
                                {"ranking_evaluation_ref", best->persisted_evaluation_ref},
                                {"verified_suboptimal", true}};
                            execution.output = response.dump();
                            execution.learning_signals.emplace_back(
                                common_learning_signal_type::semi_oracle_preference, std::string{}, std::string{},
                                "oracle.action.verify", best->persisted_evaluation_ref,
                                "host minimax proved the observed legal move strictly worse than an available move",
                                "semi-oracle", "host",
                                interaction_ref.empty() ? std::string{} :
                                    json{{"semi_oracle_interaction_ref", interaction_ref}}.dump());
                        }
                    }
                }
            }
            if (active.domain == common_semi_oracle_domain::linear_algebra &&
                    result.verdict == common_semi_oracle_verdict::violated &&
                    common_semi_oracle_learning_authorized(active, result)) {
                std::string preferred_dsl, repair_error;
                common_semi_oracle_result preferred;
                if (daemon_semi_oracle_make_algebra_repair(service, before, preferred_dsl,
                        preferred, repair_error)) {
                    // Fixed host fixtures are controls, never demonstrations:
                    // they remove generic rewrite surface form from the
                    // correctness contrast and must themselves verify.
                    const std::string control_state = R"json({"lhs":"3(x+2)","rhs":"9"})json";
                    const std::string control_dsl = "open! algebra.rewrite_equation\nlhs: 3x+6";
                    common_semi_oracle_result control;
                    if (service.verify_action(control_state, control_dsl,
                            common_semi_oracle_level::local, {}, {1, 1}, {}, control, repair_error) &&
                            control.verdict == common_semi_oracle_verdict::satisfied &&
                            control_state != before) {
                        const std::string identity = session_id + "\n" + active.id + "\n" +
                            std::to_string(active.revision) + "\n" + before + "\n" +
                            arguments["action_dsl"].get<std::string>() + "\n" + preferred_dsl;
                        common_semi_oracle_interaction interaction;
                        interaction.id = "semi-oracle://interaction/" +
                            hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
                        interaction.session_id = session_id;
                        interaction.ruleset_id = active.id;
                        interaction.ruleset_revision = active.revision;
                        interaction.action_contract_fingerprint = active.action_contract_fingerprint;
                        interaction.domain = "linear_algebra";
                        interaction.contrast_kind = "correctness_repair";
                        interaction.evidence_ref = "semi-oracle://evaluation/" +
                            hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
                        interaction.verifier_ref = preferred.evaluator_ref;
                        interaction.verifier_revision = preferred.evaluator_revision;
                        interaction.task_fingerprint = hash_sha256_hex(before.data(), before.size()).substr(0, 32);
                        interaction.state_json = before;
                        interaction.control_state_json = control_state;
                        interaction.baseline_action_dsl = arguments["action_dsl"].get<std::string>();
                        interaction.preferred_action_dsl = preferred_dsl;
                        interaction.control_action_dsl = control_dsl;
                        interaction.baseline_result_json = daemon_semi_oracle_result_json(result).dump();
                        interaction.preferred_result_json = daemon_semi_oracle_result_json(preferred).dump();
                        interaction.contrast_ref = "correctness-repair:" + interaction.task_fingerprint;
                        std::string interaction_error;
                        if (semi_oracle_store->put_interaction(interaction, interaction_error)) {
                            execution.learning_signals.emplace_back(
                                common_learning_signal_type::semi_oracle_preference, std::string{}, std::string{},
                                "oracle.action.verify", interaction.evidence_ref,
                                "host verified an algebra correction for the observed invalid rewrite",
                                "semi-oracle", "host",
                                json{{"semi_oracle_interaction_ref", interaction.id}}.dump());
                        }
                    }
                }
            }
            return execution;
        };
    } else {
        tool_request.semi_oracle_tool_call = {};
    }
    if (!request.session_id.empty() && (learning_lifecycle_store != nullptr || semi_oracle_store != nullptr)) {
        tool_request.prediction_predict_tool_call = [learning_lifecycle_store, semi_oracle_store,
                guidance_config = options.adaptation_flydelta_outcome_guidance,
                generalization = options.adaptation_outcome_guidance_generalization,
                namespace_id = request.namespace_id.empty() ? std::string("local") : request.namespace_id,
                project_id = request.project_id, session_id = request.session_id](const std::string & input) {
            return daemon_prediction_predict(learning_lifecycle_store, semi_oracle_store,
                guidance_config, generalization, namespace_id, project_id, session_id, input);
        };
    } else {
        tool_request.prediction_predict_tool_call = {};
    }
        std::string * current_plan_id = runtime != nullptr
        ? const_cast<std::string *>(&runtime->current_plan_id())
        : nullptr;
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    const std::string embedding_model = options.embedding_model;
    if (!embedding_model.empty()) {
        embedding_provider = std::make_unique<daemon_agent_embedding_provider>(
            embedding_model,
            options.n_gpu_layers);
    }

    common_agent_cli_tool_selection selection;
    if (!resolve_agent_host_tool_selection(
            memory_store,
            &plan_store,
            resource_store,
            current_plan_id,
            options.tool_profile,
            tool_request,
            query,
            embedding_provider.get(),
            selection,
            error)) {
        error = "daemon tool provider resolution failed: " + error;
        return false;
    }
    selection.embedding_provider = std::move(embedding_provider);

    tooling = std::move(selection.tooling);
    if (selection.embedding_provider) {
        auto shared_embedding = std::shared_ptr<agent_embedding_provider>(
            std::move(selection.embedding_provider));
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_embedding));
    }
    if (selection.owned_resource_store) {
        auto shared_resource_store = std::shared_ptr<agent_resource_store>(
            std::move(selection.owned_resource_store));
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_resource_store));
    }
    for (auto & mcp_client : selection.mcp_clients) {
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(
            std::shared_ptr<agent_mcp_tool_client>(std::move(mcp_client))));
    }
    for (auto & openapi_provider : selection.openapi_providers) {
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(
            std::shared_ptr<agent_tool_provider>(std::move(openapi_provider))));
    }
    if (selection.tool_view) {
        auto shared_view = std::shared_ptr<agent_tool_view>(std::move(selection.tool_view));
        tooling.tool_view = shared_view.get();
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_view));
    }
    error.clear();
    return true;
}

namespace {

std::function<bool(
        const common_agent_runtime_session_host_turn_request &,
        common_flydelta_deployment_result &,
        std::string &)>
make_daemon_flydelta_deployment_resolver(
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store,
        const common_agent_daemon_runtime & runtime) {
    if (!runtime.flydelta_sideband_registry || !runtime.model_residency) {
        return {};
    }
    const auto registry = runtime.flydelta_sideband_registry;
    const auto artifact_store = runtime.flydelta_artifact_store;
    const auto observation_budget = runtime.flydelta_observation_budget;
    const auto model_handle = runtime.flydelta_model_handle;
    const auto review_store = runtime.flydelta_sideband_review_store;
    const auto observation_journal_mutex = std::make_shared<std::mutex>();
    return [options, config_store, registry, artifact_store, observation_budget,
            observation_journal_mutex, model_handle, review_store](
            const common_agent_runtime_session_host_turn_request & request,
            common_flydelta_deployment_result & result,
            std::string & error) {
        error.clear();
        const auto current_options = config_store ? config_store->snapshot() :
            std::shared_ptr<const daemon_options>();
        const daemon_options & effective_options = current_options ? *current_options : options;
        // The resident model handle is the host-owned source of truth for the
        // effective profile.  A daemon request may legitimately omit the
        // profile id when the catalog selected its default profile; treating
        // that omission as "no FlyDelta" would silently bypass policy-mode
        // canary resolution for normal production turns.
        if (!model_handle.has_value() || !model_handle->valid()) {
            result = {};
            result.active_only = true;
            return true;
        }
        auto loaded = common_agent_runtime_loaded_model_cast(model_handle->model);
        if (!loaded || loaded->model == nullptr) {
            error = "FlyDelta deployment resolver requires the resident model handle";
            return false;
        }
        common_agent_model_profile profile;
        profile.id = model_handle->selection.profile_id;
        profile.base_model_id = model_handle->selection.base_model_id;
        profile.base_model_fingerprint = effective_options.adaptation_flydelta_model_profile_fingerprint.empty()
            ? "model:" + std::filesystem::path(model_handle->selection.path).filename().string()
            : effective_options.adaptation_flydelta_model_profile_fingerprint;
        profile.tokenizer_fingerprint = "daemon:tokenizer-unspecified-v1";
        profile.chat_template_fingerprint = "daemon:template-unspecified-v1";
        profile.context_size_tokens = model_handle->selection.context_size_tokens;
        profile.n_parallel = model_handle->selection.n_parallel;
        profile.n_sequences = model_handle->selection.n_sequences;
        profile.load_policy = model_handle->selection.load_policy;
        profile.adapters = model_handle->selection.adapters;
        profile.sidebands = model_handle->selection.sidebands;
        profile.tool_output_format = model_handle->selection.tool_output_format;
        profile.tool_output_format_configured = model_handle->selection.tool_output_format_configured;

        common_flydelta_deployment_request deployment_request;
        deployment_request.profile = std::move(profile);
        deployment_request.compatibility.base_model_id = deployment_request.profile.base_model_id;
        deployment_request.compatibility.base_model_fingerprint =
            deployment_request.profile.base_model_fingerprint;
        deployment_request.compatibility.tokenizer_fingerprint =
            deployment_request.profile.tokenizer_fingerprint;
        deployment_request.compatibility.template_fingerprint =
            deployment_request.profile.chat_template_fingerprint;
        deployment_request.compatibility.architecture = "llama";
        deployment_request.compatibility.inference_layout_revision =
            effective_options.adaptation_flydelta_capture_layout_revision;
        deployment_request.applicability = request.flydelta_applicability;
        deployment_request.authority = request.flydelta_runtime_authority;
        // Canary mode is host-owned.  Policy mode is the only configuration
        // that may grant the daemon's ordinary production turn an explicit
        // canary evaluation authority; manual mode still requires the caller
        // to put that authority on the request, while disabled always forces
        // the safe active-only path.
        if (effective_options.adaptation_flydelta_canary_policy.mode ==
                common_flydelta_canary_mode::disabled) {
            deployment_request.authority = common_flydelta_runtime_authority::active_only;
        } else if (effective_options.adaptation_flydelta_canary_policy.mode ==
                common_flydelta_canary_mode::policy) {
            deployment_request.authority = common_flydelta_runtime_authority::canary_evaluation;
        }
        deployment_request.allocation_key = request.namespace_id + "/" +
            request.project_id + "/" + request.session_id;
        deployment_request.now_epoch_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        deployment_request.model_n_embd = static_cast<size_t>(llama_model_n_embd(loaded->model));
        deployment_request.model_n_layers = static_cast<size_t>(llama_model_n_layer(loaded->model));
        deployment_request.model_profile_fingerprint =
            deployment_request.profile.base_model_fingerprint;
        deployment_request.capture_layout_revision =
            effective_options.adaptation_flydelta_capture_layout_revision;
        deployment_request.gate_config = request.flydelta_gate_config;
        deployment_request.gate_request = request.flydelta_gate_request;
        deployment_request.code = request.flydelta_code;
        deployment_request.max_artifact_weights = 1U << 20;
        deployment_request.max_artifact_bytes = 4U * 1024U * 1024U;
        deployment_request.max_overlay_bytes = 64U * 1024U * 1024U;

        common_flydelta_deployment_factory_config factory;
        factory.registry = registry.get();
        factory.open_canaries = [review_store](
                std::vector<common_flydelta_sideband_review> & reviews,
                std::string & provider_error) {
            if (!review_store) {
                reviews.clear();
                provider_error.clear();
                return true;
            }
            reviews = review_store->open_canaries(provider_error);
            return provider_error.empty();
        };
        factory.observation_counter = [observation_budget](
                const std::string & binding_key, const std::string & event_id) {
            return observation_budget ? observation_budget->reserved(binding_key, event_id) : 0;
        };
        factory.reserve_observation = [observation_budget, review_store, observation_journal_mutex](
                const common_flydelta_sideband_review & canary_review,
                const std::string & allocation_id,
                size_t max_observations, std::string & observation_id,
                std::string & reservation_error) {
            if (!observation_budget || !observation_budget->try_reserve(
                    canary_review.canary_envelope.binding_key,
                    canary_review.event_id, max_observations, reservation_error)) {
                return false;
            }
            if (review_store) {
                std::lock_guard<std::mutex> journal_lock(*observation_journal_mutex);
                common_flydelta_sideband_review reservation = canary_review;
                observation_id = canary_review.event_id + ":observation:" +
                    std::to_string(observation_budget->reserved(
                        canary_review.canary_envelope.binding_key, canary_review.event_id));
                reservation.event_id = observation_id;
                reservation.actor_id = "flydelta-runtime";
                reservation.source = common_flydelta_review_source::host_automation;
                reservation.action = common_flydelta_review_action::reserve_canary_observation;
                reservation.policy_revision = "flydelta-canary-observation-v1";
                reservation.canary_envelope_event_id = canary_review.event_id;
                reservation.observation_id = observation_id;
                reservation.allocation_id = allocation_id;
                reservation.scope_step_id = canary_review.canary_envelope.scope_fingerprint;
                if (!review_store->append(reservation, reservation_error)) return false;
            }
            if (observation_id.empty()) {
                observation_id = canary_review.event_id + ":observation:unjournaled";
            }
            return true;
        };
        factory.load_activation = [artifact_store, config_store, options](
                const common_flydelta_sideband_manifest & manifest,
                double effective_scale,
                const common_flydelta_deployment_request & request,
                common_flydelta_activation_result & activation,
                std::string & loader_error) {
            auto effective_artifact_store = artifact_store;
            const auto current_options = config_store ? config_store->snapshot() :
                std::shared_ptr<const daemon_options>();
            const daemon_options & artifact_options = current_options ? *current_options : options;
            if (!artifact_options.adaptation_flydelta_artifact_root.empty() &&
                    (!effective_artifact_store || current_options)) {
                std::error_code artifact_ec;
                auto artifact_root = std::filesystem::absolute(
                    artifact_options.adaptation_flydelta_artifact_root, artifact_ec);
                if (artifact_ec) {
                    loader_error = "could not resolve FlyDelta artifact root: " + artifact_ec.message();
                    return false;
                }
                effective_artifact_store =
                    std::make_shared<common_flydelta_artifact_store>(std::move(artifact_root));
            }
            if (!effective_artifact_store) {
                loader_error = "FlyDelta artifact store is not configured";
                return false;
            }
            common_flydelta_artifact artifact;
            if (!effective_artifact_store->read(manifest.artifact_path, artifact, loader_error)) return false;
            if (artifact.content_hash != manifest.artifact_hash) {
                loader_error = "FlyDelta artifact hash does not match its manifest";
                return false;
            }
            auto gate_request = request.gate_request;
            gate_request.requested_scale = std::min(
                gate_request.requested_scale, static_cast<float>(effective_scale));
            return common_flydelta_prepare_activation_from_artifact(
                artifact, request.compatibility, request.model_profile_fingerprint,
                request.code, manifest.id, request.gate_config, gate_request,
                request.max_artifact_weights, request.max_artifact_bytes,
                request.max_overlay_bytes, activation, loader_error);
        };
        return common_flydelta_resolve_deployment(factory, deployment_request, result, error);
    };
}

common_agent_runtime_session_host_build_config make_session_host_build_config(
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        agent_resource_store & resource_store,
        common_agent_data_store * data_store,
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store,
        const common_agent_daemon_runtime & runtime) {
    return {
        memory_store,
        plan_store,
        make_resident_request_config(options),
        make_daemon_runtime_policy(options),
        make_daemon_runtime_config(options, runtime),
        make_daemon_orchestration_config(options),
        common_memory_scope::session,
        true,
        {},
        {},
        [config_store, &memory_store, &plan_store, &resource_store, data_store,
            semi_oracle_store = runtime.semi_oracle_store.get(),
            lifecycle_store = runtime.flydelta_review_lifecycle_store](
                const common_agent_runtime_resident_runtime * runtime,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            const auto current_options = config_store->snapshot();
            return resolve_agent_daemon_tooling(
                *current_options,
                runtime,
                request,
                memory_store,
                plan_store,
                &resource_store,
                tooling,
                error,
                data_store,
                std::nullopt,
                semi_oracle_store,
                lifecycle_store.get());
        },
        runtime.model_residency,
        make_daemon_flydelta_deployment_resolver(options, config_store, runtime),
        &resource_store,
        (options.resource_blob_backend == "auto" || options.resource_blob_backend == "fs") &&
            (options.resource_metadata_backend == "cozo" ||
                (options.resource_metadata_backend == "auto" && !options.resource_metadata_db.empty())),
    };
}

bool open_daemon_memory_store(
        const daemon_options & options,
        std::unique_ptr<common_memory_store> & store,
        std::string & error) {
    const std::string backend = resolve_memory_backend(options.backend, options.memory_db, error);
    if (!error.empty()) {
        return false;
    }
    if (backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        if (options.memory_db.empty()) {
            error = "--backend cozo requires --memory-db PATH";
            return false;
        }
        store = std::make_unique<common_memory_cozo_store>();
#else
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return false;
#endif
    } else if (backend == "sqlite") {
#ifdef LLAMA_MEMORY_USE_SQLITE
        if (options.memory_db.empty()) { error = "--backend sqlite requires --memory-db PATH"; return false; }
        store = std::make_unique<common_memory_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
        return false;
#endif
    } else if (backend == "in-memory") {
        store = std::make_unique<common_memory_in_memory_store>();
    } else {
        error = "unknown memory backend: " + backend;
        return false;
    }
    return store->open(options.memory_db, error);
}

bool open_daemon_plan_store(
        const daemon_options & options,
        std::unique_ptr<common_plan_store> & store,
        std::string & error) {
    const std::string backend = resolve_plan_backend(options.plan_backend, options.plan_db, error);
    if (!error.empty()) {
        return false;
    }
    if (backend == "cozo") {
#ifdef LLAMA_PLAN_USE_COZO
        if (options.plan_db.empty()) {
            error = "--plan-backend cozo requires --plan-db PATH";
            return false;
        }
        store = std::make_unique<common_plan_cozo_store>();
#else
        error = "this binary was built without LLAMA_PLAN_COZO";
        return false;
#endif
    } else if (backend == "sqlite") {
#ifdef LLAMA_PLAN_USE_SQLITE
        if (options.plan_db.empty()) { error = "--plan-backend sqlite requires --plan-db PATH"; return false; }
        store = std::make_unique<common_plan_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
        return false;
#endif
    } else if (backend == "in-memory") {
        store = std::make_unique<common_plan_in_memory_store>();
    } else {
        error = "unknown plan backend: " + backend;
        return false;
    }
    return store->open(options.plan_db, error);
}

bool open_daemon_resource_store(
        const daemon_options & options,
        std::unique_ptr<agent_resource_store> & store,
        std::string & error) {
    store = make_agent_resource_store({
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    }, error);
    return store != nullptr;
}

bool open_daemon_data_store(
        const daemon_options & options,
        std::unique_ptr<common_agent_data_store> & store,
        std::string & error) {
    store = make_agent_data_store({options.data_backend, options.data_db}, error);
    return error.empty();
}

bool import_daemon_bootstrap_package(
        const daemon_options & options,
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        std::string & error) {
    if (options.bootstrap_package.empty()) {
        error.clear();
        return true;
    }

    common_agent_bootstrap_package package;
    if (!load_bootstrap_file(options.bootstrap_package, package, error)) {
        error = "daemon bootstrap package load failed: " + error;
        return false;
    }

    common_agent_bootstrap_config config;
    config.namespace_id = "llama-agent-host";
    config.session_id = "bootstrap";
    config.now = std::time(nullptr);
    config.install_procedures = !package.procedures.empty();

    common_agent_bootstrap_embedder embed;
    std::unique_ptr<daemon_agent_embedding_provider> embedding;
    if (!package.procedures.empty()) {
        if (options.embedding_model.empty()) {
            error = "daemon bootstrap package contains procedures; configure model.embedding_model";
            return false;
        }
        embedding = std::make_unique<daemon_agent_embedding_provider>(
            options.embedding_model, options.n_gpu_layers);
        embed = [&embedding](
                const std::string & text,
                std::vector<float> & vector,
                std::string & embed_error) {
            return embedding->embed("daemon bootstrap procedure", text, vector, embed_error);
        };
    }

    common_agent_bootstrap_result result;
    if (!common_agent_install_bootstrap_package(
            memory_store, plan_store, config, package, std::move(embed), result, error)) {
        error = "daemon bootstrap package import failed: " + error;
        return false;
    }
    std::fprintf(stderr,
        "agent bootstrap: package=%s procedures installed=%zu existing=%zu; "
        "blueprints installed=%zu existing=%zu; workflows installed=%zu existing=%zu\n",
        package.name.c_str(),
        result.installed_memory_ids.size(), result.existing_memory_ids.size(),
        result.installed_blueprint_ids.size(), result.existing_blueprint_ids.size(),
        result.installed_workflow_ids.size(), result.existing_workflow_ids.size());
    error.clear();
    return true;
}

} // namespace

bool initialize_agent_daemon_environment(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    if (!runtime.config_store) {
        runtime.config_store = std::make_shared<common_agent_daemon_config_store>(
            std::make_shared<const daemon_options>(options));
    }
    if (!open_daemon_memory_store(options, runtime.memory_store, error)) {
        return false;
    }
    if (!open_daemon_plan_store(options, runtime.plan_store, error)) {
        return false;
    }
    if (!import_daemon_bootstrap_package(
            options, *runtime.memory_store, *runtime.plan_store, error)) {
        return false;
    }
    if (!open_daemon_data_store(options, runtime.data_store, error)) {
        return false;
    }
    if (!open_daemon_resource_store(options, runtime.resource_store, error)) {
        return false;
    }
    runtime.semi_oracle_store = make_agent_semi_oracle_store(
        options.adaptation_transaction_backend,
        options.adaptation_transaction_path,
        error);
    if (!runtime.semi_oracle_store) {
        error = "SemiOracle adaptation store initialization failed: " + error;
        return false;
    }
    runtime.semi_oracle_store->set_history_projection_cache_budget(
        options.adaptation_semi_oracle_outcome_guidance.projection_cache_bytes);
    // SemiOracle uses the ordinary host-relation ingress.  The provider only
    // materializes already verified interactions; it does not evaluate moves,
    // create activations or activate an overlay.
    runtime.semi_oracle_teaching_relation_provider =
        make_daemon_semi_oracle_relation_provider(
            *runtime.semi_oracle_store, *runtime.resource_store);
    if (!open_daemon_model_residency(options, runtime.model_residency, error)) {
        return false;
    }
    if (!agent_daemon_flydelta_internal::configure_daemon_flydelta_runtime(
            options, runtime, error)) {
        return false;
    }
    if (options.adaptation_flydelta_enabled) {
        runtime.flydelta_observation_budget =
            std::make_shared<common_flydelta_observation_budget>();
        if (runtime.flydelta_sideband_review_store) {
            std::string review_error;
            for (const auto & review : runtime.flydelta_sideband_review_store->list(review_error)) {
                if (review.action != common_flydelta_review_action::reserve_canary_observation) {
                    continue;
                }
                runtime.flydelta_observation_budget->seed(
                    review.canary_envelope.binding_key.empty()
                        ? review.binding_key : review.canary_envelope.binding_key,
                    review.canary_envelope_event_id,
                    runtime.flydelta_observation_budget->reserved(
                        review.canary_envelope.binding_key.empty()
                            ? review.binding_key : review.canary_envelope.binding_key,
                        review.canary_envelope_event_id) + 1);
            }
            if (!review_error.empty()) {
                error = "FlyDelta observation journal replay failed: " + review_error;
                return false;
            }
        }
        if (!options.adaptation_flydelta_artifact_root.empty()) {
            std::error_code artifact_ec;
            auto artifact_root = std::filesystem::absolute(
                options.adaptation_flydelta_artifact_root, artifact_ec);
            if (artifact_ec) {
                error = "could not resolve FlyDelta artifact root: " + artifact_ec.message();
                return false;
            }
            runtime.flydelta_artifact_store =
                std::make_shared<common_flydelta_artifact_store>(std::move(artifact_root));
        }
    }
    if (!parse_mode(options.default_mode, runtime.default_mode)) {
        error = "unsupported default mode: " + options.default_mode;
        return false;
    }

    auto session_manager_build_config = make_session_host_build_config(
        *runtime.memory_store,
        *runtime.plan_store,
        *runtime.resource_store,
        runtime.data_store.get(),
        options,
        runtime.config_store,
        runtime);
    runtime.inference_gate = std::make_shared<common_agent_inference_capacity_gate>(
        options.inference_max_active);
    runtime.host = std::make_unique<common_agent_runtime_session_manager>(
        make_agent_runtime_session_manager_config({
            session_manager_build_config.memory_store,
            session_manager_build_config.plan_store,
            std::move(session_manager_build_config.resident_request),
            std::move(session_manager_build_config.policy),
            std::move(session_manager_build_config.runtime_config),
            std::move(session_manager_build_config.orchestration_config),
            session_manager_build_config.memory_scope,
            session_manager_build_config.memory_enabled,
            std::move(session_manager_build_config.installed_blueprint_candidates),
            std::move(session_manager_build_config.tooling),
            std::move(session_manager_build_config.tooling_resolver),
            {},
            runtime.inference_gate,
            {},
            runtime.model_residency,
            session_manager_build_config.flydelta_deployment_resolver,
            session_manager_build_config.context_resource_store,
        }));

    common_memory_store * memory_store = runtime.memory_store.get();
    common_plan_store * plan_store = runtime.plan_store.get();
    agent_resource_store * resource_store = runtime.resource_store.get();
    common_agent_data_store * data_store = runtime.data_store.get();
    runtime.tool_executor = [config_store = runtime.config_store, memory_store, plan_store,
            resource_store, data_store, semi_oracle_store = runtime.semi_oracle_store.get(),
            lifecycle_store = runtime.flydelta_review_lifecycle_store](
            const common_agent_daemon_tool_payload & payload,
            agent_tool_result & result,
            std::string & callback_error) mutable {
        if (memory_store == nullptr || plan_store == nullptr || resource_store == nullptr) {
            callback_error = "daemon tool executor stores are not initialized";
            return false;
        }
        daemon_options call_options = *config_store->snapshot();
        if (!payload.tool_profile.empty()) {
            call_options.tool_profile = payload.tool_profile;
        }
        common_agent_runtime_session_host_turn_request request;
        request.mode = common_agent_runtime_host_mode::chat;
        request.session_id = payload.session.session_id;
        request.namespace_id = payload.session.namespace_id;
        request.project_id = payload.project_id;
        request.turn_id = "mcp-tool";
        request.memory_scope = common_memory_scope::session;
        request.plan_scope = common_plan_scope::turn;

        common_agent_runtime_tooling tooling;
        if (!resolve_agent_daemon_tooling(
                call_options,
                nullptr,
                request,
                *memory_store,
                *plan_store,
                resource_store,
                tooling,
                callback_error,
                data_store,
                payload.allow_policy_gated_writes,
                semi_oracle_store,
                lifecycle_store.get())) {
            return false;
        }
        if (tooling.tool_view == nullptr) {
            callback_error = "daemon tool executor resolved no tool view";
            return false;
        }
        if (!payload.mcp_app_resource_uri.empty()) {
            if (payload.mcp_app_resource_uri.rfind("ui://llama-agent/apps/", 0) == 0) {
                const auto & chat_tools = tooling.tool_view->chat_tools();
                const auto tool = std::find_if(chat_tools.begin(), chat_tools.end(), [&](const common_chat_tool & candidate) {
                    return candidate.name == payload.tool_name;
                });
                if (tool == chat_tools.end()) {
                    callback_error = "MCP App resource owner tool is unavailable";
                    return false;
                }
                const auto metadata = nlohmann::ordered_json::parse(tool->mcp_ui_metadata, nullptr, false);
                const auto visibility = metadata.is_object()
                    ? metadata.value("visibility", nlohmann::ordered_json::array())
                    : nlohmann::ordered_json::array();
                if (!metadata.is_object() || metadata.value("resourceUri", std::string()) != payload.mcp_app_resource_uri ||
                        !visibility.is_array() || std::find(visibility.begin(), visibility.end(), "app") == visibility.end()) {
                    callback_error = "first-party MCP App resource is not linked to the requested tool";
                    return false;
                }
                nlohmann::ordered_json resource;
                if (!agent_mcp_read_first_party_app_resource(payload.mcp_app_resource_uri, resource, callback_error)) return false;
                const auto & item = resource["contents"][0];
                nlohmann::ordered_json normalized = {
                    {"uri", item.value("uri", std::string())},
                    {"name", item.value("name", std::string())},
                    {"mime_type", item.value("mimeType", std::string())},
                    {"text", item.value("text", std::string())},
                    {"_meta", item.value("_meta", nlohmann::ordered_json::object())},
                };
                result = {};
                result.ok = true;
                result.tool_name = payload.tool_name;
                result.content_json = normalized.dump();
                result.content_summary = "MCP App UI resource loaded";
                callback_error.clear();
                return true;
            }
            if (!tooling.tool_view->read_mcp_app_resource(
                    payload.tool_name, payload.mcp_app_resource_uri, result, callback_error)) {
                return false;
            }
            return result.ok;
        }
        agent_tool_call call{"daemon-mcp-tool", payload.tool_name, payload.arguments_json};
        if (payload.mcp_app_call) {
            if (!tooling.tool_view->call_mcp_app_tool(payload.mcp_app_tool_name, call, result, callback_error)) return false;
        } else {
            if (!tooling.tool_view->validate(call, callback_error)) return false;
            if (payload.read_only_only &&
                    (!tooling.tool_view->is_web_read_only(payload.tool_name) ||
                     tooling.tool_view->is_policy_gated(payload.tool_name))) {
                callback_error = "tool is not available through the read-only web tool endpoint";
                return false;
            }
            result = tooling.tool_view->call(call, callback_error);
        }
        return result.ok;
    };
    configure_agent_daemon_provider_probe(options, runtime);
    error.clear();
    return true;
}

void configure_agent_daemon_provider_probe(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime) {
    common_memory_store * probe_memory_store = runtime.memory_store.get();
    common_plan_store * probe_plan_store = runtime.plan_store.get();
    agent_resource_store * probe_resource_store = runtime.resource_store.get();
    common_agent_data_store * probe_data_store = runtime.data_store.get();
    common_semi_oracle_store * probe_semi_oracle_store = runtime.semi_oracle_store.get();
    auto lifecycle_store = runtime.flydelta_review_lifecycle_store;
    runtime.probe_mcp_providers = [
        probe_options = options,
        probe_memory_store,
        probe_plan_store,
        probe_resource_store,
        probe_data_store,
        probe_semi_oracle_store,
        lifecycle_store](
            common_agent_runtime_tooling & retained_tooling,
            std::vector<common_agent_daemon_provider_readiness> & provider_status,
            std::string & probe_error) {
        retained_tooling = {};
        provider_status.clear();
        bool all_required_ready = true;

        // The status endpoint also publishes native profile tools and their
        // MCP App metadata. Provider probing below only resolves configured
        // external MCP servers, so resolve the active native profile once
        // without providers and retain its tool view for readiness projection.
        daemon_options native_options = probe_options;
        native_options.mcp_providers.clear();
        common_agent_runtime_session_host_turn_request native_request;
        native_request.mode = common_agent_runtime_host_mode::chat;
        native_request.session_id = "daemon-native-tool-probe";
        native_request.namespace_id = "local";
        native_request.project_id = "llama-agent";
        native_request.turn_id = "daemon-native-tool-probe";
        native_request.memory_scope = common_memory_scope::session;
        native_request.plan_scope = common_plan_scope::turn;
        common_agent_runtime_tooling native_tooling;
        std::string native_error;
        if (resolve_agent_daemon_tooling(
                native_options,
                nullptr,
                native_request,
                *probe_memory_store,
                *probe_plan_store,
                probe_resource_store,
                native_tooling,
                native_error,
                probe_data_store,
                std::nullopt,
                probe_semi_oracle_store,
                lifecycle_store.get())) {
            retained_tooling = std::move(native_tooling);
        }

        for (const auto & configured : probe_options.mcp_providers) {
            if (!configured.enabled) {
                continue;
            }

            common_agent_daemon_provider_readiness status;
            status.id = configured.id.empty() ? configured.server_name : configured.id;
            status.required = configured.required;

            daemon_options provider_options = probe_options;
            provider_options.mcp_providers = {configured};
            common_agent_runtime_session_host_turn_request probe_request;
            probe_request.mode = common_agent_runtime_host_mode::chat;
            probe_request.session_id = "daemon-provider-probe";
            probe_request.namespace_id = "local";
            probe_request.project_id = "llama-agent";
            probe_request.turn_id = "daemon-provider-probe";
            probe_request.memory_scope = common_memory_scope::session;
            probe_request.plan_scope = common_plan_scope::turn;

            common_agent_runtime_tooling provider_tooling;
            std::string provider_error;
            const bool provider_ready = resolve_agent_daemon_tooling(
                provider_options,
                nullptr,
                probe_request,
                *probe_memory_store,
                *probe_plan_store,
                probe_resource_store,
                provider_tooling,
                provider_error,
                probe_data_store,
                std::nullopt,
                probe_semi_oracle_store,
                lifecycle_store.get());
            if (provider_ready) {
                status.status = "ready";
                for (auto & tool : provider_tooling.tools) {
                    retained_tooling.tools.push_back(std::move(tool));
                }
                for (auto & tool : provider_tooling.app_tools) {
                    if (std::none_of(retained_tooling.app_tools.begin(), retained_tooling.app_tools.end(),
                            [&](const common_chat_tool & existing) { return existing.name == tool.name; })) {
                        retained_tooling.app_tools.push_back(std::move(tool));
                    }
                }
                for (auto & resource : provider_tooling.owned_resources) {
                    retained_tooling.owned_resources.push_back(std::move(resource));
                }
            } else {
                status.status = "degraded";
                status.warning = provider_error.empty()
                    ? "provider probe failed"
                    : provider_error;
                if (status.required) {
                    all_required_ready = false;
                }
            }
            provider_status.push_back(std::move(status));
        }
        probe_error.clear();
        return all_required_ready;
    };
}
