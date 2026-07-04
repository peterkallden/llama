#include "agent-cli-host-adapter.h"

#include "../memory/memory-cli-memory.h"

#include "agent-plan-orchestration.h"
#include "agent-runtime-assembly.h"
#include "agent-runtime-execution.h"
#include "common/cli-scope.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>

common_agent_runtime_host_post_run make_agent_cli_runtime_post_run(
        common_memory_store & store,
        const args & options,
        bool memory_enabled) {
    return [&store, &options, memory_enabled](const common_agent_result &, std::string & hook_error) {
        if (!options.record_episode) {
            hook_error.clear();
            return true;
        }
        if (!memory_enabled) {
            std::fprintf(stderr, "warning: skipping episode recording because no query embedding could be generated\n");
            hook_error.clear();
            return true;
        }

        common_memory_record episode;
        episode.id = "episode-" + std::to_string(std::time(nullptr));
        episode.kind = common_memory_kind::episode;
        episode.content = options.prompt;
        episode.created_at = std::time(nullptr);
        episode.accessed_at = episode.created_at;
        episode.importance = 0.5f;
        episode.confidence = 0.5f;
        apply_memory_scope(options, episode);
        if (!store.put(episode, hook_error)) {
            std::fprintf(stderr, "failed to record memory episode: %s\n", hook_error.c_str());
            hook_error.clear();
        }
        return true;
    };
}

bool resolve_agent_cli_tool_selection(
        common_memory_store & store,
        common_plan_store * plan_store,
        std::string * current_plan_id,
        const args & options,
        const common_memory_query & query,
        bool memory_enabled,
        common_agent_cli_tool_selection & selection,
        std::string & error) {
    selection = {};

    if (!options.tool_profile.empty()) {
        common_tool_catalog tool_catalog;
        common_tool_bootstrap_result bootstrap;
        if (!tool_catalog.bootstrap(options.tool_profile, bootstrap, error)) {
            error = "tool bootstrap failed: " + error;
            return false;
        }

        common_native_tool_bindings bindings;
        if (!options.repository_root.empty()) {
            bindings.repository_root = std::filesystem::weakly_canonical(options.repository_root).string();
        }
        bindings.plan_store = plan_store;
        bindings.plan_id = current_plan_id;
        if (memory_enabled) {
            bindings.memory_store = &store;
            bindings.memory_query = query;
            bindings.embed_memory_query = [&options](const std::string & text, std::vector<float> & embedding, std::string & embedding_error) {
                return ensure_memory_cli_embedding(options, text, embedding, "tool query", embedding_error);
            };
        }

        native_agent_tool_provider provider(
            tool_catalog,
            [&bindings](const agent_tool_context &, common_native_tool_bindings & resolved, std::string & binding_error) {
                resolved = bindings;
                binding_error.clear();
                return true;
            });

        agent_tool_context tool_context;
        tool_context.request_id = "cli-run";
        tool_context.turn_id = options.memory_turn;
        tool_context.scope = common_cli_make_agent_scope_with_matching_plan_scope(options);
        tool_context.memory_scope = query.scope;
        tool_context.plan_scope = tool_context.scope.plan_scope;
        tool_context.profile_id = options.tool_profile;
        tool_context.repository_root = bindings.repository_root;
        tool_context.allow_network = options.tool_profile == "research";
        tool_context.allow_policy_gated_writes = options.tool_profile == "memory" || options.tool_profile == "research";
        tool_context.allow_memory_proposals = tool_context.allow_policy_gated_writes;
        tool_context.allow_plan_proposals = tool_context.allow_policy_gated_writes;
        tool_context.max_calls = options.max_tool_rounds > 0 ? options.max_tool_rounds : 1;
        if (!(selection.tool_view = provider.resolve_tools(tool_context, error))) {
            error = "tool provider resolution failed: " + error;
            return false;
        }
        selection.tooling.tools = selection.tool_view->chat_tools();
        selection.tooling.profile_tools_active = true;
        selection.tooling.tool_view = selection.tool_view.get();
        error.clear();
        return true;
    }

    error.clear();
    return true;
}

common_agent_runtime_turn_request make_agent_cli_runtime_turn_request(
        const args & options,
        const common_agent_scope & scope,
        const common_agent_orchestration_config & orchestration_config,
        common_memory_scope memory_scope,
        bool memory_enabled,
        const std::string & fallback_reason,
        common_agent_request request,
        common_agent_generation_options generation_options) {
    common_agent_runtime_turn_request turn_request;
    turn_request.request = std::move(request);
    turn_request.scope = scope;
    turn_request.inference_options = make_agent_inference_options(options);
    turn_request.policy = make_agent_runtime_policy(options);
    turn_request.runtime_config = make_agent_runtime_config(options);
    turn_request.orchestration_config = orchestration_config;
    turn_request.generation_options = generation_options;
    turn_request.memory_scope = memory_scope;
    turn_request.memory_enabled = memory_enabled;
    turn_request.fallback_reason = fallback_reason;
    return turn_request;
}

common_agent_runtime_host_inputs make_agent_cli_runtime_host_chat_inputs(
        common_memory_store & store,
        args & options,
        const std::vector<common_chat_msg> & messages,
        common_memory_scope memory_scope,
        const std::vector<common_memory_hit> & memories,
        bool memory_enabled,
        const std::string & fallback_reason,
        const common_agent_runtime_tooling & tooling,
        common_agent_runtime_host_post_run post_run) {
    common_agent_scope runtime_scope = common_cli_make_agent_scope_with_matching_plan_scope(options);
    common_agent_request request;
    request.messages = messages;
    common_agent_generation_options generation_options;
    auto turn_request = make_agent_cli_runtime_turn_request(
        options,
        runtime_scope,
        make_agent_orchestration_config(options),
        memory_scope,
        memory_enabled,
        fallback_reason,
        std::move(request),
        generation_options);
    common_agent_runtime_host_build_context build_context{
        store,
        nullptr,
        std::move(turn_request),
        nullptr,
        nullptr,
        memories,
        tooling,
    };
    auto inputs = make_agent_runtime_host_chat_inputs(build_context);
    inputs.reset_session_on_completion = true;
    inputs.post_run = std::move(post_run);
    return inputs;
}

common_agent_runtime_host_inputs make_agent_cli_runtime_host_mini_inputs(
        common_memory_store & store,
        common_plan_store & plan_store,
        args & options,
        common_agent_scope & scope,
        std::string & current_plan_id,
        const std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
        const common_agent_orchestration_config & orchestration_config,
        common_memory_scope memory_scope,
        const std::vector<common_memory_hit> & memories,
        bool memory_enabled,
        const std::string & fallback_reason,
        const common_agent_runtime_tooling & tooling,
        common_agent_runtime_host_post_run post_run) {
    auto turn_request = make_agent_cli_runtime_turn_request(
        options,
        scope,
        orchestration_config,
        memory_scope,
        memory_enabled,
        fallback_reason);
    common_agent_runtime_host_build_context build_context{
        store,
        &plan_store,
        std::move(turn_request),
        &current_plan_id,
        &installed_blueprint_candidates,
        memories,
        tooling,
    };
    auto inputs = make_agent_runtime_host_mini_inputs(build_context, orchestration_config);
    inputs.reset_session_on_completion = true;
    inputs.post_run = std::move(post_run);
    return inputs;
}

int finish_agent_cli_runtime_result(const common_agent_result & result) {
    std::printf("%s\n", result.response.c_str());
    std::fprintf(stderr, "decoded %d tokens\n", result.total_decoded_tokens);
    return 0;
}
