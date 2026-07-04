#include "agent-cli-host-adapter.h"

#include "../memory/memory-cli-memory.h"

#include "agent-plan-orchestration.h"
#include "agent-runtime-assembly.h"
#include "agent-runtime-execution.h"
#include "common/cli-scope.h"

#include <cstdio>
#include <ctime>

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

common_agent_chat_tool_handler make_agent_cli_runtime_tool_handler(
        common_memory_store & store,
        const args & options) {
    return [&store, &options](const common_chat_tool_call & call) {
        if (call.name == "memory_search") {
            return memory_search_tool_result(store, options, call.arguments);
        }
        if (call.name == "memory_remember") {
            return memory_remember_tool_result(store, options, call.arguments);
        }
        std::fprintf(stderr, "warning: rejected unsupported memory tool call: %s\n", call.name.c_str());
        return std::string(R"({"ok":false,"error":"unsupported memory tool"})");
    };
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
        const std::vector<common_chat_tool> & tools,
        bool profile_tools_active,
        agent_tool_view * tool_view,
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
        tools,
        profile_tools_active,
        tool_view,
        make_agent_cli_runtime_tool_handler(store, options),
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
        const std::vector<common_chat_tool> & tools,
        bool profile_tools_active,
        agent_tool_view * tool_view,
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
        tools,
        profile_tools_active,
        tool_view,
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
