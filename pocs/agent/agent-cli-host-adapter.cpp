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

namespace {

class cli_agent_embedding_provider final : public agent_embedding_provider {
public:
    explicit cli_agent_embedding_provider(const args & options)
        : options(options) {}

    bool embed(
            const std::string & purpose,
            const std::string & text,
            std::vector<float> & embedding,
            std::string & error) override {
        return ensure_memory_cli_embedding(options, text, embedding, purpose.c_str(), error);
    }

private:
    const args & options;
};

agent_tool_context make_agent_cli_tool_context(
        const args & options,
        const common_memory_query & query,
        const std::string & repository_root) {
    agent_tool_context tool_context;
    tool_context.request_id = "cli-run";
    tool_context.turn_id = options.memory_turn;
    tool_context.scope = common_cli_make_agent_scope_with_matching_plan_scope(options);
    tool_context.memory_scope = query.scope;
    tool_context.plan_scope = tool_context.scope.plan_scope;
    tool_context.profile_id = options.tool_profile;
    tool_context.repository_root = repository_root;
    tool_context.allow_network = options.tool_profile == "research" || !options.mcp_tool_command.empty();
    tool_context.allow_policy_gated_writes = options.tool_profile == "memory" || options.tool_profile == "research";
    tool_context.allow_memory_proposals = tool_context.allow_policy_gated_writes;
    tool_context.allow_plan_proposals = tool_context.allow_policy_gated_writes;
    tool_context.max_calls = options.max_tool_rounds > 0 ? options.max_tool_rounds : 1;
    return tool_context;
}

} // namespace

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

    common_tool_catalog tool_catalog;
    std::unique_ptr<native_agent_tool_provider> native_provider;
    if (memory_enabled) {
        selection.embedding_provider = std::make_unique<cli_agent_embedding_provider>(options);
    }
    if (!options.tool_profile.empty()) {
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
            bindings.embed_memory_query = [provider = selection.embedding_provider.get()](const std::string & text, std::vector<float> & embedding, std::string & embedding_error) {
                return provider != nullptr &&
                    provider->embed("tool query", text, embedding, embedding_error);
            };
        }

        native_provider = std::make_unique<native_agent_tool_provider>(
            tool_catalog,
            [bindings](const agent_tool_context &, common_native_tool_bindings & resolved, std::string & binding_error) mutable {
                resolved = bindings;
                binding_error.clear();
                return true;
            });
    }

    std::unique_ptr<mcp_agent_tool_provider> mcp_provider;
    if (!options.mcp_tool_command.empty()) {
        std::vector<std::string> command_line;
        command_line.push_back(options.mcp_tool_command);
        command_line.insert(command_line.end(), options.mcp_tool_args.begin(), options.mcp_tool_args.end());

        selection.mcp_client = std::make_unique<agent_mcp_stdio_client>(agent_mcp_stdio_client_config{
            options.mcp_tool_server_name,
            std::move(command_line),
            {},
        });
        mcp_provider = std::make_unique<mcp_agent_tool_provider>(
            options.mcp_tool_server_name,
            *selection.mcp_client,
            options.mcp_tool_prefix);
    }

    if (native_provider || mcp_provider) {
        const auto repository_root = !options.repository_root.empty()
            ? std::filesystem::weakly_canonical(options.repository_root).string()
            : std::string();
        const auto tool_context = make_agent_cli_tool_context(options, query, repository_root);

        composite_agent_tool_provider provider;
        if (native_provider) {
            provider.add_provider(*native_provider);
        }
        if (mcp_provider) {
            provider.add_provider(*mcp_provider);
        }

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
        agent_embedding_provider * embedding_provider,
        common_agent_request request,
        common_agent_generation_options generation_options) {
    common_agent_runtime_turn_request turn_request;
    turn_request.request = std::move(request);
    turn_request.scope = scope;
    turn_request.inference_options = make_agent_inference_options({
        options.model,
        options.n_predict,
        options.n_gpu_layers,
        true,
    });
    turn_request.policy = make_agent_runtime_policy({
        options.agent_inference_backend,
        options.tool_profile,
        options.memory_learn,
        options.memory_learn_show_candidate,
        options.plan_show_summary,
        options.agent_trace,
        options.reflection_mode,
        static_cast<size_t>(options.max_tool_rounds),
    });
    turn_request.runtime_config = make_agent_runtime_config({
        {options.n_predict},
        options.memory_learn == "post-turn",
        {options.memory_learn_min_confidence, options.memory_learn_min_reuse},
        [embedding_provider](const std::string & text, std::vector<float> & embedding, std::string & error) {
            return embedding_provider != nullptr &&
                embedding_provider->embed("memory candidate", text, embedding, error);
        },
    });
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
        agent_embedding_provider * embedding_provider,
        common_agent_runtime_host_post_run post_run) {
    common_agent_scope runtime_scope = common_cli_make_agent_scope_with_matching_plan_scope(options);
    common_agent_request request;
    request.messages = messages;
    common_agent_generation_options generation_options;
    auto turn_request = make_agent_cli_runtime_turn_request(
        options,
        runtime_scope,
        make_agent_orchestration_config({
            options.prompt,
            options.agent_plan,
            options.agent_blueprint,
            options.agent_bootstrap,
            options.agent_import,
            options.agent_export,
        }),
        memory_scope,
        memory_enabled,
        fallback_reason,
        embedding_provider,
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
        agent_embedding_provider * embedding_provider,
        common_agent_runtime_host_post_run post_run) {
    auto turn_request = make_agent_cli_runtime_turn_request(
        options,
        scope,
        orchestration_config,
        memory_scope,
        memory_enabled,
        fallback_reason,
        embedding_provider);
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
