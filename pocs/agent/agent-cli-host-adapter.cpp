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

agent_host_tool_selection_request make_agent_cli_tool_selection_request(
        const args & options,
        const common_memory_query & query,
        const std::string & repository_root) {
    agent_host_tool_selection_request request;
    request.tool_context = make_agent_cli_tool_context(options, query, repository_root);
    request.repository_root = repository_root;
    request.resource_store_config = {
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    };
    append_legacy_stdio_mcp_provider(
        options.mcp_tool_command,
        options.mcp_tool_args,
        options.mcp_tool_server_name,
        options.mcp_tool_prefix,
        request.mcp_providers);
    return request;
}

} // namespace

bool has_enabled_stdio_mcp_provider(
        const std::vector<agent_host_mcp_provider_config> & providers) {
    for (const auto & provider : providers) {
        if (provider.enabled &&
                provider.type == "mcp" &&
                provider.transport == "stdio" &&
                !provider.command.empty()) {
            return true;
        }
    }
    return false;
}

void append_configured_stdio_mcp_providers(
        const std::vector<agent_host_mcp_provider_config> & configured_providers,
        std::vector<agent_host_stdio_mcp_provider_request> & request_providers) {
    for (const auto & provider : configured_providers) {
        if (!provider.enabled || provider.type != "mcp" || provider.transport != "stdio" || provider.command.empty()) {
            continue;
        }
        agent_host_stdio_mcp_provider_request request_provider;
        request_provider.server_name = provider.server_name.empty() ? provider.id : provider.server_name;
        request_provider.exposed_name_prefix = provider.prefix;
        request_provider.command_line = provider.command;
        request_providers.push_back(std::move(request_provider));
    }
}

void append_legacy_stdio_mcp_provider(
        const std::string & command,
        const std::vector<std::string> & args,
        const std::string & server_name,
        const std::string & prefix,
        std::vector<agent_host_stdio_mcp_provider_request> & request_providers) {
    if (command.empty()) {
        return;
    }
    agent_host_stdio_mcp_provider_request provider;
    provider.server_name = server_name;
    provider.exposed_name_prefix = prefix;
    provider.command_line.push_back(command);
    provider.command_line.insert(provider.command_line.end(), args.begin(), args.end());
    request_providers.push_back(std::move(provider));
}

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

bool resolve_agent_host_tool_selection(
        common_memory_store & store,
        common_plan_store * plan_store,
        agent_resource_store * resource_store,
        std::string * current_plan_id,
        const std::string & tool_profile,
        const agent_host_tool_selection_request & request,
        const common_memory_query & query,
        agent_embedding_provider * embedding_provider,
        common_agent_cli_tool_selection & selection,
        std::string & error) {
    selection = {};

    common_tool_catalog tool_catalog;
    std::unique_ptr<native_agent_tool_provider> native_provider;
    if (!tool_profile.empty()) {
        common_tool_bootstrap_result bootstrap;
        if (!tool_catalog.bootstrap(tool_profile, bootstrap, error)) {
            error = "tool bootstrap failed: " + error;
            return false;
        }

        common_native_tool_bindings bindings;
        if (!request.repository_root.empty()) {
            bindings.repository_root = request.repository_root;
        }
        bindings.plan_store = plan_store;
        bindings.plan_id = current_plan_id;
        if (resource_store == nullptr) {
            selection.owned_resource_store = make_agent_resource_store(
                request.resource_store_config,
                error);
            if (!selection.owned_resource_store) {
                error = "resource store setup failed: " + error;
                return false;
            }
            resource_store = selection.owned_resource_store.get();
        }
        bindings.resource_runtime.store = resource_store;
        bindings.memory_store = &store;
        bindings.memory_query = query;
        if (embedding_provider != nullptr) {
            bindings.embed_memory_query = [provider = embedding_provider](const std::string & text, std::vector<float> & embedding, std::string & embedding_error) {
                return provider != nullptr &&
                    provider->embed("tool query", text, embedding, embedding_error);
            };
        }

        native_provider = std::make_unique<native_agent_tool_provider>(
            tool_catalog,
            [bindings](const agent_tool_context & context, common_native_tool_bindings & resolved, std::string & binding_error) mutable {
                resolved = bindings;
                resolved.resource_runtime.namespace_id = context.scope.namespace_id;
                resolved.resource_runtime.session_id = context.scope.session_id;
                resolved.resource_runtime.project_id = context.scope.project_id;
                resolved.resource_runtime.turn_id = context.scope.turn_id;
                binding_error.clear();
                return true;
            });
    }

    std::vector<std::unique_ptr<mcp_agent_tool_provider>> mcp_providers;
    for (const auto & provider_request : request.mcp_providers) {
        if (provider_request.command_line.empty()) {
            error = "MCP provider command line must not be empty";
            return false;
        }
        auto client = std::make_unique<agent_mcp_stdio_client>(agent_mcp_stdio_client_config{
            provider_request.server_name,
            provider_request.command_line,
            {},
        });
        auto provider = std::make_unique<mcp_agent_tool_provider>(
            provider_request.server_name,
            *client,
            provider_request.exposed_name_prefix);
        selection.mcp_clients.push_back(std::move(client));
        mcp_providers.push_back(std::move(provider));
    }

    if (native_provider || !mcp_providers.empty()) {
        composite_agent_tool_provider provider;
        if (native_provider) {
            provider.add_provider(*native_provider);
        }
        for (const auto & mcp_provider : mcp_providers) {
            provider.add_provider(*mcp_provider);
        }

        if (!(selection.tool_view = provider.resolve_tools(request.tool_context, error))) {
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

bool resolve_agent_cli_tool_selection(
        common_memory_store & store,
        common_plan_store * plan_store,
        agent_resource_store * resource_store,
        std::string * current_plan_id,
        const args & options,
        const common_memory_query & query,
        bool memory_enabled,
        common_agent_cli_tool_selection & selection,
        std::string & error) {
    selection = {};
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    if (memory_enabled) {
        embedding_provider = std::make_unique<cli_agent_embedding_provider>(options);
    }

    const auto repository_root = !options.repository_root.empty()
        ? std::filesystem::weakly_canonical(options.repository_root).string()
        : std::string();
    const auto request = make_agent_cli_tool_selection_request(options, query, repository_root);
    const bool ok = resolve_agent_host_tool_selection(
        store,
        plan_store,
        resource_store,
        current_plan_id,
        options.tool_profile,
        request,
        query,
        embedding_provider.get(),
        selection,
        error);
    if (ok) {
        selection.embedding_provider = std::move(embedding_provider);
    }
    return ok;
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
