#include "agent-cli-host-adapter.h"

#include "../memory/memory-cli-memory.h"

#include "agent-plan-orchestration.h"
#include "agent-runtime-assembly.h"
#include "agent-runtime-execution.h"
#include "common/cli-scope.h"

#include <cstdio>
#include <ctime>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

agent_tool_result make_legacy_memory_failure(
        const agent_tool_call & call,
        std::string code,
        common_tool_failure_class failure_class,
        std::string summary,
        std::string raw = {}) {
    agent_tool_result result;
    result.ok = false;
    result.tool_call_id = call.id;
    result.tool_name = call.name;
    result.failure_code = std::move(code);
    result.failure_class = failure_class;
    result.safe_summary = std::move(summary);
    result.raw_diagnostic = std::move(raw);
    result.content_json = json({
        {"ok", false},
        {"error", {
            {"code", result.failure_code},
            {"message", result.safe_summary},
            {"retryable", false},
            {"class", common_tool_failure_class_name(result.failure_class)},
        }},
    }).dump();
    return result;
}

agent_tool_result normalize_legacy_memory_result(
        const agent_tool_call & call,
        const std::string & raw_result) {
    const auto parsed = json::parse(raw_result, nullptr, false);
    if (parsed.is_discarded()) {
        agent_tool_result result;
        result.ok = true;
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        result.content_json = json({{"ok", true}, {"result_text", raw_result}}).dump();
        return result;
    }

    if (parsed.is_object() && parsed.value("ok", false)) {
        agent_tool_result result;
        result.ok = true;
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        result.content_json = json({{"ok", true}, {"result", parsed}}).dump();
        return result;
    }

    std::string summary = "The tool call was rejected by its legacy memory contract.";
    if (parsed.is_object() && parsed.contains("error") && parsed["error"].is_string()) {
        summary = parsed["error"].get<std::string>();
    }
    return make_legacy_memory_failure(
        call,
        "tool_call_rejected",
        common_tool_failure_class::execution,
        std::move(summary),
        raw_result);
}

class legacy_memory_agent_tool_view final : public agent_tool_view {
public:
    legacy_memory_agent_tool_view(
            common_memory_store & store,
            const args & options,
            bool enable_memory_search_tool,
            bool enable_memory_remember_tool)
        : store(store)
        , options(options)
        , max_calls(options.max_tool_rounds > 0 ? options.max_tool_rounds : 1) {
        if (enable_memory_search_tool) {
            tool_defs.push_back(memory_search_tool_definition());
        }
        if (enable_memory_remember_tool) {
            tool_defs.push_back(memory_remember_tool_definition());
        }
    }

    const std::vector<common_chat_tool> & chat_tools() const override {
        return tool_defs;
    }

    bool exposes_tool(const std::string & name) const override {
        for (const auto & tool : tool_defs) {
            if (tool.name == name) {
                return true;
            }
        }
        return false;
    }

    bool is_read_only(const std::string & name) const override {
        return name == "memory_search" && exposes_tool(name);
    }

    bool validate(const agent_tool_call & call, std::string & error) const override {
        if (!exposes_tool(call.name)) {
            error = "tool is unavailable in this runtime view";
            return false;
        }
        const auto parsed = json::parse(call.arguments_json, nullptr, false);
        if (!parsed.is_object()) {
            error = "tool arguments must be a JSON object";
            return false;
        }
        error.clear();
        return true;
    }

    agent_tool_result call(
            const agent_tool_call & call,
            std::string & error) override {
        if (call_count >= max_calls) {
            error = "tool call limit reached";
            return make_legacy_memory_failure(
                call,
                "tool_call_limit_reached",
                common_tool_failure_class::limit,
                "The runtime tool call limit has been reached.");
        }
        if (!exposes_tool(call.name)) {
            error = "tool is unavailable in this runtime view";
            return make_legacy_memory_failure(
                call,
                "tool_unavailable",
                common_tool_failure_class::not_found,
                "The requested tool is not available in this runtime view.",
                error);
        }
        if (!validate(call, error)) {
            return make_legacy_memory_failure(
                call,
                "tool.invalid_arguments",
                common_tool_failure_class::validation,
                "Tool arguments do not satisfy the registered contract.",
                error);
        }

        ++call_count;
        if (call.name == "memory_search") {
            error.clear();
            return normalize_legacy_memory_result(call, memory_search_tool_result(store, options, call.arguments_json));
        }
        if (call.name == "memory_remember") {
            error.clear();
            return normalize_legacy_memory_result(call, memory_remember_tool_result(store, options, call.arguments_json));
        }
        error.clear();
        return make_legacy_memory_failure(
            call,
            "tool.execution_failed",
            common_tool_failure_class::execution,
            "The legacy memory tool dispatch did not recognize the requested tool name.");
    }

private:
    common_memory_store & store;
    const args & options;
    std::vector<common_chat_tool> tool_defs;
    size_t max_calls = 1;
    size_t call_count = 0;
};

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

std::unique_ptr<agent_tool_view> make_agent_cli_legacy_memory_tool_view(
        common_memory_store & store,
        const args & options,
        bool enable_memory_search_tool,
        bool enable_memory_remember_tool) {
    if (!enable_memory_search_tool && !enable_memory_remember_tool) {
        return nullptr;
    }
    return std::make_unique<legacy_memory_agent_tool_view>(
        store,
        options,
        enable_memory_search_tool,
        enable_memory_remember_tool);
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
