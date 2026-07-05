#include "agent-tool-provider.h"

#include "agent/tool-registry.h"

#include <algorithm>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

size_t effective_result_limit(
        const agent_tool_context & context,
        const common_tool_definition & definition) {
    return std::min(context.default_max_result_bytes, definition.max_result_bytes);
}

bool is_definition_allowed(
        const common_tool_definition & definition,
        const common_tool_registry & registry,
        const agent_tool_context & context) {
    if (!context.allowed_tool_names.empty() &&
            std::find(context.allowed_tool_names.begin(), context.allowed_tool_names.end(), definition.name) == context.allowed_tool_names.end()) {
        return false;
    }
    if (!registry.matches_binding(definition.name, definition.version, definition.executor_id)) {
        return false;
    }

    switch (definition.risk_class) {
        case common_tool_risk_class::local_read:
            return registry.is_read_only(definition.name);

        case common_tool_risk_class::network_read:
            return context.allow_network && registry.is_read_only(definition.name);

        case common_tool_risk_class::memory_proposal:
            return context.allow_memory_proposals &&
                context.allow_policy_gated_writes &&
                definition.requires_confirmation &&
                registry.is_policy_gated(definition.name);

        case common_tool_risk_class::plan_proposal:
            return context.allow_plan_proposals &&
                context.allow_policy_gated_writes &&
                definition.requires_confirmation &&
                registry.is_policy_gated(definition.name);
    }

    return false;
}

agent_tool_result make_failure_result(
        const agent_tool_call & call,
        std::string failure_code,
        common_tool_failure_class failure_class,
        bool retryable,
        std::string safe_summary,
        std::string raw_diagnostic = {}) {
    agent_tool_result result;
    result.ok = false;
    result.tool_call_id = call.id;
    result.tool_name = call.name;
    result.failure_code = std::move(failure_code);
    result.failure_class = failure_class;
    result.retryable = retryable;
    result.safe_summary = std::move(safe_summary);
    result.raw_diagnostic = std::move(raw_diagnostic);
    result.content_json = json({
        {"ok", false},
        {"error", {
            {"code", result.failure_code.empty() ? "tool_call_rejected" : result.failure_code},
            {"message", result.safe_summary.empty() ? "The tool call was rejected by its native contract or executor." : result.safe_summary},
            {"retryable", result.retryable},
            {"class", common_tool_failure_class_name(result.failure_class)},
        }}
    }).dump();
    return result;
}

agent_tool_result normalize_execution_result(
        const agent_tool_context & context,
        const common_tool_definition & definition,
        const agent_tool_call & call,
        const common_tool_execution_result & execution) {
    if (!execution.ok) {
        return make_failure_result(
            call,
            execution.failure_code.empty() ? "tool_call_rejected" : execution.failure_code,
            execution.failure_class,
            execution.retryable,
            execution.safe_summary.empty() ? "The tool call was rejected by its native contract or executor." : execution.safe_summary,
            execution.raw_diagnostic);
    }

    const size_t result_limit = effective_result_limit(context, definition);
    if (execution.output.size() > result_limit) {
        return make_failure_result(
            call,
            "tool_result_too_large",
            common_tool_failure_class::limit,
            false,
            "The tool result exceeded the configured result limit.",
            "tool result bytes exceeded configured limit");
    }

    agent_tool_result result;
    result.ok = true;
    result.tool_call_id = call.id;
    result.tool_name = call.name;

    const auto value = json::parse(execution.output, nullptr, false);
    result.content_json = value.is_discarded()
        ? json({{"ok", true}, {"result_text", execution.output}}).dump()
        : json({{"ok", true}, {"result", value}}).dump();
    return result;
}

bool is_mcp_definition_allowed(
        const mcp_agent_tool_definition & definition,
        const agent_tool_context & context) {
    if (!context.allowed_tool_names.empty() &&
            std::find(context.allowed_tool_names.begin(), context.allowed_tool_names.end(), definition.name) == context.allowed_tool_names.end()) {
        return false;
    }
    if (definition.uses_network && !context.allow_network) {
        return false;
    }
    if (definition.writes_memory &&
            (!context.allow_memory_proposals || !context.allow_policy_gated_writes || !definition.requires_confirmation)) {
        return false;
    }
    if (definition.writes_plan &&
            (!context.allow_plan_proposals || !context.allow_policy_gated_writes || !definition.requires_confirmation)) {
        return false;
    }
    if (!definition.read_only &&
            !definition.writes_memory &&
            !definition.writes_plan &&
            (!context.allow_policy_gated_writes || !definition.requires_confirmation)) {
        return false;
    }
    return true;
}

std::string make_mcp_exposed_name(
        const std::string & prefix,
        const std::string & tool_name) {
    if (prefix.empty()) {
        return tool_name;
    }
    return prefix + "_" + tool_name;
}

agent_tool_result normalize_mcp_execution_result(
        const agent_tool_call & call,
        const mcp_agent_tool_call_result & execution) {
    if (!execution.ok) {
        return make_failure_result(
            call,
            execution.failure_code.empty() ? "mcp.tool_call_failed" : execution.failure_code,
            execution.failure_class,
            execution.retryable,
            execution.safe_summary.empty() ? "The MCP tool call failed." : execution.safe_summary,
            execution.raw_diagnostic);
    }

    agent_tool_result result;
    result.ok = true;
    result.tool_call_id = call.id;
    result.tool_name = call.name;

    if (!execution.structured_content_json.empty()) {
        const auto value = json::parse(execution.structured_content_json, nullptr, false);
        result.content_json = value.is_discarded()
            ? json({{"ok", true}, {"result_text", execution.structured_content_json}}).dump()
            : json({{"ok", true}, {"result", value}}).dump();
        return result;
    }

    result.content_json = json({{"ok", true}, {"result_text", execution.text_content}}).dump();
    return result;
}

class native_agent_tool_view : public agent_tool_view {
public:
    native_agent_tool_view(
            agent_tool_context context,
            common_tool_registry registry,
            std::vector<common_chat_tool> chat_tools,
            std::map<std::string, common_tool_definition> definitions)
        : context(std::move(context))
        , registry(std::move(registry))
        , chat_tool_list(std::move(chat_tools))
        , definitions(std::move(definitions)) {}

    const std::vector<common_chat_tool> & chat_tools() const override {
        return chat_tool_list;
    }

    bool exposes_tool(const std::string & name) const override {
        return definitions.find(name) != definitions.end();
    }

    bool is_read_only(const std::string & name) const override {
        return exposes_tool(name) && registry.is_read_only(name);
    }

    bool validate(const agent_tool_call & call, std::string & error) const override {
        if (!exposes_tool(call.name)) {
            error = "tool is unavailable in this runtime view";
            return false;
        }
        return registry.validate({call.name, call.arguments_json}, error);
    }

    agent_tool_result call(
            const agent_tool_call & call,
            std::string & error) override {
        if (call_count >= context.max_calls) {
            error = "tool call limit reached";
            return make_failure_result(
                call,
                "tool_call_limit_reached",
                common_tool_failure_class::limit,
                false,
                "The runtime tool call limit has been reached.");
        }

        auto definition_it = definitions.find(call.name);
        if (definition_it == definitions.end()) {
            error = "tool is unavailable in this runtime view";
            return make_failure_result(
                call,
                "tool_unavailable",
                common_tool_failure_class::not_found,
                false,
                "The requested tool is not available in this runtime view.",
                error);
        }

        std::string validation_error;
        if (!validate(call, validation_error)) {
            error = validation_error;
            return make_failure_result(
                call,
                "tool.invalid_arguments",
                common_tool_failure_class::validation,
                false,
                "Tool arguments do not satisfy the registered contract.",
                validation_error);
        }

        ++call_count;
        const auto execution = registry.execute({call.name, call.arguments_json});
        auto result = normalize_execution_result(context, definition_it->second, call, execution);
        error = result.ok ? std::string() : result.raw_diagnostic;
        return result;
    }

private:
    agent_tool_context context;
    common_tool_registry registry;
    std::vector<common_chat_tool> chat_tool_list;
    std::map<std::string, common_tool_definition> definitions;
    size_t call_count = 0;
};

class mcp_agent_tool_view : public agent_tool_view {
public:
    mcp_agent_tool_view(
            agent_tool_context context,
            agent_mcp_tool_client & client,
            std::vector<common_chat_tool> chat_tools,
            std::map<std::string, mcp_agent_tool_definition> definitions)
        : context(std::move(context))
        , client(client)
        , chat_tool_list(std::move(chat_tools))
        , definitions(std::move(definitions)) {}

    const std::vector<common_chat_tool> & chat_tools() const override {
        return chat_tool_list;
    }

    bool exposes_tool(const std::string & name) const override {
        return definitions.find(name) != definitions.end();
    }

    bool is_read_only(const std::string & name) const override {
        auto it = definitions.find(name);
        return it != definitions.end() && it->second.read_only;
    }

    bool validate(const agent_tool_call & call, std::string & error) const override {
        auto it = definitions.find(call.name);
        if (it == definitions.end()) {
            error = "tool is unavailable in this MCP runtime view";
            return false;
        }

        const auto arguments = json::parse(call.arguments_json, nullptr, false);
        if (arguments.is_discarded() || !arguments.is_object()) {
            error = "tool arguments must be a JSON object";
            return false;
        }

        error.clear();
        return true;
    }

    agent_tool_result call(
            const agent_tool_call & call,
            std::string & error) override {
        if (call_count >= context.max_calls) {
            error = "tool call limit reached";
            return make_failure_result(
                call,
                "tool_call_limit_reached",
                common_tool_failure_class::limit,
                false,
                "The runtime tool call limit has been reached.");
        }

        auto it = definitions.find(call.name);
        if (it == definitions.end()) {
            error = "tool is unavailable in this MCP runtime view";
            return make_failure_result(
                call,
                "tool_unavailable",
                common_tool_failure_class::not_found,
                false,
                "The requested tool is not available in this MCP runtime view.",
                error);
        }

        std::string validation_error;
        if (!validate(call, validation_error)) {
            error = validation_error;
            return make_failure_result(
                call,
                "tool.invalid_arguments",
                common_tool_failure_class::validation,
                false,
                "Tool arguments do not satisfy the MCP tool contract.",
                validation_error);
        }

        ++call_count;
        mcp_agent_tool_call_result execution;
        if (!client.call_tool(context, it->second, call.arguments_json, execution, error)) {
            return make_failure_result(
                call,
                "mcp.call_failed",
                common_tool_failure_class::execution,
                false,
                "The MCP tool client failed to execute the requested tool.",
                error);
        }

        auto result = normalize_mcp_execution_result(call, execution);
        error = result.ok ? std::string() : result.raw_diagnostic;
        return result;
    }

private:
    agent_tool_context context;
    agent_mcp_tool_client & client;
    std::vector<common_chat_tool> chat_tool_list;
    std::map<std::string, mcp_agent_tool_definition> definitions;
    size_t call_count = 0;
};

class composite_agent_tool_view : public agent_tool_view {
public:
    composite_agent_tool_view(
            std::vector<std::unique_ptr<agent_tool_view>> views,
            std::vector<common_chat_tool> chat_tools,
            std::map<std::string, size_t> tool_owners)
        : views(std::move(views))
        , chat_tool_list(std::move(chat_tools))
        , tool_owners(std::move(tool_owners)) {}

    const std::vector<common_chat_tool> & chat_tools() const override {
        return chat_tool_list;
    }

    bool exposes_tool(const std::string & name) const override {
        return tool_owners.find(name) != tool_owners.end();
    }

    bool is_read_only(const std::string & name) const override {
        const auto * view = find_owner(name);
        return view != nullptr && view->is_read_only(name);
    }

    bool validate(const agent_tool_call & call, std::string & error) const override {
        const auto * view = find_owner(call.name);
        if (view == nullptr) {
            error = "tool is unavailable in this composite runtime view";
            return false;
        }
        return view->validate(call, error);
    }

    agent_tool_result call(
            const agent_tool_call & call,
            std::string & error) override {
        auto * view = find_owner(call.name);
        if (view == nullptr) {
            error = "tool is unavailable in this composite runtime view";
            return make_failure_result(
                call,
                "tool_unavailable",
                common_tool_failure_class::not_found,
                false,
                "The requested tool is not available in this runtime view.",
                error);
        }
        return view->call(call, error);
    }

private:
    const agent_tool_view * find_owner(const std::string & name) const {
        const auto it = tool_owners.find(name);
        if (it == tool_owners.end() || it->second >= views.size()) {
            return nullptr;
        }
        return views[it->second].get();
    }

    agent_tool_view * find_owner(const std::string & name) {
        const auto it = tool_owners.find(name);
        if (it == tool_owners.end() || it->second >= views.size()) {
            return nullptr;
        }
        return views[it->second].get();
    }

    std::vector<std::unique_ptr<agent_tool_view>> views;
    std::vector<common_chat_tool> chat_tool_list;
    std::map<std::string, size_t> tool_owners;
};

} // namespace

native_agent_tool_provider::native_agent_tool_provider(
        const common_tool_catalog & catalog,
        binding_factory make_bindings)
    : catalog(catalog)
    , make_bindings(std::move(make_bindings)) {}

std::unique_ptr<agent_tool_view> native_agent_tool_provider::resolve_tools(
        const agent_tool_context & context,
        std::string & error) {
    common_native_tool_bindings bindings;
    if (make_bindings && !make_bindings(context, bindings, error)) {
        return nullptr;
    }

    common_tool_registry registry;
    common_tool_adapter_result adapter_result;
    if (!common_register_native_tool_adapters(
            catalog,
            context.profile_id,
            bindings,
            registry,
            adapter_result,
            error)) {
        return nullptr;
    }

    const auto definitions = catalog.load_profile(context.profile_id, error);
    if (!error.empty()) {
        return nullptr;
    }

    std::vector<common_chat_tool> chat_tools;
    std::map<std::string, common_tool_definition> resolved_definitions;
    for (const auto & definition : definitions) {
        if (!definition.enabled || !is_definition_allowed(definition, registry, context)) {
            continue;
        }
        chat_tools.push_back({
            definition.name,
            definition.description,
            definition.input_schema_json,
        });
        resolved_definitions.emplace(definition.name, definition);
    }

    error.clear();
    return std::make_unique<native_agent_tool_view>(
        context,
        std::move(registry),
        std::move(chat_tools),
        std::move(resolved_definitions));
}

mcp_agent_tool_provider::mcp_agent_tool_provider(
        std::string provider_id,
        agent_mcp_tool_client & client,
        std::string exposed_name_prefix)
    : provider_id(std::move(provider_id))
    , client(client)
    , exposed_name_prefix(std::move(exposed_name_prefix)) {}

std::unique_ptr<agent_tool_view> mcp_agent_tool_provider::resolve_tools(
        const agent_tool_context & context,
        std::string & error) {
    std::vector<mcp_agent_tool_definition> listed_tools;
    if (!client.list_tools(context, listed_tools, error)) {
        return nullptr;
    }

    std::vector<common_chat_tool> chat_tools;
    std::map<std::string, mcp_agent_tool_definition> resolved_definitions;
    for (auto definition : listed_tools) {
        if (definition.provider_id.empty()) {
            definition.provider_id = provider_id;
        }
        if (!is_mcp_definition_allowed(definition, context)) {
            continue;
        }

        const std::string exposed_name = make_mcp_exposed_name(
            exposed_name_prefix.empty() ? definition.provider_id : exposed_name_prefix,
            definition.name);
        chat_tools.push_back({
            exposed_name,
            definition.description,
            definition.input_schema_json,
        });
        resolved_definitions.emplace(exposed_name, std::move(definition));
    }

    error.clear();
    return std::make_unique<mcp_agent_tool_view>(
        context,
        client,
        std::move(chat_tools),
        std::move(resolved_definitions));
}

void composite_agent_tool_provider::add_provider(agent_tool_provider & provider) {
    providers.push_back(&provider);
}

std::unique_ptr<agent_tool_view> composite_agent_tool_provider::resolve_tools(
        const agent_tool_context & context,
        std::string & error) {
    std::vector<std::unique_ptr<agent_tool_view>> resolved_views;
    std::vector<common_chat_tool> merged_chat_tools;
    std::map<std::string, size_t> tool_owners;

    for (auto * provider : providers) {
        if (provider == nullptr) {
            continue;
        }

        std::unique_ptr<agent_tool_view> view = provider->resolve_tools(context, error);
        if (!view) {
            return nullptr;
        }

        const size_t owner_index = resolved_views.size();
        for (const auto & tool : view->chat_tools()) {
            if (tool_owners.find(tool.name) != tool_owners.end()) {
                error = "duplicate tool exposed in composite provider: " + tool.name;
                return nullptr;
            }
            tool_owners.emplace(tool.name, owner_index);
            merged_chat_tools.push_back(tool);
        }
        resolved_views.push_back(std::move(view));
    }

    error.clear();
    return std::make_unique<composite_agent_tool_view>(
        std::move(resolved_views),
        std::move(merged_chat_tools),
        std::move(tool_owners));
}

bool agent_dispatch_chat_tool_calls(
        common_chat_msg & assistant_message,
        agent_tool_view & tool_view,
        size_t max_calls,
        common_tool_chat_dispatch_result & result,
        std::string & error) {
    result = {};
    if (assistant_message.role != "assistant") {
        error = "only assistant messages may contain tool calls";
        return false;
    }
    if (assistant_message.tool_calls.size() > max_calls) {
        error = "tool call batch exceeds configured limit";
        return false;
    }

    for (size_t index = 0; index < assistant_message.tool_calls.size(); ++index) {
        auto & call = assistant_message.tool_calls[index];
        if (call.id.empty()) {
            call.id = "tool-call-" + std::to_string(index + 1);
        }

        const auto tool_result = tool_view.call({call.id, call.name, call.arguments}, error);
        common_chat_msg tool_message;
        tool_message.role = "tool";
        tool_message.tool_name = call.name;
        tool_message.tool_call_id = call.id;
        tool_message.content = tool_result.content_json;
        if (tool_result.ok) {
            ++result.executed;
        }
        result.tool_messages.push_back(std::move(tool_message));
    }

    error.clear();
    return true;
}
