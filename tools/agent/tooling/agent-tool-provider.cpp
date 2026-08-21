#include "agent-tool-provider.h"
#include "plan/plan-bindings.h"
#include "agent-tool-result-json-contracts.h"
#include "agent/agent-runtime.h"

#include "agent/tooling/registry/tool-registry.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

void fill_repair_skeleton(
        common_agent_tool_repair_context & result,
        const std::string & schema_json) {
    const auto schema = json::parse(schema_json, nullptr, false);
    if (!schema.is_object() || !schema.contains("properties") || !schema["properties"].is_object()) return;
    json skeleton = json::object();
    std::vector<std::string> required;
    if (schema.contains("required") && schema["required"].is_array()) {
        for (const auto & item : schema["required"]) {
            if (item.is_string()) required.push_back(item.get<std::string>());
        }
    }
    size_t count = 0;
    for (const auto & name : required) {
        if (count++ >= 24 || !schema["properties"].contains(name)) break;
        const auto & property = schema["properties"][name];
        if (property.contains("default")) skeleton[name] = property["default"];
        else if (property.contains("minimum") &&
                (property.value("type", "") == "integer" || property.value("type", "") == "number")) {
            skeleton[name] = property["minimum"];
        }
        else if (property.value("type", "") == "string") skeleton[name] = "";
        else if (property.value("type", "") == "integer" || property.value("type", "") == "number") skeleton[name] = 0;
        else if (property.value("type", "") == "boolean") skeleton[name] = false;
        else if (property.value("type", "") == "array") skeleton[name] = json::array();
        else skeleton[name] = json::object();
    }
    result.arguments_skeleton = skeleton.dump();
}

size_t effective_result_limit(
        const agent_tool_context & context,
        const common_tool_definition & definition) {
    return std::min(context.default_max_result_bytes, definition.max_result_bytes);
}

bool tool_async_enabled(
        const agent_tool_context & context,
        const std::string & tool_name) {
    return std::find(
        context.async_exposed_tool_names.begin(),
        context.async_exposed_tool_names.end(),
        tool_name) != context.async_exposed_tool_names.end();
}

std::chrono::steady_clock::time_point effective_async_deadline(
        const agent_tool_context & context,
        uint32_t timeout_ms) {
    auto deadline = context.execution_control.deadline.value_or(
        std::chrono::steady_clock::time_point::max());
    if (timeout_ms > 0) {
        deadline = std::min(
            deadline,
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms));
    }
    return deadline;
}

bool is_definition_allowed(
        const common_tool_definition & definition,
        const common_tool_registry & registry,
        const agent_tool_context & context) {
    if (!context.allowed_exposed_tool_names.empty() &&
            std::find(context.allowed_exposed_tool_names.begin(), context.allowed_exposed_tool_names.end(), definition.name) == context.allowed_exposed_tool_names.end()) {
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

        case common_tool_risk_class::sandbox_execution:
            return context.allow_policy_gated_writes &&
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
    return make_agent_tool_failure_result(
        call,
        std::move(failure_code),
        failure_class,
        retryable,
        std::move(safe_summary),
        std::move(raw_diagnostic));
}

agent_tool_result make_execution_control_failure_result(
        const agent_tool_context & context,
        const agent_tool_call & call) {
    if (context.execution_control.is_cancel_requested()) {
        return make_failure_result(
            call,
            "tool_call_cancelled",
            common_tool_failure_class::execution,
            false,
            "The tool call was cancelled by the host runtime.",
            context.execution_control.stop_reason());
    }
    if (context.execution_control.is_deadline_exceeded()) {
        return make_failure_result(
            call,
            "tool_call_deadline_exceeded",
            common_tool_failure_class::timeout,
            false,
            "The tool call exceeded the host turn deadline.",
            context.execution_control.stop_reason());
    }
    return {};
}

uint32_t effective_timeout_ms(
        const agent_tool_context & context,
        const common_tool_definition & definition) {
    return definition.timeout_ms > 0 ? definition.timeout_ms : context.default_timeout_ms;
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

    return make_agent_tool_json_success_result(
        call,
        execution.output,
        execution.content_summary.empty() ? definition.description : execution.content_summary,
        execution.resource_refs);
}

bool is_mcp_definition_allowed(
        const mcp_agent_tool_definition & definition,
        const std::string & exposed_name,
        const agent_tool_context & context) {
    if (!context.allowed_exposed_tool_names.empty() &&
            std::find(context.allowed_exposed_tool_names.begin(), context.allowed_exposed_tool_names.end(), exposed_name) == context.allowed_exposed_tool_names.end()) {
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
        const mcp_agent_tool_call_result & execution,
        size_t result_limit) {
    if (!execution.ok) {
        return make_failure_result(
            call,
            execution.failure_code.empty() ? "mcp.tool_call_failed" : execution.failure_code,
            execution.failure_class,
            execution.retryable,
            execution.safe_summary.empty() ? "The MCP tool call failed." : execution.safe_summary,
            execution.raw_diagnostic);
    }

    if (execution.structured_content_json.size() > result_limit ||
            execution.text_content.size() > result_limit - execution.structured_content_json.size()) {
        return make_failure_result(
            call,
            "tool_result_too_large",
            common_tool_failure_class::limit,
            false,
            "The MCP tool result exceeded the configured result limit.",
            "MCP tool result bytes exceeded configured limit");
    }

    if (!execution.structured_content_json.empty()) {
        return make_agent_tool_structured_success_result(
            call,
            execution.structured_content_json,
            execution.text_content,
            execution.resource_refs);
    }

    return make_agent_tool_text_success_result(
        call,
        execution.text_content,
        execution.text_content,
        execution.resource_refs);
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

    bool describe_tool_dataflow(
            const std::string & name,
            common_plan_tool_dataflow_contract & contract,
            std::string & error) const override {
        const auto it = definitions.find(name);
        if (it == definitions.end()) { error.clear(); return false; }
        return common_plan_dataflow_contract_from_schemas(
            it->second.name,
            it->second.input_schema_json,
            it->second.result_schema_json,
            contract,
            error);
    }

    common_agent_tool_repair_context make_repair_context(
            const std::string & name,
            const std::string &,
            const std::string & validation_error) const override {
        common_agent_tool_repair_context result;
        result.tool_name = name;
        result.validation_error = validation_error;
        for (const auto & tool : chat_tool_list) {
            if (tool.name == name) {
                result.available_tools.push_back(tool.name);
                result.compact_contract = tool.description;
            }
        }
        const auto it = definitions.find(name);
        if (it != definitions.end()) fill_repair_skeleton(result, it->second.input_schema_json);
        std::sort(result.available_tools.begin(), result.available_tools.end());
        return result;
    }

    bool exposes_tool(const std::string & name) const override {
        return definitions.find(name) != definitions.end();
    }

    bool is_read_only(const std::string & name) const override {
        return exposes_tool(name) && registry.is_read_only(name);
    }

    bool is_policy_gated(const std::string & name) const override {
        return exposes_tool(name) && registry.is_policy_gated(name);
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
        return execute_sync(call, error);
    }

    bool supports_async_call(const std::string & name) const override {
        return exposes_tool(name) && tool_async_enabled(context, name);
    }

    bool begin_call_async(
            const agent_tool_call & call,
            agent_tool_pending_call & pending,
            std::string & error) override {
        if (!supports_async_call(call.name)) {
            error = "tool is not configured for asynchronous execution in this runtime view";
            return false;
        }
        if (!reserve_call_slot(call, error)) {
            return false;
        }

        pending = {};
        pending.kind = common_runtime_operation_kind::tool;
        pending.subject_name = call.name;
        pending.operation_id = "native-tool-op-" + std::to_string(++next_async_operation_id);
        const auto cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
        agent_tool_context async_context = context;
        async_context.execution_control.cancellation = cancellation;
        async_context.execution_control.deadline = effective_async_deadline(
            context,
            effective_timeout_ms(context, definitions.at(call.name)));
        pending.deadline = *async_context.execution_control.deadline;
        {
            std::lock_guard<std::mutex> lock(async_mutex);
            async_calls.emplace(
                pending.operation_id,
                std::async(std::launch::async, [this, call, async_context]() mutable {
                    std::string ignored_error;
                    return execute_validated_call(call, async_context, ignored_error);
                }));
            async_cancellations.emplace(pending.operation_id, cancellation);
        }
        error.clear();
        return true;
    }

    bool poll_call_async(
            const agent_tool_pending_call & pending,
            bool & ready,
            agent_tool_result & result,
            std::string & error) override {
        std::lock_guard<std::mutex> lock(async_mutex);
        const auto it = async_calls.find(pending.operation_id);
        if (it == async_calls.end()) {
            error = "async tool operation is unknown";
            ready = false;
            return false;
        }
        if (it->second.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            ready = false;
            error.clear();
            return true;
        }
        ready = true;
        result = it->second.get();
        error = result.ok ? std::string() : result.raw_diagnostic;
        async_calls.erase(it);
        async_cancellations.erase(pending.operation_id);
        return true;
    }

    bool cancel_call_async(
            const agent_tool_pending_call & pending,
            std::string & error) override {
        std::lock_guard<std::mutex> lock(async_mutex);
        const auto it = async_cancellations.find(pending.operation_id);
        if (it == async_cancellations.end()) {
            error = "async tool operation is unknown";
            return false;
        }
        it->second->request_cancel("tool operation cancelled");
        error.clear();
        return true;
    }

private:
    bool reserve_call_slot(
            const agent_tool_call & call,
            std::string & error) {
        if (context.execution_control.should_stop()) {
            auto result = make_execution_control_failure_result(context, call);
            error = result.raw_diagnostic;
            return false;
        }
        auto definition_it = definitions.find(call.name);
        if (definition_it == definitions.end()) {
            error = "tool is unavailable in this runtime view";
            return false;
        }

        std::string validation_error;
        if (!validate(call, validation_error)) {
            error = validation_error;
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (call_count >= context.max_calls) {
                error = "tool call limit reached";
                return false;
            }
            ++call_count;
        }
        error.clear();
        return true;
    }

    agent_tool_result execute_sync(
            const agent_tool_call & call,
            std::string & error) {
        if (!reserve_call_slot(call, error)) {
            if (context.execution_control.should_stop()) {
                auto result = make_execution_control_failure_result(context, call);
                error = result.raw_diagnostic;
                return result;
            }
            if (definitions.find(call.name) == definitions.end()) {
                return make_failure_result(
                    call,
                    "tool_unavailable",
                    common_tool_failure_class::not_found,
                    false,
                    "The requested tool is not available in this runtime view.",
                    error);
            }
            if (error == "tool call limit reached") {
                return make_failure_result(
                    call,
                    "tool_call_limit_reached",
                    common_tool_failure_class::limit,
                    false,
                    "The runtime tool call limit has been reached.");
            }
            return make_failure_result(
                call,
                "tool.invalid_arguments",
                common_tool_failure_class::validation,
                false,
                "Tool arguments do not satisfy the registered contract.",
                error);
        }
        return execute_validated_call(call, error);
    }

    agent_tool_result execute_validated_call(
            const agent_tool_call & call,
            std::string & error) {
        return execute_validated_call(call, context, error);
    }

    agent_tool_result execute_validated_call(
            const agent_tool_call & call,
            const agent_tool_context & execution_context,
            std::string & error) {
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

        const auto started_at = std::chrono::steady_clock::now();
        const auto execution = registry.execute({call.name, call.arguments_json});
        if (execution_context.execution_control.should_stop()) {
            auto result = make_execution_control_failure_result(execution_context, call);
            error = result.raw_diagnostic;
            return result;
        }
        const uint32_t timeout_ms = effective_timeout_ms(execution_context, definition_it->second);
        const auto elapsed_ms = (uint32_t) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started_at).count();
        if (timeout_ms > 0 && elapsed_ms > timeout_ms) {
            error = "tool execution exceeded timeout";
            return make_failure_result(
                call,
                "tool_call_timeout",
                common_tool_failure_class::timeout,
                false,
                "The tool call exceeded the configured timeout.",
                error);
        }
        auto result = normalize_execution_result(execution_context, definition_it->second, call, execution);
        error = result.ok ? std::string() : result.raw_diagnostic;
        return result;
    }
    agent_tool_context context;
    common_tool_registry registry;
    std::vector<common_chat_tool> chat_tool_list;
    std::map<std::string, common_tool_definition> definitions;
    size_t call_count = 0;
    mutable std::mutex state_mutex;
    mutable std::mutex async_mutex;
    std::map<std::string, std::future<agent_tool_result>> async_calls;
    std::map<std::string, std::shared_ptr<common_agent_runtime_cancellation_state>> async_cancellations;
    std::atomic<uint64_t> next_async_operation_id = 0;
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

    bool is_policy_gated(const std::string & name) const override {
        auto it = definitions.find(name);
        return it != definitions.end() &&
            !it->second.read_only &&
            it->second.requires_confirmation;
    }

    bool validate(const agent_tool_call & call, std::string & error) const override {
        auto it = definitions.find(call.name);
        if (it == definitions.end()) {
            error = "tool is unavailable in this MCP runtime view";
            return false;
        }

        std::string normalized_arguments;
        if (!normalize_arguments(call, it->second, normalized_arguments, error)) {
            return false;
        }

        error.clear();
        return true;
    }

    agent_tool_result call(
            const agent_tool_call & call,
            std::string & error) override {
        return execute_sync(call, error);
    }

    bool supports_async_call(const std::string & name) const override {
        return exposes_tool(name) && tool_async_enabled(context, name);
    }

    bool begin_call_async(
            const agent_tool_call & call,
            agent_tool_pending_call & pending,
            std::string & error) override {
        if (!supports_async_call(call.name)) {
            error = "tool is not configured for asynchronous execution in this MCP runtime view";
            return false;
        }
        if (!reserve_call_slot(call, error)) {
            return false;
        }

        pending = {};
        pending.kind = common_runtime_operation_kind::tool;
        pending.subject_name = call.name;
        pending.operation_id = "mcp-tool-op-" + std::to_string(++next_async_operation_id);
        const auto cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
        agent_tool_context async_context = context;
        async_context.execution_control.cancellation = cancellation;
        async_context.execution_control.deadline = effective_async_deadline(
            context,
            context.default_timeout_ms);
        pending.deadline = *async_context.execution_control.deadline;
        {
            std::lock_guard<std::mutex> lock(async_mutex);
            async_calls.emplace(
                pending.operation_id,
                std::async(std::launch::async, [this, call, async_context]() mutable {
                    std::string ignored_error;
                    return execute_validated_call(call, async_context, ignored_error);
                }));
            async_cancellations.emplace(pending.operation_id, cancellation);
        }
        error.clear();
        return true;
    }

    bool poll_call_async(
            const agent_tool_pending_call & pending,
            bool & ready,
            agent_tool_result & result,
            std::string & error) override {
        std::lock_guard<std::mutex> lock(async_mutex);
        const auto it = async_calls.find(pending.operation_id);
        if (it == async_calls.end()) {
            error = "async MCP tool operation is unknown";
            ready = false;
            return false;
        }
        if (it->second.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            ready = false;
            error.clear();
            return true;
        }
        ready = true;
        result = it->second.get();
        error = result.ok ? std::string() : result.raw_diagnostic;
        async_calls.erase(it);
        async_cancellations.erase(pending.operation_id);
        return true;
    }

    bool cancel_call_async(
            const agent_tool_pending_call & pending,
            std::string & error) override {
        std::lock_guard<std::mutex> lock(async_mutex);
        const auto it = async_cancellations.find(pending.operation_id);
        if (it == async_cancellations.end()) {
            error = "async MCP tool operation is unknown";
            return false;
        }
        it->second->request_cancel("MCP tool operation cancelled");
        error.clear();
        return true;
    }

private:
    bool reserve_call_slot(
            const agent_tool_call & call,
            std::string & error) {
        if (context.execution_control.should_stop()) {
            auto result = make_execution_control_failure_result(context, call);
            error = result.raw_diagnostic;
            return false;
        }
        auto it = definitions.find(call.name);
        if (it == definitions.end()) {
            error = "tool is unavailable in this MCP runtime view";
            return false;
        }

        std::string validation_error;
        if (!validate(call, validation_error)) {
            error = validation_error;
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (call_count >= context.max_calls) {
                error = "tool call limit reached";
                return false;
            }
            ++call_count;
        }
        error.clear();
        return true;
    }

    agent_tool_result execute_sync(
            const agent_tool_call & call,
            std::string & error) {
        if (!reserve_call_slot(call, error)) {
            if (context.execution_control.should_stop()) {
                auto result = make_execution_control_failure_result(context, call);
                error = result.raw_diagnostic;
                return result;
            }
            if (definitions.find(call.name) == definitions.end()) {
                return make_failure_result(
                    call,
                    "tool_unavailable",
                    common_tool_failure_class::not_found,
                    false,
                    "The requested tool is not available in this MCP runtime view.",
                    error);
            }
            if (error == "tool call limit reached") {
                return make_failure_result(
                    call,
                    "tool_call_limit_reached",
                    common_tool_failure_class::limit,
                    false,
                    "The runtime tool call limit has been reached.");
            }
            return make_failure_result(
                call,
                "tool.invalid_arguments",
                common_tool_failure_class::validation,
                false,
                "Tool arguments do not satisfy the MCP tool contract.",
                error);
        }
        return execute_validated_call(call, error);
    }

    agent_tool_result execute_validated_call(
            const agent_tool_call & call,
            std::string & error) {
        return execute_validated_call(call, context, error);
    }

    agent_tool_result execute_validated_call(
            const agent_tool_call & call,
            const agent_tool_context & execution_context,
            std::string & error) {
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

        const auto started_at = std::chrono::steady_clock::now();
        mcp_agent_tool_call_result execution;
        std::string normalized_arguments;
        if (!normalize_arguments(call, it->second, normalized_arguments, error)) {
            return make_failure_result(
                call,
                "tool.invalid_arguments",
                common_tool_failure_class::validation,
                false,
                "MCP tool arguments do not satisfy the registered contract.",
                error);
        }
        if (!client.call_tool(execution_context, it->second, normalized_arguments, execution, error)) {
            if (execution_context.execution_control.should_stop()) {
                auto result = make_execution_control_failure_result(execution_context, call);
                error = result.raw_diagnostic;
                return result;
            }
            return make_failure_result(
                call,
                "mcp.call_failed",
                common_tool_failure_class::execution,
                false,
                "The MCP tool client failed to execute the requested tool.",
                error);
        }
        if (execution_context.execution_control.should_stop()) {
            auto result = make_execution_control_failure_result(execution_context, call);
            error = result.raw_diagnostic;
            return result;
        }
        const auto elapsed_ms = (uint32_t) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started_at).count();
        if (execution_context.default_timeout_ms > 0 && elapsed_ms > execution_context.default_timeout_ms) {
            error = "MCP tool execution exceeded timeout";
            return make_failure_result(
                call,
                "mcp.tool_timeout",
                common_tool_failure_class::timeout,
                false,
                "The MCP tool call exceeded the configured timeout.",
                error);
        }

        const size_t result_limit = std::min(
            execution_context.default_max_result_bytes,
            it->second.max_result_bytes);
        auto result = normalize_mcp_execution_result(call, execution, result_limit);
        error = result.ok ? std::string() : result.raw_diagnostic;
        return result;
    }

    bool normalize_arguments(
            const agent_tool_call & call,
            const mcp_agent_tool_definition & definition,
            std::string & normalized_arguments,
            std::string & error) const {
        return common_schema_normalize_and_validate_object(
            call.arguments_json,
            definition.input_schema_json,
            normalized_arguments,
            error);
    }
    agent_tool_context context;
    agent_mcp_tool_client & client;
    std::vector<common_chat_tool> chat_tool_list;
    std::map<std::string, mcp_agent_tool_definition> definitions;
    size_t call_count = 0;
    mutable std::mutex state_mutex;
    mutable std::mutex async_mutex;
    std::map<std::string, std::future<agent_tool_result>> async_calls;
    std::map<std::string, std::shared_ptr<common_agent_runtime_cancellation_state>> async_cancellations;
    std::atomic<uint64_t> next_async_operation_id = 0;
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

    bool describe_tool_dataflow(
            const std::string & name,
            common_plan_tool_dataflow_contract & contract,
            std::string & error) const override {
        const auto * view = find_owner(name);
        return view != nullptr && view->describe_tool_dataflow(name, contract, error);
    }

    common_agent_tool_repair_context make_repair_context(
            const std::string & name,
            const std::string & arguments_json,
            const std::string & validation_error) const override {
        const auto * view = find_owner(name);
        if (view != nullptr) return view->make_repair_context(name, arguments_json, validation_error);
        common_agent_tool_repair_context result;
        result.tool_name = name;
        result.validation_error = validation_error;
        return result;
    }

    bool exposes_tool(const std::string & name) const override {
        return tool_owners.find(name) != tool_owners.end();
    }

    bool is_read_only(const std::string & name) const override {
        const auto * view = find_owner(name);
        return view != nullptr && view->is_read_only(name);
    }

    bool is_policy_gated(const std::string & name) const override {
        const auto * view = find_owner(name);
        return view != nullptr && view->is_policy_gated(name);
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

    bool supports_async_call(const std::string & name) const override {
        const auto * view = find_owner(name);
        return view != nullptr && view->supports_async_call(name);
    }

    bool begin_call_async(
            const agent_tool_call & call,
            agent_tool_pending_call & pending,
            std::string & error) override {
        auto * view = find_owner(call.name);
        if (view == nullptr) {
            error = "tool is unavailable in this composite runtime view";
            return false;
        }
        return view->begin_call_async(call, pending, error);
    }

    bool poll_call_async(
            const agent_tool_pending_call & pending,
            bool & ready,
            agent_tool_result & result,
            std::string & error) override {
        auto * view = find_owner(pending.subject_name);
        if (view == nullptr) {
            error = "tool is unavailable in this composite runtime view";
            ready = false;
            return false;
        }
        return view->poll_call_async(pending, ready, result, error);
    }

    bool cancel_call_async(
            const agent_tool_pending_call & pending,
            std::string & error) override {
        auto * view = find_owner(pending.subject_name);
        if (view == nullptr) {
            error = "tool is unavailable in this composite runtime view";
            return false;
        }
        return view->cancel_call_async(pending, error);
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

common_agent_tool_repair_context agent_tool_view::make_repair_context(
        const std::string & name,
        const std::string &,
        const std::string & validation_error) const {
    common_agent_tool_repair_context result;
    result.tool_name = name;
    result.validation_error = validation_error;
    for (const auto & tool : chat_tools()) {
        if (tool.name == name) {
            result.available_tools.push_back(tool.name);
            fill_repair_skeleton(result, tool.parameters);
            result.compact_contract = tool.description;
        }
    }
    std::sort(result.available_tools.begin(), result.available_tools.end());
    return result;
}

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

    std::vector<common_tool_definition> definitions;
    if (context.profile_snapshot) {
        if (context.profile_snapshot->id != context.profile_id) {
            error = "tool profile snapshot id does not match runtime profile";
            return nullptr;
        }
        definitions = context.profile_snapshot->tools;
    } else {
        definitions = catalog.load_profile(context.profile_id, error);
    }
    if (!error.empty()) {
        return nullptr;
    }

    std::vector<common_chat_tool> chat_tools;
    std::map<std::string, common_tool_definition> resolved_definitions;
    for (const auto & definition : definitions) {
        if (!definition.enabled || !is_definition_allowed(definition, registry, context)) {
            continue;
        }
        std::string compact_error;
        const auto model_description = common_render_compact_tool_description(
            definition.name,
            definition.description,
            common_tool_model_input_schema(definition),
            common_tool_model_result_schema(definition),
            compact_error);
        if (!compact_error.empty()) {
            error = compact_error;
            return nullptr;
        }
        chat_tools.push_back({
            definition.name,
            model_description,
            common_tool_model_input_schema(definition),
            common_tool_model_result_schema(definition),
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
        const std::string exposed_name = make_mcp_exposed_name(
            exposed_name_prefix.empty() ? definition.provider_id : exposed_name_prefix,
            definition.name);
        if (!is_mcp_definition_allowed(definition, exposed_name, context)) {
            continue;
        }
        std::string compact_error;
        const auto model_description = common_render_compact_tool_description(
            exposed_name,
            definition.description,
            definition.input_schema_json,
            R"({"type":"object"})",
            compact_error);
        if (!compact_error.empty()) {
            error = compact_error;
            return nullptr;
        }
        chat_tools.push_back({
            exposed_name,
            model_description,
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
