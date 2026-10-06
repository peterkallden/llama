#include "agent-tool-runtime-adapter.h"
#include "plan/plan-bindings.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

std::string normalized_tool_name(const std::string & value) {
    std::string result;
    bool separator = false;
    for (const unsigned char character : value) {
        if (std::isalnum(character)) {
            if (separator && !result.empty()) result.push_back('.');
            result.push_back(static_cast<char>(std::tolower(character)));
            separator = false;
        } else {
            separator = true;
        }
    }
    return result;
}

size_t edit_distance(const std::string & left, const std::string & right) {
    std::vector<size_t> previous(right.size() + 1);
    std::vector<size_t> current(right.size() + 1);
    for (size_t column = 0; column <= right.size(); ++column) previous[column] = column;
    for (size_t row = 1; row <= left.size(); ++row) {
        current[0] = row;
        for (size_t column = 1; column <= right.size(); ++column) {
            current[column] = std::min({
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (left[row - 1] == right[column - 1] ? 0 : 1)});
        }
        previous.swap(current);
    }
    return previous[right.size()];
}

double tool_name_score(const std::string & requested, const std::string & candidate) {
    const std::string normalized_requested = normalized_tool_name(requested);
    const std::string normalized_candidate = normalized_tool_name(candidate);
    if (normalized_requested.empty() || normalized_candidate.empty()) return 0.0;
    if (normalized_requested == normalized_candidate) return 1.0;
    const std::string suffix = "." + normalized_requested;
    if (normalized_candidate.size() > suffix.size() &&
            normalized_candidate.compare(normalized_candidate.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return 0.92;
    }
    const size_t distance = edit_distance(normalized_requested, normalized_candidate);
    const size_t length = std::max(normalized_requested.size(), normalized_candidate.size());
    return length == 0 ? 0.0 : 1.0 - static_cast<double>(distance) / static_cast<double>(length);
}

class provider_agent_tool_runtime final : public common_agent_tool_runtime {
public:
    explicit provider_agent_tool_runtime(agent_tool_view & tool_view)
        : tool_view(tool_view) {}

    bool is_read_only(const std::string & tool_name) const override {
        return tool_view.is_read_only(tool_name);
    }

    bool is_policy_gated(const std::string & tool_name) const override {
        return tool_view.is_policy_gated(tool_name);
    }

    common_learning_tool_metadata learning_metadata(const std::string & tool_name) const override {
        return tool_view.learning_metadata(tool_name);
    }

    bool describe_tool_dataflow(
            const std::string & tool_name,
            common_plan_tool_dataflow_contract & contract,
            std::string & error) const override {
        return tool_view.describe_tool_dataflow(tool_name, contract, error);
    }

    bool validate_plan(const common_plan_state & plan, std::string & error) const override {
        return tool_view.validate_plan(plan, error);
    }

    bool is_available(const std::string & tool_name) const override {
        return tool_view.exposes_tool(tool_name);
    }

    bool resolve_tool_name(
            const std::string & requested,
            std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear();
        if (tool_view.exposes_tool(requested)) {
            resolved = requested;
            return true;
        }
        struct scored_tool { std::string name; double score = 0.0; };
        std::vector<scored_tool> scored;
        for (const auto & tool : tool_view.chat_tools()) {
            const double score = tool_name_score(requested, tool.name);
            if (score >= 0.55) scored.push_back({tool.name, score});
        }
        std::sort(scored.begin(), scored.end(), [](const auto & left, const auto & right) {
            if (left.score != right.score) return left.score > right.score;
            return left.name < right.name;
        });
        for (const auto & item : scored) candidates.push_back(item.name);
        if (scored.empty() || scored.front().score < 0.80) return false;
        if (scored.size() > 1 && scored.front().score - scored[1].score < 0.12) return false;
        resolved = scored.front().name;
        candidates.clear();
        return true;
    }

    common_agent_tool_repair_context make_repair_context(
            const common_agent_tool_call & call,
            const std::string & validation_error) const override {
        return tool_view.make_repair_context(call.name, call.arguments_json, validation_error);
    }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        return tool_view.validate({"", call.name, call.arguments_json}, error);
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        std::string error;
        const auto result = tool_view.call({"", call.name, call.arguments_json}, error);
        if (result.ok) {
            return common_tool_execution_result::success(
                result.content_json,
                result.content_summary,
                result.resource_refs);
        }
        return common_tool_execution_result::failure(
            result.failure_code.empty() ? "tool.execution_failed" : result.failure_code,
            result.failure_class,
            result.retryable,
            result.safe_summary.empty() ? "The tool failed." : result.safe_summary,
            result.raw_diagnostic);
    }

    bool supports_async(const common_agent_tool_call & call) const override {
        return tool_view.supports_async_call(call.name);
    }

    bool begin_async(
            const common_agent_tool_call & call,
            common_runtime_operation_ref & pending,
            std::string & error) const override {
        if (!tool_view.begin_call_async(
                {"", call.name, call.arguments_json}, pending, error)) {
            return false;
        }
        error.clear();
        return true;
    }

    bool poll_async(
            const common_runtime_operation_ref & pending,
            bool & ready,
            common_tool_execution_result & output,
            std::string & error) const override {
        agent_tool_result result;
        if (!tool_view.poll_call_async(pending, ready, result, error)) {
            return false;
        }
        if (!ready) {
            error.clear();
            return true;
        }
        if (result.ok) {
            output = common_tool_execution_result::success(
                result.content_json,
                result.content_summary,
                result.resource_refs);
        } else {
            output = common_tool_execution_result::failure(
                result.failure_code.empty() ? "tool.execution_failed" : result.failure_code,
                result.failure_class,
                result.retryable,
                result.safe_summary.empty() ? "The tool failed." : result.safe_summary,
                result.raw_diagnostic);
        }
        error.clear();
        return true;
    }

    bool cancel_async(
            const common_runtime_operation_ref & pending,
            std::string & error) const override {
        return tool_view.cancel_call_async(pending, error);
    }

private:
    agent_tool_view & tool_view;
};

class scoped_agent_tool_runtime final : public common_agent_tool_runtime {
public:
    scoped_agent_tool_runtime(
            std::unique_ptr<common_agent_tool_runtime> provider,
            common_agent_execution_envelope envelope,
            common_agent_execution_phase phase)
        : provider(std::move(provider)), envelope(std::move(envelope)), phase(phase) {}

    bool is_read_only(const std::string & tool_name) const override {
        return allowed(tool_name) && provider->is_read_only(tool_name);
    }

    bool is_policy_gated(const std::string & tool_name) const override {
        return allowed(tool_name) && provider->is_policy_gated(tool_name);
    }

    bool describe_tool_dataflow(
            const std::string & tool_name,
            common_plan_tool_dataflow_contract & contract,
            std::string & error) const override {
        if (!require(tool_name, error)) return false;
        return provider->describe_tool_dataflow(tool_name, contract, error);
    }

    bool validate_plan(const common_plan_state & plan, std::string & error) const override {
        if (!provider->validate_plan(plan, error)) return false;
        for (const auto & step : plan.steps) {
            const auto name = step.tool_call ? step.tool_call->name :
                (step.selected_tool ? *step.selected_tool : std::string());
            if (!name.empty() && !allowed(name)) {
                error = "route execution envelope denies tool in plan: " + name;
                return false;
            }
        }
        error.clear();
        return true;
    }

    bool is_available(const std::string & tool_name) const override {
        return allowed(tool_name) && provider->is_available(tool_name);
    }

    bool resolve_tool_name(
            const std::string & requested,
            std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear();
        if (allowed(requested) && provider->is_available(requested)) {
            resolved = requested;
            return true;
        }
        std::vector<std::string> provider_candidates;
        if (!provider->resolve_tool_name(requested, resolved, provider_candidates)) return false;
        if (!allowed(resolved)) return false;
        for (const auto & candidate : provider_candidates) if (allowed(candidate)) {
            candidates.push_back(candidate);
        }
        return true;
    }

    common_agent_tool_repair_context make_repair_context(
            const common_agent_tool_call & call,
            const std::string & validation_error) const override {
        if (!allowed(call.name)) {
            return {call.name, "tool is outside the route execution envelope", {}, {}, {}, call.arguments_json, false, {}};
        }
        auto result = provider->make_repair_context(call, validation_error);
        result.available_tools.erase(std::remove_if(result.available_tools.begin(), result.available_tools.end(),
            [&](const auto & name) { return !allowed(name); }), result.available_tools.end());
        result.candidate_tools.erase(std::remove_if(result.candidate_tools.begin(), result.candidate_tools.end(),
            [&](const auto & name) { return !allowed(name); }), result.candidate_tools.end());
        return result;
    }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        if (!require(call.name, error)) return false;
        return provider->validate(call, error);
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        std::string error;
        if (!require(call.name, error)) {
            return common_tool_execution_result::failure(
                "tool.outside_execution_envelope",
                common_tool_failure_class::policy,
                false,
                "The selected route does not permit this tool.",
                error);
        }
        return provider->execute(call);
    }

    bool supports_async(const common_agent_tool_call & call) const override {
        return allowed(call.name) && provider->supports_async(call);
    }

    bool begin_async(
            const common_agent_tool_call & call,
            common_runtime_operation_ref & pending,
            std::string & error) const override {
        if (!require(call.name, error)) return false;
        return provider->begin_async(call, pending, error);
    }

    bool poll_async(
            const common_runtime_operation_ref & pending,
            bool & ready,
            common_tool_execution_result & output,
            std::string & error) const override {
        return provider->poll_async(pending, ready, output, error);
    }

    bool cancel_async(
            const common_runtime_operation_ref & pending,
            std::string & error) const override {
        return provider->cancel_async(pending, error);
    }

    bool set_execution_phase(common_agent_execution_phase next) const override {
        if (common_agent_execution_phase_policy_for(envelope, next) == nullptr) return false;
        phase = next;
        return true;
    }

    bool has_execution_envelope() const override { return true; }

private:
    bool allowed(const std::string & name) const {
        return common_agent_execution_envelope_allows(envelope, phase, name);
    }

    bool require(const std::string & name, std::string & error) const {
        if (allowed(name)) {
            error.clear();
            return true;
        }
        error = "tool outside route execution envelope: " + name +
            " phase=" + common_agent_execution_phase_name(phase);
        return false;
    }

    std::unique_ptr<common_agent_tool_runtime> provider;
    common_agent_execution_envelope envelope;
    mutable common_agent_execution_phase phase;
};

} // namespace

std::unique_ptr<common_agent_tool_runtime> make_provider_agent_tool_runtime(
        agent_tool_view & tool_view) {
    return std::make_unique<provider_agent_tool_runtime>(tool_view);
}

std::unique_ptr<common_agent_tool_runtime> make_scoped_agent_tool_runtime(
        std::unique_ptr<common_agent_tool_runtime> provider_runtime,
        common_agent_execution_envelope envelope,
        common_agent_execution_phase initial_phase) {
    return std::make_unique<scoped_agent_tool_runtime>(
        std::move(provider_runtime), std::move(envelope), initial_phase);
}
