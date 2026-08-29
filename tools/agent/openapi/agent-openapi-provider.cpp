#include "agent-openapi-provider.h"

#include <algorithm>
#include <nlohmann/json.hpp>

class agent_openapi_tool_provider::client : public agent_mcp_tool_client {
public:
    client(const agent_openapi_catalog & catalog, agent_openapi_executor executor,
            agent_openapi_result_materializer materializer)
        : catalog(catalog), executor(std::move(executor)), materializer(std::move(materializer)) {}

    bool list_tools(const agent_tool_context &, std::vector<mcp_agent_tool_definition> & tools,
                    std::string &) override {
        tools.clear();
        for (const auto & operation : catalog.operations) {
            mcp_agent_tool_definition definition;
            definition.provider_id = catalog.provider_id;
            definition.name = operation.operation_id;
            definition.description = operation.description.empty() ? operation.summary : operation.description;
            for (const auto & relation : catalog.relations) {
                if (relation.collection_operation_id == operation.operation_id) {
                    definition.description += " workflow: " + catalog.prefix + "." +
                        relation.collection_operation_id + " -> choose/bind " +
                        relation.item_parameter + " -> " + catalog.prefix + "." +
                        relation.item_operation_id + ".";
                } else if (relation.item_operation_id == operation.operation_id) {
                    definition.description += " workflow input: " + catalog.prefix + "." +
                        relation.collection_operation_id + " produces " +
                        relation.item_parameter + ".";
                }
            }
            definition.input_schema_json = operation.input_schema_json;
            definition.result_schema_json = operation.result_schema_json;
            definition.read_only = operation.read_only;
            definition.requires_confirmation = operation.requires_confirmation;
            definition.uses_network = true;
            tools.push_back(std::move(definition));
        }
        return true;
    }

    bool call_tool(const agent_tool_context & context, const mcp_agent_tool_definition & tool,
                   const std::string & arguments_json, mcp_agent_tool_call_result & result,
                   std::string & error) override {
        const auto it = std::find_if(catalog.operations.begin(), catalog.operations.end(),
            [&](const agent_openapi_operation & operation) { return operation.operation_id == tool.name; });
        if (it == catalog.operations.end()) {
            error = "OpenAPI operation is not in the filtered catalog";
            return false;
        }
        agent_openapi_execution_result execution;
        if (!executor || !executor(context, *it, arguments_json, execution, error)) return false;
        if (materializer && !materializer(context, *it, arguments_json, execution, error)) return false;
        result.ok = execution.ok;
        result.structured_content_json = execution.structured_content_json;
        result.text_content = execution.text_content;
        result.resource_refs = execution.resource_refs;
        result.dataset_refs = execution.dataset_refs;
        result.failure_code = execution.failure_code;
        result.failure_class = execution.failure_class;
        result.retryable = execution.retryable;
        result.safe_summary = execution.safe_summary;
        result.raw_diagnostic = execution.raw_diagnostic;
        return true;
    }

private:
    const agent_openapi_catalog & catalog;
    agent_openapi_executor executor;
    agent_openapi_result_materializer materializer;
};

agent_openapi_tool_provider::agent_openapi_tool_provider(
        agent_openapi_catalog catalog, agent_openapi_executor executor,
        agent_openapi_result_materializer materializer)
    : catalog(std::move(catalog))
    , executor(std::move(executor))
    , materializer(std::move(materializer))
    , client_impl(std::make_unique<client>(this->catalog, this->executor, this->materializer))
    , delegate(std::make_unique<mcp_agent_tool_provider>(
        this->catalog.provider_id,
        *client_impl,
        this->catalog.prefix + ".",
        [this](const common_plan_state & plan, std::string & error) {
            for (const auto & relation : this->catalog.relations) {
                const std::string item_name = this->catalog.prefix + "." + relation.item_operation_id;
                const std::string collection_name = this->catalog.prefix + "." + relation.collection_operation_id;
                for (size_t index = 0; index < plan.steps.size(); ++index) {
                    const auto & step = plan.steps[index];
                    if (!step.tool_call || step.tool_call->name != item_name ||
                            step.tool_call->arguments_json.find("\"$") == std::string::npos) continue;
                    bool producer_before_item = false;
                    for (size_t previous = 0; previous < index; ++previous) {
                        if (plan.steps[previous].tool_call &&
                                plan.steps[previous].tool_call->name == collection_name) {
                            producer_before_item = true;
                            break;
                        }
                    }
                    if (!producer_before_item) {
                        error = "workflow.required_producer_missing: " + item_name +
                            " requires " + collection_name + " before dynamic " +
                            relation.item_parameter + " binding";
                        return false;
                    }
                }
            }
            error.clear();
            return true;
        },
        "openapi")) {}

agent_openapi_tool_provider::~agent_openapi_tool_provider() = default;

std::unique_ptr<agent_tool_view> agent_openapi_tool_provider::resolve_tools(
        const agent_tool_context & context, std::string & error) {
    return delegate->resolve_tools(context, error);
}
