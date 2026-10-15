#include "tools/agent/tooling/agent-tool-provider.h"
#include "plan/plan-bindings.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

class fake_mcp_tool_client final : public agent_mcp_tool_client {
public:
    std::string last_arguments;
    std::vector<mcp_agent_tool_definition> listed_tools;

    bool list_tools(
            const agent_tool_context &,
            std::vector<mcp_agent_tool_definition> & tools,
            std::string & error) override {
        tools = {
            {
                "github",
                "create_issue",
                "Create a GitHub issue.",
                R"({"type":"object","additionalProperties":false,"required":["title"],"properties":{"title":{"type":"string","minLength":1}}})",
                std::string(),
                false,
                true,
                true,
                false,
                false,
            },
            {
                "github",
                "search_issues",
                "Search GitHub issues.",
                R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1}}})",
                std::string(),
                true,
                false,
                true,
                false,
                false,
            },
            {
                "github",
                "get_issue",
                "Get one issue by its host-provided issue reference.",
                R"({"type":"object","additionalProperties":false,"required":["issue_id"],"properties":{"issue_id":{"type":"string","x-agent-type":"issue_ref"}}})",
                std::string(),
                true,
                false,
                true,
                false,
                false,
            },
        };
        tools[1].result_schema_json = R"({"type":"object","required":["issue_id"],"properties":{"issue_id":{"type":"string","x-agent-type":"issue_ref"},"items":{"type":"array"}}})";
        listed_tools = tools;
        error.clear();
        return true;
    }

    bool call_tool(
            const agent_tool_context &,
            const mcp_agent_tool_definition & tool,
            const std::string & arguments_json,
            mcp_agent_tool_call_result & result,
            std::string & error) override {
        last_arguments = arguments_json;
        if (tool.name == "search_issues") {
            result.ok = true;
            result.structured_content_json = R"({"issue_id":"issue-123","items":[{"title":"stub issue"}]})";
            common_agent_dataset_ref dataset;
            dataset.uri = "dataset://mcp/github/issues";
            dataset.name = "github issues";
            dataset.row_count = 1;
            dataset.column_count = 1;
            dataset.source_resource_uri = "agent-resource://mcp/github/issues";
            dataset.source_representation = "mcp:json";
            if (arguments_json.find("invalid") != std::string::npos) {
                dataset.source_resource_uri.clear();
            }
            result.dataset_refs = {std::move(dataset)};
            error.clear();
            return true;
        }

        if (tool.name == "get_issue") {
            result.ok = true;
            const auto arguments = nlohmann::ordered_json::parse(arguments_json);
            result.structured_content_json = nlohmann::ordered_json{
                {"id", arguments.at("issue_id")},
            }.dump();
            error.clear();
            return true;
        }

        if (tool.name == "create_issue") {
            result.ok = true;
            result.text_content = "created issue #123";
            error.clear();
            return true;
        }

        error = "unexpected MCP tool call: " + tool.name + " with args " + arguments_json;
        return false;
    }
};

} // namespace

int main() {
    std::string error;
    fake_mcp_tool_client client;
    mcp_agent_tool_provider provider("github", client);

    agent_tool_context read_only_context;
    read_only_context.request_id = "mcp-provider-smoke";
    read_only_context.turn_id = "turn-1";
    read_only_context.allow_network = true;

    std::unique_ptr<agent_tool_view> read_only_view = provider.resolve_tools(read_only_context, error);
    if (!read_only_view) {
        std::fprintf(stderr, "MCP provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(read_only_view->chat_tools(), "github_search_issues")) {
        std::fprintf(stderr, "search_issues was not exposed through the MCP tool view\n");
        return 1;
    }
    const auto search_model_tool = std::find_if(
        read_only_view->chat_tools().begin(), read_only_view->chat_tools().end(),
        [](const auto & tool) { return tool.name == "github_search_issues"; });
    if (search_model_tool == read_only_view->chat_tools().end() ||
            search_model_tool->output_schema != R"({"type":"object","required":["issue_id"],"properties":{"issue_id":{"type":"string","x-agent-type":"issue_ref"},"items":{"type":"array"}}})") {
        // The fake MCP client retains the exact outputSchema parsed from the
        // remote server; the provider must expose it independently of the
        // compact model-facing result projection.
        std::fprintf(stderr, "MCP provider dropped the canonical output schema\n");
        return 1;
    }
    if (has_tool(read_only_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "create_issue was exposed despite writes being disabled\n");
        return 1;
    }
    const auto mcp_metadata = read_only_view->learning_metadata("github_search_issues");
    if (mcp_metadata.tool_family != "github" || mcp_metadata.provider_kind != "mcp") {
        std::fprintf(stderr, "MCP learning metadata was not canonical\n");
        return 1;
    }

    const auto search_result = read_only_view->call({
        "call-1",
        "github_search_issues",
        R"({ "query": "resident inference" })",
    }, error);
    if (!search_result.ok || search_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "search_issues did not return the expected result: %s\n", search_result.content_json.c_str());
        return 1;
    }
    if (client.last_arguments != R"({"query":"resident inference"})") {
        std::fprintf(stderr, "MCP provider did not execute normalized search arguments: %s\n", client.last_arguments.c_str());
        return 1;
    }

    common_plan_tool_dataflow_contract search_flow;
    common_plan_tool_dataflow_contract get_issue_flow;
    if (!read_only_view->describe_tool_dataflow("github_search_issues", search_flow, error) ||
            !read_only_view->describe_tool_dataflow("github_get_issue", get_issue_flow, error) ||
            search_flow.outputs.empty() || get_issue_flow.inputs.empty()) {
        std::fprintf(stderr, "MCP outputSchema did not reach provider dataflow contracts: %s\n", error.c_str());
        return 1;
    }
    common_plan_state bound_plan;
    common_plan_step source_step{"lookup", "Lookup", "Search for an issue"};
    source_step.status = common_plan_step_status::completed;
    source_step.tool_call = common_plan_tool_call{"github_search_issues", R"({"query":"resident inference"})"};
    bound_plan.steps.push_back(source_step);
    bound_plan.observations.push_back({
        "tool:lookup:1", "github_search_issues", search_result.content_json, 1.0f, {}, {}, 0});
    common_plan_step bound_step{"fetch", "Fetch", "Read the discovered issue"};
    bound_step.depends_on = {"lookup"};
    bound_step.tool_call = common_plan_tool_call{
        "github_get_issue", R"({"issue_id":{"$from_step":"lookup","$json_pointer":"/issue_id"}})"};
    const auto resolve_flow = [&](const std::string & name,
            common_plan_tool_dataflow_contract & contract,
            std::string & flow_error) {
        return read_only_view->describe_tool_dataflow(name, contract, flow_error);
    };
    std::string bound_arguments;
    if (!common_plan_materialize_tool_arguments(
            bound_plan, bound_step, bound_step.tool_call->arguments_json,
            bound_arguments, error, resolve_flow)) {
        std::fprintf(stderr, "MCP result binding failed: %s\n", error.c_str());
        return 1;
    }
    const auto bound_result = read_only_view->call({"call-bound", "github_get_issue", bound_arguments}, error);
    if (!bound_result.ok || client.last_arguments != R"({"issue_id":"issue-123"})") {
        std::fprintf(stderr, "bound MCP value was not passed to the next executed call: args=%s error=%s\n",
            client.last_arguments.c_str(), error.c_str());
        return 1;
    }
    if (search_result.dataset_refs.size() != 1 ||
            search_result.dataset_refs.front().uri != "dataset://mcp/github/issues" ||
            search_result.content_json.find("dataset_refs") == std::string::npos) {
        std::fprintf(stderr, "MCP dataset reference was not propagated in the compact result\n");
        return 1;
    }
    const auto invalid_dataset_result = read_only_view->call({
        "call-invalid-dataset",
        "github_search_issues",
        R"({"query":"invalid"})",
    }, error);
    if (invalid_dataset_result.ok ||
            invalid_dataset_result.failure_code != "tool_dataset_ref_invalid" ||
            invalid_dataset_result.failure_class != common_tool_failure_class::validation) {
        std::fprintf(stderr, "invalid MCP dataset reference was not rejected\n");
        return 1;
    }

    agent_tool_context limited_context = read_only_context;
    limited_context.turn_id = "turn-limited-result";
    limited_context.default_max_result_bytes = 8;
    std::unique_ptr<agent_tool_view> limited_view = provider.resolve_tools(limited_context, error);
    if (!limited_view) {
        std::fprintf(stderr, "limited MCP provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    const auto limited_result = limited_view->call({
        "call-limited",
        "github_search_issues",
        R"({"query":"x"})",
    }, error);
    if (limited_result.ok || limited_result.failure_code != "tool_result_too_large") {
        std::fprintf(stderr, "MCP result-size limit was not enforced\n");
        return 1;
    }

    agent_tool_context filtered_context;
    filtered_context.request_id = "mcp-provider-smoke";
    filtered_context.turn_id = "turn-filtered";
    filtered_context.allow_network = true;
    filtered_context.allowed_exposed_tool_names = {"github_search_issues"};

    std::unique_ptr<agent_tool_view> filtered_view = provider.resolve_tools(filtered_context, error);
    if (!filtered_view) {
        std::fprintf(stderr, "filtered MCP provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(filtered_view->chat_tools(), "github_search_issues")) {
        std::fprintf(stderr, "filtered MCP provider dropped the allowed exposed tool\n");
        return 1;
    }
    if (has_tool(filtered_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "filtered MCP provider did not enforce exposed-name filtering\n");
        return 1;
    }

    agent_tool_context write_context;
    write_context.request_id = "mcp-provider-smoke";
    write_context.turn_id = "turn-2";
    write_context.allow_network = true;
    write_context.allow_policy_gated_writes = true;

    std::unique_ptr<agent_tool_view> write_view = provider.resolve_tools(write_context, error);
    if (!write_view) {
        std::fprintf(stderr, "MCP write-capable provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(write_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "create_issue was not exposed when policy-gated writes were enabled\n");
        return 1;
    }

    const auto create_result = write_view->call({
        "call-2",
        "github_create_issue",
        R"({"title":"Add MCP provider smoke"})",
    }, error);
    if (!create_result.ok || create_result.content_json.find("created issue #123") == std::string::npos) {
        std::fprintf(stderr, "create_issue did not return the expected result: %s\n", create_result.content_json.c_str());
        return 1;
    }

    const auto invalid_result = write_view->call({
        "call-3",
        "github_search_issues",
        R"("not-an-object")",
    }, error);
    if (invalid_result.ok || invalid_result.failure_class != common_tool_failure_class::validation) {
        std::fprintf(stderr, "invalid MCP arguments were not rejected correctly\n");
        return 1;
    }

    const auto missing_required_result = write_view->call({
        "call-4",
        "github_search_issues",
        R"({})",
    }, error);
    if (missing_required_result.ok || missing_required_result.failure_class != common_tool_failure_class::validation) {
        std::fprintf(stderr, "MCP required-field validation was not enforced\n");
        return 1;
    }

    const auto wrong_type_result = write_view->call({
        "call-5",
        "github_search_issues",
        R"({"query":42})",
    }, error);
    if (wrong_type_result.ok || wrong_type_result.failure_class != common_tool_failure_class::validation) {
        std::fprintf(stderr, "MCP property-type validation was not enforced\n");
        return 1;
    }

    std::printf("mcp_exposed_tools=%zu\n", write_view->chat_tools().size());
    std::printf("mcp_search_result=%s\n", search_result.content_json.c_str());
    std::printf("mcp_create_result=%s\n", create_result.content_json.c_str());
    return 0;
}
