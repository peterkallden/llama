#include "agent-mcp-server-tool-registry.h"

#include "agent/dataset-contracts.h"
#include "agent/tooling/contracts/schema-contract.h"

#include <nlohmann/json.hpp>

#include <exception>

namespace {

const agent_mcp_server_tool * find_tool(
        const std::vector<agent_mcp_server_tool> & tools,
        const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return &tool;
        }
    }
    return nullptr;
}

} // namespace

agent_mcp_json agent_mcp_canonical_output_schema(const std::string & result_schema_json) {
    // MCP outputSchema validates structuredContent itself. The transport
    // envelope (content/isError/structuredContent) is outside that object,
    // and native tool output is the canonical result object, not a second
    // synthetic {ok,result,...} wrapper.
    auto schema = result_schema_json.empty()
        ? agent_mcp_json::object()
        : agent_mcp_json::parse(result_schema_json, nullptr, false);
    return schema.is_object() ? schema : agent_mcp_json::object();
}

agent_mcp_json agent_mcp_first_party_app_metadata(const std::string & tool_name) {
    if (tool_name == "dataset.schema" || tool_name == "dataset.sample" || tool_name == "dataset.inspect") {
        return {{"resourceUri", "ui://llama-agent/apps/data-explorer"}, {"visibility", {"model", "app"}}};
    }
    if (tool_name == "plan.get") {
        return {{"resourceUri", "ui://llama-agent/apps/workflow-inspector"}, {"visibility", {"model", "app"}}};
    }
    return agent_mcp_json::object();
}

agent_mcp_json agent_mcp_first_party_app_resources() {
    const auto make_resource = [](const std::string & uri, const std::string & name, const std::string & description) {
        const agent_mcp_json ui_metadata = {
            {"csp", {
                {"connectDomains", agent_mcp_json::array()},
                {"resourceDomains", agent_mcp_json::array()},
                {"frameDomains", agent_mcp_json::array()},
            }},
        };
        return agent_mcp_json{
            {"uri", uri}, {"name", name}, {"description", description},
            {"mimeType", "text/html;profile=mcp-app"},
            {"_meta", {{"ui", ui_metadata}}},
        };
    };
    return {{"resources", agent_mcp_json::array({
        make_resource("ui://llama-agent/apps/data-explorer", "Data Explorer", "Read-only dataset schema and sample viewer."),
        make_resource("ui://llama-agent/apps/workflow-inspector", "Workflow Inspector", "Read-only view of the active plan."),
    })}};
}

bool agent_mcp_read_first_party_app_resource(
        const std::string & uri,
        agent_mcp_json & content,
        std::string & error) {
    const std::string data_app = R"HTML(<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:14px system-ui;background:#101827;color:#e5edf7;padding:18px}input,button{padding:8px;margin:4px;background:#1b2b40;color:inherit;border:1px solid #55708e;border-radius:6px}pre{white-space:pre-wrap;background:#09111d;padding:10px}</style></head><body><h2>Data Explorer</h2><p>Read a host-authorized dataset schema or bounded sample.</p><input id="dataset" placeholder="dataset://local/name"><button onclick="run('dataset.schema')">Schema</button><button onclick="run('dataset.sample')">Sample</button><pre id="out">Enter a dataset reference.</pre><script>let seq=1;function send(method,params){const id=seq++;parent.postMessage({jsonrpc:'2.0',id,method,params},'*');return id}addEventListener('message',e=>{const m=e.data;if(m?.id===1&&m.result?.protocolVersion){send('ui/notifications/initialized',{});return}if(m?.result?.structuredContent)document.querySelector('#out').textContent=JSON.stringify(m.result.structuredContent,null,2);if(m?.error)document.querySelector('#out').textContent=m.error.message});send('ui/initialize',{protocolVersion:'2026-01-26',appInfo:{name:'llama-agent-data-explorer',version:'0.1'},appCapabilities:{tools:{}}});function run(name){const dataset=document.querySelector('#dataset').value;send('tools/call',{name,arguments:name==='dataset.sample'?{dataset,rows:8}:{dataset}})}</script></body></html>)HTML";
    const std::string workflow_app = R"HTML(<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:14px system-ui;background:#101827;color:#e5edf7;padding:18px}input,button{padding:8px;margin:4px;background:#1b2b40;color:inherit;border:1px solid #55708e;border-radius:6px}pre{white-space:pre-wrap;background:#09111d;padding:10px}</style></head><body><h2>Workflow Inspector</h2><p>Load a host-owned plan by its plan ID. Missing verification fields are not treated as success.</p><input id="plan" placeholder="plan ID"><button onclick="loadPlan()">Load plan</button><pre id="out">Enter a plan ID.</pre><script>let seq=1;function send(method,params){const id=seq++;parent.postMessage({jsonrpc:'2.0',id,method,params},'*');return id}addEventListener('message',e=>{const m=e.data;if(m?.id===1&&m.result?.protocolVersion){send('ui/notifications/initialized',{});return}if(m?.result?.structuredContent)document.querySelector('#out').textContent=JSON.stringify(m.result.structuredContent,null,2);if(m?.error)document.querySelector('#out').textContent=m.error.message});send('ui/initialize',{protocolVersion:'2026-01-26',appInfo:{name:'llama-agent-workflow-inspector',version:'0.1'},appCapabilities:{tools:{}}});function loadPlan(){send('tools/call',{name:'plan.get',arguments:{plan_id:document.querySelector('#plan').value}})}</script></body></html>)HTML";
    const char * text = nullptr;
    if (uri == "ui://llama-agent/apps/data-explorer") text = data_app.c_str();
    else if (uri == "ui://llama-agent/apps/workflow-inspector") text = workflow_app.c_str();
    else {
        error = "unknown first-party MCP App resource URI";
        return false;
    }
    const agent_mcp_json ui_metadata = {
        {"csp", {
            {"connectDomains", agent_mcp_json::array()},
            {"resourceDomains", agent_mcp_json::array()},
            {"frameDomains", agent_mcp_json::array()},
        }},
    };
    const agent_mcp_json resource_content = {
        {"uri", uri},
        {"name", uri.find("data-explorer") != std::string::npos ? "Data Explorer" : "Workflow Inspector"},
        {"mimeType", "text/html;profile=mcp-app"},
        {"text", text},
        {"_meta", {{"ui", ui_metadata}}},
    };
    content = {{"contents", agent_mcp_json::array({resource_content})}};
    error.clear();
    return true;
}

bool agent_mcp_server_tool_registry::register_tool(
        agent_mcp_server_tool tool,
        std::string & error) {
    if (tool.name.empty()) {
        error = "MCP server tool name is required";
        return false;
    }
    if (!tool.handler) {
        error = "MCP server tool handler is required";
        return false;
    }
    if (find_tool(tools_, tool.name) != nullptr) {
        error = "duplicate MCP server tool: " + tool.name;
        return false;
    }
    tools_.push_back(std::move(tool));
    error.clear();
    return true;
}

std::vector<agent_mcp_server_tool> agent_mcp_server_tool_registry::list_tools() const {
    return tools_;
}

bool agent_mcp_server_tool_registry::contains_tool(const std::string & name) const {
    return find_tool(tools_, name) != nullptr;
}

bool agent_mcp_server_tool_registry::validate_tool_arguments(
        const std::string & name,
        const agent_mcp_json & arguments,
        std::string & error) const {
    const auto * tool = find_tool(tools_, name);
    if (tool == nullptr) {
        error = "unknown MCP server tool: " + name;
        return false;
    }
    if (!arguments.is_object()) {
        error = "MCP tool arguments must be a JSON object";
        return false;
    }
    auto normalized_input = arguments;
    if (!normalize_common_agent_dataset_tool_arguments(name, normalized_input, error)) {
        return false;
    }
    std::string normalized_arguments;
    if (!common_schema_normalize_and_validate_object(
            normalized_input.dump(),
            tool->input_schema_json,
            normalized_arguments,
            error)) {
        return false;
    }
    error.clear();
    return true;
}

bool agent_mcp_server_tool_registry::call_tool(
        const std::string & name,
        const agent_mcp_json & arguments,
        agent_mcp_server_tool_result & result,
        std::string & error) const {
    result = {};
    const auto * tool = find_tool(tools_, name);
    if (tool == nullptr) {
        error = "unknown MCP server tool: " + name;
        return false;
    }
    if (!validate_tool_arguments(name, arguments, error)) {
        return false;
    }

    auto normalized_input = arguments;
    if (!normalize_common_agent_dataset_tool_arguments(name, normalized_input, error)) {
        return false;
    }
    std::string normalized_arguments;
    if (!common_schema_normalize_and_validate_object(
            normalized_input.dump(), tool->input_schema_json, normalized_arguments, error)) {
        return false;
    }

    const auto normalized = agent_mcp_json::parse(normalized_arguments, nullptr, false);
    if (normalized.is_discarded() || !normalized.is_object()) {
        error = "MCP tool schema validator returned invalid normalized arguments";
        return false;
    }
    try {
        if (!tool->handler(normalized, result, error)) {
            result.ok = false;
            if (result.failure_code.empty()) {
                result.failure_code = "tool.execution_failed";
            }
            if (result.failure_class.empty()) result.failure_class = "execution";
            if (result.safe_summary.empty()) result.safe_summary = "The MCP server tool failed.";
            if (result.content.empty()) {
                result.content = agent_mcp_json::array({
                    {{"type", "text"}, {"text", result.safe_summary}},
                });
            }
            if (error.empty()) error = result.raw_diagnostic.empty()
                ? result.safe_summary : result.raw_diagnostic;
            return false;
        }
        return true;
    } catch (const std::exception & exception) {
        result.ok = false;
        result.failure_code = "tool.executor_exception";
        result.failure_class = "execution";
        result.retryable = false;
        result.safe_summary = "The tool failed unexpectedly.";
        result.raw_diagnostic = exception.what();
        result.content = agent_mcp_json::array({
            {{"type", "text"}, {"text", result.safe_summary}},
        });
        error = result.raw_diagnostic;
        return false;
    } catch (...) {
        result.ok = false;
        result.failure_code = "tool.executor_exception";
        result.failure_class = "execution";
        result.retryable = false;
        result.safe_summary = "The tool failed unexpectedly.";
        result.raw_diagnostic = "MCP server tool handler threw a non-standard exception";
        result.content = agent_mcp_json::array({
            {{"type", "text"}, {"text", result.safe_summary}},
        });
        error = result.raw_diagnostic;
        return false;
    }
}

agent_mcp_json agent_mcp_render_tools_list_result(
        const agent_mcp_server_tool_registry & registry) {
    agent_mcp_json tools = agent_mcp_json::array();
    for (const auto & tool : registry.list_tools()) {
        agent_mcp_json schema = agent_mcp_json::parse(tool.input_schema_json, nullptr, false);
        if (schema.is_discarded() || !schema.is_object()) {
            schema = agent_mcp_json::object();
        }
        tools.push_back({
            {"name", tool.name},
            {"description", tool.description},
            {"inputSchema", std::move(schema)},
            {"annotations", {
                {"readOnlyHint", tool.read_only},
            }},
            {"hostPolicy", {
                {"requiresConfirmation", tool.requires_confirmation},
                {"usesNetwork", tool.uses_network},
                {"writesMemory", tool.writes_memory},
                {"writesPlan", tool.writes_plan},
            }},
        });
        if (!tool.output_schema_json.empty()) {
            const auto output_schema = agent_mcp_json::parse(
                tool.output_schema_json, nullptr, false);
            if (output_schema.is_object()) {
                tools.back()["outputSchema"] = agent_mcp_canonical_output_schema(
                    tool.output_schema_json);
            }
        }
        if (tool.ui_metadata.is_object() && !tool.ui_metadata.empty()) {
            tools.back()["_meta"] = {{"ui", tool.ui_metadata}};
        }
    }
    return {{"tools", std::move(tools)}};
}

bool agent_mcp_register_native_tool_view(
        const std::vector<common_chat_tool> & tools,
        const std::function<bool(const std::string &)> & is_read_only,
        const std::function<bool(const std::string &)> & is_policy_gated,
        agent_mcp_server_tool_registry & registry,
        std::string & error) {
    for (const auto & tool : tools) {
        if (tool.name.empty()) {
            error = "native MCP tool view contains an unnamed tool";
            return false;
        }
        agent_mcp_server_tool server_tool{
                tool.name,
                tool.description,
                tool.parameters,
                is_read_only ? is_read_only(tool.name) : true,
                is_policy_gated ? is_policy_gated(tool.name) : false,
                false,
                false,
                false,
                [](const agent_mcp_json &, agent_mcp_server_tool_result &, std::string & handler_error) {
                    handler_error = "native MCP tool requires the host dispatcher executor";
                    return false;
                },
            };
        server_tool.output_schema_json = tool.output_schema;
        if (!tool.mcp_ui_metadata.empty()) {
            const auto ui = agent_mcp_json::parse(tool.mcp_ui_metadata, nullptr, false);
            if (ui.is_object()) server_tool.ui_metadata = ui;
        }
        if (server_tool.ui_metadata.empty()) {
            server_tool.ui_metadata = agent_mcp_first_party_app_metadata(tool.name);
        }
        if (!registry.register_tool(std::move(server_tool), error)) {
            return false;
        }
    }
    error.clear();
    return true;
}

agent_mcp_json agent_mcp_render_tool_call_result(
        const agent_mcp_server_tool_result & result) {
    agent_mcp_json rendered = {
        {"content", result.content.is_array() ? result.content : agent_mcp_json::array()},
        {"isError", !result.ok},
    };
    if (!result.structured_content.is_null() && !result.structured_content.empty()) {
        rendered["structuredContent"] = result.structured_content;
    }
    if (!result.ok) {
        rendered["errorInfo"] = {
            {"code", result.failure_code.empty() ? "mcp.tool_error" : result.failure_code},
            {"class", result.failure_class.empty() ? "execution" : result.failure_class},
            {"retryable", result.retryable},
            {"safeSummary", result.safe_summary.empty() ? "The MCP server tool failed." : result.safe_summary},
        };
    }
    return rendered;
}

void agent_mcp_set_tool_call_result(
        agent_mcp_server_tool_result & result,
        bool ok,
        agent_mcp_json structured_content,
        std::string safe_summary,
        std::string failure_code,
        std::string failure_class,
        bool retryable,
        std::string raw_diagnostic,
        agent_mcp_json content) {
    result.ok = ok;
    result.structured_content = structured_content.is_null() || structured_content.is_discarded()
        ? agent_mcp_json::object() : std::move(structured_content);
    result.safe_summary = std::move(safe_summary);
    result.failure_code = std::move(failure_code);
    result.failure_class = std::move(failure_class);
    result.retryable = retryable;
    result.raw_diagnostic = std::move(raw_diagnostic);
    if (!ok) {
        if (result.failure_code.empty()) result.failure_code = "tool.execution_failed";
        if (result.failure_class.empty()) result.failure_class = "execution";
        if (result.safe_summary.empty()) result.safe_summary = "The MCP server tool failed.";
    }
    if (content.is_array() && !content.empty()) {
        result.content = std::move(content);
    } else if (!result.safe_summary.empty()) {
        result.content = agent_mcp_json::array({
            {{"type", "text"}, {"text", result.safe_summary}},
        });
    } else {
        result.content = agent_mcp_json::array();
    }
}
