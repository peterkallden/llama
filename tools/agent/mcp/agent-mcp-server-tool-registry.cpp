#include "agent-mcp-server-tool-registry.h"

#include "agent/dataset-contracts.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/contracts/tool-result-schema.h"

#include <nlohmann/json.hpp>

#include <exception>

#ifdef LLAMA_AGENT_INVESTIGATION_WORKSPACE_RESOURCE
#include "agent-investigation-workspace-html.h"
#endif
#ifdef LLAMA_AGENT_DOCUMENT_EDITOR_RESOURCE
#include "agent-document-editor-html.h"
#endif

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
    // envelope (content/isError/structuredContent) is outside that object.
    // Native registration below derives the common success payload schema;
    // directly registered handlers supply their own structuredContent schema.
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
#ifdef LLAMA_AGENT_INVESTIGATION_WORKSPACE_RESOURCE
    if (tool_name == "visualization.snapshot") {
        return {{"resourceUri", "ui://llama-agent/apps/investigation-workspace"}, {"visibility", {"app"}}};
    }
    if (tool_name.rfind("investigation.", 0) == 0 ||
            tool_name.rfind("visualization.", 0) == 0) {
        return {{"resourceUri", "ui://llama-agent/apps/investigation-workspace"}, {"visibility", {"model", "app"}}};
    }
#endif
    if (tool_name == "document.create" || tool_name == "document.open" || tool_name == "document.import" || tool_name == "document.save" ||
            tool_name == "document.propose_patch" || tool_name == "document.apply_patch") {
#ifdef LLAMA_AGENT_DOCUMENT_EDITOR_RESOURCE
        return {{"resourceUri", "ui://llama-agent/apps/document-editor"}, {"visibility", {"model", "app"}}};
#else
        return agent_mcp_json::object();
#endif
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
    auto resources=agent_mcp_json::array({
        make_resource("ui://llama-agent/apps/data-explorer", "Data Explorer", "Read-only dataset schema and sample viewer."),
        make_resource("ui://llama-agent/apps/workflow-inspector", "Workflow Inspector", "Read-only view of the active plan."),
    });
#ifdef LLAMA_AGENT_INVESTIGATION_WORKSPACE_RESOURCE
    resources.push_back(make_resource("ui://llama-agent/apps/investigation-workspace",
        "Investigation Workspace", "Explore a scoped case graph, timeline and source evidence."));
#endif
#ifdef LLAMA_AGENT_DOCUMENT_EDITOR_RESOURCE
    resources.push_back(make_resource("ui://llama-agent/apps/document-editor", "Document Editor", "Edit a saved report revision and request an agent review."));
#endif
    return {{"resources", std::move(resources)}};
}

bool agent_mcp_read_first_party_app_resource(
        const std::string & uri,
        agent_mcp_json & content,
        std::string & error) {
    const std::string data_app = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>
body{font:14px system-ui;background:#101827;color:#e5edf7;padding:18px}
input,select,button{padding:8px;margin:4px;background:#1b2b40;color:inherit;border:1px solid #55708e;border-radius:6px}
button{cursor:pointer}button:disabled{opacity:.5}input#dataset{width:min(520px,85%)}
.scroll{overflow:auto;max-height:480px;margin-top:14px}table{border-collapse:collapse;width:100%;text-align:left}
td,th{padding:8px 12px;border-bottom:1px solid #30445f;white-space:pre-wrap;overflow-wrap:anywhere;vertical-align:top}
th{position:sticky;top:0;background:#1b2b40}th button{border:0;margin:0;padding:4px;font:inherit;font-weight:bold}
tr:nth-child(even){background:#152238}pre{white-space:pre-wrap;background:#09111d;padding:10px}#out{white-space:pre-wrap}
</style></head><body><h2>Data Explorer</h2><p>Browse a dataset, uploaded resource or repository file.</p>
<label>Source type <select id="source"><option value="dataset">Dataset reference</option><option value="resource">Resource reference</option><option value="path">Repository-relative path</option></select></label>
<label>Source <input id="dataset" placeholder="dataset://local/name"></label>
<div><button id="inspect">Inspect</button><button id="schema">Schema</button><label>Sample rows <input id="rows" type="number" value="8" min="1" max="100"></label><button id="sample">Sample</button></div>
<p id="out" role="status">Enter a source.</p><button id="raw-toggle" hidden aria-pressed="false">Show raw JSON</button>
<div id="table" class="scroll"></div><pre id="raw" hidden></pre><script>
let seq=1,generation=0,current=null,columns=[],tableRows=[],sortColumn='',descending=false,rawVisible=false;
const pending=new Map(),q=s=>document.querySelector(s),collator=new Intl.Collator(undefined,{numeric:true,sensitivity:'base'});
function send(method,params){const id=seq++;parent.postMessage({jsonrpc:'2.0',id,method,params},'*');return id}
function cell(value){return value==null?'—':typeof value==='object'?JSON.stringify(value):String(value)}
function numericValue(value){if(typeof value==='number'&&Number.isFinite(value))return value;if(typeof value==='string'&&/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())){const parsed=Number(value);if(Number.isFinite(parsed))return parsed}return undefined}
function setBusy(value){for(const id of ['inspect','schema','sample'])q('#'+id).disabled=value}
function reset(){generation++;pending.clear();current=null;columns=[];tableRows=[];sortColumn='';descending=false;q('#table').replaceChildren();q('#raw').textContent='';q('#raw').hidden=true;q('#raw-toggle').hidden=true;q('#out').textContent='Enter a source.';setBusy(false)}
function sortBy(column){descending=sortColumn===column?!descending:false;sortColumn=column;drawTable()}
function drawTable(){
  const table=document.createElement('table'),head=document.createElement('thead'),header=document.createElement('tr');
  for(const column of columns){const th=document.createElement('th'),button=document.createElement('button');th.setAttribute('scope','col');th.setAttribute('aria-sort',sortColumn===column?(descending?'descending':'ascending'):'none');button.textContent=column+(sortColumn===column?(descending?' ↓':' ↑'):'');button.addEventListener('click',()=>sortBy(column));th.appendChild(button);header.appendChild(th)}
  head.appendChild(header);table.appendChild(head);const body=document.createElement('tbody');
  const ordered=tableRows.map((row,index)=>({row,index}));
  if(sortColumn)ordered.sort((a,b)=>{const left=a.row[sortColumn],right=b.row[sortColumn];if(left==null||right==null)return left==null&&right==null?a.index-b.index:left==null?1:-1;const ln=numericValue(left),rn=numericValue(right);const value=ln!==undefined&&rn!==undefined?ln-rn:collator.compare(cell(left),cell(right));return(descending?-value:value)||a.index-b.index});
  for(const {row} of ordered){const tr=document.createElement('tr');for(const column of columns){const td=document.createElement('td');td.textContent=cell(row[column]);tr.appendChild(td)}body.appendChild(tr)}
  table.appendChild(body);q('#table').replaceChildren(table);q('#table').hidden=rawVisible;q('#raw').hidden=!rawVisible;
}
function renderResult(response){
  if(!response)return;
  let value=response.structuredContent??response.structured_content;
  if(response.isError||value?.ok===false){reset();q('#out').textContent=value?.error?.safe_summary||(response.content||[]).filter(c=>c.type==='text').map(c=>c.text).join('\n')||'Tool execution failed.';return}
  if(value?.ok===true&&value.result&&typeof value.result==='object')value=value.result;
  if(!value||typeof value!=='object'||Array.isArray(value)){reset();q('#out').textContent=(response.content||[]).filter(c=>c.type==='text').map(c=>c.text).join('\n')||'No structured result returned.';return}
  current=value;sortColumn='';descending=false;
  if(Array.isArray(value.rows)){
    columns=Array.isArray(value.columns)?value.columns.map(c=>typeof c==='string'?c:c?.name).filter(Boolean):[];
    tableRows=value.rows.map(row=>Array.isArray(row)?Object.fromEntries(row.map((v,i)=>[columns[i]??String(i+1),v])):row&&typeof row==='object'?row:{value:row});
    for(const row of tableRows)for(const key of Object.keys(row))if(!columns.includes(key))columns.push(key);
  }else if(Array.isArray(value.columns)){tableRows=value.columns.map(c=>typeof c==='string'?{name:c}:c);columns=[...new Set(tableRows.flatMap(row=>Object.keys(row??{})))];}
  else{columns=['Field','Value'];tableRows=Object.entries(value).map(([Field,Value])=>({Field,Value}));}
  q('#raw').textContent=JSON.stringify(value,null,2);q('#raw-toggle').hidden=false;
  q('#out').textContent=tableRows.length+' displayed rows. Click a column heading to sort the displayed rows.'+(value.scan_truncated||value.result_truncated?' This is a bounded sample; more rows exist outside this view.':'');
  drawTable();
}
function run(name){
  const source=q('#source').value,value=q('#dataset').value.trim();
  if(!value){reset();q('#out').textContent='Enter a source first.';return}
  generation++;pending.clear();setBusy(true);q('#out').textContent='Loading…';q('#table').replaceChildren();q('#raw').hidden=true;q('#raw-toggle').hidden=true;
  const args={[source]:value};if(name==='dataset.sample'){const count=Number(q('#rows').value);args.rows=Number.isFinite(count)?Math.max(1,Math.min(100,Math.trunc(count))):8}
  pending.set(send('tools/call',{name,arguments:args}),generation);
}
q('#source').addEventListener('change',()=>{reset();q('#dataset').placeholder=q('#source').value==='path'?'datasets/example.csv':q('#source').value==='resource'?'agent-resource://resource/…':'dataset://local/name'});
q('#dataset').addEventListener('input',reset);
for(const [id,name] of Object.entries({inspect:'dataset.inspect',schema:'dataset.schema',sample:'dataset.sample'}))q('#'+id).addEventListener('click',()=>run(name));
q('#raw-toggle').addEventListener('click',()=>{rawVisible=!rawVisible;q('#raw-toggle').textContent=rawVisible?'Show table':'Show raw JSON';q('#raw-toggle').setAttribute('aria-pressed',String(rawVisible));q('#table').hidden=rawVisible;q('#raw').hidden=!rawVisible});
addEventListener('message',event=>{
  if(event.source!==parent)return;const message=event.data;if(message?.jsonrpc!=='2.0')return;
  if(message.id===1&&message.result?.protocolVersion){parent.postMessage({jsonrpc:'2.0',method:'ui/notifications/initialized',params:{}},'*');return}
  if(message.method==='ui/notifications/tool-input'){reset();const args=message.params?.arguments;for(const source of ['dataset','resource','path'])if(typeof args?.[source]==='string'){q('#source').value=source;q('#dataset').value=args[source];break}return}
  if(message.method==='ui/notifications/tool-result'){renderResult(message.params);return}
  if(pending.get(message.id)!==generation)return;pending.delete(message.id);setBusy(false);
  if(message.error){reset();q('#out').textContent=message.error.message;return}if(message.result)renderResult(message.result);
});
send('ui/initialize',{protocolVersion:'2026-01-26',appInfo:{name:'llama-agent-data-explorer',version:'0.2'},appCapabilities:{tools:{}}});
</script></body></html>)HTML";
    const std::string workflow_app = R"HTML(<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:14px system-ui;background:#101827;color:#e5edf7;padding:18px}input,button{padding:8px;margin:4px;background:#1b2b40;color:inherit;border:1px solid #55708e;border-radius:6px}pre{white-space:pre-wrap;background:#09111d;padding:10px}</style></head><body><h2>Workflow Inspector</h2><p>Load a host-owned plan by its plan ID. Missing verification fields are not treated as success.</p><input id="plan" placeholder="plan ID"><button onclick="loadPlan()">Load plan</button><pre id="out">Enter a plan ID.</pre><script>let seq=1;function send(method,params){const id=seq++;parent.postMessage({jsonrpc:'2.0',id,method,params},'*');return id}function renderResult(r){if(!r)return;const out=document.querySelector('#out');if(r.isError)out.textContent=(r.content||[]).filter(c=>c.type==='text').map(c=>c.text).join('\n')||'Tool execution failed.';else if(r.structuredContent!==undefined)out.textContent=JSON.stringify(r.structuredContent,null,2);else if(Array.isArray(r.content))out.textContent=r.content.filter(c=>c.type==='text').map(c=>c.text).join('\n')}addEventListener('message',e=>{if(e.source!==parent)return;const m=e.data;if(m?.jsonrpc!=='2.0')return;if(m.id===1&&m.result?.protocolVersion){send('ui/notifications/initialized',{});return}if(m.method==='ui/notifications/tool-input'){const value=m.params?.arguments?.plan_id;if(typeof value==='string')document.querySelector('#plan').value=value;return}if(m.method==='ui/notifications/tool-result'){renderResult(m.params);return}if(m.result)renderResult(m.result);if(m.error)document.querySelector('#out').textContent=m.error.message});send('ui/initialize',{protocolVersion:'2026-01-26',appInfo:{name:'llama-agent-workflow-inspector',version:'0.1'},appCapabilities:{tools:{}}});function loadPlan(){send('tools/call',{name:'plan.get',arguments:{plan_id:document.querySelector('#plan').value}})}</script></body></html>)HTML";
    const char * text = nullptr;
    if (uri == "ui://llama-agent/apps/data-explorer") text = data_app.c_str();
    else if (uri == "ui://llama-agent/apps/workflow-inspector") text = workflow_app.c_str();
#ifdef LLAMA_AGENT_DOCUMENT_EDITOR_RESOURCE
    else if (uri == "ui://llama-agent/apps/document-editor") text = LLAMA_AGENT_DOCUMENT_EDITOR_HTML;
#endif
#ifdef LLAMA_AGENT_INVESTIGATION_WORKSPACE_RESOURCE
    else if (uri == "ui://llama-agent/apps/investigation-workspace") text = LLAMA_AGENT_INVESTIGATION_WORKSPACE_HTML;
#endif
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
        {"name", uri.find("data-explorer") != std::string::npos ? "Data Explorer" : uri.find("workflow-inspector") != std::string::npos ? "Workflow Inspector" : "Document Editor"},
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
        if (!tool.output_schema.empty()) {
            const auto schema = common_tool_success_payload_schema(tool.output_schema);
            if (schema.is_object()) server_tool.output_schema_json = schema.dump();
        }
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
