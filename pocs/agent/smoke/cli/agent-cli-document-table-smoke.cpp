#include "tools/agent/cli/agent-cli-host-adapter.h"

#include "memory/memory-in-memory.h"

#include <iostream>
#include <string>
#include <vector>

class smoke_data_store final : public common_agent_data_store {
public:
    bool put_row(const std::string & dataset, const std::string & row_id,
            const std::string & row_json, std::string & error) override {
        last_dataset = dataset;
        rows.push_back(row_id + "=" + row_json);
        error.clear();
        return true;
    }

    bool put_dataset_descriptor(const common_agent_dataset_descriptor & value,
            std::string & error) override {
        descriptor = value;
        error.clear();
        return true;
    }

    bool get_dataset_descriptor(const std::string & uri,
            common_agent_dataset_descriptor & value, std::string & error) override {
        if (descriptor.ref.uri != uri) {
            error = "dataset was not materialized";
            return false;
        }
        value = descriptor;
        error.clear();
        return true;
    }

    bool execute(const std::string &, const std::string &, std::string &,
            std::string & error) override {
        error = "not used by document table smoke";
        return false;
    }

    common_agent_dataset_descriptor descriptor;
    std::string last_dataset;
    std::vector<std::string> rows;
};

static bool has_chat_tool(const agent_tool_view & view, const std::string & name) {
    for (const auto & tool : view.chat_tools()) {
        if (tool.name == name) return true;
    }
    return false;
}

int main() {
    common_memory_in_memory_store memory;
    std::string error;
    const auto fail = [](const std::string & message) {
        std::cerr << message << "\n";
        return 1;
    };
    if (!memory.open("", error)) return fail("memory setup failed: " + error);

    common_tool_profile profile;
    profile.id = "cli-document-tables";
    profile.members = {
        {"document.tables", 1, true, "{}"},
        {"document.table", 1, true, "{}"},
    };

    agent_host_tool_selection_request request;
    request.tool_context.request_id = "document-table-smoke";
    request.tool_context.turn_id = "turn-1";
    request.tool_context.profile_id = profile.id;
    request.tool_context.scope.namespace_id = "local";
    request.tool_context.scope.session_id = "session-1";
    request.tool_context.scope.project_id = "project-1";
    request.tool_context.scope.turn_id = "turn-1";
    request.tool_profiles.emplace(profile.id, profile);
    smoke_data_store data;
    request.data_store = &data;

    common_agent_cli_tool_selection selection;
    common_memory_query query;
    query.scope = common_memory_scope::session;
    query.session_id = "session-1";
    if (!resolve_agent_host_tool_selection(
            memory, nullptr, nullptr, nullptr, profile.id, request, query, nullptr,
            selection, error)) return fail("tool selection failed: " + error);
    if (selection.tool_view == nullptr) return fail("tool view was not created");
    if (selection.owned_resource_store == nullptr) return fail("resource store was not created");
    if (!has_chat_tool(*selection.tool_view, "document.tables") ||
            !has_chat_tool(*selection.tool_view, "document.table"))
        return fail("document table tools were not exposed to the model tool view");

    const std::string document_json = R"({"blocks":[
      {"t":"Header","c":[1,[],[{"t":"Str","c":"Budget summary"}]]},
      {"t":"Table","c":[[],[],[],
        [[],[[[],[[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"category"}]}]],[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"amount"}]}]]]]]],
        [[[],0,[],[
          [[],[[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"Travel"}]}]],[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"125"}]}]]]]]],[]]
      ]}
    ]})";
    agent_resource_put_request put;
    put.name = "report.document.json";
    put.description = "Structured document representation for table lookup smoke.";
    put.mime_type = "application/json";
    put.scope = common_runtime_resource_scope::turn;
    put.namespace_id = "local";
    put.session_id = "session-1";
    put.project_id = "project-1";
    put.turn_id = "turn-1";
    put.bytes = document_json;
    agent_resource_descriptor document;
    if (!selection.owned_resource_store->put_bytes(put, document, error))
        return fail("document resource setup failed: " + error);

    // Reproduce the CLI host lifetime transfer: the resolved tool view keeps
    // non-owning bindings, while the resource store is retained by tooling
    // for the complete runtime operation.
    common_agent_runtime_tooling tooling = std::move(selection.tooling);
    auto tool_view = std::move(selection.tool_view);
    auto owned_resource_store = std::move(selection.owned_resource_store);
    if (!tool_view || !owned_resource_store) return fail("CLI ownership transfer failed");
    tooling.tool_view = tool_view.get();
    auto shared_resource_store = std::shared_ptr<agent_resource_store>(
        std::move(owned_resource_store));
    tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_resource_store));

    auto listed = tool_view->call({
        "list-tables", "document.tables",
        std::string("{\"resource\":\"") + document.uri + "\"}",
    }, error);
    if (!listed.ok) {
        std::cerr << "document.tables failed: code=" << listed.failure_code
                  << " summary=" << listed.safe_summary
                  << " detail=" << listed.raw_diagnostic << "\n";
    }
    if (!listed.ok) return fail("document.tables returned an invalid result");
    if (listed.content_json.find("Budget summary") == std::string::npos)
        return fail("document.tables omitted the table name");
    if (listed.content_json.find("document-node://table/0") == std::string::npos)
        return fail("document.tables omitted the table node");

    auto selected = tool_view->call({
        "select-table", "document.table",
        std::string("{\"resource\":\"") + document.uri + "\",\"table\":\" budget   SUMMARY \"}",
    }, error);
    if (!selected.ok) return fail("document.table returned an invalid result");
    if (selected.content_json.find("\"dataset\":\"dataset://import/") == std::string::npos ||
            selected.content_json.find("/0/Budget summary") == std::string::npos)
        return fail("document.table omitted the dataset reference");
    if (data.descriptor.origin.kind != "document_table")
        return fail("dataset origin kind was not preserved");
    if (data.descriptor.origin.source_representation_uri != document.uri)
        return fail("dataset source representation was not preserved");
    if (data.descriptor.origin.source_node_id != "document-node://table/0")
        return fail("dataset source node was not preserved");
    if (data.last_dataset != data.descriptor.ref.uri)
        return fail("dataset row and descriptor references differ");
    if (data.rows.size() != 1) return fail("unexpected materialized row count");

    auto invalid = tool_view->call({
        "invalid-table", "document.table",
        std::string("{\"resource\":\"") + document.uri + "\",\"table\":\"Budget summary\",\"table_index\":0}",
    }, error);
    if (invalid.ok) return fail("invalid table locator was accepted");
    if (invalid.failure_code != "tool.document.table.invalid_locator")
        return fail("invalid table locator returned the wrong failure code");
    return 0;
}
