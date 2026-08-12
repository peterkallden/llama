#include "tools/agent/cli/agent-cli-host-adapter.h"

#include "memory/memory-in-memory.h"

#include <cassert>
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

int main() {
    common_memory_in_memory_store memory;
    std::string error;
    assert(memory.open("", error));

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
    request.tool_profiles.emplace(profile.id, profile);
    smoke_data_store data;
    request.data_store = &data;

    common_agent_cli_tool_selection selection;
    common_memory_query query;
    query.scope = common_memory_scope::session;
    query.session_id = "session-1";
    assert(resolve_agent_host_tool_selection(
        memory, nullptr, nullptr, nullptr, profile.id, request, query, nullptr,
        selection, error));
    assert(selection.tool_view != nullptr);
    assert(selection.owned_resource_store != nullptr);

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
    put.scope = common_runtime_resource_scope::session;
    put.namespace_id = "local";
    put.session_id = "session-1";
    put.project_id = "project-1";
    put.bytes = document_json;
    agent_resource_descriptor document;
    assert(selection.owned_resource_store->put_bytes(put, document, error));

    auto listed = selection.tool_view->call({
        "list-tables", "document.tables",
        std::string("{\"resource\":\"") + document.uri + "\"}",
    }, error);
    if (!listed.ok) {
        std::cerr << "document.tables failed: code=" << listed.failure_code
                  << " summary=" << listed.safe_summary
                  << " detail=" << listed.raw_diagnostic << "\n";
    }
    assert(listed.ok);
    assert(listed.content_json.find("Budget summary") != std::string::npos);
    assert(listed.content_json.find("document-node://table/0") != std::string::npos);

    auto selected = selection.tool_view->call({
        "select-table", "document.table",
        std::string("{\"resource\":\"") + document.uri + "\",\"table\":\" budget   SUMMARY \"}",
    }, error);
    assert(selected.ok);
    assert(selected.content_json.find("dataset://import/0/Budget summary") != std::string::npos);
    assert(data.descriptor.origin.kind == "document_table");
    assert(data.descriptor.origin.source_representation_uri == document.uri);
    assert(data.descriptor.origin.source_node_id == "document-node://table/0");
    assert(data.last_dataset == data.descriptor.ref.uri);
    assert(data.rows.size() == 1);

    auto invalid = selection.tool_view->call({
        "invalid-table", "document.table",
        std::string("{\"resource\":\"") + document.uri + "\",\"table\":\"Budget summary\",\"table_index\":0}",
    }, error);
    assert(!invalid.ok);
    assert(invalid.failure_code == "tool.document.table.invalid_locator");
    return 0;
}
