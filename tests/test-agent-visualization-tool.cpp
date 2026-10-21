#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/visualization/visualization-contract.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <map>

class visualization_data_store final : public common_agent_data_store {
public:
    visualization_data_store() {
        descriptor.ref.uri = "dataset://scope/links";
        descriptor.ref.name = "Links";
        descriptor.ref.row_count = 3;
        descriptor.ref.column_count = 3;
        descriptor.ref.source_resource_uri = "resource://source/links";
        descriptor.ref.source_representation = "tabular-dataset";
        descriptor.import_processor_id = "test-dataset-import-v1";
        descriptor.import_processor_version = "1";
        descriptor.ref.count_kind = common_agent_dataset_count_kind::exact;
        descriptor.columns = {{"source", common_agent_dataset_column_type::string, false},
            {"target", common_agent_dataset_column_type::string, false},
            {"relation", common_agent_dataset_column_type::string, true}};
    }

    bool execute(const std::string &, const std::string &, std::string &, std::string & error) override {
        error = "visualization must use scoped reads";
        return false;
    }
    bool execute_scoped(const std::string & operation, const std::string & request,
            const common_agent_scope &, bool, std::string & result, std::string & error) override {
        if (operation != "data.query") { error = "unexpected operation"; return false; }
        observed_request = nlohmann::json::parse(request);
        result = R"({"columns":["source","target","relation"],"rows":[["a","b","auth"],["a","b","auth"],["b","c","network"]],"scan_truncated":false,"result_truncated":false})";
        error.clear(); return true;
    }
    bool get_dataset_descriptor(const std::string & uri,
            common_agent_dataset_descriptor & out, std::string & error) override {
        if (uri != descriptor.ref.uri) { error = "dataset unavailable"; return false; }
        out = descriptor; error.clear(); return true;
    }
    common_agent_dataset_descriptor descriptor;
    nlohmann::json observed_request;
};

class visualization_resource_store final : public agent_resource_store {
public:
    bool put_text(const agent_resource_put_request & request, agent_resource_descriptor & out,
            std::string & error) override {
        saved_request = request;
        saved = request.text;
        out.uri = "resource://session/visualization-1";
        out.name = request.name;
        out.mime_type = request.mime_type;
        out.scope = request.scope;
        error.clear(); return true;
    }
    bool read_text(const std::string &, const agent_resource_read_authority &, size_t,
            std::string &, std::string & error) const override { error = "not used"; return false; }
    bool stat(const std::string &, const agent_resource_read_authority &,
            agent_resource_descriptor &, std::string & error) const override { error = "not used"; return false; }
    bool list(const agent_resource_read_authority &, std::vector<agent_resource_descriptor> &,
            std::string & error) const override { error = "not used"; return false; }
    agent_resource_put_request saved_request;
    std::string saved;
};

int main() {
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    assert(catalog.bootstrap("analysis", bootstrap, error));
    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-a";
    visualization_data_store data;
    visualization_resource_store resources;
    common_native_tool_bindings bindings;
    bindings.tool_scope = scope;
    bindings.data_store = &data;
    bindings.resource_runtime.store = &resources;
    bindings.resource_runtime.namespace_id = scope.namespace_id;
    bindings.resource_runtime.session_id = scope.session_id;
    common_tool_registry registry;
    common_tool_adapter_result adapters;
    assert(common_register_native_tool_adapters(catalog, "analysis", bindings,
        registry, adapters, error));
    assert(registry.contains("visualization.create"));
    assert(!registry.is_read_only("visualization.create"));
    const auto result = registry.execute({"visualization.create",
        R"({"dataset":"dataset://scope/links","title":"Connections","source_field":"source","target_field":"target","edge_label_field":"relation","max_rows":2})"});
    if (!result.ok) std::cerr << result.failure_code << ": " << result.raw_diagnostic << " / " << result.output << "\n";
    assert(result.ok);
    assert(data.observed_request.value("limit", 0) == 3);
    assert(resources.saved_request.mime_type == "application/vnd.llama-agent.visualization+json");
    assert(resources.saved_request.scope == common_runtime_resource_scope::session);
    assert(resources.saved_request.lineage.parent_uri == "dataset://scope/links");
    const auto output = nlohmann::json::parse(result.output);
    assert(output.at("visualization_uri") == "resource://session/visualization-1");
    assert(output.at("truncated").get<bool>() && !output.at("complete").get<bool>());
    common_agent_visualization_relationship graph;
    assert(common_agent_visualization_relationship_from_json(output.at("visualization"), graph, error));
    assert(graph.nodes.size() == 3 && graph.edges.size() == 2);
    assert(graph.edges.front().label == "auth");
    assert(!graph.limitations.empty());

    const auto saved_spec = resources.saved;
    const auto invalid_reference = registry.execute({"visualization.create",
        R"({"dataset":"file:///tmp/links","title":"Invalid","source_field":"source","target_field":"target"})"});
    assert(!invalid_reference.ok && resources.saved == saved_spec);
    return 0;
}
