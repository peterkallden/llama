#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/visualization/visualization-contract.h"
#include "base64.hpp"
#include "hash/hash.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <map>
#include <string>

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
        result = page;
        error.clear(); return true;
    }
    bool get_dataset_descriptor(const std::string & uri,
            common_agent_dataset_descriptor & out, std::string & error) override {
        if (uri != descriptor.ref.uri) { error = "dataset unavailable"; return false; }
        out = descriptor; error.clear(); return true;
    }
    common_agent_dataset_descriptor descriptor;
    nlohmann::json observed_request;
    std::string page = R"({"columns":["source","target","relation"],"rows":[["a","b","auth"],["a","b","auth"],["b","c","network"]],"scan_truncated":false,"result_truncated":false})";
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
        out.namespace_id = request.namespace_id;
        out.session_id = request.session_id;
        out.project_id = request.project_id;
        out.sha256 = hash_sha256_hex(request.text.data(), request.text.size());
        out.metadata = request.metadata;
        out.lineage = request.lineage;
        resources[out.uri] = out;
        text_resources[out.uri] = request.text;
        descriptor = out;
        error.clear(); return true;
    }
    bool put_bytes(const agent_resource_put_request & request, agent_resource_descriptor & out,
            std::string & error) override {
        saved_request = request;
        image = request.bytes;
        out.uri = "resource://session/snapshot-1";
        out.name = request.name;
        out.mime_type = request.mime_type;
        out.scope = request.scope;
        out.namespace_id = request.namespace_id;
        out.session_id = request.session_id;
        out.project_id = request.project_id;
        out.sha256 = hash_sha256_hex(request.bytes.data(), request.bytes.size());
        out.size_bytes = request.bytes.size();
        out.metadata = request.metadata;
        out.lineage = request.lineage;
        resources[out.uri] = out;
        descriptor = out;
        error.clear(); return true;
    }
    bool read_text(const std::string & uri, const agent_resource_read_authority & authority, size_t max_bytes,
            std::string & out, std::string & error) const override {
        const auto it = text_resources.find(uri);
        if (!authorized(uri, authority) || it == text_resources.end() || it->second.size() > max_bytes) { error = "resource unavailable"; return false; }
        out = it->second; error.clear(); return true;
    }
    bool read_bytes(const std::string & uri, const agent_resource_read_authority & authority, size_t max_bytes,
            std::string & out, std::string & error) const override {
        if (!authorized(uri, authority) || uri != "resource://session/snapshot-1" || image.size() > max_bytes) { error = "resource unavailable"; return false; }
        out = image; error.clear(); return true;
    }
    bool stat(const std::string & uri, const agent_resource_read_authority & authority,
            agent_resource_descriptor & out, std::string & error) const override {
        const auto it = resources.find(uri);
        if (!authorized(uri, authority) || it == resources.end()) { error = "resource unavailable"; return false; }
        out = it->second; error.clear(); return true;
    }
    bool list(const agent_resource_read_authority &, std::vector<agent_resource_descriptor> &,
            std::string & error) const override { error = "not used"; return false; }
    agent_resource_put_request saved_request;
    std::string saved;
    std::string image;
    agent_resource_descriptor descriptor;
    std::map<std::string, agent_resource_descriptor> resources;
    std::map<std::string, std::string> text_resources;
private:
    bool authorized(const std::string & uri, const agent_resource_read_authority & authority) const {
        const auto it = resources.find(uri);
        if (it == resources.end()) return false;
        const auto & resource = it->second;
        return authority.namespace_id == resource.namespace_id &&
            (resource.scope == common_runtime_resource_scope::project
                ? authority.project_id == resource.project_id
                : authority.project_id.empty() && authority.session_id == resource.session_id);
    }
};

int main() {
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    assert(catalog.bootstrap("investigation", bootstrap, error));
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
    {
        // Registered handlers must own the binding values after setup returns.
        const auto registration_bindings = bindings;
        assert(common_register_native_tool_adapters(catalog, "investigation", registration_bindings,
            registry, adapters, error));
    }
    assert(registry.contains("visualization.create"));
    assert(!registry.is_read_only("visualization.create"));
    assert(registry.is_policy_gated("visualization.create"));
    assert(registry.is_policy_gated("visualization.snapshot"));
    assert(registry.contains("visualization.open"));
    assert(registry.is_read_only("visualization.open"));
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
    assert(graph.nodes.size() == 2 && graph.edges.size() == 1);
    assert(graph.edges.front().label == "auth");
    assert(!graph.limitations.empty());
    const auto reopened = registry.execute({"visualization.open",
        R"({"visualization_uri":"resource://session/visualization-1"})"});
    assert(reopened.ok);
    const auto reopened_json = nlohmann::json::parse(reopened.output);
    assert(reopened_json.at("visualization_uri") == "resource://session/visualization-1");
    common_agent_visualization_relationship reopened_graph;
    assert(common_agent_visualization_relationship_from_json(reopened_json.at("visualization"), reopened_graph, error));
    assert(reopened_graph.data_revision == graph.data_revision && reopened_graph.nodes.size() == 2);
    const auto foreign_resource = registry.execute({"visualization.open",
        R"({"visualization_uri":"resource://session/another-user-graph"})"});
    assert(!foreign_resource.ok);
    const auto extra_argument = registry.execute({"visualization.open",
        R"({"visualization_uri":"resource://session/visualization-1","dataset":"dataset://scope/links"})"});
    assert(!extra_argument.ok);

    const auto saved_spec = resources.saved;
    const auto invalid_reference = registry.execute({"visualization.create",
        R"({"dataset":"file:///tmp/links","title":"Invalid","source_field":"source","target_field":"target"})"});
    assert(!invalid_reference.ok && resources.saved == saved_spec);
    const auto project = [&](const std::string & rows, const std::string & flags = "") {
        data.page = "{\"columns\":[\"source\",\"target\",\"relation\"],\"rows\":" + rows + flags + "}";
        return registry.execute({"visualization.create",
            R"({"dataset":"dataset://scope/links","title":"Connections","source_field":"source","target_field":"target","edge_label_field":"relation"})"});
    };
    auto partial = project(R"([["a","b","auth"],[null,"c","network"]])");
    assert(partial.ok && !nlohmann::json::parse(partial.output).at("complete").get<bool>());
    partial = project(R"([["a","b","auth"]])", ",\"complete\":false");
    assert(partial.ok && !nlohmann::json::parse(partial.output).at("complete").get<bool>());
    const auto typed = project(R"([[1,"b","auth"],["1","b","auth"]])");
    assert(typed.ok && nlohmann::json::parse(typed.output).at("visualization").at("nodes").size() == 3);
    const auto malformed = project(R"([["a","b","auth"]])", ",\"complete\":\"yes\"");
    assert(!malformed.ok);
    std::string label(511, 'x'); label += "€tail";
    data.page = nlohmann::json{{"columns", {"source", "target", "relation"}},
        {"rows", {{"a", "b", label}, {"a", "b", label + "different"}}}}.dump();
    const auto unicode = registry.execute({"visualization.create",
        R"({"dataset":"dataset://scope/links","title":"Unicode","source_field":"source","target_field":"target","edge_label_field":"relation"})"});
    assert(unicode.ok);
    const auto unicode_graph = nlohmann::json::parse(unicode.output).at("visualization");
    assert(unicode_graph.at("edges").size() == 2);
    assert(unicode_graph.at("edges")[0].at("label").get<std::string>().size() == 511);
    const auto snapshot = registry.execute({"visualization.snapshot", nlohmann::json{
        {"visualization_uri", "resource://session/visualization-1"},
        {"image_base64", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="}, {"title", "Connections"}}.dump()});
    if (!snapshot.ok) std::cerr << snapshot.failure_code << ": " << snapshot.raw_diagnostic << " / " << snapshot.output << "\n";
    assert(snapshot.ok && resources.saved_request.mime_type == "image/png" &&
        resources.saved_request.bytes_are_authoritative && resources.saved_request.lineage.parent_uri == "resource://session/visualization-1");
    const auto snapshot_json = nlohmann::json::parse(snapshot.output);
    assert(snapshot_json.at("snapshot_uri") == "resource://session/snapshot-1" &&
        snapshot_json.at("mime_type") == "image/png" && snapshot_json.at("size_bytes").get<size_t>() >= 45);
    const auto case_snapshot = registry.execute({"visualization.snapshot", nlohmann::json{
        {"visualization", unicode_graph}, {"image_base64", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="}, {"title", "Case view"}}.dump()});
    assert(case_snapshot.ok);
    const auto case_snapshot_json = nlohmann::json::parse(case_snapshot.output);
    assert(case_snapshot_json.at("visualization_uri") == "resource://session/visualization-1" &&
        resources.saved_request.lineage.parent_uri == "resource://session/visualization-1" &&
        resources.resources.at("resource://session/visualization-1").metadata.purpose.find("client supplied") != std::string::npos);
    const auto invalid_snapshot = registry.execute({"visualization.snapshot", nlohmann::json{
        {"visualization_uri", "resource://session/visualization-1"}, {"image_base64", "bm90cG5n"}}.dump()});
    assert(!invalid_snapshot.ok);
    return 0;
}
