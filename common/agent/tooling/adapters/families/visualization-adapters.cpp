#include "agent/tooling/adapters/families/visualization-adapters.h"

#include "agent/tooling/adapters/families/data-adapters.h"
#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/visualization/visualization-contract.h"
#include "hash/hash.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace {
using json = common_adapter_json;

std::string scalar_text(const json & value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean() || value.is_number()) return value.dump();
    return {};
}

bool read_field(const json & arguments, const char * name, std::string & value,
        bool required, std::string & error, size_t maximum = 128) {
    if (!arguments.contains(name)) {
        if (required) error = std::string("missing required field: ") + name;
        value.clear();
        return !required;
    }
    if (!arguments[name].is_string()) {
        error = std::string(name) + " must be a string";
        return false;
    }
    value = arguments[name].get<std::string>();
    if ((required && value.empty()) || value.size() > maximum) {
        error = std::string(name) + " is empty or exceeds its size limit";
        return false;
    }
    return true;
}

common_tool_execution_result create_relationship(
        const common_native_tool_bindings & bindings, const std::string & input) {
    json arguments;
    std::string error;
    if (!common_adapter_parse_object(input, arguments, error))
        return common_adapter_validation_failure("tool.visualization.create.invalid_arguments", std::move(error));

    static const std::set<std::string> allowed{
        "dataset", "title", "source_field", "target_field", "source_label_field",
        "target_label_field", "edge_label_field", "source_kind", "target_kind", "layout", "max_rows"};
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
        if (allowed.count(it.key()) == 0)
            return common_adapter_validation_failure("tool.visualization.create.invalid_arguments",
                "unsupported visualization.create argument: " + it.key());
    }

    std::string dataset_uri, title, source_field, target_field, source_label_field,
        target_label_field, edge_label_field, source_kind, target_kind, layout;
    const auto read = [&](const char * key, std::string & value, bool required) {
        return read_field(arguments, key, value, required, error);
    };
    if (!read_field(arguments, "dataset", dataset_uri, true, error, 512) ||
            !read_field(arguments, "title", title, true, error, 256) ||
            !read("source_field", source_field, true) ||
            !read("target_field", target_field, true) ||
            !read("source_label_field", source_label_field, false) ||
            !read("target_label_field", target_label_field, false) ||
            !read("edge_label_field", edge_label_field, false) ||
            !read_field(arguments, "source_kind", source_kind, false, error, 64) ||
            !read_field(arguments, "target_kind", target_kind, false, error, 64) ||
            !read("layout", layout, false)) {
        return common_adapter_validation_failure("tool.visualization.create.invalid_arguments", std::move(error));
    }
    if (layout.empty()) layout = "cose";
    if (source_kind.empty()) source_kind = "entity";
    if (target_kind.empty()) target_kind = source_kind;
    size_t max_rows = 100;
    if (arguments.contains("max_rows")) {
        if (!arguments["max_rows"].is_number_integer())
            return common_adapter_validation_failure("tool.visualization.create.invalid_arguments", "max_rows must be an integer");
        const auto requested = arguments["max_rows"].get<long long>();
        if (requested < 1 || requested > 100)
            return common_adapter_validation_failure("tool.visualization.create.invalid_arguments", "max_rows must be between 1 and 100");
        max_rows = static_cast<size_t>(requested);
    }

    common_agent_dataset_descriptor descriptor;
    common_tool_execution_result failure;
    if (!common_authorize_dataset_reference(bindings, dataset_uri, descriptor, failure)) return failure;
    std::set<std::string> columns;
    for (const auto & column : descriptor.columns) columns.insert(column.name);
    const std::vector<std::string> mapped_fields{source_field, target_field, source_label_field,
        target_label_field, edge_label_field};
    for (const auto & field : mapped_fields) {
        if (!field.empty() && !columns.empty() && columns.count(field) == 0)
            return common_adapter_validation_failure("tool.visualization.create.field_unmapped",
                "field is not present in the selected dataset schema: " + field);
    }

    json select = json::array();
    for (const auto & field : mapped_fields) {
        if (!field.empty() && std::find(select.begin(), select.end(), field) == select.end()) select.push_back(field);
    }
    // Ask for one extra row so a full page is never presented as complete.
    json order_by = json::array({
        {{"field", source_field}, {"direction", "asc"}},
        {{"field", target_field}, {"direction", "asc"}}});
    if (!edge_label_field.empty() && edge_label_field != source_field && edge_label_field != target_field)
        order_by.push_back({{"field", edge_label_field}, {"direction", "asc"}});
    const json query{{"dataset", dataset_uri}, {"select", select}, {"order_by", order_by}, {"limit", max_rows + 1},
        {"max_result_rows", max_rows + 1}, {"max_scan_rows", max_rows + 1}};
    std::string response, query_error;
    if (!bindings.data_store->execute_scoped("data.query", query.dump(), bindings.tool_scope,
            bindings.allow_network, response, query_error)) {
        return common_adapter_execution_failure("tool.visualization.create.query_failed",
            std::move(query_error), "The selected dataset could not be read.");
    }
    const auto page = json::parse(response, nullptr, false);
    if (!page.is_object() || !page.contains("columns") || !page["columns"].is_array() ||
            !page.contains("rows") || !page["rows"].is_array())
        return common_adapter_execution_failure("tool.visualization.create.invalid_dataset_result",
            "data.query returned an invalid positional page", "The dataset rows could not be projected.");
    for (const char * flag : {"scan_truncated", "result_truncated"}) {
        if (page.contains(flag) && !page[flag].is_boolean())
            return common_adapter_execution_failure("tool.visualization.create.invalid_dataset_result",
                std::string("data.query returned a non-boolean ") + flag,
                "The dataset rows could not be projected.");
    }
    std::vector<std::string> returned_columns;
    for (const auto & column : page["columns"]) {
        if (!column.is_string()) return common_adapter_execution_failure(
            "tool.visualization.create.invalid_dataset_result", "data.query returned a non-string column name",
            "The dataset rows could not be projected.");
        returned_columns.push_back(column.get<std::string>());
    }
    const auto column_index = [&](const std::string & name) -> size_t {
        const auto it = std::find(returned_columns.begin(), returned_columns.end(), name);
        return it == returned_columns.end() ? returned_columns.size() : static_cast<size_t>(it - returned_columns.begin());
    };
    const size_t source_index = column_index(source_field), target_index = column_index(target_field);
    const size_t source_label_index = source_label_field.empty() ? returned_columns.size() : column_index(source_label_field);
    const size_t target_label_index = target_label_field.empty() ? returned_columns.size() : column_index(target_label_field);
    const size_t edge_label_index = edge_label_field.empty() ? returned_columns.size() : column_index(edge_label_field);
    if (source_index == returned_columns.size() || target_index == returned_columns.size() ||
            (!source_label_field.empty() && source_label_index == returned_columns.size()) ||
            (!target_label_field.empty() && target_label_index == returned_columns.size()) ||
            (!edge_label_field.empty() && edge_label_index == returned_columns.size()))
        return common_adapter_execution_failure("tool.visualization.create.invalid_dataset_result",
            "data.query omitted a requested column", "The dataset rows could not be projected.");

    common_agent_visualization_relationship spec;
    spec.title = title;
    spec.view.layout = layout;
    std::map<std::string, common_agent_visualization_node> nodes;
    std::map<std::tuple<std::string, std::string, std::string>, size_t> edges;
    size_t invalid_rows = 0;
    bool label_truncated = false;
    for (const auto & row : page["rows"]) {
        if (!row.is_array() || row.size() != returned_columns.size()) {
            ++invalid_rows;
            continue;
        }
        const auto at = [&](size_t index) -> std::string {
            return index < row.size() ? scalar_text(row[index]) : std::string();
        };
        const std::string source_value = at(source_index), target_value = at(target_index);
        if (source_value.empty() || target_value.empty()) { ++invalid_rows; continue; }
        const std::string source_id = "node:" + source_kind + ":" + source_value;
        const std::string target_id = "node:" + target_kind + ":" + target_value;
        if (source_id.size() > 256 || target_id.size() > 256) {
            return common_adapter_validation_failure("tool.visualization.create.identity_too_long",
                "mapped entity identity exceeds the visualization contract limit");
        }
        std::string source_label = at(source_label_index).empty() ? source_value : at(source_label_index);
        std::string target_label = at(target_label_index).empty() ? target_value : at(target_label_index);
        if (source_label.size() > 512) { source_label.resize(512); label_truncated = true; }
        if (target_label.size() > 512) { target_label.resize(512); label_truncated = true; }
        const auto add_node = [&](const std::string & id, const std::string & label, const std::string & kind) {
            const auto inserted = nodes.emplace(id, common_agent_visualization_node{id, label, kind});
            if (!inserted.second && label < inserted.first->second.label) inserted.first->second.label = label;
        };
        add_node(source_id, source_label, source_kind);
        add_node(target_id, target_label, target_kind);
        const std::string edge_label = at(edge_label_index);
        ++edges[{source_id, target_id, edge_label}];
    }
    const bool source_truncated = page.value("scan_truncated", false) || page.value("result_truncated", false);
    const bool extra_row = page["rows"].size() > max_rows;
    if (extra_row) spec.truncated = true;
    if (source_truncated) spec.truncated = true;
    if (invalid_rows) {
        spec.complete = false;
        spec.limitations.push_back("Rows with missing identities or an invalid row shape were skipped.");
    }
    if (label_truncated) spec.limitations.push_back("Some display labels were shortened to fit the visualization contract.");
    if (extra_row) spec.limitations.push_back("The dataset projection reached its configured row limit.");
    if (source_truncated) spec.limitations.push_back("The source reported a truncated query result.");
    if (descriptor.ref.count_kind != common_agent_dataset_count_kind::exact)
        spec.limitations.push_back("Dataset total coverage is not exact; this is a bounded projection.");
    spec.complete = !spec.truncated && descriptor.ref.count_kind == common_agent_dataset_count_kind::exact;
    size_t emitted_edges = 0;
    for (const auto & [id, node] : nodes) {
        if (spec.nodes.size() == COMMON_AGENT_VISUALIZATION_MAX_NODES) { spec.truncated = true; break; }
        spec.nodes.push_back(node);
    }
    std::set<std::string> included_nodes;
    for (const auto & node : spec.nodes) included_nodes.insert(node.id);
    for (const auto & [key, count] : edges) {
        const auto & [source, target, label] = key;
        if (!included_nodes.count(source) || !included_nodes.count(target)) continue;
        if (emitted_edges == COMMON_AGENT_VISUALIZATION_MAX_EDGES) { spec.truncated = true; break; }
        std::string bounded_edge_label = label;
        if (bounded_edge_label.size() > 512) {
            bounded_edge_label.resize(512);
            label_truncated = true;
        }
        const auto serialized_key = json::array({source, target, bounded_edge_label}).dump();
        const std::string edge_id = "edge:" + hash_sha256_hex(serialized_key.data(), serialized_key.size());
        spec.edges.push_back({edge_id, source, target,
            bounded_edge_label.empty() ? (count == 1 ? "observed relation" : "observed relation (" + std::to_string(count) + ")") : bounded_edge_label,
            true});
        ++emitted_edges;
    }
    if (spec.truncated && std::find(spec.limitations.begin(), spec.limitations.end(),
            "The projected graph reached its contract limits.") == spec.limitations.end() &&
            (nodes.size() > spec.nodes.size() || edges.size() > spec.edges.size()))
        spec.limitations.push_back("The projected graph reached its contract limits.");
    if (spec.truncated) spec.complete = false;
    if (label_truncated && std::find(spec.limitations.begin(), spec.limitations.end(),
            "Some display labels were shortened to fit the visualization contract.") == spec.limitations.end())
        spec.limitations.push_back("Some display labels were shortened to fit the visualization contract.");
    if (!spec.complete && spec.limitations.empty()) spec.limitations.push_back("The projection is incomplete.");
    json revision_basis{{"dataset", dataset_uri}, {"source_field", source_field},
        {"target_field", target_field}, {"source_kind", source_kind}, {"target_kind", target_kind},
        {"nodes", json::array()}, {"edges", json::array()}, {"complete", spec.complete}};
    for (const auto & node : spec.nodes) revision_basis["nodes"].push_back(
        {{"id", node.id}, {"label", node.label}, {"kind", node.kind}});
    for (const auto & edge : spec.edges) revision_basis["edges"].push_back(
        {{"id", edge.id}, {"source", edge.source}, {"target", edge.target}, {"label", edge.label}});
    const auto revision_json = revision_basis.dump();
    spec.data_revision = hash_sha256_hex(revision_json.data(), revision_json.size());
    std::string validation_error;
    if (!common_agent_visualization_relationship_validate(spec, validation_error))
        return common_adapter_execution_failure("tool.visualization.create.projection_invalid",
            std::move(validation_error), "The dataset could not be represented as a valid graph.");
    const auto serialized = common_agent_visualization_relationship_to_json(spec).dump();
    if (serialized.size() > COMMON_AGENT_VISUALIZATION_MAX_BYTES)
        return common_adapter_failure("tool.visualization.create.byte_limit", common_tool_failure_class::limit,
            false, "The graph exceeds the visualization resource size limit.", "serialized visualization exceeded byte limit");
    if (bindings.resource_runtime.store == nullptr)
        return common_adapter_execution_failure("tool.visualization.create.store_unavailable",
            "resource store is unavailable", "The visualization could not be saved.");

    agent_resource_put_request put;
    put.name = title;
    put.description = "Frozen relationship visualization projected from an authorized dataset.";
    put.mime_type = "application/vnd.llama-agent.visualization+json";
    put.text = serialized;
    put.scope = bindings.resource_runtime.project_id.empty()
        ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
    put.source_provider = "native";
    put.source_tool = "visualization.create";
    put.metadata.content_summary = "Versioned relationship visualization.";
    put.metadata.purpose = "Present a bounded projection of an authorized structured dataset.";
    for (const auto & limitation : spec.limitations) put.metadata.limitations += limitation + "\n";
    put.lineage.parent_uri = dataset_uri;
    put.lineage.chunk_count = 1;
    put.lineage.byte_length = serialized.size();
    put.lineage.derivation = "visualization.create:relationship-v1";
    apply_agent_resource_runtime(bindings.resource_runtime, put);
    agent_resource_descriptor resource;
    std::string store_error;
    if (!bindings.resource_runtime.store->put_text(put, resource, store_error))
        return common_adapter_execution_failure("tool.visualization.create.store_failed",
            std::move(store_error), "The visualization could not be saved.");
    return common_adapter_success_json({{"visualization_uri", resource.uri},
        {"dataset_uri", dataset_uri}, {"visualization", common_agent_visualization_relationship_to_json(spec)},
        {"complete", spec.complete}, {"truncated", spec.truncated}, {"limitations", spec.limitations}});
}
}

bool common_try_register_visualization_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    if (definition.executor_id != "builtin.visualization.create") return false;
    installed = common_adapter_register_definition(definition, registry, [&bindings](const std::string & input) {
        return create_relationship(bindings, input);
    }, error, false, false);
    return true;
}
