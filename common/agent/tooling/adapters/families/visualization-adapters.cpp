#include "agent/tooling/adapters/families/visualization-adapters.h"

#include "agent/tooling/adapters/families/data-adapters.h"
#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/visualization/visualization-contract.h"
#include "base64.hpp"
#include "hash/hash.h"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <iterator>
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

bool shorten_label(std::string & value) {
    if (value.size() <= 512) return false;
    size_t end = 512;
    while (end > 0 && (static_cast<unsigned char>(value[end]) & 0xc0) == 0x80) --end;
    value.resize(end);
    return true;
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

uint32_t png_u32(const std::string & bytes, size_t offset) {
    return (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) << 24) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1])) << 16) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2])) << 8) |
        static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3]));
}

bool bounded_png(const std::string & bytes) {
    static constexpr unsigned char signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (bytes.size() < 45 || !std::equal(std::begin(signature), std::end(signature),
            reinterpret_cast<const unsigned char *>(bytes.data()))) return false;
    size_t offset = 8;
    bool first = true, has_idat = false, has_iend = false;
    while (offset + 12 <= bytes.size()) {
        const uint32_t length = png_u32(bytes, offset);
        if (static_cast<uint64_t>(length) + 12 > bytes.size() - offset) return false;
        const std::string type = bytes.substr(offset + 4, 4);
        if (first) {
            if (type != "IHDR" || length != 13) return false;
            const auto width = png_u32(bytes, offset + 8), height = png_u32(bytes, offset + 12);
            const unsigned char depth = static_cast<unsigned char>(bytes[offset + 16]);
            const unsigned char color = static_cast<unsigned char>(bytes[offset + 17]);
            if (width == 0 || height == 0 || width > 4096 || height > 4096 ||
                    (depth != 1 && depth != 2 && depth != 4 && depth != 8 && depth != 16) ||
                    (color != 0 && color != 2 && color != 3 && color != 4 && color != 6) ||
                    bytes[offset + 18] != 0 || bytes[offset + 19] != 0 ||
                    (bytes[offset + 20] != 0 && bytes[offset + 20] != 1)) return false;
            first = false;
        } else if (type == "IHDR") return false;
        if (type == "IDAT") has_idat = true;
        offset += static_cast<size_t>(length) + 12;
        if (type == "IEND") { has_iend = length == 0 && offset == bytes.size(); break; }
    }
    return !first && has_idat && has_iend;
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
    common_agent_visualization_relationship empty_spec;
    empty_spec.title = title;
    empty_spec.view.layout = layout;
    if (!common_agent_visualization_relationship_validate(empty_spec, error))
        return common_adapter_validation_failure("tool.visualization.create.invalid_arguments", std::move(error));
    if (bindings.resource_runtime.store == nullptr)
        return common_adapter_execution_failure("tool.visualization.create.store_unavailable",
            "resource store is unavailable", "The visualization could not be saved.");
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
    if (response.size() > COMMON_AGENT_VISUALIZATION_MAX_BYTES)
        return common_adapter_limit_failure("tool.visualization.create.source_byte_limit",
            "dataset page exceeded the visualization input byte limit", "The dataset page is too large to visualize.");
    const auto page = json::parse(response, nullptr, false);
    if (!page.is_object() || !page.contains("columns") || !page["columns"].is_array() ||
            !page.contains("rows") || !page["rows"].is_array())
        return common_adapter_execution_failure("tool.visualization.create.invalid_dataset_result",
            "data.query returned an invalid positional page", "The dataset rows could not be projected.");
    for (const char * flag : {"scan_truncated", "result_truncated", "complete", "has_more"}) {
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
        const auto name = column.get<std::string>();
        if (std::find(returned_columns.begin(), returned_columns.end(), name) != returned_columns.end())
            return common_adapter_execution_failure("tool.visualization.create.invalid_dataset_result",
                "data.query returned duplicate column names", "The dataset rows could not be projected.");
        returned_columns.push_back(name);
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
    size_t consumed_rows = 0;
    for (const auto & row : page["rows"]) {
        // The lookahead row detects truncation; it is not part of the projection.
        if (consumed_rows++ == max_rows) break;
        if (!row.is_array() || row.size() != returned_columns.size()) {
            ++invalid_rows;
            continue;
        }
        const auto at = [&](size_t index) -> std::string {
            return index < row.size() ? scalar_text(row[index]) : std::string();
        };
        const std::string source_value = at(source_index), target_value = at(target_index);
        if (source_value.empty() || target_value.empty()) { ++invalid_rows; continue; }
        // Preserve scalar type, dataset domain and kind without delimiter collisions.
        const auto node_id = [&](size_t index, const std::string & kind) {
            const auto key = json::array({dataset_uri, kind, row[index]}).dump();
            return "node:" + hash_sha256_hex(key.data(), key.size());
        };
        const std::string source_id = node_id(source_index, source_kind);
        const std::string target_id = node_id(target_index, target_kind);
        std::string source_label = at(source_label_index).empty() ? source_value : at(source_label_index);
        std::string target_label = at(target_label_index).empty() ? target_value : at(target_label_index);
        label_truncated = shorten_label(source_label) || label_truncated;
        label_truncated = shorten_label(target_label) || label_truncated;
        const auto add_node = [&](const std::string & id, const std::string & label, const std::string & kind) {
            const auto inserted = nodes.emplace(id, common_agent_visualization_node{id, label, kind});
            if (!inserted.second && label < inserted.first->second.label) inserted.first->second.label = label;
        };
        add_node(source_id, source_label, source_kind);
        add_node(target_id, target_label, target_kind);
        const std::string edge_label = at(edge_label_index);
        ++edges[{source_id, target_id, edge_label}];
    }
    const bool source_truncated = page.value("scan_truncated", false) || page.value("result_truncated", false) ||
        page.value("has_more", false);
    const bool source_incomplete = !page.value("complete", true);
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
    if (source_incomplete) spec.limitations.push_back("The source reported incomplete query coverage.");
    if (descriptor.ref.count_kind != common_agent_dataset_count_kind::exact)
        spec.limitations.push_back("Dataset total coverage is not exact; this is a bounded projection.");
    spec.complete = !spec.truncated && !source_incomplete && invalid_rows == 0 &&
        descriptor.ref.count_kind == common_agent_dataset_count_kind::exact;
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
        label_truncated = shorten_label(bounded_edge_label) || label_truncated;
        const auto serialized_key = json::array({source, target, label}).dump();
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

common_tool_execution_result open_relationship(
        const common_native_tool_bindings & bindings, const std::string & input) {
    json arguments;
    std::string error;
    if (!common_adapter_parse_object(input, arguments, error))
        return common_adapter_validation_failure("tool.visualization.open.invalid_arguments", std::move(error));
    if (arguments.size() != 1 || !arguments.contains("visualization_uri") ||
            !arguments["visualization_uri"].is_string())
        return common_adapter_validation_failure("tool.visualization.open.invalid_arguments",
            "visualization.open requires only a visualization_uri string");
    const auto uri = arguments["visualization_uri"].get<std::string>();
    if (uri.empty() || uri.size() > 512)
        return common_adapter_validation_failure("tool.visualization.open.invalid_arguments",
            "visualization_uri is empty or exceeds its size limit");
    auto * store = bindings.resource_runtime.store;
    if (store == nullptr)
        return common_adapter_execution_failure("tool.visualization.open.store_unavailable",
            "resource store is unavailable", "The saved visualization could not be opened.");

    agent_resource_descriptor descriptor;
    const auto authority = make_agent_resource_read_authority(
        bindings.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
    if (!store->stat(uri, authority, descriptor, error))
        return common_adapter_not_found_failure("tool.visualization.open.unavailable", std::move(error),
            "The visualization is unavailable in the current runtime scope.");
    if (descriptor.mime_type != "application/vnd.llama-agent.visualization+json")
        return common_adapter_validation_failure("tool.visualization.open.wrong_resource_type",
            "the selected resource is not a relationship visualization");
    std::string serialized;
    if (!store->read_text(descriptor.uri, authority, COMMON_AGENT_VISUALIZATION_MAX_BYTES,
            serialized, error))
        return common_adapter_execution_failure("tool.visualization.open.read_failed", std::move(error),
            "The saved visualization could not be read.");
    if (serialized.size() > COMMON_AGENT_VISUALIZATION_MAX_BYTES)
        return common_adapter_limit_failure("tool.visualization.open.byte_limit",
            "saved visualization exceeded its byte limit", "The saved visualization is too large to open.");
    const auto value = nlohmann::json::parse(serialized, nullptr, false);
    common_agent_visualization_relationship spec;
    if (value.is_discarded() || !common_agent_visualization_relationship_from_json(value, spec, error))
        return common_adapter_execution_failure("tool.visualization.open.invalid_resource",
            error.empty() ? "saved visualization is not valid JSON" : std::move(error),
            "The saved visualization is invalid and cannot be rendered.");
    return common_adapter_success_json({{"visualization_uri", descriptor.uri},
        {"visualization", common_agent_visualization_relationship_to_json(spec)},
        {"complete", spec.complete}, {"truncated", spec.truncated}, {"limitations", spec.limitations}});
}

common_tool_execution_result save_relationship_snapshot(
        const common_native_tool_bindings & bindings, const std::string & input) {
    json arguments;
    std::string error;
    if (!common_adapter_parse_object(input, arguments, error))
        return common_adapter_validation_failure("tool.visualization.snapshot.invalid_arguments", std::move(error));
    static const std::set<std::string> allowed{"visualization_uri", "visualization", "image_base64", "title", "view_json"};
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
        if (allowed.count(it.key()) == 0)
            return common_adapter_validation_failure("tool.visualization.snapshot.invalid_arguments",
                "unsupported visualization.snapshot argument: " + it.key());
    }
    std::string uri, encoded, title, view_json;
    if (!read_field(arguments, "visualization_uri", uri, false, error, 512) ||
            !read_field(arguments, "image_base64", encoded, true, error, 699052) ||
            !read_field(arguments, "title", title, false, error, 128) ||
            !read_field(arguments, "view_json", view_json, false, error, 8192))
        return common_adapter_validation_failure("tool.visualization.snapshot.invalid_arguments", std::move(error));
    if (arguments.contains("visualization_uri") && uri.empty())
        return common_adapter_validation_failure("tool.visualization.snapshot.invalid_arguments",
            "visualization_uri cannot be empty when supplied");
    if (bindings.resource_runtime.store == nullptr)
        return common_adapter_execution_failure("tool.visualization.snapshot.store_unavailable",
            "resource store is unavailable", "The visualization snapshot could not be saved.");

    auto * store = bindings.resource_runtime.store;
    const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
        static_cast<int64_t>(std::time(nullptr)));
    const bool has_uri = !uri.empty();
    const bool has_inline_projection = arguments.contains("visualization");
    if (has_uri == has_inline_projection)
        return common_adapter_validation_failure("tool.visualization.snapshot.source_invalid",
            "provide exactly one of visualization_uri or visualization");
    agent_resource_descriptor source;
    json parsed;
    if (has_uri) {
        if (has_inline_projection)
            return common_adapter_validation_failure("tool.visualization.snapshot.source_invalid",
                "provide exactly one of visualization_uri or visualization");
        if (!store->stat(uri, authority, source, error))
            return common_adapter_not_found_failure("tool.visualization.snapshot.source_unavailable", std::move(error),
                "The source visualization is unavailable in the current runtime scope.");
        if (source.mime_type != "application/vnd.llama-agent.visualization+json")
            return common_adapter_validation_failure("tool.visualization.snapshot.source_type_invalid",
                "the selected resource is not a saved relationship visualization");
        std::string source_text;
        if (!store->read_text(source.uri, authority, COMMON_AGENT_VISUALIZATION_MAX_BYTES, source_text, error))
            return common_adapter_execution_failure("tool.visualization.snapshot.source_read_failed", std::move(error),
                "The source visualization could not be verified.");
        if (!source.sha256.empty() && source.sha256 != hash_sha256_hex(source_text.data(), source_text.size()))
            return common_adapter_execution_failure("tool.visualization.snapshot.source_hash_mismatch",
                "saved visualization bytes did not match the resource descriptor hash",
                "The source visualization failed integrity validation.");
        parsed = json::parse(source_text, nullptr, false);
    } else {
        parsed = arguments["visualization"];
        if (!parsed.is_object() || parsed.dump().size() > COMMON_AGENT_VISUALIZATION_MAX_BYTES)
            return common_adapter_validation_failure("tool.visualization.snapshot.source_invalid",
                "visualization must be a bounded relationship specification");
    }
    common_agent_visualization_relationship spec;
    if (parsed.is_discarded() || !common_agent_visualization_relationship_from_json(parsed, spec, error))
        return common_adapter_execution_failure("tool.visualization.snapshot.source_invalid",
            error.empty() ? "saved visualization is invalid" : std::move(error),
            "The source visualization could not be verified.");
    if (!view_json.empty()) {
        const auto view = json::parse(view_json, nullptr, false);
        if (!view.is_object() || view.dump().size() > 8192)
            return common_adapter_validation_failure("tool.visualization.snapshot.view_invalid",
                "view_json must be a bounded JSON object");
    }
    std::string bytes;
    try { bytes = base64::decode(encoded); }
    catch (...) { return common_adapter_validation_failure("tool.visualization.snapshot.image_invalid",
        "image_base64 is not valid base64"); }
    constexpr size_t max_snapshot_bytes = 512 * 1024;
    if (bytes.size() > max_snapshot_bytes || !bounded_png(bytes))
        return common_adapter_validation_failure("tool.visualization.snapshot.image_invalid",
            "snapshot must be a structurally valid PNG with dimensions up to 4096 pixels per side and size up to 512 KiB");

    if (!has_uri) {
        const auto serialized = common_agent_visualization_relationship_to_json(spec).dump();
        agent_resource_put_request graph_put;
        graph_put.name = title.empty() ? spec.title : title;
        graph_put.description = "Validated relationship projection supplied for a persistent PNG snapshot.";
        graph_put.mime_type = "application/vnd.llama-agent.visualization+json";
        graph_put.text = serialized;
        graph_put.scope = bindings.resource_runtime.project_id.empty()
            ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        graph_put.source_provider = "native";
        graph_put.source_tool = "visualization.snapshot";
        graph_put.metadata.purpose = "Validated graph projection for a snapshot; client supplied, not independent evidence.";
        graph_put.metadata.content_hash = "sha256:" + hash_sha256_hex(serialized.data(), serialized.size());
        graph_put.lineage.chunk_count = 1;
        graph_put.lineage.byte_length = serialized.size();
        graph_put.lineage.derivation = "visualization.snapshot:client-projection-v1";
        apply_agent_resource_runtime(bindings.resource_runtime, graph_put);
        if (!store->put_text(graph_put, source, error))
            return common_adapter_execution_failure("tool.visualization.snapshot.source_store_failed", std::move(error),
                "The validated graph projection could not be persisted.");
        uri = source.uri;
    }

    const std::string fingerprint = "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
    json provenance{{"visualization_uri", source.uri}, {"data_revision", spec.data_revision},
        {"visualization_fingerprint", source.sha256}, {"image_fingerprint", fingerprint},
        {"renderer", "cytoscape-relationship-v1"}};
    if (!view_json.empty()) provenance["view"] = json::parse(view_json);
    agent_resource_put_request put;
    put.name = title.empty() ? spec.title + ".png" : title + ".png";
    put.description = "Immutable PNG snapshot of a saved relationship visualization.";
    put.mime_type = "image/png";
    put.bytes = std::move(bytes);
    put.bytes_are_authoritative = true;
    put.scope = bindings.resource_runtime.project_id.empty()
        ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
    put.source_provider = "native";
    put.source_tool = "visualization.snapshot";
    put.metadata.purpose = "Frozen visual presentation of a saved relationship visualization.";
    put.metadata.content_summary = "Snapshot of " + spec.title + " at data revision " + spec.data_revision;
    put.metadata.source_operation = "visualization.snapshot";
    put.metadata.source_request_json = provenance.dump();
    put.metadata.content_hash = fingerprint;
    put.lineage.parent_uri = source.uri;
    put.lineage.chunk_count = 1;
    put.lineage.byte_length = put.bytes.size();
    put.lineage.derivation = "visualization.snapshot:png-v1";
    apply_agent_resource_runtime(bindings.resource_runtime, put);
    agent_resource_descriptor snapshot;
    if (!store->put_bytes(put, snapshot, error))
        return common_adapter_execution_failure("tool.visualization.snapshot.store_failed", std::move(error),
            "The PNG snapshot could not be persisted.");
    return common_adapter_success_json({{"snapshot_uri", snapshot.uri}, {"visualization_uri", source.uri},
        {"data_revision", spec.data_revision}, {"content_fingerprint", fingerprint},
        {"mime_type", snapshot.mime_type}, {"size_bytes", snapshot.size_bytes}});
}
}

bool common_try_register_visualization_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    if (definition.executor_id == "builtin.visualization.create") {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            return create_relationship(bindings, input);
        }, error, false, definition.requires_confirmation);
        return true;
    }
    if (definition.executor_id == "builtin.visualization.open") {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            return open_relationship(bindings, input);
        }, error);
        return true;
    }
    if (definition.executor_id == "builtin.visualization.snapshot") {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            return save_relationship_snapshot(bindings, input);
        }, error, false, definition.requires_confirmation);
        return true;
    }
    return false;
}
