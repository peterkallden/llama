#include "agent-data-store-cozo.h"
#include "agent-data-store-cozo-aggregate.h"
#include "agent-data-store-cozo-encoding.h"
#include "agent-data-store-cozo-join.h"
#include "agent-data-store-cozo-reader.h"
#include "agent-data-store-cozo-query.h"
#include "agent-data-store-cozo-schema.h"

extern "C" {
#include <cozo_c.h>
}
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>

using json = nlohmann::ordered_json;

namespace {

std::string normalized_dataset_name(const std::string & value) {
    std::string result;
    for (const unsigned char ch : value) {
        if (!std::isspace(ch)) result += static_cast<char>(std::tolower(ch));
    }
    return result;
}

common_agent_dataset_column_type infer_dataset_column_type(const json & value) {
    if (value.is_null()) return common_agent_dataset_column_type::null_;
    if (value.is_boolean()) return common_agent_dataset_column_type::boolean;
    if (value.is_number_integer()) return common_agent_dataset_column_type::integer;
    if (value.is_number_float()) return common_agent_dataset_column_type::decimal;
    if (value.is_string()) return common_agent_dataset_column_type::string;
    return common_agent_dataset_column_type::unknown;
}

bool materialize_data_result(
        common_agent_cozo_data_store & store,
        const std::string & operation,
        const json & request,
        std::string & result_json,
        std::string & error) {
    if (!request.value("materialize", false)) return true;
    if (!request.contains("result_dataset") || !request["result_dataset"].is_string() ||
            request["result_dataset"].get<std::string>().empty()) {
        error = "materialized data operation requires result_dataset";
        return false;
    }
    const auto result = json::parse(result_json, nullptr, false);
    if (!result.is_object() || result.value("scan_truncated", false) || result.value("result_truncated", false)) {
        error = "cannot materialize a truncated data operation result";
        return false;
    }
    if (!result.contains("rows") || !result["rows"].is_array()) {
        error = "data operation did not produce materializable rows";
        return false;
    }
    std::vector<std::string> parents;
    if (operation == "data.join") {
        parents = {request.value("left", std::string()), request.value("right", std::string())};
    } else {
        parents = {request.value("dataset", std::string())};
    }
    common_agent_dataset_descriptor first_parent;
    if (parents.empty() || parents.front().empty() || !store.get_dataset_descriptor(parents.front(), first_parent, error)) {
        if (error.empty()) error = "materialized data operation requires parent dataset metadata";
        return false;
    }
    common_agent_dataset_descriptor descriptor;
    descriptor.ref.uri = request["result_dataset"].get<std::string>();
    descriptor.ref.name = descriptor.ref.uri.substr(descriptor.ref.uri.find_last_of('/') + 1);
    descriptor.ref.row_count = result["rows"].size();
    descriptor.ref.source_resource_uri = first_parent.ref.source_resource_uri;
    descriptor.ref.source_representation = "derived-dataset";
    descriptor.lineage.parent_dataset_uris = parents;
    descriptor.lineage.operation = operation;
    descriptor.lineage.operation_summary = "Materialized bounded result of " + operation;
    descriptor.origin.kind = "derived";
    descriptor.origin.source_representation_uri = first_parent.origin.source_representation_uri;
    descriptor.origin.source_node_id = first_parent.origin.source_node_id;
    descriptor.origin.table_index = first_parent.origin.table_index;
    descriptor.origin.section_path = first_parent.origin.section_path;
    descriptor.origin.caption = first_parent.origin.caption;
    descriptor.origin.notes = first_parent.origin.notes;
    descriptor.import_processor_id = "cozo-data-operation-v1";
    descriptor.import_processor_version = "1";
    std::vector<std::string> column_names;
    if (result.contains("columns") && result["columns"].is_array()) {
        for (const auto & column : result["columns"]) if (column.is_string()) column_names.push_back(column.get<std::string>());
    }
    std::set<std::string> column_set(column_names.begin(), column_names.end());
    for (const auto & row : result["rows"]) if (row.is_object()) for (auto it = row.begin(); it != row.end(); ++it) column_set.insert(it.key());
    if (column_set.empty()) for (const auto & column : first_parent.columns) column_set.insert(column.name);
    column_names.assign(column_set.begin(), column_set.end());
    for (const auto & name : column_names) {
        common_agent_dataset_column column;
        column.name = name;
        column.type = common_agent_dataset_column_type::unknown;
        for (size_t row_index = 0; row_index < result["rows"].size(); ++row_index) {
            const auto & row = result["rows"][row_index];
            json cell;
            if (row.is_object() && row.contains(name)) cell = row[name];
            else if (row.is_array() && result.contains("columns") && result["columns"].is_array()) {
                const auto column_index = std::find(column_names.begin(), column_names.end(), name);
                if (column_index != column_names.end()) {
                    const auto index = static_cast<size_t>(std::distance(column_names.begin(), column_index));
                    if (index < row.size()) cell = row[index];
                }
            }
            const auto type = infer_dataset_column_type(cell);
            if (type != common_agent_dataset_column_type::null_) { column.type = type; break; }
        }
        descriptor.columns.push_back(std::move(column));
    }
    descriptor.ref.column_count = descriptor.columns.size();
    if (!validate_common_agent_dataset_descriptor(descriptor, common_agent_dataset_limits{}, error)) return false;
    for (size_t index = 0; index < result["rows"].size(); ++index) {
        json row = result["rows"][index];
        if (row.is_array()) {
            json object = json::object();
            for (size_t column_index = 0; column_index < row.size() && column_index < column_names.size(); ++column_index) object[column_names[column_index]] = row[column_index];
            row = std::move(object);
        }
        if (!row.is_object() || !store.put_row(descriptor.ref.uri, std::to_string(index), row.dump(), error)) return false;
    }
    if (!store.put_dataset_descriptor(descriptor, error)) return false;
    result_json = json({
        {"dataset", descriptor.ref.uri}, {"name", descriptor.ref.name},
        {"rows", descriptor.ref.row_count}, {"columns", descriptor.ref.column_count},
        {"source", descriptor.ref.source_resource_uri},
        {"lineage", {{"parents", descriptor.lineage.parent_dataset_uris},
                      {"operation", descriptor.lineage.operation}}},
        {"materialized", true}}).dump();
    return true;
}
}

common_agent_cozo_data_store::~common_agent_cozo_data_store() { close(); }

bool common_agent_cozo_data_store::run(const std::string & script, const std::string & params_json, std::string & result_json, std::string & error) const {
    if (db_id_ < 0) { error = "Cozo data store is not open"; return false; }
    char * result = cozo_run_query(db_id_, script.c_str(), params_json.empty() ? "{}" : params_json.c_str(), false);
    if (!result) { error = "Cozo query failed without diagnostic output"; return false; }
    result_json = result; cozo_free_str(result);
    const auto parsed = json::parse(result_json, nullptr, false);
    if (parsed.is_object() && parsed.value("ok", true) == false) { error = parsed.value("message", std::string("Cozo query failed")); return false; }
    return true;
}

bool common_agent_cozo_data_store::open(const std::string & path, std::string & error) {
    close();
    char * open_error = cozo_open_db("sqlite", path.c_str(), "{}", &db_id_);
    if (open_error) { error = open_error; cozo_free_str(open_error); db_id_ = -1; return false; }
    std::string relations;
    if (!run("::relations", "{}", relations, error)) { close(); return false; }
    const auto parsed = json::parse(relations, nullptr, false);
    bool rows_present = false, values_present = false, order_present = false, dataset_present = false;
    if (parsed.is_object() && parsed.contains("rows")) for (const auto & row : parsed["rows"]) if (row.is_array() && !row.empty() && row[0].is_string()) {
        rows_present |= row[0] == "agent_data_rows";
        values_present |= row[0] == "agent_data_values";
        order_present |= row[0] == "agent_data_row_order";
        dataset_present |= row[0] == "agent_dataset_metadata";
    }
    if (!rows_present) { std::string ignored; if (!run(agent_cozo_schema_script(), "{}", ignored, error)) { close(); return false; } }
    else if (!values_present) { std::string ignored; if (!run(agent_cozo_values_schema_script(), "{}", ignored, error)) { close(); return false; } }
    if (rows_present && !order_present) { std::string ignored; if (!run(agent_cozo_order_schema_script(), "{}", ignored, error)) { close(); return false; } }
    if (rows_present && !order_present) {
        std::string raw;
        if (!run("?[dataset, row_id] := *agent_data_rows[dataset, row_id, row_json]", "{}", raw, error)) { close(); return false; }
        const auto existing = json::parse(raw, nullptr, false);
        if (!existing.is_object() || !existing.contains("rows")) { error = "Cozo row-order migration returned invalid rows"; close(); return false; }
        std::map<std::string, int64_t> next_sequence;
        json order_rows = json::array();
        for (const auto & row : existing["rows"]) if (row.is_array() && row.size() >= 2 && row[0].is_string() && row[1].is_string()) {
            const auto dataset = row[0].get<std::string>();
            order_rows.push_back({dataset, row[1].get<std::string>(), ++next_sequence[dataset]});
        }
        if (!order_rows.empty() && !run("?[dataset, row_id, row_seq] <- $rows :put agent_data_row_order { dataset, row_id => row_seq }", json({{"rows", order_rows}}).dump(), raw, error)) { close(); return false; }
    }
    if (!dataset_present) { std::string ignored; if (!run(agent_cozo_dataset_schema_script(), "{}", ignored, error)) { close(); return false; } }
    return true;
}

void common_agent_cozo_data_store::close() { if (db_id_ >= 0) { cozo_close_db(db_id_); db_id_ = -1; } }

bool common_agent_cozo_data_store::put_row(const std::string & dataset, const std::string & row_id, const std::string & row_json, std::string & error) {
    const auto parsed = json::parse(row_json, nullptr, false);
    if (!parsed.is_object() || dataset.empty() || row_id.empty()) { error = "data row requires dataset, row id and JSON object"; return false; }
    const json params = {{"dataset", dataset}, {"row_id", row_id}, {"row_json", row_json}};
    std::string result;
    if (!run("?[dataset, row_id, row_json] <- [[$dataset, $row_id, $row_json]] :put agent_data_rows { dataset, row_id => row_json }", params.dump(), result, error)) return false;
    std::string order_raw;
    if (!run("?[row_seq] := *agent_data_row_order[$dataset, $row_id, row_seq]", json({{"dataset", dataset}, {"row_id", row_id}}).dump(), order_raw, error)) return false;
    const auto existing_order = json::parse(order_raw, nullptr, false);
    if (!existing_order.is_object() || !existing_order.contains("rows")) { error = "Cozo row-order query returned invalid rows"; return false; }
    if (existing_order["rows"].empty()) {
        std::string max_raw;
        if (!run("?[max(row_seq)] := *agent_data_row_order[$dataset, row_id, row_seq]", json({{"dataset", dataset}}).dump(), max_raw, error)) return false;
        const auto maximum = json::parse(max_raw, nullptr, false);
        int64_t next_sequence = 1;
        if (maximum.is_object() && maximum.contains("rows") && !maximum["rows"].empty() && !maximum["rows"][0].empty() && maximum["rows"][0][0].is_number()) next_sequence += maximum["rows"][0][0].get<int64_t>();
        if (!run("?[dataset, row_id, row_seq] <- [[$dataset, $row_id, $row_seq]] :put agent_data_row_order { dataset, row_id => row_seq }", json({{"dataset", dataset}, {"row_id", row_id}, {"row_seq", next_sequence}}).dump(), result, error)) return false;
    }
    const auto values = agent_cozo_encode_row_values(dataset, row_id, parsed);
    if (!values.empty() && !run("?[dataset, row_id, field, value_kind, value_text, value_number] <- $rows :put agent_data_values { dataset, row_id, field => value_kind, value_text, value_number }", json({{"rows", values}}).dump(), result, error)) return false;
    return true;
}

bool common_agent_cozo_data_store::put_dataset_descriptor(
        const common_agent_dataset_descriptor & descriptor,
        std::string & error) {
    if (!validate_common_agent_dataset_descriptor(descriptor, common_agent_dataset_limits{}, error)) return false;
    json columns = json::array();
    for (const auto & column : descriptor.columns) columns.push_back({
        {"name", column.name}, {"type", common_agent_dataset_column_type_name(column.type)},
        {"nullable", column.nullable}});
    const json value = {
        {"uri", descriptor.ref.uri}, {"name", descriptor.ref.name},
        {"row_count", descriptor.ref.row_count}, {"column_count", descriptor.ref.column_count},
        {"source_resource_uri", descriptor.ref.source_resource_uri},
        {"source_representation", descriptor.ref.source_representation},
        {"source_provider", descriptor.ref.source_provider},
        {"source_operation", descriptor.ref.source_operation},
        {"source_request_json", descriptor.ref.source_request_json},
        {"retrieved_at", descriptor.ref.retrieved_at},
        {"content_hash", descriptor.ref.content_hash},
        {"columns", columns}, {"source_workbook_name", descriptor.source_workbook_name},
        {"source_sheet_name", descriptor.source_sheet_name},
        {"source_sheet_index", descriptor.source_sheet_index ? json(*descriptor.source_sheet_index) : json()},
        {"source_range", descriptor.source_range}, {"source_object", descriptor.source_object},
        {"import_processor_id", descriptor.import_processor_id},
        {"import_processor_version", descriptor.import_processor_version},
        {"origin_kind", descriptor.origin.kind},
        {"origin_source_representation_uri", descriptor.origin.source_representation_uri},
        {"origin_source_node_id", descriptor.origin.source_node_id},
        {"origin_table_index", descriptor.origin.table_index},
        {"origin_section_path", descriptor.origin.section_path},
        {"origin_caption", descriptor.origin.caption},
        {"origin_notes", descriptor.origin.notes},
        {"origin_header_mode", common_agent_table_header_mode_name(descriptor.origin.header_mode)},
        {"origin_header_confidence", descriptor.origin.header_confidence},
        {"origin_header_reason", descriptor.origin.header_reason},
        {"parent_dataset_uris", descriptor.lineage.parent_dataset_uris},
        {"operation", descriptor.lineage.operation},
        {"operation_summary", descriptor.lineage.operation_summary},
    };
    std::string result;
    return run("?[dataset_uri, descriptor_json] <- [[$dataset_uri, $descriptor_json]] :put agent_dataset_metadata { dataset_uri => descriptor_json }",
        json({{"dataset_uri", descriptor.ref.uri}, {"descriptor_json", value.dump()}}).dump(), result, error);
}

bool common_agent_cozo_data_store::get_dataset_descriptor(
        const std::string & dataset_uri,
        common_agent_dataset_descriptor & descriptor,
        std::string & error) {
    descriptor = {};
    if (dataset_uri.empty()) { error = "dataset uri must not be empty"; return false; }
    std::string raw;
    if (!run("?[descriptor_json] := *agent_dataset_metadata[$dataset_uri, descriptor_json]",
            json({{"dataset_uri", dataset_uri}}).dump(), raw, error)) return false;
    const auto result = json::parse(raw, nullptr, false);
    if (!result.is_object() || !result.contains("rows") || result["rows"].empty() ||
            !result["rows"][0].is_array() || result["rows"][0].empty()) {
        error = "dataset descriptor was not found";
        return false;
    }
    const auto value = json::parse(result["rows"][0][0].get<std::string>(), nullptr, false);
    if (!value.is_object()) { error = "dataset descriptor is invalid"; return false; }
    descriptor.ref.uri = value.value("uri", dataset_uri);
    descriptor.ref.name = value.value("name", std::string());
    descriptor.ref.row_count = value.value("row_count", size_t(0));
    descriptor.ref.column_count = value.value("column_count", size_t(0));
    descriptor.ref.source_resource_uri = value.value("source_resource_uri", std::string());
    descriptor.ref.source_representation = value.value("source_representation", std::string());
    descriptor.ref.source_provider = value.value("source_provider", std::string());
    descriptor.ref.source_operation = value.value("source_operation", std::string());
    descriptor.ref.source_request_json = value.value("source_request_json", std::string());
    descriptor.ref.retrieved_at = value.value("retrieved_at", int64_t(0));
    descriptor.ref.content_hash = value.value("content_hash", std::string());
    for (const auto & column : value.value("columns", json::array())) if (column.is_object()) {
        const auto type = column.value("type", std::string("unknown"));
        common_agent_dataset_column_type column_type = common_agent_dataset_column_type::unknown;
        for (int i = 0; i <= static_cast<int>(common_agent_dataset_column_type::unknown); ++i) {
            const auto candidate = static_cast<common_agent_dataset_column_type>(i);
            if (type == common_agent_dataset_column_type_name(candidate)) { column_type = candidate; break; }
        }
        descriptor.columns.push_back({column.value("name", std::string()), column_type, column.value("nullable", true)});
    }
    descriptor.source_workbook_name = value.value("source_workbook_name", std::string());
    descriptor.source_sheet_name = value.value("source_sheet_name", std::string());
    if (value.contains("source_sheet_index") && value["source_sheet_index"].is_number_unsigned()) descriptor.source_sheet_index = value["source_sheet_index"].get<size_t>();
    descriptor.source_range = value.value("source_range", std::string());
    descriptor.source_object = value.value("source_object", std::string());
    descriptor.import_processor_id = value.value("import_processor_id", std::string());
    descriptor.import_processor_version = value.value("import_processor_version", std::string());
    descriptor.origin.kind = value.value("origin_kind", std::string());
    descriptor.origin.source_representation_uri = value.value("origin_source_representation_uri", std::string());
    descriptor.origin.source_node_id = value.value("origin_source_node_id", std::string());
    descriptor.origin.table_index = value.value("origin_table_index", size_t(0));
    descriptor.origin.section_path = value.value("origin_section_path", std::vector<std::string>());
    descriptor.origin.caption = value.value("origin_caption", std::string());
    descriptor.origin.notes = value.value("origin_notes", std::vector<std::string>());
    const auto origin_header_mode = value.value("origin_header_mode", std::string("none"));
    if (origin_header_mode == "explicit") descriptor.origin.header_mode = common_agent_table_header_mode::explicit_;
    else if (origin_header_mode == "first_row") descriptor.origin.header_mode = common_agent_table_header_mode::first_row;
    else if (origin_header_mode == "first_column") descriptor.origin.header_mode = common_agent_table_header_mode::first_column;
    else if (origin_header_mode == "both") descriptor.origin.header_mode = common_agent_table_header_mode::both;
    else if (origin_header_mode == "ambiguous") descriptor.origin.header_mode = common_agent_table_header_mode::ambiguous;
    descriptor.origin.header_confidence = value.value("origin_header_confidence", 0.0);
    descriptor.origin.header_reason = value.value("origin_header_reason", std::string());
    descriptor.lineage.parent_dataset_uris = value.value("parent_dataset_uris", std::vector<std::string>());
    descriptor.lineage.operation = value.value("operation", std::string());
    descriptor.lineage.operation_summary = value.value("operation_summary", std::string());
    return validate_common_agent_dataset_descriptor(descriptor, common_agent_dataset_limits{}, error);
}

bool common_agent_cozo_data_store::list_dataset_descriptors(
        std::vector<common_agent_dataset_descriptor> & descriptors,
        std::string & error) {
    descriptors.clear();
    std::string raw;
    if (!run("?[dataset_uri] := *agent_dataset_metadata[dataset_uri, descriptor_json]", "{}", raw, error)) return false;
    const auto result = json::parse(raw, nullptr, false);
    if (!result.is_object() || !result.contains("rows")) {
        error = "dataset descriptor listing returned invalid data";
        return false;
    }
    std::vector<std::string> uris;
    for (const auto & row : result["rows"]) {
        if (row.is_array() && !row.empty() && row[0].is_string()) uris.push_back(row[0].get<std::string>());
    }
    std::sort(uris.begin(), uris.end());
    for (const auto & uri : uris) {
        common_agent_dataset_descriptor descriptor;
        if (!get_dataset_descriptor(uri, descriptor, error)) return false;
        descriptors.push_back(std::move(descriptor));
    }
    error.clear();
    return true;
}

bool common_agent_cozo_data_store::find_dataset_by_name(
        const std::string & name,
        common_agent_dataset_descriptor & descriptor,
        std::string & error) {
    descriptor = {};
    std::vector<common_agent_dataset_descriptor> descriptors;
    if (!list_dataset_descriptors(descriptors, error)) return false;
    std::string wanted;
    for (const char ch : name) if (!std::isspace(static_cast<unsigned char>(ch))) wanted += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (wanted.empty()) { error = "dataset name must not be empty"; return false; }
    const common_agent_dataset_descriptor * match = nullptr;
    for (const auto & candidate : descriptors) {
        std::string candidate_name;
        for (const char ch : candidate.ref.name) if (!std::isspace(static_cast<unsigned char>(ch))) candidate_name += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (candidate_name != wanted) continue;
        if (match != nullptr) {
            error = "dataset name is ambiguous; choose one of: " + match->ref.name + " (" + match->ref.uri + ")";
            for (const auto & option : descriptors) {
                if (normalized_dataset_name(option.ref.name) == wanted && option.ref.uri != match->ref.uri) {
                    error += ", " + option.ref.name + " (" + option.ref.uri + ")";
                }
            }
            return false;
        }
        match = &candidate;
    }
    if (match == nullptr) { error = "dataset name was not found"; return false; }
    descriptor = *match;
    error.clear();
    return true;
}

bool common_agent_cozo_data_store::execute(const std::string & operation, const std::string & request_json, std::string & result_json, std::string & error) {
    const auto request = json::parse(request_json, nullptr, false);
    if (!request.is_object()) { error = "data operation requires an object request"; return false; }
    const size_t max_scan_rows = std::min<size_t>(request.value("max_scan_rows", 10000), 100000);
    const size_t max_result_rows = std::min<size_t>(request.value("max_result_rows", 1000), 10000);
    if (max_scan_rows == 0 || max_result_rows == 0) { error = "max_scan_rows and max_result_rows must be greater than zero"; return false; }
    const bool is_join = operation == "data.join";
    if (is_join) {
        if (!request.contains("left") || !request["left"].is_string() || !request.contains("right") || !request["right"].is_string() || !request.contains("on") || !request["on"].is_array()) {
            error = "data.join requires left, right and on"; return false;
        }
    } else if (!request.contains("dataset") || !request["dataset"].is_string()) {
        error = operation + " requires dataset"; return false;
    }
    const agent_cozo_query_runner run_query = [this](
            const std::string & script,
            const std::string & params_json,
            std::string & raw,
            std::string & query_error) {
        return run(script, params_json, raw, query_error);
    };
    const agent_cozo_scan_metadata_reader read_metadata = [&run_query](
            const std::string & dataset_name,
            size_t scan_limit,
            size_t & scanned,
            bool & truncated,
            std::string & metadata_error) {
        return agent_cozo_read_scan_metadata(dataset_name, scan_limit, scanned, truncated, run_query, metadata_error);
    };
    if (operation == "data.aggregate") {
        if (!agent_cozo_execute_native_aggregate(request, max_scan_rows, max_result_rows, run_query, read_metadata, result_json, error)) return false;
        return materialize_data_result(*this, operation, request, result_json, error);
    }
    if (operation == "data.join") {
        if (!agent_cozo_execute_native_join(request, max_scan_rows, max_result_rows, run_query, read_metadata, result_json, error)) return false;
        return materialize_data_result(*this, operation, request, result_json, error);
    }

    std::vector<json> rows;
    size_t scanned_rows = 0;
    bool scan_truncated = false;
    if (!is_join) {
        const auto conditions = operation == "data.filter" ? request.value("conditions", json::array()) : request.value("where", json::array());
        bool loaded = conditions.is_array() && !conditions.empty()
            ? agent_cozo_read_dataset_with_conditions(request["dataset"].get<std::string>(), conditions, max_scan_rows, rows, scanned_rows, scan_truncated, run_query, error)
            : agent_cozo_read_dataset(request["dataset"].get<std::string>(), max_scan_rows, rows, scanned_rows, scan_truncated, run_query, error);
        if (!loaded && error == "unsupported condition") {
            error.clear();
            loaded = agent_cozo_read_dataset(request["dataset"].get<std::string>(), max_scan_rows, rows, scanned_rows, scan_truncated, run_query, error);
        }
        if (!loaded) return false;
    }

    if (operation == "data.filter") {
        for (auto it = rows.begin(); it != rows.end();) {
            bool keep = true; for (const auto & condition : request.value("conditions", json::array())) if (!agent_cozo_match_condition(*it, condition)) keep = false;
            if (!keep) it = rows.erase(it); else ++it;
        }
        const auto limit = std::min<size_t>(request.value("limit", max_result_rows), max_result_rows);
        const bool result_truncated = rows.size() > limit;
        if (result_truncated) rows.resize(limit);
        result_json = json({{"rows", rows}, {"scanned_rows", scanned_rows}, {"row_count", rows.size()}, {"scan_truncated", scan_truncated}, {"result_truncated", result_truncated}}).dump();
        return materialize_data_result(*this, operation, request, result_json, error);
    } else if (operation == "data.query") {
        for (auto it = rows.begin(); it != rows.end();) {
            bool keep = true; for (const auto & condition : request.value("where", json::array())) if (!agent_cozo_match_condition(*it, condition)) keep = false;
            if (!keep) it = rows.erase(it); else ++it;
        }
        agent_cozo_sort_rows(rows, request.value("order_by", json::array()));
    } else if (operation == "statistics.describe") {
        const auto group_by = request.value("group_by", json::array());
        if (!group_by.is_array() || group_by.size() > 16) {
            error = "statistics.describe group_by must be an array with at most 16 fields";
            return false;
        }
        for (const auto & field : group_by) {
            if (!field.is_string() || field.get<std::string>().empty()) {
                error = "statistics.describe group_by fields must be non-empty strings";
                return false;
            }
        }
        json requested_columns = request.value("columns", json::array());
        if (requested_columns.empty()) {
            common_agent_dataset_descriptor descriptor;
            if (!get_dataset_descriptor(request["dataset"].get<std::string>(), descriptor, error)) return false;
            for (const auto & column : descriptor.columns) {
                if (column.type == common_agent_dataset_column_type::integer ||
                        column.type == common_agent_dataset_column_type::decimal) {
                    requested_columns.push_back(column.name);
                    if (requested_columns.size() >= 32) break;
                }
            }
        }
        struct statistic_accumulator {
            size_t count = 0;
            size_t null_count = 0;
            double sum = 0;
            double sum_squares = 0;
            double minimum = 0;
            double maximum = 0;
        };
        struct group_accumulator {
            json values = json::object();
            std::map<std::string, statistic_accumulator> columns;
        };
        std::map<std::string, group_accumulator> groups;
        for (const auto & row : rows) {
            if (!row.is_object()) continue;
            json group_values = json::object();
            std::string group_key;
            for (const auto & field : group_by) {
                const auto name = field.get<std::string>();
                const auto value = row.contains(name) ? row[name] : json();
                group_values[name] = value;
                group_key += value.dump() + "\x1f";
            }
            auto & group = groups[group_key];
            group.values = std::move(group_values);
            for (const auto & name : requested_columns) {
                if (!name.is_string()) continue;
                const auto column = name.get<std::string>();
                auto & accumulator = group.columns[column];
                if (!row.contains(column) || row[column].is_null()) {
                    ++accumulator.null_count;
                    continue;
                }
                if (!row[column].is_number()) continue;
                const double value = row[column].get<double>();
                if (accumulator.count == 0) accumulator.minimum = accumulator.maximum = value;
                else {
                    accumulator.minimum = std::min(accumulator.minimum, value);
                    accumulator.maximum = std::max(accumulator.maximum, value);
                }
                accumulator.sum += value;
                accumulator.sum_squares += value * value;
                ++accumulator.count;
            }
        }
        json columns = json::array();
        json grouped = json::array();
        for (auto & entry : groups) {
            json group = entry.second.values;
            json described = json::array();
            for (const auto & name : requested_columns) {
                if (!name.is_string()) continue;
                const auto column = name.get<std::string>();
                const auto & accumulator = entry.second.columns[column];
                const double variance = accumulator.count == 0 ? 0.0
                    : std::max(0.0, accumulator.sum_squares / accumulator.count -
                        (accumulator.sum / accumulator.count) * (accumulator.sum / accumulator.count));
                described.push_back({{"name", column}, {"count", accumulator.count},
                    {"null_count", accumulator.null_count},
                    {"min", accumulator.count ? json(accumulator.minimum) : json()},
                    {"max", accumulator.count ? json(accumulator.maximum) : json()},
                    {"mean", accumulator.count ? json(accumulator.sum / accumulator.count) : json()},
                    {"stddev", accumulator.count ? json(std::sqrt(variance)) : json()}});
            }
            if (group_by.empty()) columns = std::move(described);
            else {
                group["columns"] = std::move(described);
                grouped.push_back(std::move(group));
            }
        }
        result_json = json({{"columns", columns}, {"groups", grouped},
            {"group_by", group_by}, {"scanned_rows", scanned_rows},
            {"scan_truncated", scan_truncated}}).dump(); return true;
    } else if (operation == "statistics.outliers") {
        const auto method = request.value("method", std::string("iqr"));
        if (method != "iqr") { error = "statistics.outliers only supports the iqr method"; return false; }
        const double multiplier = request.value("multiplier", 1.5);
        if (!(multiplier >= 0.1 && multiplier <= 10.0)) {
            error = "statistics.outliers multiplier must be between 0.1 and 10.0";
            return false;
        }
        const auto group_by = request.value("group_by", json::array());
        if (!group_by.is_array() || group_by.size() > 16) {
            error = "statistics.outliers group_by must be an array with at most 16 fields";
            return false;
        }
        for (const auto & field : group_by) {
            if (!field.is_string() || field.get<std::string>().empty()) {
                error = "statistics.outliers group_by fields must be non-empty strings";
                return false;
            }
        }
        json requested_columns = request.value("columns", json::array());
        if (requested_columns.empty() && request.contains("column")) {
            requested_columns = json::array({request["column"]});
        }
        if (requested_columns.empty()) {
            common_agent_dataset_descriptor descriptor;
            if (!get_dataset_descriptor(request["dataset"].get<std::string>(), descriptor, error)) return false;
            for (const auto & column : descriptor.columns) {
                if (column.type == common_agent_dataset_column_type::integer ||
                        column.type == common_agent_dataset_column_type::decimal) {
                    requested_columns.push_back(column.name);
                    if (requested_columns.size() >= 32) break;
                }
            }
        }
        for (const auto & column : requested_columns) {
            if (!column.is_string() || column.get<std::string>().empty()) {
                error = "statistics.outliers columns must contain non-empty strings";
                return false;
            }
        }
        struct outlier_group {
            json values = json::object();
            std::map<std::string, std::vector<std::pair<double, json>>> observations;
        };
        std::map<std::string, outlier_group> groups;
        for (const auto & row : rows) {
            if (!row.is_object()) continue;
            json group_values = json::object();
            std::string group_key;
            for (const auto & field : group_by) {
                const auto name = field.get<std::string>();
                const auto value = row.contains(name) ? row[name] : json();
                group_values[name] = value;
                group_key += value.dump() + "\x1f";
            }
            auto & group = groups[group_key];
            group.values = std::move(group_values);
            for (const auto & column : requested_columns) {
                const auto name = column.get<std::string>();
                if (row.contains(name) && row[name].is_number()) {
                    group.observations[name].push_back({row[name].get<double>(), row});
                }
            }
        }
        const auto quantile = [](std::vector<double> values, double probability) {
            std::sort(values.begin(), values.end());
            if (values.empty()) return 0.0;
            const double position = probability * static_cast<double>(values.size() - 1);
            const size_t lower = static_cast<size_t>(position);
            const size_t upper = std::min(values.size() - 1, lower + 1);
            return values[lower] + (values[upper] - values[lower]) * (position - static_cast<double>(lower));
        };
        json output_columns = json::array();
        for (const auto & column : requested_columns) {
            const auto name = column.get<std::string>();
            json output_groups = json::array();
            for (const auto & entry : groups) {
                const auto found = entry.second.observations.find(name);
                if (found == entry.second.observations.end()) continue;
                const auto & observations = found->second;
                json output_group = entry.second.values;
                output_group["count"] = observations.size();
                output_group["outliers"] = json::array();
                if (observations.size() >= 4) {
                    std::vector<double> values;
                    values.reserve(observations.size());
                    for (const auto & observation : observations) values.push_back(observation.first);
                    const double q1 = quantile(values, 0.25);
                    const double q3 = quantile(values, 0.75);
                    const double iqr = q3 - q1;
                    const double lower = q1 - multiplier * iqr;
                    const double upper = q3 + multiplier * iqr;
                    output_group["q1"] = q1;
                    output_group["q3"] = q3;
                    output_group["iqr"] = iqr;
                    output_group["lower"] = lower;
                    output_group["upper"] = upper;
                    for (const auto & observation : observations) {
                        if (observation.first < lower || observation.first > upper) {
                            output_group["outliers"].push_back({{"value", observation.first}, {"row", observation.second}});
                        }
                    }
                }
                output_groups.push_back(std::move(output_group));
            }
            output_columns.push_back({{"name", name}, {"groups", output_groups}});
        }
        result_json = json({{"method", method}, {"multiplier", multiplier},
            {"group_by", group_by}, {"columns", output_columns},
            {"scanned_rows", scanned_rows}, {"scan_truncated", scan_truncated}}).dump();
        return true;
    } else if (operation == "statistics.value_counts") {
        if (!request.contains("column") || !request["column"].is_string() || request["column"].get<std::string>().empty()) {
            error = "statistics.value_counts requires a non-empty column";
            return false;
        }
        const auto column = request["column"].get<std::string>();
        const size_t limit = std::min<size_t>(request.value("limit", max_result_rows), 1000);
        if (limit == 0) { error = "statistics.value_counts limit must be greater than zero"; return false; }
        struct value_count { json value; size_t count = 0; };
        std::map<std::string, value_count> counts;
        size_t null_count = 0;
        for (const auto & row : rows) {
            if (!row.is_object() || !row.contains(column) || row[column].is_null()) { ++null_count; continue; }
            const auto key = row[column].dump();
            auto & entry = counts[key];
            entry.value = row[column];
            ++entry.count;
        }
        std::vector<value_count> ordered;
        ordered.reserve(counts.size());
        for (auto & entry : counts) ordered.push_back(std::move(entry.second));
        std::sort(ordered.begin(), ordered.end(), [](const value_count & left, const value_count & right) {
            if (left.count != right.count) return left.count > right.count;
            return left.value.dump() < right.value.dump();
        });
        const bool result_truncated = ordered.size() > limit;
        if (result_truncated) ordered.resize(limit);
        json values = json::array();
        for (const auto & entry : ordered) values.push_back({{"value", entry.value}, {"count", entry.count}});
        result_json = json({{"column", column}, {"values", values}, {"distinct_count", counts.size()},
            {"null_count", null_count}, {"scanned_rows", scanned_rows}, {"scan_truncated", scan_truncated},
            {"result_truncated", result_truncated}}).dump();
        return true;
    } else if (operation == "data.transform") {
        for (auto & row : rows) for (const auto & transform : request.value("operations", json::array())) if (transform.is_object()) {
            const auto type = transform.value("type", std::string());
            if (type == "rename" && transform.contains("from") && transform.contains("to")) { const auto from = transform["from"].get<std::string>(), to = transform["to"].get<std::string>(); if (row.contains(from)) { row[to] = row[from]; row.erase(from); } }
            else if (type == "drop" && transform.contains("column")) row.erase(transform["column"].get<std::string>());
        }
        const bool result_truncated = rows.size() > max_result_rows;
        if (result_truncated) rows.resize(max_result_rows);
        result_json = json({{"rows", rows}, {"scanned_rows", scanned_rows}, {"row_count", rows.size()}, {"scan_truncated", scan_truncated}, {"result_truncated", result_truncated}}).dump(); return materialize_data_result(*this, operation, request, result_json, error);
    }

    const auto selected = request.value("select", json::array());
    const size_t limit = std::min<size_t>(request.value("limit", max_result_rows), max_result_rows);
    json columns = json::array(), output_rows = json::array();
    if (!selected.empty()) columns = selected;
    else if (!rows.empty()) for (auto it = rows[0].begin(); it != rows[0].end(); ++it) columns.push_back(it.key());
    const size_t offset = std::min<size_t>(request.value("offset", 0), rows.size());
    std::set<std::string> distinct_rows;
    for (size_t index = offset; index < rows.size() && output_rows.size() < limit; ++index) {
        const auto & row = rows[index];
        json projected;
        if (selected.empty()) projected = row;
        else { projected = json::array(); for (const auto & column : selected) projected.push_back(row.value(column.get<std::string>(), json())); }
        if (request.value("distinct", false) && !distinct_rows.insert(projected.dump()).second) continue;
        output_rows.push_back(std::move(projected));
    }
        result_json = json({{"columns", columns}, {"rows", output_rows}, {"scanned_rows", scanned_rows}, {"row_count", output_rows.size()}, {"scan_truncated", scan_truncated}, {"result_truncated", rows.size() > output_rows.size()}}).dump();
    return materialize_data_result(*this, operation, request, result_json, error);
}
