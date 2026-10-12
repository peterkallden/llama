#include "agent-data-store-sqlite.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <map>

using json = nlohmann::ordered_json;

namespace {

std::string text_column(common_sqlite_statement & statement, int index) {
    const auto * value = statement.column_text(index);
    return value ? reinterpret_cast<const char *>(value) : std::string();
}

json descriptor_json(const common_agent_dataset_descriptor & descriptor) {
    json columns = json::array();
    for (const auto & column : descriptor.columns) columns.push_back({{"name", column.name}, {"type", common_agent_dataset_column_type_name(column.type)}, {"nullable", column.nullable}});
    json value = common_agent_dataset_ref_to_json(
        descriptor.ref, common_agent_dataset_ref_json_projection::full);
    value["columns"] = std::move(columns);
    value["source_workbook_name"] = descriptor.source_workbook_name;
    value["source_sheet_name"] = descriptor.source_sheet_name;
    value["source_sheet_index"] = descriptor.source_sheet_index;
    value["source_range"] = descriptor.source_range;
    value["source_object"] = descriptor.source_object;
    value["import_processor_id"] = descriptor.import_processor_id;
    value["import_processor_version"] = descriptor.import_processor_version;
    value["parent_dataset_uris"] = descriptor.lineage.parent_dataset_uris;
    value["operation"] = descriptor.lineage.operation;
    value["operation_summary"] = descriptor.lineage.operation_summary;
    value["origin_kind"] = descriptor.origin.kind;
    value["origin_source_representation_uri"] = descriptor.origin.source_representation_uri;
    value["origin_source_node_id"] = descriptor.origin.source_node_id;
    value["origin_table_index"] = descriptor.origin.table_index;
    value["origin_section_path"] = descriptor.origin.section_path;
    value["origin_caption"] = descriptor.origin.caption;
    value["origin_notes"] = descriptor.origin.notes;
    value["origin_header_mode"] = common_agent_table_header_mode_name(descriptor.origin.header_mode);
    value["origin_header_confidence"] = descriptor.origin.header_confidence;
    value["origin_header_reason"] = descriptor.origin.header_reason;
    return value;
}

bool parse_descriptor(const std::string & text, common_agent_dataset_descriptor & descriptor, std::string & error) {
    const auto value = json::parse(text, nullptr, false);
    if (!value.is_object()) { error = "sqlite dataset descriptor is invalid"; return false; }
    descriptor = {};
    if (!common_agent_dataset_ref_from_json(value, descriptor.ref, error)) return false;
    for (const auto & item : value.value("columns", json::array())) {
        common_agent_dataset_column column; column.name = item.value("name", std::string{}); column.nullable = item.value("nullable", true); const auto type = item.value("type", std::string("unknown"));
        for (int index = 0; index <= static_cast<int>(common_agent_dataset_column_type::unknown); ++index) { const auto candidate = static_cast<common_agent_dataset_column_type>(index); if (type == common_agent_dataset_column_type_name(candidate)) { column.type = candidate; break; } }
        descriptor.columns.push_back(std::move(column));
    }
    descriptor.source_workbook_name = value.value("source_workbook_name", std::string{}); descriptor.source_sheet_name = value.value("source_sheet_name", std::string{}); if (value.contains("source_sheet_index") && value["source_sheet_index"].is_number_unsigned()) descriptor.source_sheet_index = value["source_sheet_index"].get<size_t>(); descriptor.source_range = value.value("source_range", std::string{}); descriptor.source_object = value.value("source_object", std::string{}); descriptor.import_processor_id = value.value("import_processor_id", std::string{}); descriptor.import_processor_version = value.value("import_processor_version", std::string{}); descriptor.lineage.parent_dataset_uris = value.value("parent_dataset_uris", std::vector<std::string>{}); descriptor.lineage.operation = value.value("operation", std::string{}); descriptor.lineage.operation_summary = value.value("operation_summary", std::string{}); descriptor.origin.kind = value.value("origin_kind", std::string{}); descriptor.origin.source_representation_uri = value.value("origin_source_representation_uri", std::string{}); descriptor.origin.source_node_id = value.value("origin_source_node_id", std::string{}); descriptor.origin.table_index = value.value("origin_table_index", size_t(0)); descriptor.origin.section_path = value.value("origin_section_path", std::vector<std::string>{}); descriptor.origin.caption = value.value("origin_caption", std::string{}); descriptor.origin.notes = value.value("origin_notes", std::vector<std::string>{}); descriptor.origin.header_confidence = value.value("origin_header_confidence", 0.0); descriptor.origin.header_reason = value.value("origin_header_reason", std::string{}); const auto header_mode = value.value("origin_header_mode", std::string("none")); for (int index = 0; index <= static_cast<int>(common_agent_table_header_mode::ambiguous); ++index) { const auto candidate = static_cast<common_agent_table_header_mode>(index); if (header_mode == common_agent_table_header_mode_name(candidate)) { descriptor.origin.header_mode = candidate; break; } }
    return validate_common_agent_dataset_descriptor(descriptor, common_agent_dataset_limits{}, error);
}

std::string normalized_name(const std::string & value) { std::string result; for (unsigned char c : value) if (!std::isspace(c)) result += static_cast<char>(std::tolower(c)); return result; }

bool condition_matches(const json & row, const json & condition) {
    const std::string field = condition.value("field", std::string{}); const std::string op = condition.value("operator", std::string("=")); if (field.empty()) return false; if (op == "is_null") return !row.contains(field) || row[field].is_null(); if (op == "not_null") return row.contains(field) && !row[field].is_null(); if (!row.contains(field)) return false; const auto & actual = row[field]; const auto expected = condition.value("value", json()); if (op == "=") return actual == expected; if (op == "!=") return actual != expected; if (!actual.is_number() || !expected.is_number()) return false; const double a = actual.get<double>(), b = expected.get<double>(); return op == ">" ? a > b : op == ">=" ? a >= b : op == "<" ? a < b : op == "<=" ? a <= b : false;
}

}

common_agent_sqlite_data_store::~common_agent_sqlite_data_store() { close(); }

bool common_agent_sqlite_data_store::open(const std::string & path, std::string & error) { close(); if (!database_.open(path, error) || !database_.execute("PRAGMA journal_mode = WAL;", error) || !ensure_schema(error)) { close(); return false; } return true; }
void common_agent_sqlite_data_store::close() { database_.close(); }
bool common_agent_sqlite_data_store::ensure_schema(std::string & error) { return database_.execute("CREATE TABLE IF NOT EXISTS agent_dataset_metadata(dataset_uri TEXT PRIMARY KEY, descriptor_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS agent_data_rows(dataset TEXT NOT NULL, row_id TEXT NOT NULL, row_seq INTEGER NOT NULL, row_json TEXT NOT NULL, PRIMARY KEY(dataset,row_id)); CREATE INDEX IF NOT EXISTS agent_data_rows_order ON agent_data_rows(dataset,row_seq);", error); }

bool common_agent_sqlite_data_store::put_row(const std::string & dataset, const std::string & row_id, const std::string & row_json, std::string & error) {
    const auto value = json::parse(row_json, nullptr, false); if (!value.is_object() || dataset.empty() || row_id.empty()) { error = "data row requires dataset, row id and JSON object"; return false; }
    common_sqlite_statement statement; if (!database_.prepare("INSERT INTO agent_data_rows(dataset,row_id,row_seq,row_json) VALUES(?,?,COALESCE((SELECT MAX(row_seq)+1 FROM agent_data_rows WHERE dataset=?),1),?) ON CONFLICT(dataset,row_id) DO UPDATE SET row_json=excluded.row_json;", statement, error) || !statement.bind_text(1, dataset, error) || !statement.bind_text(2, row_id, error) || !statement.bind_text(3, dataset, error) || !statement.bind_text(4, row_json, error)) return false; bool row = false; return statement.step(row, error);
}

bool common_agent_sqlite_data_store::put_dataset_descriptor(const common_agent_dataset_descriptor & descriptor, std::string & error) {
    if (!validate_common_agent_dataset_descriptor(descriptor, common_agent_dataset_limits{}, error)) return false; common_sqlite_statement statement; const auto serialized = descriptor_json(descriptor).dump(); if (!database_.prepare("INSERT INTO agent_dataset_metadata(dataset_uri,descriptor_json) VALUES(?,?) ON CONFLICT(dataset_uri) DO UPDATE SET descriptor_json=excluded.descriptor_json;", statement, error) || !statement.bind_text(1, descriptor.ref.uri, error) || !statement.bind_text(2, serialized, error)) return false; bool row = false; return statement.step(row, error);
}

bool common_agent_sqlite_data_store::get_dataset_descriptor(const std::string & dataset_uri, common_agent_dataset_descriptor & descriptor, std::string & error) {
    common_sqlite_statement statement; if (!database_.prepare("SELECT descriptor_json FROM agent_dataset_metadata WHERE dataset_uri=?;", statement, error) || !statement.bind_text(1, dataset_uri, error)) return false; bool row = false; if (!statement.step(row, error) || !row) { error = "dataset descriptor was not found"; return false; } return parse_descriptor(text_column(statement, 0), descriptor, error);
}

bool common_agent_sqlite_data_store::list_dataset_descriptors(std::vector<common_agent_dataset_descriptor> & descriptors, std::string & error) {
    descriptors.clear(); common_sqlite_statement statement; if (!database_.prepare("SELECT descriptor_json FROM agent_dataset_metadata ORDER BY dataset_uri;", statement, error)) return false; bool row = false; while (statement.step(row, error) && row) { common_agent_dataset_descriptor descriptor; if (!parse_descriptor(text_column(statement, 0), descriptor, error)) return false; descriptors.push_back(std::move(descriptor)); } return error.empty();
}

bool common_agent_sqlite_data_store::find_dataset_by_name(const std::string & name, common_agent_dataset_descriptor & descriptor, std::string & error) {
    const auto wanted = normalized_name(name); if (wanted.empty()) { error = "dataset name must not be empty"; return false; } std::vector<common_agent_dataset_descriptor> descriptors; if (!list_dataset_descriptors(descriptors, error)) return false; const common_agent_dataset_descriptor * match = nullptr; for (const auto & candidate : descriptors) if (normalized_name(candidate.ref.name) == wanted) { if (match) { error = "dataset name is ambiguous; choose one of: " + match->ref.name + " (" + match->ref.uri + ")"; for (const auto & option : descriptors) if (normalized_name(option.ref.name) == wanted && option.ref.uri != match->ref.uri) error += ", " + option.ref.name + " (" + option.ref.uri + ")"; return false; } match = &candidate; } if (!match) { error = "dataset name was not found"; return false; } descriptor = *match; return true;
}

bool common_agent_sqlite_data_store::execute(const std::string & operation, const std::string & request_json, std::string & result_json, std::string & error) {
    const auto request = json::parse(request_json, nullptr, false); if (!request.is_object()) { error = "data operation requires an object request"; return false; }
    if (operation != "data.query" && operation != "data.filter") { error = "sqlite data backend does not support operation: " + operation; return false; }
    const std::string dataset = request.value("dataset", std::string{}); if (dataset.empty()) { error = operation + " requires dataset"; return false; }
    const size_t scan_limit = std::min<size_t>(request.value("max_scan_rows", 10000), 100000); const size_t result_limit = std::min<size_t>(request.value("limit", request.value("max_result_rows", 1000)), 10000); if (!scan_limit || !result_limit) { error = "query limits must be greater than zero"; return false; }
    common_sqlite_statement statement; if (!database_.prepare("SELECT row_json FROM agent_data_rows WHERE dataset=? ORDER BY row_seq LIMIT ?;", statement, error) || !statement.bind_text(1, dataset, error) || !statement.bind_int64(2, static_cast<int64_t>(scan_limit + 1), error)) return false;
    std::vector<json> rows; bool scan_truncated = false; bool row = false; size_t scanned = 0;
    // The bounded scan is deliberately implemented through the existing host
    // JSON contract first. SQL remains authoritative for persistence/order;
    // richer predicates can be moved into parameterized SQL incrementally.
    while (statement.step(row, error) && row) { ++scanned; if (scanned > scan_limit) { scan_truncated = true; break; } const auto value = json::parse(text_column(statement, 0), nullptr, false); if (!value.is_object()) continue; const auto conditions = operation == "data.filter" ? request.value("conditions", json::array()) : request.value("where", json::array()); bool keep = true; if (conditions.is_array()) for (const auto & condition : conditions) if (!condition_matches(value, condition)) keep = false; if (keep) rows.push_back(value); }
    if (!error.empty()) return false; const bool result_truncated = rows.size() > result_limit; if (result_truncated) rows.resize(result_limit);
    result_json = json({{"rows", rows}, {"row_count", rows.size()}, {"scanned_rows", scanned}, {"scan_truncated", scan_truncated}, {"result_truncated", result_truncated}}).dump(); return true;
}
