#include "agent-data-store-cozo-reader.h"

#include <algorithm>

bool agent_cozo_read_dataset(
        const std::string & dataset,
        size_t max_scan_rows,
        std::vector<agent_cozo_aggregate_json> & rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        const agent_cozo_query_runner & run,
        std::string & error) {
    if (dataset.empty()) { error = "dataset name must not be empty"; return false; }
    const auto scan_limit = std::to_string(max_scan_rows + 1);
    std::string raw;
    if (!run("?[row_id, row_json] := *agent_data_rows[$dataset, row_id, row_json] :limit " + scan_limit,
            agent_cozo_aggregate_json({{"dataset", dataset}}).dump(), raw, error)) return false;
    const auto rows_value = agent_cozo_aggregate_json::parse(raw, nullptr, false);
    if (!rows_value.is_object() || !rows_value.contains("rows")) { error = "Cozo data query returned invalid rows"; return false; }
    rows.clear();
    for (const auto & item : rows_value["rows"]) {
        if (!item.is_array() || item.size() < 2 || !item[1].is_string()) continue;
        const auto row = agent_cozo_aggregate_json::parse(item[1].get<std::string>(), nullptr, false);
        if (row.is_object()) rows.push_back(row);
    }
    scan_truncated = rows.size() > max_scan_rows;
    if (scan_truncated) { rows.resize(max_scan_rows); scanned_rows = max_scan_rows; }
    else scanned_rows = rows.size();
    return true;
}

bool agent_cozo_read_dataset_with_conditions(
        const std::string & dataset,
        const agent_cozo_aggregate_json & conditions,
        size_t max_scan_rows,
        std::vector<agent_cozo_aggregate_json> & rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        const agent_cozo_query_runner & run,
        std::string & error) {
    if (!conditions.is_array() || conditions.empty()) return agent_cozo_read_dataset(dataset, max_scan_rows, rows, scanned_rows, scan_truncated, run, error);
    std::string script = "?[row_id, row_json] := ";
    agent_cozo_aggregate_json params = {{"dataset", dataset}};
    size_t index = 0;
    for (const auto & condition : conditions) {
        if (!condition.is_object() || !condition.value("field", std::string()).size()) { error = "unsupported condition"; return false; }
        const auto op = condition.value("operator", std::string("="));
        const auto field_name = condition["field"].get<std::string>();
        const auto field = "field" + std::to_string(index);
        const auto kind = "kind" + std::to_string(index);
        const auto text = "text" + std::to_string(index);
        const auto number = "number" + std::to_string(index);
        const auto expected = condition.value("value", agent_cozo_aggregate_json());
        if (op != "=" && op != "!=" && op != ">" && op != ">=" && op != "<" && op != "<=") { error = "unsupported condition"; return false; }
        if (!(expected.is_string() || expected.is_number() || expected.is_boolean())) { error = "unsupported condition"; return false; }
        params[field] = field_name;
        if (expected.is_number()) { params[kind] = "number"; params[number] = expected.get<double>(); script += "*agent_data_values[$dataset, row_id, $" + field + ", " + kind + ", " + text + ", " + number + "], " + kind + " == $" + kind + ", " + number + " " + op + " $" + number; }
        else { params[kind] = expected.is_boolean() ? "boolean" : "string"; params[text] = expected.is_boolean() ? (expected.get<bool>() ? "true" : "false") : expected.get<std::string>(); script += "*agent_data_values[$dataset, row_id, $" + field + ", " + kind + ", " + text + ", " + number + "], " + kind + " == $" + kind + ", " + text + " " + op + " $" + text; }
        script += ", ";
        ++index;
    }
    script += "*agent_data_rows[$dataset, row_id, row_json] :limit " + std::to_string(max_scan_rows + 1);
    std::string raw;
    if (!run(script, params.dump(), raw, error)) return false;
    const auto rows_value = agent_cozo_aggregate_json::parse(raw, nullptr, false);
    if (!rows_value.is_object() || !rows_value.contains("rows")) { error = "Cozo filtered query returned invalid rows"; return false; }
    rows.clear();
    for (const auto & item : rows_value["rows"]) if (item.is_array() && item.size() >= 2 && item[1].is_string()) {
        const auto row = agent_cozo_aggregate_json::parse(item[1].get<std::string>(), nullptr, false);
        if (row.is_object()) rows.push_back(row);
    }
    scan_truncated = rows.size() > max_scan_rows;
    if (scan_truncated) { rows.resize(max_scan_rows); scanned_rows = max_scan_rows; }
    else scanned_rows = rows.size();
    return true;
}

bool agent_cozo_read_scan_metadata(
        const std::string & dataset,
        size_t max_scan_rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        const agent_cozo_query_runner & run,
        std::string & error) {
    std::string raw;
    if (!run("?[count(row_id)] := *agent_data_row_order[$dataset, row_id, row_seq], row_seq <= $scan_limit", agent_cozo_aggregate_json({{"dataset", dataset}, {"scan_limit", max_scan_rows + 1}}).dump(), raw, error)) return false;
    const auto parsed = agent_cozo_aggregate_json::parse(raw, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows") || parsed["rows"].empty() || !parsed["rows"][0].is_array() || parsed["rows"][0].empty()) { error = "Cozo scan metadata query returned invalid rows"; return false; }
    const auto count = parsed["rows"][0][0].is_number_unsigned() ? parsed["rows"][0][0].get<size_t>() : static_cast<size_t>(std::max<int64_t>(0, parsed["rows"][0][0].get<int64_t>()));
    scan_truncated = count > max_scan_rows;
    scanned_rows = std::min(count, max_scan_rows);
    return true;
}
