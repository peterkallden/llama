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
#include <map>
#include <set>

using json = nlohmann::ordered_json;

namespace {
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
    bool rows_present = false, values_present = false, order_present = false;
    if (parsed.is_object() && parsed.contains("rows")) for (const auto & row : parsed["rows"]) if (row.is_array() && !row.empty() && row[0].is_string()) {
        rows_present |= row[0] == "agent_data_rows";
        values_present |= row[0] == "agent_data_values";
        order_present |= row[0] == "agent_data_row_order";
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
    if (operation == "data.aggregate") return agent_cozo_execute_native_aggregate(
        request, max_scan_rows, max_result_rows, run_query, read_metadata, result_json, error);
    if (operation == "data.join") return agent_cozo_execute_native_join(
        request, max_scan_rows, max_result_rows, run_query, read_metadata, result_json, error);

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
        return true;
    } else if (operation == "data.query") {
        for (auto it = rows.begin(); it != rows.end();) {
            bool keep = true; for (const auto & condition : request.value("where", json::array())) if (!agent_cozo_match_condition(*it, condition)) keep = false;
            if (!keep) it = rows.erase(it); else ++it;
        }
        agent_cozo_sort_rows(rows, request.value("order_by", json::array()));
    } else if (operation == "statistics.describe") {
        json columns = json::array();
        for (const auto & name : request.value("columns", json::array())) { if (!name.is_string()) continue; double sum = 0; size_t count = 0; for (const auto & row : rows) if (row.contains(name) && row[name].is_number()) { sum += row[name].get<double>(); ++count; } columns.push_back({{"name", name}, {"count", count}, {"mean", count ? sum / count : 0.0}}); }
        result_json = json({{"columns", columns}, {"scanned_rows", scanned_rows}, {"scan_truncated", scan_truncated}}).dump(); return true;
    } else if (operation == "data.transform") {
        for (auto & row : rows) for (const auto & transform : request.value("operations", json::array())) if (transform.is_object()) {
            const auto type = transform.value("type", std::string());
            if (type == "rename" && transform.contains("from") && transform.contains("to")) { const auto from = transform["from"].get<std::string>(), to = transform["to"].get<std::string>(); if (row.contains(from)) { row[to] = row[from]; row.erase(from); } }
            else if (type == "drop" && transform.contains("column")) row.erase(transform["column"].get<std::string>());
        }
        const bool result_truncated = rows.size() > max_result_rows;
        if (result_truncated) rows.resize(max_result_rows);
        result_json = json({{"rows", rows}, {"scanned_rows", scanned_rows}, {"row_count", rows.size()}, {"scan_truncated", scan_truncated}, {"result_truncated", result_truncated}}).dump(); return true;
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
    return true;
}
