#include "agent-data-store-cozo.h"
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

bool common_agent_cozo_data_store::read_dataset(
        const std::string & dataset,
        size_t max_scan_rows,
        std::vector<json> & rows,
        size_t & scanned_rows,
        bool & scan_truncated,
    std::string & error) const {
    if (dataset.empty()) { error = "dataset name must not be empty"; return false; }
    const auto scan_limit = std::to_string(max_scan_rows + 1);
    std::string raw;
    if (!run("?[row_id, row_json] := *agent_data_rows[$dataset, row_id, row_json] :limit " + scan_limit,
            json({{"dataset", dataset}}).dump(), raw, error)) return false;
    const auto rows_value = json::parse(raw, nullptr, false);
    if (!rows_value.is_object() || !rows_value.contains("rows")) { error = "Cozo data query returned invalid rows"; return false; }
    rows.clear();
    for (const auto & item : rows_value["rows"]) {
        if (!item.is_array() || item.size() < 2 || !item[1].is_string()) continue;
        const auto row = json::parse(item[1].get<std::string>(), nullptr, false);
        if (row.is_object()) rows.push_back(row);
    }
    scan_truncated = rows.size() > max_scan_rows;
    if (scan_truncated) {
        rows.resize(max_scan_rows);
        scanned_rows = max_scan_rows;
    } else {
        scanned_rows = rows.size();
    }
    return true;
}

bool common_agent_cozo_data_store::read_dataset_with_conditions(
        const std::string & dataset,
        const json & conditions,
        size_t max_scan_rows,
        std::vector<json> & rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        std::string & error) const {
    if (!conditions.is_array() || conditions.empty()) return read_dataset(dataset, max_scan_rows, rows, scanned_rows, scan_truncated, error);
    std::string script = "?[row_id, row_json] := ";
    json params = {{"dataset", dataset}};
    size_t index = 0;
    for (const auto & condition : conditions) {
        if (!condition.is_object() || !condition.value("field", std::string()).size()) { error = "unsupported condition"; return false; }
        const auto op = condition.value("operator", std::string("="));
        const auto field_name = condition["field"].get<std::string>();
        const auto field = "field" + std::to_string(index);
        const auto kind = "kind" + std::to_string(index);
        const auto text = "text" + std::to_string(index);
        const auto number = "number" + std::to_string(index);
        const auto expected = condition.value("value", json());
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
    const auto rows_value = json::parse(raw, nullptr, false);
    if (!rows_value.is_object() || !rows_value.contains("rows")) { error = "Cozo filtered query returned invalid rows"; return false; }
    rows.clear();
    for (const auto & item : rows_value["rows"]) if (item.is_array() && item.size() >= 2 && item[1].is_string()) {
        const auto row = json::parse(item[1].get<std::string>(), nullptr, false);
        if (row.is_object()) rows.push_back(row);
    }
    scan_truncated = rows.size() > max_scan_rows;
    if (scan_truncated) { rows.resize(max_scan_rows); scanned_rows = max_scan_rows; }
    else scanned_rows = rows.size();
    return true;
}

bool common_agent_cozo_data_store::read_scan_metadata(
        const std::string & dataset,
        size_t max_scan_rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        std::string & error) const {
    std::string raw;
    if (!run("?[count(row_id)] := *agent_data_row_order[$dataset, row_id, row_seq], row_seq <= $scan_limit", json({{"dataset", dataset}, {"scan_limit", max_scan_rows + 1}}).dump(), raw, error)) return false;
    const auto parsed = json::parse(raw, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows") || parsed["rows"].empty() || !parsed["rows"][0].is_array() || parsed["rows"][0].empty()) {
        error = "Cozo scan metadata query returned invalid rows";
        return false;
    }
    const auto count = parsed["rows"][0][0].is_number_unsigned()
        ? parsed["rows"][0][0].get<size_t>()
        : static_cast<size_t>(std::max<int64_t>(0, parsed["rows"][0][0].get<int64_t>()));
    scan_truncated = count > max_scan_rows;
    scanned_rows = std::min(count, max_scan_rows);
    return true;
}

bool common_agent_cozo_data_store::execute_native_aggregate(
        const json & request,
        size_t max_scan_rows,
        size_t max_result_rows,
        std::string & result_json,
        std::string & error) const {
    const auto dataset = request.value("dataset", std::string());
    const auto group_by = request.value("group_by", json::array());
    const auto measures = request.value("measures", json::array());
    if (dataset.empty() || !group_by.is_array() || !measures.is_array() || measures.empty()) { error = "data.aggregate requires dataset, group_by and measures"; return false; }
    for (const auto & field : group_by) if (!field.is_string() || field.get<std::string>().empty()) { error = "data.aggregate group_by fields must be strings"; return false; }

    json output_rows = json::array();
    std::map<std::string, size_t> output_index;
    for (const auto & measure : measures) {
        if (!measure.is_object()) { error = "data.aggregate measures must be objects"; return false; }
        const auto function = measure.value("function", std::string());
        const auto column = measure.value("column", std::string());
        const auto name = measure.value("as", function + (column.empty() ? std::string() : "_" + column));
        const bool count = function == "count";
        const bool numeric = function == "sum" || function == "avg" || function == "min" || function == "max";
        if ((!count && !numeric) || (!count && column.empty())) { error = "unsupported data.aggregate function"; return false; }

        json params = {{"dataset", dataset}, {"scan_limit", max_scan_rows}};
        std::string script = "?[";
        for (size_t i = 0; i < group_by.size(); ++i) script += "gkind" + std::to_string(i) + ", gtext" + std::to_string(i) + ", ";
        script += count ? "count(row_id)] := " : (function == "avg" ? "mean(mnumber)] := " : function + "(mnumber)] := ");
        for (size_t i = 0; i < group_by.size(); ++i) {
            const auto field = "gfield" + std::to_string(i);
            params[field] = group_by[i].get<std::string>();
            script += "*agent_data_row_order[$dataset, row_id, row_seq], row_seq <= $scan_limit, *agent_data_values[$dataset, row_id, $" + field + ", gkind" + std::to_string(i) + ", gtext" + std::to_string(i) + ", gnumber" + std::to_string(i) + "], ";
        }
        if (count) script += "*agent_data_row_order[$dataset, row_id, row_seq], row_seq <= $scan_limit, *agent_data_rows[$dataset, row_id, row_json]";
        else {
            params["mfield"] = column;
            script += "*agent_data_row_order[$dataset, row_id, row_seq], row_seq <= $scan_limit, *agent_data_values[$dataset, row_id, $mfield, mkind, mtext, mnumber], mkind == 'number'";
        }
        script += " :limit " + std::to_string(max_result_rows);
        std::string raw;
        if (!run(script, params.dump(), raw, error)) return false;
        const auto parsed = json::parse(raw, nullptr, false);
        if (!parsed.is_object() || !parsed.contains("rows")) { error = "Cozo aggregate query returned invalid rows"; return false; }
        for (const auto & row : parsed["rows"]) {
            if (!row.is_array() || row.size() < (group_by.size() * 2 + 1)) continue;
            json output = json::object();
            std::string key;
            size_t position = 0;
            for (size_t i = 0; i < group_by.size(); ++i) {
                const auto kind = row[position++].get<std::string>();
                const auto text = row[position++].get<std::string>();
                json value = text;
                if (kind == "number") value = json::parse(text, nullptr, false);
                else if (kind == "boolean") value = text == "true";
                output[group_by[i].get<std::string>()] = value;
                key += kind + ":" + text + "\x1f";
            }
            if (output_index.find(key) == output_index.end()) { output_index[key] = output_rows.size(); output_rows.push_back(std::move(output)); }
            output_rows[output_index[key]][name] = row.back();
        }
    }
    size_t scanned_rows = 0;
    bool scan_truncated = false;
    if (!read_scan_metadata(dataset, max_scan_rows, scanned_rows, scan_truncated, error)) return false;
    result_json = json({{"rows", output_rows}, {"row_count", output_rows.size()}, {"scanned_rows", scanned_rows}, {"scan_truncated", scan_truncated}, {"scan_mode", "native_bounded"}, {"result_truncated", output_rows.size() >= max_result_rows}}).dump();
    return true;
}

bool common_agent_cozo_data_store::execute_native_join(
        const json & request,
        size_t max_scan_rows,
        size_t max_result_rows,
        std::string & result_json,
        std::string & error) const {
    const auto left = request.value("left", std::string());
    const auto right = request.value("right", std::string());
    const auto on = request.value("on", json::array());
    const auto type = request.value("type", std::string("inner"));
    if (left.empty() || right.empty() || !on.is_array() || on.empty()) { error = "data.join requires left, right and on"; return false; }
    if (type != "inner" && type != "left") { error = "unsupported data.join type"; return false; }
    const auto query_limit = max_result_rows + 1;
    std::string script = "?[left_json, right_json] := ";
    json params = {{"left", left}, {"right", right}, {"scan_limit", max_scan_rows}};
    for (size_t i = 0; i < on.size(); ++i) {
        if (!on[i].is_object() || !on[i].contains("left") || !on[i].contains("right")) { error = "data.join on entries require left and right"; return false; }
        params["left_field" + std::to_string(i)] = on[i]["left"];
        params["right_field" + std::to_string(i)] = on[i]["right"];
        script += "*agent_data_values[$left, left_id, $left_field" + std::to_string(i) + ", left_kind" + std::to_string(i) + ", left_text" + std::to_string(i) + ", left_number" + std::to_string(i) + "], ";
        script += "*agent_data_values[$right, right_id, $right_field" + std::to_string(i) + ", right_kind" + std::to_string(i) + ", right_text" + std::to_string(i) + ", right_number" + std::to_string(i) + "], ";
        script += "left_kind" + std::to_string(i) + " == right_kind" + std::to_string(i) + ", left_text" + std::to_string(i) + " == right_text" + std::to_string(i) + ", ";
    }
    script += "*agent_data_row_order[$left, left_id, left_seq], left_seq <= $scan_limit, *agent_data_rows[$left, left_id, left_json], *agent_data_row_order[$right, right_id, right_seq], right_seq <= $scan_limit, *agent_data_rows[$right, right_id, right_json] :limit " + std::to_string(query_limit);
    std::string raw;
    if (!run(script, params.dump(), raw, error)) return false;
    const auto parsed = json::parse(raw, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows")) { error = "Cozo join query returned invalid rows"; return false; }
    json joined = json::array();
    for (const auto & row : parsed["rows"]) if (row.is_array() && row.size() >= 2) {
        const auto left_row = json::parse(row[0].get<std::string>(), nullptr, false);
        const auto right_row = json::parse(row[1].get<std::string>(), nullptr, false);
        if (!left_row.is_object() || !right_row.is_object()) continue;
        json merged = left_row;
        for (auto it = right_row.begin(); it != right_row.end(); ++it) if (!merged.contains(it.key())) merged[it.key()] = it.value();
        joined.push_back(std::move(merged));
    }
    bool result_truncated = joined.size() > max_result_rows;
    if (joined.size() > max_result_rows) {
        json limited = json::array();
        for (size_t i = 0; i < max_result_rows; ++i) limited.push_back(joined[i]);
        joined = std::move(limited);
    }

    if (type == "left" && !result_truncated) {
        std::string unmatched_script;
        unmatched_script.reserve(512);
        unmatched_script = "left_match[left_id] := ";
        for (size_t i = 0; i < on.size(); ++i) {
            unmatched_script += "*agent_data_values[$left, left_id, $left_field" + std::to_string(i) + ", left_kind" + std::to_string(i) + ", left_text" + std::to_string(i) + ", left_number" + std::to_string(i) + "], ";
            unmatched_script += "*agent_data_row_order[$right, right_id, right_seq], right_seq <= $scan_limit, *agent_data_values[$right, right_id, $right_field" + std::to_string(i) + ", right_kind" + std::to_string(i) + ", right_text" + std::to_string(i) + ", right_number" + std::to_string(i) + "], ";
            unmatched_script += "left_kind" + std::to_string(i) + " == right_kind" + std::to_string(i) + ", left_text" + std::to_string(i) + " == right_text" + std::to_string(i) + ", ";
        }
        if (!unmatched_script.empty() && unmatched_script.back() == ' ') unmatched_script.pop_back();
        unmatched_script += "\n?[left_json] := *agent_data_row_order[$left, left_id, left_seq], left_seq <= $scan_limit, *agent_data_rows[$left, left_id, left_json], not left_match[left_id] :limit " + std::to_string(query_limit);

        std::string unmatched_raw;
        if (!run(unmatched_script, params.dump(), unmatched_raw, error)) return false;
        const auto unmatched = json::parse(unmatched_raw, nullptr, false);
        if (!unmatched.is_object() || !unmatched.contains("rows")) { error = "Cozo left join query returned invalid rows"; return false; }
        for (const auto & row : unmatched["rows"]) {
            if (!row.is_array() || row.empty() || !row[0].is_string()) continue;
            const auto left_row = json::parse(row[0].get<std::string>(), nullptr, false);
            if (!left_row.is_object()) continue;
            if (joined.size() < max_result_rows) joined.push_back(left_row);
            else { result_truncated = true; break; }
        }
    }
    size_t left_scanned = 0, right_scanned = 0;
    bool left_truncated = false, right_truncated = false;
    if (!read_scan_metadata(left, max_scan_rows, left_scanned, left_truncated, error) || !read_scan_metadata(right, max_scan_rows, right_scanned, right_truncated, error)) return false;
    result_json = json({{"rows", joined}, {"row_count", joined.size()}, {"scanned_rows", left_scanned + right_scanned}, {"scan_truncated", left_truncated || right_truncated}, {"scan_mode", "native_bounded"}, {"result_truncated", result_truncated}}).dump();
    return true;
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
    json values = json::array();
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
        const auto & value = it.value();
        std::string kind = "null";
        std::string text;
        double number = 0.0;
        if (value.is_string()) { kind = "string"; text = value.get<std::string>(); }
        else if (value.is_number()) { kind = "number"; text = value.dump(); number = value.get<double>(); }
        else if (value.is_boolean()) { kind = "boolean"; text = value.get<bool>() ? "true" : "false"; }
        else text = value.dump();
        values.push_back({dataset, row_id, it.key(), kind, text, number});
    }
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
    if (operation == "data.aggregate") return execute_native_aggregate(request, max_scan_rows, max_result_rows, result_json, error);
    if (operation == "data.join") return execute_native_join(request, max_scan_rows, max_result_rows, result_json, error);

    std::vector<json> rows;
    size_t scanned_rows = 0;
    bool scan_truncated = false;
    if (!is_join) {
        const auto conditions = operation == "data.filter" ? request.value("conditions", json::array()) : request.value("where", json::array());
        bool loaded = conditions.is_array() && !conditions.empty()
            ? read_dataset_with_conditions(request["dataset"].get<std::string>(), conditions, max_scan_rows, rows, scanned_rows, scan_truncated, error)
            : read_dataset(request["dataset"].get<std::string>(), max_scan_rows, rows, scanned_rows, scan_truncated, error);
        if (!loaded && error == "unsupported condition") {
            error.clear();
            loaded = read_dataset(request["dataset"].get<std::string>(), max_scan_rows, rows, scanned_rows, scan_truncated, error);
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
