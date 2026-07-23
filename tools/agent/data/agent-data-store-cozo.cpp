#include "agent-data-store-cozo.h"

extern "C" {
#include <cozo_c.h>
}
#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <set>

using json = nlohmann::ordered_json;

namespace {
const char * schema = R"COZO(
    {
        ?[dataset, row_id, row_json] <- [['__probe__', '__probe__', '{}']]
        :create agent_data_rows {
            dataset: String,
            row_id: String =>
            row_json: String
        }
    }
    {
        ?[dataset, row_id] <- [['__probe__', '__probe__']]
        :delete agent_data_rows { dataset, row_id }
    }
)COZO";

bool match_condition(const json & row, const json & condition) {
    if (!condition.is_object() || !condition.value("field", std::string()).size()) return false;
    const auto field = condition["field"].get<std::string>();
    const auto op = condition.value("operator", std::string("="));
    if (op == "is_null") return !row.contains(field) || row[field].is_null();
    if (op == "not_null") return row.contains(field) && !row[field].is_null();
    if (!row.contains(field)) return false;
    const auto & value = row[field];
    const auto expected = condition.value("value", json());
    if (op == "=") return value == expected;
    if (op == "!=") return value != expected;
    if ((op == ">" || op == ">=" || op == "<" || op == "<=") && value.is_number() && expected.is_number()) {
        const double a = value.get<double>(), b = expected.get<double>();
        return op == ">" ? a > b : op == ">=" ? a >= b : op == "<" ? a < b : a <= b;
    }
    return false;
}

void sort_rows(std::vector<json> & rows, const json & order_by) {
    if (!order_by.is_array() || order_by.empty()) return;
    std::stable_sort(rows.begin(), rows.end(), [&order_by](const json & left, const json & right) {
        for (const auto & item : order_by) {
            if (!item.is_object() || !item.value("field", std::string()).size()) continue;
            const auto field = item["field"].get<std::string>();
            const auto direction = item.value("direction", std::string("asc"));
            const auto l = left.value(field, json());
            const auto r = right.value(field, json());
            if (l == r) continue;
            const bool less = l < r;
            return direction == "desc" ? !less : less;
        }
        return false;
    });
}
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
    bool present = false;
    if (parsed.is_object() && parsed.contains("rows")) for (const auto & row : parsed["rows"]) if (row.is_array() && !row.empty() && row[0] == "agent_data_rows") present = true;
    if (!present) { std::string ignored; if (!run(schema, "{}", ignored, error)) { close(); return false; } }
    return true;
}

void common_agent_cozo_data_store::close() { if (db_id_ >= 0) { cozo_close_db(db_id_); db_id_ = -1; } }

bool common_agent_cozo_data_store::put_row(const std::string & dataset, const std::string & row_id, const std::string & row_json, std::string & error) {
    const auto parsed = json::parse(row_json, nullptr, false);
    if (!parsed.is_object() || dataset.empty() || row_id.empty()) { error = "data row requires dataset, row id and JSON object"; return false; }
    const json params = {{"dataset", dataset}, {"row_id", row_id}, {"row_json", row_json}};
    std::string result;
    return run("?[dataset, row_id, row_json] <- [[$dataset, $row_id, $row_json]] :put agent_data_rows { dataset, row_id => row_json }", params.dump(), result, error);
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

    std::vector<json> rows;
    size_t scanned_rows = 0;
    bool scan_truncated = false;
    if (!is_join && !read_dataset(request["dataset"].get<std::string>(), max_scan_rows, rows, scanned_rows, scan_truncated, error)) return false;

    if (operation == "data.filter") {
        for (auto it = rows.begin(); it != rows.end();) {
            bool keep = true; for (const auto & condition : request.value("conditions", json::array())) if (!match_condition(*it, condition)) keep = false;
            if (!keep) it = rows.erase(it); else ++it;
        }
        const auto limit = std::min<size_t>(request.value("limit", max_result_rows), max_result_rows);
        const bool result_truncated = rows.size() > limit;
        if (result_truncated) rows.resize(limit);
        result_json = json({{"rows", rows}, {"scanned_rows", scanned_rows}, {"row_count", rows.size()}, {"scan_truncated", scan_truncated}, {"result_truncated", result_truncated}}).dump();
        return true;
    } else if (operation == "data.query") {
        for (auto it = rows.begin(); it != rows.end();) {
            bool keep = true; for (const auto & condition : request.value("where", json::array())) if (!match_condition(*it, condition)) keep = false;
            if (!keep) it = rows.erase(it); else ++it;
        }
        sort_rows(rows, request.value("order_by", json::array()));
    } else if (operation == "data.aggregate") {
        std::map<std::string, std::vector<json>> groups;
        const auto group_by = request.value("group_by", json::array());
        for (const auto & row : rows) {
            std::string key;
            for (const auto & field : group_by) key += json(row.value(field.get<std::string>(), json())).dump() + "\x1f";
            groups[key].push_back(row);
        }
        if (groups.empty()) groups[""] = {};
        if (groups.size() == 1 && groups.begin()->second.empty()) groups.begin()->second = rows;
        json output_rows = json::array();
        for (const auto & group : groups) {
            json output = json::object();
            if (!group_by.empty() && !group.second.empty()) for (const auto & field : group_by) output[field.get<std::string>()] = group.second.front().value(field.get<std::string>(), json());
            for (const auto & measure : request.value("measures", json::array())) {
                if (!measure.is_object()) continue;
                const auto function = measure.value("function", std::string()); const auto column = measure.value("column", std::string());
                const auto name = measure.value("as", function + (column.empty() ? std::string() : "_" + column));
                if (function == "count") output[name] = group.second.size();
                else if ((function == "sum" || function == "avg") && !column.empty()) { double sum = 0; size_t count = 0; for (const auto & row : group.second) if (row.contains(column) && row[column].is_number()) { sum += row[column].get<double>(); ++count; } output[name] = function == "avg" && count ? sum / count : sum; }
                else if (function == "min" || function == "max") { json value; for (const auto & row : group.second) if (row.contains(column) && row[column].is_number() && (!value.is_number() || (function == "min" ? row[column] < value : row[column] > value))) value = row[column]; output[name] = value; }
            }
            output_rows.push_back(std::move(output));
            if (output_rows.size() >= max_result_rows) break;
        }
        result_json = json({{"rows", output_rows}, {"scanned_rows", scanned_rows}, {"row_count", output_rows.size()}, {"scan_truncated", scan_truncated}, {"result_truncated", groups.size() > output_rows.size()}}).dump(); return true;
    } else if (operation == "statistics.describe") {
        json columns = json::array();
        for (const auto & name : request.value("columns", json::array())) { if (!name.is_string()) continue; double sum = 0; size_t count = 0; for (const auto & row : rows) if (row.contains(name) && row[name].is_number()) { sum += row[name].get<double>(); ++count; } columns.push_back({{"name", name}, {"count", count}, {"mean", count ? sum / count : 0.0}}); }
        result_json = json({{"columns", columns}, {"scanned_rows", scanned_rows}, {"scan_truncated", scan_truncated}}).dump(); return true;
    } else if (operation == "data.join") {
        bool left_truncated = false, right_truncated = false; size_t left_scanned = 0, right_scanned = 0;
        if (!read_dataset(request["left"].get<std::string>(), max_scan_rows, rows, left_scanned, left_truncated, error)) return false;
        std::vector<json> right_rows;
        if (!read_dataset(request["right"].get<std::string>(), max_scan_rows, right_rows, right_scanned, right_truncated, error)) return false;
        json joined = json::array();
        for (const auto & left : rows) for (const auto & right : right_rows) {
            bool match = true;
            for (const auto & key : request.value("on", json::array())) if (key.is_object() && key.contains("left") && key.contains("right")) match = match && left.value(key["left"].get<std::string>(), json()) == right.value(key["right"].get<std::string>(), json());
            if (match) { json row = left; for (auto it = right.begin(); it != right.end(); ++it) if (!row.contains(it.key())) row[it.key()] = it.value(); joined.push_back(std::move(row)); }
            if (joined.size() >= max_result_rows) break;
        }
        if (request.value("type", std::string("inner")) == "left") for (const auto & left : rows) {
            bool matched = false; for (const auto & row : joined) { bool same = true; for (const auto & key : request["on"]) if (key.is_object() && left.value(key["left"].get<std::string>(), json()) != row.value(key["left"].get<std::string>(), json())) same = false; if (same) { matched = true; break; } }
            if (!matched && joined.size() < max_result_rows) joined.push_back(left);
        }
        result_json = json({{"rows", joined}, {"scanned_rows", left_scanned + right_scanned}, {"row_count", joined.size()}, {"scan_truncated", left_truncated || right_truncated}, {"result_truncated", joined.size() >= max_result_rows}}).dump(); return true;
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
