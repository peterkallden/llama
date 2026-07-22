#include "agent-data-store-cozo.h"

extern "C" {
#include <cozo_c.h>
}
#include <nlohmann/json.hpp>

#include <algorithm>
#include <set>

using json = nlohmann::ordered_json;

namespace {
const char * schema = R"COZO(
    ?[dataset, row_id, row_json] <- [['__probe__', '__probe__', '{}']]
    :create agent_data_rows {
        dataset: String,
        row_id: String =>
        row_json: String
    }
    ?[dataset, row_id] <- [['__probe__', '__probe__']]
    :delete agent_data_rows { dataset, row_id }
)COZO";

bool match_condition(const json & row, const json & condition) {
    if (!condition.is_object() || !condition.value("field", std::string()).size()) return false;
    const auto field = condition["field"].get<std::string>();
    const auto op = condition.value("operator", std::string("="));
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
    if (!request.is_object() || !request.contains("dataset")) { error = "data operation requires dataset"; return false; }
    const auto dataset = request["dataset"].get<std::string>();
    const json params = {{"dataset", dataset}};
    std::string raw;
    if (!run("?[row_id, row_json] := *agent_data_rows[$dataset, row_id, row_json]", params.dump(), raw, error)) return false;
    const auto rows_value = json::parse(raw, nullptr, false);
    if (!rows_value.is_object() || !rows_value.contains("rows")) { error = "Cozo data query returned invalid rows"; return false; }
    std::vector<json> rows;
    for (const auto & item : rows_value["rows"]) if (item.is_array() && item.size() >= 2) { const auto row = json::parse(item[1].get<std::string>(), nullptr, false); if (row.is_object()) rows.push_back(row); }

    if (operation == "data.filter") {
        for (auto it = rows.begin(); it != rows.end();) {
            bool keep = true; for (const auto & condition : request.value("conditions", json::array())) if (!match_condition(*it, condition)) keep = false;
            if (!keep) it = rows.erase(it); else ++it;
        }
    } else if (operation == "data.query") {
        for (auto it = rows.begin(); it != rows.end();) {
            bool keep = true; for (const auto & condition : request.value("where", json::array())) if (!match_condition(*it, condition)) keep = false;
            if (!keep) it = rows.erase(it); else ++it;
        }
    } else if (operation == "data.aggregate") {
        json output = json::object();
        for (const auto & measure : request.value("measures", json::array())) {
            if (!measure.is_object()) continue;
            const auto function = measure.value("function", std::string()); const auto column = measure.value("column", std::string());
            if (function == "count") output[measure.value("as", std::string("count"))] = rows.size();
            else if ((function == "sum" || function == "avg") && !column.empty()) { double sum = 0; size_t count = 0; for (const auto & row : rows) if (row.contains(column) && row[column].is_number()) { sum += row[column].get<double>(); ++count; } output[measure.value("as", function + "_" + column)] = function == "avg" && count ? sum / count : sum; }
        }
        result_json = json({{"rows", json::array({output})}, {"row_count", 1}}).dump(); return true;
    } else if (operation == "statistics.describe") {
        json columns = json::array();
        for (const auto & name : request.value("columns", json::array())) { if (!name.is_string()) continue; double sum = 0; size_t count = 0; for (const auto & row : rows) if (row.contains(name) && row[name].is_number()) { sum += row[name].get<double>(); ++count; } columns.push_back({{"name", name}, {"count", count}, {"mean", count ? sum / count : 0.0}}); }
        result_json = json({{"columns", columns}}).dump(); return true;
    } else if (operation == "data.join") {
        if (!request.contains("right") || !request["right"].is_string()) { error = "data.join requires a right dataset"; return false; }
        std::string right_raw;
        if (!run("?[row_id, row_json] := *agent_data_rows[$dataset, row_id, row_json]", json({{"dataset", request["right"]}}).dump(), right_raw, error)) return false;
        const auto right_value = json::parse(right_raw, nullptr, false); std::vector<json> right_rows;
        if (!right_value.is_object() || !right_value.contains("rows")) { error = "Cozo join returned invalid right rows"; return false; }
        for (const auto & item : right_value["rows"]) if (item.is_array() && item.size() >= 2) { const auto row = json::parse(item[1].get<std::string>(), nullptr, false); if (row.is_object()) right_rows.push_back(row); }
        json joined = json::array();
        for (const auto & left : rows) for (const auto & right : right_rows) {
            bool match = true;
            for (const auto & key : request.value("on", json::array())) if (key.is_object() && key.contains("left") && key.contains("right")) match = match && left.value(key["left"].get<std::string>(), json()) == right.value(key["right"].get<std::string>(), json());
            if (match) { json row = left; for (auto it = right.begin(); it != right.end(); ++it) if (!row.contains(it.key())) row[it.key()] = it.value(); joined.push_back(std::move(row)); }
        }
        result_json = json({{"rows", joined}, {"row_count", joined.size()}}).dump(); return true;
    } else if (operation == "data.transform") {
        for (auto & row : rows) for (const auto & transform : request.value("operations", json::array())) if (transform.is_object()) {
            const auto type = transform.value("type", std::string());
            if (type == "rename" && transform.contains("from") && transform.contains("to")) { const auto from = transform["from"].get<std::string>(), to = transform["to"].get<std::string>(); if (row.contains(from)) { row[to] = row[from]; row.erase(from); } }
            else if (type == "drop" && transform.contains("column")) row.erase(transform["column"].get<std::string>());
        }
        result_json = json({{"rows", rows}, {"row_count", rows.size()}}).dump(); return true;
    }

    const auto selected = request.value("select", json::array());
    const size_t limit = std::min<size_t>(request.value("limit", 1000), 1000);
    json columns = json::array(), output_rows = json::array();
    if (!selected.empty()) columns = selected;
    else if (!rows.empty()) for (auto it = rows[0].begin(); it != rows[0].end(); ++it) columns.push_back(it.key());
    for (const auto & row : rows) { if (output_rows.size() >= limit) break; if (selected.empty()) output_rows.push_back(row); else { json projected = json::array(); for (const auto & column : selected) projected.push_back(row.value(column.get<std::string>(), json())); output_rows.push_back(projected); } }
    result_json = json({{"columns", columns}, {"rows", output_rows}, {"row_count", output_rows.size()}, {"truncated", rows.size() > output_rows.size()}}).dump();
    return true;
}
