#include "agent-data-store-cozo-aggregate.h"

#include <map>

bool agent_cozo_execute_native_aggregate(
        const agent_cozo_aggregate_json & request,
        size_t max_scan_rows,
        size_t max_result_rows,
        const agent_cozo_query_runner & run,
        const agent_cozo_scan_metadata_reader & read_scan_metadata,
        std::string & result_json,
        std::string & error) {
    const auto dataset = request.value("dataset", std::string());
    const auto group_by = request.value("group_by", agent_cozo_aggregate_json::array());
    const auto measures = request.value("measures", agent_cozo_aggregate_json::array());
    if (dataset.empty() || !group_by.is_array() || !measures.is_array() || measures.empty()) { error = "data.aggregate requires dataset, group_by and measures"; return false; }
    for (const auto & field : group_by) if (!field.is_string() || field.get<std::string>().empty()) { error = "data.aggregate group_by fields must be strings"; return false; }

    agent_cozo_aggregate_json output_rows = agent_cozo_aggregate_json::array();
    std::map<std::string, size_t> output_index;
    for (const auto & measure : measures) {
        if (!measure.is_object()) { error = "data.aggregate measures must be objects"; return false; }
        const auto function = measure.value("function", std::string());
        const auto column = measure.value("column", std::string());
        const auto name = measure.value("as", function + (column.empty() ? std::string() : "_" + column));
        const bool count = function == "count";
        const bool numeric = function == "sum" || function == "avg" || function == "min" || function == "max";
        if ((!count && !numeric) || (!count && column.empty())) { error = "unsupported data.aggregate function"; return false; }

        agent_cozo_aggregate_json params = {{"dataset", dataset}, {"scan_limit", max_scan_rows}};
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
        const auto parsed = agent_cozo_aggregate_json::parse(raw, nullptr, false);
        if (!parsed.is_object() || !parsed.contains("rows")) { error = "Cozo aggregate query returned invalid rows"; return false; }
        for (const auto & row : parsed["rows"]) {
            if (!row.is_array() || row.size() < (group_by.size() * 2 + 1)) continue;
            agent_cozo_aggregate_json output = agent_cozo_aggregate_json::object();
            std::string key;
            size_t position = 0;
            for (size_t i = 0; i < group_by.size(); ++i) {
                const auto kind = row[position++].get<std::string>();
                const auto text = row[position++].get<std::string>();
                agent_cozo_aggregate_json value = text;
                if (kind == "number") value = agent_cozo_aggregate_json::parse(text, nullptr, false);
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
    result_json = agent_cozo_aggregate_json({{"rows", output_rows}, {"row_count", output_rows.size()}, {"scanned_rows", scanned_rows}, {"scan_truncated", scan_truncated}, {"scan_mode", "native_bounded"}, {"result_truncated", output_rows.size() >= max_result_rows}}).dump();
    return true;
}
