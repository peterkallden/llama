#include "agent-data-store-cozo-join.h"

bool agent_cozo_execute_native_join(
        const agent_cozo_aggregate_json & request,
        size_t max_scan_rows,
        size_t max_result_rows,
        const agent_cozo_query_runner & run,
        const agent_cozo_scan_metadata_reader & read_scan_metadata,
        std::string & result_json,
        std::string & error) {
    const auto left = request.value("left", std::string());
    const auto right = request.value("right", std::string());
    const auto on = request.value("on", agent_cozo_aggregate_json::array());
    const auto type = request.value("type", std::string("inner"));
    if (left.empty() || right.empty() || !on.is_array() || on.empty()) { error = "data.join requires left, right and on"; return false; }
    if (type != "inner" && type != "left") { error = "unsupported data.join type"; return false; }
    const auto query_limit = max_result_rows + 1;
    std::string script = "?[left_json, right_json] := ";
    agent_cozo_aggregate_json params = {{"left", left}, {"right", right}, {"scan_limit", max_scan_rows}};
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
    const auto parsed = agent_cozo_aggregate_json::parse(raw, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows")) { error = "Cozo join query returned invalid rows"; return false; }
    agent_cozo_aggregate_json joined = agent_cozo_aggregate_json::array();
    for (const auto & row : parsed["rows"]) if (row.is_array() && row.size() >= 2) {
        const auto left_row = agent_cozo_aggregate_json::parse(row[0].get<std::string>(), nullptr, false);
        const auto right_row = agent_cozo_aggregate_json::parse(row[1].get<std::string>(), nullptr, false);
        if (!left_row.is_object() || !right_row.is_object()) continue;
        agent_cozo_aggregate_json merged = left_row;
        for (auto it = right_row.begin(); it != right_row.end(); ++it) if (!merged.contains(it.key())) merged[it.key()] = it.value();
        joined.push_back(std::move(merged));
    }
    bool result_truncated = joined.size() > max_result_rows;
    if (joined.size() > max_result_rows) {
        agent_cozo_aggregate_json limited = agent_cozo_aggregate_json::array();
        for (size_t i = 0; i < max_result_rows; ++i) limited.push_back(joined[i]);
        joined = std::move(limited);
    }

    if (type == "left" && !result_truncated) {
        std::string unmatched_script = "left_match[left_id] := ";
        for (size_t i = 0; i < on.size(); ++i) {
            unmatched_script += "*agent_data_values[$left, left_id, $left_field" + std::to_string(i) + ", left_kind" + std::to_string(i) + ", left_text" + std::to_string(i) + ", left_number" + std::to_string(i) + "], ";
            unmatched_script += "*agent_data_row_order[$right, right_id, right_seq], right_seq <= $scan_limit, *agent_data_values[$right, right_id, $right_field" + std::to_string(i) + ", right_kind" + std::to_string(i) + ", right_text" + std::to_string(i) + ", right_number" + std::to_string(i) + "], ";
            unmatched_script += "left_kind" + std::to_string(i) + " == right_kind" + std::to_string(i) + ", left_text" + std::to_string(i) + " == right_text" + std::to_string(i) + ", ";
        }
        if (!unmatched_script.empty() && unmatched_script.back() == ' ') unmatched_script.pop_back();
        unmatched_script += "\n?[left_json] := *agent_data_row_order[$left, left_id, left_seq], left_seq <= $scan_limit, *agent_data_rows[$left, left_id, left_json], not left_match[left_id] :limit " + std::to_string(query_limit);
        std::string unmatched_raw;
        if (!run(unmatched_script, params.dump(), unmatched_raw, error)) return false;
        const auto unmatched = agent_cozo_aggregate_json::parse(unmatched_raw, nullptr, false);
        if (!unmatched.is_object() || !unmatched.contains("rows")) { error = "Cozo left join query returned invalid rows"; return false; }
        for (const auto & row : unmatched["rows"]) {
            if (!row.is_array() || row.empty() || !row[0].is_string()) continue;
            const auto left_row = agent_cozo_aggregate_json::parse(row[0].get<std::string>(), nullptr, false);
            if (!left_row.is_object()) continue;
            if (joined.size() < max_result_rows) joined.push_back(left_row);
            else { result_truncated = true; break; }
        }
    }
    size_t left_scanned = 0, right_scanned = 0;
    bool left_truncated = false, right_truncated = false;
    if (!read_scan_metadata(left, max_scan_rows, left_scanned, left_truncated, error) || !read_scan_metadata(right, max_scan_rows, right_scanned, right_truncated, error)) return false;
    result_json = agent_cozo_aggregate_json({{"rows", joined}, {"row_count", joined.size()}, {"scanned_rows", left_scanned + right_scanned}, {"scan_truncated", left_truncated || right_truncated}, {"scan_mode", "native_bounded"}, {"result_truncated", result_truncated}}).dump();
    return true;
}
