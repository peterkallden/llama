#pragma once

#include <nlohmann/json.hpp>

#include <functional>
#include <string>

using agent_cozo_aggregate_json = nlohmann::ordered_json;
using agent_cozo_query_runner = std::function<bool(
        const std::string &, const std::string &, std::string &, std::string &)>;
using agent_cozo_scan_metadata_reader = std::function<bool(
        const std::string &, size_t, size_t &, bool &, std::string &)>;

bool agent_cozo_execute_native_aggregate(
        const agent_cozo_aggregate_json & request,
        size_t max_scan_rows,
        size_t max_result_rows,
        const agent_cozo_query_runner & run,
        const agent_cozo_scan_metadata_reader & read_scan_metadata,
        std::string & result_json,
        std::string & error);
