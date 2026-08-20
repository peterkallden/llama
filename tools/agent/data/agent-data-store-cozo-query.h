#pragma once

#include <nlohmann/json.hpp>

#include <vector>

using agent_cozo_json = nlohmann::ordered_json;

bool agent_cozo_match_condition(const agent_cozo_json & row, const agent_cozo_json & condition);
void agent_cozo_sort_rows(std::vector<agent_cozo_json> & rows, const agent_cozo_json & order_by);
