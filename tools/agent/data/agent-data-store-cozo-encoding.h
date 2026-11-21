#pragma once

#include <nlohmann/json.hpp>

using agent_cozo_encoding_json = nlohmann::ordered_json;

agent_cozo_encoding_json agent_cozo_encode_row_values(
        const std::string & dataset,
        const std::string & row_id,
        const agent_cozo_encoding_json & row);
