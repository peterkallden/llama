#include "agent-data-store-cozo-encoding.h"

agent_cozo_encoding_json agent_cozo_encode_row_values(
        const std::string & dataset,
        const std::string & row_id,
        const agent_cozo_encoding_json & row) {
    agent_cozo_encoding_json values = agent_cozo_encoding_json::array();
    for (auto it = row.begin(); it != row.end(); ++it) {
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
    return values;
}
