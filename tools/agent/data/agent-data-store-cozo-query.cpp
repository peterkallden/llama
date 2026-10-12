#include "agent-data-store-cozo-query.h"

#include <algorithm>

bool agent_cozo_match_condition(const agent_cozo_json & row, const agent_cozo_json & condition) {
    if (!condition.is_object() || !condition.value("field", std::string()).size()) return false;
    const auto field = condition["field"].get<std::string>();
    const auto op = condition.value("operator", std::string("="));
    if (op == "is_null") return !row.contains(field) || row[field].is_null();
    if (op == "not_null") return row.contains(field) && !row[field].is_null();
    if (!row.contains(field)) return false;
    const auto & value = row[field];
    const auto expected = condition.value("value", agent_cozo_json());
    if (op == "=") return value == expected;
    if (op == "!=") return value != expected;
    if ((op == ">" || op == ">=" || op == "<" || op == "<=") && value.is_number() && expected.is_number()) {
        const double a = value.get<double>(), b = expected.get<double>();
        return op == ">" ? a > b : op == ">=" ? a >= b : op == "<" ? a < b : a <= b;
    }
    return false;
}

void agent_cozo_sort_rows(std::vector<agent_cozo_json> & rows, const agent_cozo_json & order_by) {
    if (!order_by.is_array() || order_by.empty()) return;
    std::stable_sort(rows.begin(), rows.end(), [&order_by](const agent_cozo_json & left, const agent_cozo_json & right) {
        for (const auto & item : order_by) {
            if (!item.is_object() || !item.value("field", std::string()).size()) continue;
            const auto field = item["field"].get<std::string>();
            const auto direction = item.value("direction", std::string("asc"));
            const auto l = left.value(field, agent_cozo_json());
            const auto r = right.value(field, agent_cozo_json());
            if (l == r) continue;
            const bool less = l < r;
            return direction == "desc" ? !less : less;
        }
        return false;
    });
}
