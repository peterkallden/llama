#include "agent/adaptation/flydelta/flydelta-semantic-decision.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <regex>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char character) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    });
    return value;
}

bool ends_with(const std::string & value, const char * suffix) {
    const std::string suffix_value = suffix;
    return value.size() >= suffix_value.size() &&
        value.compare(value.size() - suffix_value.size(), suffix_value.size(), suffix_value) == 0;
}

std::string unquote(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2 &&
            ((value.front() == '\'' && value.back() == '\'') ||
             (value.front() == '"' && value.back() == '"'))) {
        value = value.substr(1, value.size() - 2);
    }
    return trim(std::move(value));
}

bool string_value(const json & object, const char * key, std::string & value) {
    if (!object.contains(key) || !object.at(key).is_string()) return false;
    value = object.at(key).get<std::string>();
    return true;
}

bool string_array(const json & object, const char * key, std::vector<std::string> & values) {
    if (!object.contains(key)) return false;
    const auto & source = object.at(key);
    if (source.is_string()) {
        values = {trim(source.get<std::string>())};
        return !values.front().empty();
    }
    if (!source.is_array()) return false;
    values.clear();
    for (const auto & item : source) {
        if (!item.is_string() || trim(item.get<std::string>()).empty()) return false;
        values.push_back(trim(item.get<std::string>()));
    }
    return true;
}

bool normalize_predicate_text(
        const std::string & source,
        common_flydelta_semantic_predicate & predicate) {
    static const std::regex pattern(
        R"(^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*(==|=|!=|>=|<=|>|<|equals|is)\s*(.*?)\s*$)",
        std::regex::icase);
    std::smatch match;
    if (!std::regex_match(source, match, pattern) || match.size() != 4) return false;
    predicate.field = trim(match[1].str());
    predicate.operation = lower(trim(match[2].str()));
    if (predicate.operation == "=" || predicate.operation == "==" ||
            predicate.operation == "is") predicate.operation = "eq";
    if (predicate.operation == "equals") predicate.operation = "eq";
    predicate.value = unquote(match[3].str());
    return !predicate.field.empty() && !predicate.value.empty();
}

bool normalize_predicates(const json & object, common_flydelta_semantic_decision & decision) {
    if (object.contains("predicate")) {
        if (!object.at("predicate").is_string()) return false;
        common_flydelta_semantic_predicate predicate;
        if (!normalize_predicate_text(object.at("predicate").get<std::string>(), predicate)) return false;
        decision.predicates.push_back(std::move(predicate));
        return true;
    }
    if (!object.contains("conditions") || !object.at("conditions").is_array()) return false;
    for (const auto & item : object.at("conditions")) {
        if (!item.is_object()) return false;
        std::string field;
        std::string operation;
        std::string value;
        if (!string_value(item, "field", field) || !string_value(item, "value", value)) return false;
        if (!string_value(item, "operator", operation) && !string_value(item, "operation", operation)) return false;
        common_flydelta_semantic_predicate predicate;
        if (!normalize_predicate_text(field + " " + operation + " " + value, predicate)) return false;
        decision.predicates.push_back(std::move(predicate));
    }
    return !decision.predicates.empty();
}

bool normalize_order(const json & object, common_flydelta_semantic_decision & decision) {
    if (!object.contains("order_by")) return false;
    if (object.at("order_by").is_string()) {
        const std::string value = trim(object.at("order_by").get<std::string>());
        const std::string lowered = lower(value);
        if (ends_with(lowered, " newest first") || ends_with(lowered, " latest first")) {
            const size_t suffix = 13;
            decision.order_by_field = trim(value.substr(0, value.size() - suffix));
            decision.order_by_direction = "desc";
        } else if (ends_with(lowered, " oldest first") || ends_with(lowered, " earliest first")) {
            const size_t suffix = ends_with(lowered, " oldest first") ? 13 : 14;
            decision.order_by_field = trim(value.substr(0, value.size() - suffix));
            decision.order_by_direction = "asc";
        } else {
            const auto separator = value.find_last_of(" \t");
            if (separator == std::string::npos) {
                decision.order_by_field = value;
                decision.order_by_direction = "asc";
            } else {
                decision.order_by_field = trim(value.substr(0, separator));
                decision.order_by_direction = lower(trim(value.substr(separator + 1)));
            }
        }
    } else if (object.at("order_by").is_object()) {
        if (!string_value(object.at("order_by"), "field", decision.order_by_field)) return false;
        if (!string_value(object.at("order_by"), "direction", decision.order_by_direction)) {
            decision.order_by_direction = "asc";
        }
        decision.order_by_direction = lower(decision.order_by_direction);
    } else {
        return false;
    }
    if (decision.order_by_direction == "descending" || decision.order_by_direction == "newest_first") {
        decision.order_by_direction = "desc";
    } else if (decision.order_by_direction == "ascending" || decision.order_by_direction == "oldest_first") {
        decision.order_by_direction = "asc";
    }
    return !decision.order_by_field.empty();
}

bool parse_json_text(const std::string & text, json & parsed) {
    const auto first = text.find('{');
    const auto last = text.rfind('}');
    if (first == std::string::npos || last < first) return false;
    parsed = json::parse(text.substr(first, last - first + 1), nullptr, false);
    return !parsed.is_discarded() && parsed.is_object();
}

} // namespace

const char * common_flydelta_semantic_decision_status_name(
        common_flydelta_semantic_decision_status status) {
    switch (status) {
        case common_flydelta_semantic_decision_status::valid: return "valid";
        case common_flydelta_semantic_decision_status::parse_failure: return "parse_failure";
        case common_flydelta_semantic_decision_status::unsupported_operation: return "unsupported_operation";
        case common_flydelta_semantic_decision_status::missing_field: return "missing_field";
        case common_flydelta_semantic_decision_status::invalid_value: return "invalid_value";
    }
    return "unknown";
}

bool common_flydelta_semantic_decision_validate(
        const common_flydelta_semantic_decision & decision,
        std::string & error) {
    error.clear();
    if (decision.schema_version != 1 || decision.operation.empty()) {
        error = "semantic decision requires a schema version and operation";
        return false;
    }
    if (decision.operation == "aggregate") {
        if (decision.group_by.empty() || decision.aggregate_function.empty() ||
                decision.aggregate_field.empty()) {
            error = "aggregate decision requires group_by and aggregate measure";
            return false;
        }
    } else if (decision.operation == "filter") {
        if (decision.predicates.empty()) {
            error = "filter decision requires a predicate";
        }
    } else if (decision.operation == "query") {
        if (decision.order_by_field.empty() || decision.order_by_direction.empty() || decision.limit <= 0) {
            error = "query decision requires order_by and a positive limit";
        }
    } else {
        error = "unsupported semantic decision operation";
    }
    return error.empty();
}

bool common_flydelta_parse_semantic_decision(
        const std::string & text,
        common_flydelta_semantic_decision & decision,
        common_flydelta_semantic_decision_status & status,
        std::string & error) {
    decision = {};
    status = common_flydelta_semantic_decision_status::parse_failure;
    error.clear();
    json root;
    if (!parse_json_text(text, root)) {
        error = "semantic decision is not a JSON object";
        return false;
    }
    json object = root;
    if (root.contains("name")) {
        if (!root.at("name").is_string() || !root.contains("arguments") || !root.at("arguments").is_object()) {
            error = "tool decision requires name and object arguments";
            return false;
        }
        object = root.at("arguments");
        const std::string name = root.at("name").get<std::string>();
        if (name == "data.aggregate") decision.operation = "aggregate";
        else if (name == "data.filter") decision.operation = "filter";
        else if (name == "data.query") decision.operation = "query";
        else {
            status = common_flydelta_semantic_decision_status::unsupported_operation;
            error = "unsupported dataset tool in semantic decision";
            return false;
        }
    } else {
        if (!string_value(root, "operation", decision.operation)) {
            error = "semantic decision requires operation";
            return false;
        }
        decision.operation = lower(trim(decision.operation));
    }
    if (object.contains("dataset")) {
        if (!object.at("dataset").is_string()) {
            status = common_flydelta_semantic_decision_status::invalid_value;
            error = "semantic decision dataset must be a string";
            return false;
        }
        decision.dataset = trim(object.at("dataset").get<std::string>());
    }
    if (decision.operation == "aggregate") {
        if (!string_array(object, "group_by", decision.group_by)) {
            status = common_flydelta_semantic_decision_status::missing_field;
            error = "aggregate decision requires group_by";
            return false;
        }
        if (object.contains("measure") && object.at("measure").is_string()) {
            decision.aggregate_function = "sum";
            decision.aggregate_field = trim(object.at("measure").get<std::string>());
        } else if (object.contains("measures") && object.at("measures").is_array() &&
                object.at("measures").size() == 1 && object.at("measures").front().is_object()) {
            const auto & measure = object.at("measures").front();
            if (!string_value(measure, "function", decision.aggregate_function) ||
                    (!string_value(measure, "column", decision.aggregate_field) &&
                     !string_value(measure, "field", decision.aggregate_field))) {
                status = common_flydelta_semantic_decision_status::missing_field;
                error = "aggregate decision requires one function and field";
                return false;
            }
            decision.aggregate_function = lower(trim(decision.aggregate_function));
            decision.aggregate_field = trim(decision.aggregate_field);
        } else {
            status = common_flydelta_semantic_decision_status::missing_field;
            error = "aggregate decision requires measure";
            return false;
        }
    } else if (decision.operation == "filter") {
        if (!normalize_predicates(object, decision)) {
            status = common_flydelta_semantic_decision_status::missing_field;
            error = "filter decision requires a supported predicate";
            return false;
        }
    } else if (decision.operation == "query") {
        if (!normalize_order(object, decision) || !object.contains("limit") || !object.at("limit").is_number_integer()) {
            status = common_flydelta_semantic_decision_status::missing_field;
            error = "query decision requires order_by and integer limit";
            return false;
        }
        decision.limit = object.at("limit").get<int>();
    } else {
        status = common_flydelta_semantic_decision_status::unsupported_operation;
        error = "unsupported semantic decision operation";
        return false;
    }
    if (!common_flydelta_semantic_decision_validate(decision, error)) {
        status = error.find("unsupported") != std::string::npos
            ? common_flydelta_semantic_decision_status::unsupported_operation
            : common_flydelta_semantic_decision_status::invalid_value;
        return false;
    }
    status = common_flydelta_semantic_decision_status::valid;
    return true;
}

bool common_flydelta_semantic_decision_equal(
        const common_flydelta_semantic_decision & left,
        const common_flydelta_semantic_decision & right) {
    return left.schema_version == right.schema_version && left.operation == right.operation &&
        left.dataset == right.dataset && left.group_by == right.group_by &&
        left.aggregate_function == right.aggregate_function && left.aggregate_field == right.aggregate_field &&
        left.predicates.size() == right.predicates.size() &&
        std::equal(left.predicates.begin(), left.predicates.end(), right.predicates.begin(),
            [](const auto & a, const auto & b) {
                return a.field == b.field && a.operation == b.operation && a.value == b.value;
            }) && left.order_by_field == right.order_by_field &&
        left.order_by_direction == right.order_by_direction && left.limit == right.limit;
}
