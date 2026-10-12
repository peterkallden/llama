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

namespace {

int progress_rank(const common_flydelta_semantic_progress_field_state state) {
    switch (state) {
        case common_flydelta_semantic_progress_field_state::matched: return 2;
        case common_flydelta_semantic_progress_field_state::partial: return 1;
        case common_flydelta_semantic_progress_field_state::mismatch:
        case common_flydelta_semantic_progress_field_state::missing: return 0;
        case common_flydelta_semantic_progress_field_state::unknown: return -1;
    }
    return -1;
}

common_flydelta_semantic_progress_field_state compare_value(
        const std::string & actual, const std::string & expected,
        const bool partial = false) {
    if (actual.empty()) return common_flydelta_semantic_progress_field_state::missing;
    if (expected.empty()) return common_flydelta_semantic_progress_field_state::unknown;
    if (lower(trim(actual)) == lower(trim(expected))) {
        return common_flydelta_semantic_progress_field_state::matched;
    }
    return partial ? common_flydelta_semantic_progress_field_state::partial :
        common_flydelta_semantic_progress_field_state::mismatch;
}

std::string json_string_or_empty(const json & object, const char * key) {
    return object.contains(key) && object.at(key).is_string()
        ? trim(object.at(key).get<std::string>()) : std::string{};
}

bool json_string_array_or_first(
        const json & object, const char * key, std::string & value) {
    if (!object.contains(key)) return false;
    const auto & item = object.at(key);
    if (item.is_string()) {
        value = trim(item.get<std::string>());
        return !value.empty();
    }
    if (!item.is_array() || item.empty() || !item.front().is_string()) return false;
    value = trim(item.front().get<std::string>());
    return !value.empty();
}

bool observe_fields(
        const std::string & generated,
        common_flydelta_semantic_decision & observed,
        bool & strict_valid,
        bool & grouping_alias,
        bool & measure_alias) {
    common_flydelta_semantic_decision_status status;
    std::string strict_error;
    strict_valid = common_flydelta_parse_semantic_decision(
        generated, observed, status, strict_error);
    if (strict_valid) return true;

    json root;
    if (!parse_json_text(generated, root)) return false;
    json object = root;
    if (root.contains("name")) {
        if (!root.at("name").is_string() || !root.contains("arguments") ||
                !root.at("arguments").is_object()) return false;
        object = root.at("arguments");
        const std::string name = root.at("name").get<std::string>();
        if (name == "data.aggregate") observed.operation = "aggregate";
        else if (name == "data.filter") observed.operation = "filter";
        else if (name == "data.query") observed.operation = "query";
        else {
            // Keep unsupported operations observable for diagnostic progress.  This
            // is deliberately not a relaxed execution path: strict_valid remains
            // false and the normal host verifier still rejects the call.
            const auto separator = name.rfind('.');
            observed.operation = name.substr(separator == std::string::npos ? 0 : separator + 1);
        }
    } else if (object.contains("operation") && object.at("operation").is_string()) {
        observed.operation = lower(trim(object.at("operation").get<std::string>()));
    } else {
        return false;
    }
    observed.dataset = json_string_or_empty(object, "dataset");
    std::string grouping_value;
    if (object.contains("group_by")) {
        grouping_alias = json_string_array_or_first(object, "group_by", grouping_value);
    } else if (object.contains("groupby")) {
        grouping_alias = json_string_array_or_first(object, "groupby", grouping_value);
    }
    if (!grouping_value.empty()) observed.group_by = {grouping_value};
    if (observed.operation == "aggregate") {
        if (object.contains("measure") && object.at("measure").is_string()) {
            observed.aggregate_field = trim(object.at("measure").get<std::string>());
        } else if (object.contains("measures") && object.at("measures").is_array() &&
                !object.at("measures").empty() && object.at("measures").front().is_object()) {
            const auto & measure = object.at("measures").front();
            observed.aggregate_field = json_string_or_empty(measure, "column");
            if (observed.aggregate_field.empty()) observed.aggregate_field =
                json_string_or_empty(measure, "field");
        } else if (object.contains("columns")) {
            measure_alias = json_string_array_or_first(object, "columns", observed.aggregate_field);
        }
    } else if (object.contains("columns")) {
        // A failed/competing tool often still carries the requested field.  Keep
        // that evidence so a later candidate can be reported as partial progress.
        measure_alias = json_string_array_or_first(object, "columns", observed.aggregate_field);
    }
    return !observed.operation.empty() || !observed.dataset.empty() ||
        !observed.group_by.empty() || !observed.aggregate_field.empty();
}

} // namespace

const char * common_flydelta_semantic_progress_field_state_name(
        common_flydelta_semantic_progress_field_state state) {
    switch (state) {
        case common_flydelta_semantic_progress_field_state::unknown: return "unknown";
        case common_flydelta_semantic_progress_field_state::missing: return "missing";
        case common_flydelta_semantic_progress_field_state::mismatch: return "mismatch";
        case common_flydelta_semantic_progress_field_state::partial: return "partial";
        case common_flydelta_semantic_progress_field_state::matched: return "matched";
    }
    return "unknown";
}

const char * common_flydelta_semantic_progress_outcome_name(
        common_flydelta_semantic_progress_outcome outcome) {
    switch (outcome) {
        case common_flydelta_semantic_progress_outcome::unknown: return "unknown";
        case common_flydelta_semantic_progress_outcome::unchanged: return "unchanged";
        case common_flydelta_semantic_progress_outcome::improved: return "improved";
        case common_flydelta_semantic_progress_outcome::solved: return "solved";
        case common_flydelta_semantic_progress_outcome::regressed: return "regressed";
    }
    return "unknown";
}

bool common_flydelta_observe_semantic_progress(
        const std::string & generated,
        const common_flydelta_semantic_decision & expected,
        common_flydelta_semantic_progress_observation & observation,
        std::string & error) {
    observation = {};
    error.clear();
    if (!common_flydelta_semantic_decision_validate(expected, error)) return false;

    common_flydelta_semantic_decision observed;
    bool strict_valid = false;
    bool grouping_alias = false;
    bool measure_alias = false;
    if (!observe_fields(generated, observed, strict_valid, grouping_alias, measure_alias)) {
        error = "semantic progress observation found no bounded decision fields";
        return false;
    }
    observation.available = true;
    observation.contract_valid = strict_valid;
    observation.operation = observed.operation;
    observation.dataset = observed.dataset;
    if (!observed.group_by.empty()) {
        observation.grouping = observed.group_by.front();
    }
    observation.measure = observed.aggregate_field;
    observation.operation_state = compare_value(
        observation.operation, expected.operation);
    observation.dataset_state = compare_value(observation.dataset, expected.dataset);
    observation.grouping_state = compare_value(
        observation.grouping, expected.group_by.empty() ? std::string{} : expected.group_by.front(),
        grouping_alias);
    observation.measure_state = compare_value(
        observation.measure,
        expected.aggregate_field,
        measure_alias);
    observation.contract_state = strict_valid
        ? common_flydelta_semantic_progress_field_state::matched
        : common_flydelta_semantic_progress_field_state::partial;
    observation.score = progress_rank(observation.operation_state) +
        progress_rank(observation.dataset_state) +
        progress_rank(observation.grouping_state) +
        progress_rank(observation.measure_state) +
        progress_rank(observation.contract_state);
    return true;
}

common_flydelta_semantic_progress common_flydelta_compare_semantic_progress(
        const common_flydelta_semantic_progress_observation & baseline,
        const common_flydelta_semantic_progress_observation & candidate,
        const bool baseline_strict_passed,
        const bool candidate_strict_passed) {
    common_flydelta_semantic_progress result;
    result.baseline_score = baseline.score;
    result.candidate_score = candidate.score;
    result.comparable = baseline.available && candidate.available;
    if (candidate_strict_passed) {
        result.outcome = common_flydelta_semantic_progress_outcome::solved;
    } else if (baseline_strict_passed && candidate.available) {
        result.outcome = common_flydelta_semantic_progress_outcome::regressed;
    } else if (!result.comparable) {
        result.outcome = common_flydelta_semantic_progress_outcome::unknown;
    } else if (candidate.score > baseline.score) {
        result.outcome = common_flydelta_semantic_progress_outcome::improved;
    } else if (candidate.score < baseline.score) {
        result.outcome = common_flydelta_semantic_progress_outcome::regressed;
    } else {
        result.outcome = common_flydelta_semantic_progress_outcome::unchanged;
    }

    const auto add_dimension = [&](const char * name,
            const common_flydelta_semantic_progress_field_state before,
            const common_flydelta_semantic_progress_field_state after) {
        if (progress_rank(after) > progress_rank(before)) result.improved_dimensions.emplace_back(name);
        if (progress_rank(after) < progress_rank(before)) result.regressed_dimensions.emplace_back(name);
    };
    add_dimension("operation", baseline.operation_state, candidate.operation_state);
    add_dimension("dataset", baseline.dataset_state, candidate.dataset_state);
    add_dimension("grouping", baseline.grouping_state, candidate.grouping_state);
    add_dimension("measure", baseline.measure_state, candidate.measure_state);
    add_dimension("contract_shape", baseline.contract_state, candidate.contract_state);
    if (candidate.grouping_state != common_flydelta_semantic_progress_field_state::matched) {
        result.residual_dimensions.emplace_back("grouping");
    }
    if (candidate.contract_state != common_flydelta_semantic_progress_field_state::matched) {
        result.residual_dimensions.emplace_back("contract_shape");
    }
    return result;
}
