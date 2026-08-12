#include "dataset-contracts.h"

#include <limits>
#include <cctype>
#include <sstream>

namespace {

bool normalize_condition_value(
        const std::string & field,
        const nlohmann::ordered_json & value,
        nlohmann::ordered_json & output,
        std::string & error) {
    if (field.empty() || field.size() > 128) {
        error = "dataset condition field is out of bounds";
        return false;
    }
    if (value.is_object()) {
        if (value.size() != 1 || !value.begin().key().size()) {
            error = "dataset condition operator object must contain exactly one operator";
            return false;
        }
        const auto operator_name = value.begin().key();
        const auto & operand = value.begin().value();
        const auto canonical_operator =
            operator_name == "eq" ? "=" : operator_name == "neq" ? "!=" :
            operator_name == "gt" ? ">" : operator_name == "gte" ? ">=" :
            operator_name == "lt" ? "<" : operator_name == "lte" ? "<=" :
            operator_name == "in" ? "in" : operator_name == "is_null" ? "is_null" :
            operator_name == "not_null" ? "not_null" : nullptr;
        if (canonical_operator == nullptr) {
            error = "unsupported dataset condition operator: " + operator_name;
            return false;
        }
        if ((std::string(canonical_operator) == "in" &&
                    (!operand.is_array() || operand.empty() || operand.size() > 64)) ||
                ((std::string(canonical_operator) == "is_null" ||
                  std::string(canonical_operator) == "not_null") && !operand.is_boolean())) {
            error = "invalid operand for dataset condition operator: " + operator_name;
            return false;
        }
        output = { {"field", field}, {"operator", canonical_operator} };
        if (std::string(canonical_operator) != "is_null" && std::string(canonical_operator) != "not_null") {
            output["value"] = operand;
        }
        return true;
    }
    output = {{"field", field}, {"operator", "="}, {"value", value}};
    return true;
}

bool normalize_condition_list(
        const nlohmann::ordered_json & input,
        nlohmann::ordered_json & output,
        std::string & error) {
    output = nlohmann::ordered_json::array();
    if (input.is_array()) {
        if (input.size() > 16) { error = "dataset condition list is too large"; return false; }
        for (const auto & condition : input) {
            if (!condition.is_object() || !condition.contains("field") || !condition["field"].is_string() ||
                    condition["field"].get<std::string>().empty() || !condition.contains("operator")) {
                error = "dataset condition must use field, operator and value";
                return false;
            }
            const auto field = condition["field"].get<std::string>();
            const auto op = condition["operator"];
            if (!op.is_string()) { error = "dataset condition operator must be a string"; return false; }
            const auto name = op.get<std::string>();
            const auto canonical = name == "eq" ? "=" : name == "neq" ? "!=" : name == "gt" ? ">" :
                name == "gte" ? ">=" : name == "lt" ? "<" : name == "lte" ? "<=" : name;
            if (canonical != "=" && canonical != "!=" && canonical != ">" && canonical != ">=" &&
                    canonical != "<" && canonical != "<=" && canonical != "in" && canonical != "is_null" && canonical != "not_null") {
                error = "unsupported dataset condition operator: " + name;
                return false;
            }
            nlohmann::ordered_json normalized = {{"field", field}, {"operator", canonical}};
            if (canonical != "is_null" && canonical != "not_null") {
                if (!condition.contains("value")) { error = "dataset condition requires value"; return false; }
                normalized["value"] = condition["value"];
                if (canonical == "in" && (!normalized["value"].is_array() || normalized["value"].empty() || normalized["value"].size() > 64)) {
                    error = "dataset condition in value must be a bounded non-empty array";
                    return false;
                }
            }
            if (field.size() > 128) { error = "dataset condition field is out of bounds"; return false; }
            output.push_back(std::move(normalized));
        }
        return true;
    }
    if (input.is_string()) {
        const auto expression = input.get<std::string>();
        if (expression.empty() || expression.size() > 2048) { error = "dataset condition expression is out of bounds"; return false; }
        std::string part;
        bool quoted = false;
        size_t bracket_depth = 0;
        std::vector<std::string> parts;
        for (size_t i = 0, start = 0; i <= expression.size(); ++i) {
            const char ch = i < expression.size() ? expression[i] : '\0';
            if (ch == '"' && (i == 0 || expression[i - 1] != '\\')) quoted = !quoted;
            if (!quoted && ch == '[') ++bracket_depth;
            if (!quoted && ch == ']' && bracket_depth > 0) --bracket_depth;
            const bool separator = !quoted && bracket_depth == 0 && i + 5 <= expression.size() &&
                expression.compare(i, 5, " and ") == 0;
            if ((separator || ch == '\0') && !quoted && bracket_depth == 0) {
                part = expression.substr(start, i - start);
                if (part.empty()) { error = "dataset condition expression contains an empty predicate"; return false; }
                parts.push_back(part);
                if (separator) { i += 4; start = i + 1; }
            }
        }
        if (parts.empty() || parts.size() > 16) { error = "dataset condition expression has too many predicates"; return false; }
        for (const auto & predicate : parts) {
            std::istringstream stream(predicate);
            std::string field, operator_name;
            if (!(stream >> field >> operator_name) || field.size() > 128) { error = "dataset condition expression requires field and operator"; return false; }
            std::string operand;
            std::getline(stream, operand);
            const auto first = operand.find_first_not_of(" \t");
            if (first == std::string::npos) { error = "dataset condition expression requires a value"; return false; }
            operand.erase(0, first);
            const auto canonical = operator_name == "eq" || operator_name == "-eq" ? "=" :
                operator_name == "neq" || operator_name == "-neq" ? "!=" :
                operator_name == "gt" || operator_name == "-gt" ? ">" :
                operator_name == "gte" || operator_name == "-gte" ? ">=" :
                operator_name == "lt" || operator_name == "-lt" ? "<" :
                operator_name == "lte" || operator_name == "-lte" ? "<=" : operator_name;
            if (canonical != "=" && canonical != "!=" && canonical != ">" && canonical != ">=" &&
                    canonical != "<" && canonical != "<=" && canonical != "in") {
                error = "unsupported dataset condition expression operator: " + operator_name;
                return false;
            }
            auto value = nlohmann::ordered_json::parse(operand, nullptr, false);
            if (value.is_discarded()) value = operand;
            nlohmann::ordered_json normalized = {{"field", field}, {"operator", canonical}, {"value", value}};
            if (canonical == "in" && (!value.is_array() || value.empty() || value.size() > 64)) { error = "dataset condition expression in value must be a bounded array"; return false; }
            output.push_back(std::move(normalized));
        }
        return true;
    }
    if (!input.is_object() || input.empty() || input.size() > 16) {
        error = "dataset conditions must be an array or a bounded object";
        return false;
    }
    for (auto it = input.begin(); it != input.end(); ++it) {
        nlohmann::ordered_json normalized;
        if (!normalize_condition_value(it.key(), it.value(), normalized, error)) return false;
        output.push_back(std::move(normalized));
    }
    return true;
}

} // namespace

const char * common_agent_dataset_column_type_name(
        common_agent_dataset_column_type type) {
    switch (type) {
        case common_agent_dataset_column_type::null_: return "null";
        case common_agent_dataset_column_type::boolean: return "boolean";
        case common_agent_dataset_column_type::integer: return "integer";
        case common_agent_dataset_column_type::decimal: return "decimal";
        case common_agent_dataset_column_type::string: return "string";
        case common_agent_dataset_column_type::date: return "date";
        case common_agent_dataset_column_type::datetime: return "datetime";
        case common_agent_dataset_column_type::binary: return "binary";
        case common_agent_dataset_column_type::unknown: return "unknown";
    }
    return "unknown";
}

bool validate_common_agent_dataset_ref(
        const common_agent_dataset_ref & ref,
        std::string & error) {
    if (ref.uri.empty() || ref.name.empty()) {
        error = "dataset reference requires uri and name";
        return false;
    }
    if (ref.source_resource_uri.empty()) {
        error = "dataset reference requires source resource provenance";
        return false;
    }
    if (ref.column_count == 0 && ref.row_count != 0) {
        error = "dataset reference cannot have rows without columns";
        return false;
    }
    error.clear();
    return true;
}

bool validate_common_agent_dataset_descriptor(
        const common_agent_dataset_descriptor & descriptor,
        const common_agent_dataset_limits & limits,
        std::string & error) {
    if (!validate_common_agent_dataset_ref(descriptor.ref, error)) return false;
    if (descriptor.ref.row_count > limits.max_rows ||
            descriptor.ref.column_count > limits.max_columns) {
        error = "dataset shape exceeds host limits";
        return false;
    }
    if (descriptor.ref.column_count != descriptor.columns.size()) {
        error = "dataset column count does not match schema";
        return false;
    }
    if (descriptor.ref.row_count != 0 &&
            descriptor.ref.column_count > std::numeric_limits<size_t>::max() / descriptor.ref.row_count) {
        error = "dataset cell count overflows host limits";
        return false;
    }
    if (descriptor.ref.row_count * descriptor.ref.column_count > limits.max_cells) {
        error = "dataset cell count exceeds host limits";
        return false;
    }
    for (const auto & column : descriptor.columns) {
        if (column.name.empty()) {
            error = "dataset schema contains an unnamed column";
            return false;
        }
    }
    if (descriptor.import_processor_id.empty()) {
        error = "dataset descriptor requires import processor provenance";
        return false;
    }
    error.clear();
    return true;
}

bool normalize_common_agent_dataset_tool_arguments(
        const std::string & tool_name,
        nlohmann::ordered_json & arguments,
        std::string & error) {
    if (tool_name == "data.query") {
        if (arguments.contains("where")) {
            nlohmann::ordered_json normalized;
            if (!normalize_condition_list(arguments["where"], normalized, error)) return false;
            arguments["where"] = std::move(normalized);
        }
    } else if (tool_name == "data.filter") {
        if (!arguments.contains("conditions")) { error = "data.filter requires conditions"; return false; }
        nlohmann::ordered_json normalized;
        if (!normalize_condition_list(arguments["conditions"], normalized, error)) return false;
        arguments["conditions"] = std::move(normalized);
    }
    error.clear();
    return true;
}
