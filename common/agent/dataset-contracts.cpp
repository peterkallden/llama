#include "dataset-contracts.h"

#include <limits>
#include <cctype>
#include <sstream>
#include <algorithm>
#include <cstdlib>

namespace {

std::string normalized_table_name(const std::string & value) {
    std::string result;
    bool pending_space = false;
    for (const unsigned char character : value) {
        if (std::isspace(character)) {
            pending_space = !result.empty();
            continue;
        }
        if (pending_space) result.push_back(' ');
        pending_space = false;
        result.push_back(static_cast<char>(std::tolower(character)));
    }
    return result;
}

std::string trim_copy(const std::string & value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) --last;
    return value.substr(first, last - first);
}

bool parse_aggregate_expression(const std::string & expression, nlohmann::ordered_json & measure) {
    const auto text = trim_copy(expression);
    const auto open = text.find('(');
    if (open == std::string::npos || text.empty() || text.back() != ')' ||
            text.find('(', open + 1) != std::string::npos) return false;
    const auto function = normalized_table_name(text.substr(0, open));
    const auto column = trim_copy(text.substr(open + 1, text.size() - open - 2));
    if (column.empty() || (function != "count" && function != "sum" && function != "avg" &&
            function != "min" && function != "max") || (function == "count" && column != "*")) return false;
    measure = {{"function", function}};
    measure["column"] = function == "count" ? "*" : column;
    return true;
}

} // namespace

bool resolve_common_agent_document_table(
        const common_agent_document_table_catalog & catalog,
        const common_agent_document_table_locator & locator,
        common_agent_document_table_entry & resolved,
        std::string & error) {
    if (locator.table_index.has_value() && !locator.node_id.empty()) {
        error = "table locator must not combine table_index and node_id";
        return false;
    }
    if (locator.table_index.has_value()) {
        for (const auto & table : catalog.tables) {
            if (table.table_index == locator.table_index.value()) {
                resolved = table;
                error.clear();
                return true;
            }
        }
        error = "document table index was not found";
        return false;
    }
    if (!locator.node_id.empty()) {
        for (const auto & table : catalog.tables) {
            if (table.node_id == locator.node_id) {
                resolved = table;
                error.clear();
                return true;
            }
        }
        error = "document table node was not found";
        return false;
    }
    const auto wanted = normalized_table_name(locator.name);
    if (wanted.empty()) {
        error = "document table locator requires index, name or node_id";
        return false;
    }
    const common_agent_document_table_entry * match = nullptr;
    for (const auto & table : catalog.tables) {
        if (normalized_table_name(table.name) != wanted &&
                normalized_table_name(table.caption) != wanted) continue;
        if (match != nullptr) {
            error = "document table name is ambiguous";
            return false;
        }
        match = &table;
    }
    if (match == nullptr) {
        error = "document table name was not found";
        return false;
    }
    resolved = *match;
    error.clear();
    return true;
}

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

nlohmann::ordered_json common_agent_dataset_ref_to_json(
        const common_agent_dataset_ref & ref,
        common_agent_dataset_ref_json_projection projection) {
    nlohmann::ordered_json value = {
        {"uri", ref.uri},
        {"name", ref.name},
        {"row_count", ref.row_count},
        {"column_count", ref.column_count},
        {"source_resource_uri", ref.source_resource_uri},
        {"source_representation", ref.source_representation},
    };
    if (projection == common_agent_dataset_ref_json_projection::full) {
        value["source_provider"] = ref.source_provider;
        value["source_operation"] = ref.source_operation;
        value["source_request_json"] = ref.source_request_json;
        value["retrieved_at"] = ref.retrieved_at;
        value["content_hash"] = ref.content_hash;
    }
    return value;
}

bool common_agent_dataset_ref_from_json(
        const nlohmann::ordered_json & value,
        common_agent_dataset_ref & ref,
        std::string & error) {
    if (!value.is_object()) {
        error = "dataset ref JSON must be an object";
        return false;
    }
    ref = {};
    ref.uri = value.value("uri", std::string{});
    ref.name = value.value("name", std::string{});
    ref.row_count = value.value("row_count", size_t(0));
    ref.column_count = value.value("column_count", size_t(0));
    ref.source_resource_uri = value.value("source_resource_uri", std::string{});
    ref.source_representation = value.value("source_representation", std::string{});
    ref.source_provider = value.value("source_provider", std::string{});
    ref.source_operation = value.value("source_operation", std::string{});
    ref.source_request_json = value.value("source_request_json", std::string{});
    ref.retrieved_at = value.value("retrieved_at", int64_t(0));
    ref.content_hash = value.value("content_hash", std::string{});
    return validate_common_agent_dataset_ref(ref, error);
}

const char * common_agent_table_header_mode_name(
        common_agent_table_header_mode mode) {
    switch (mode) {
        case common_agent_table_header_mode::explicit_: return "explicit";
        case common_agent_table_header_mode::first_row: return "first_row";
        case common_agent_table_header_mode::first_column: return "first_column";
        case common_agent_table_header_mode::both: return "both";
        case common_agent_table_header_mode::none: return "none";
        case common_agent_table_header_mode::ambiguous: return "ambiguous";
    }
    return "ambiguous";
}

namespace {

bool looks_numeric(const std::string & value) {
    if (value.empty()) return false;
    char * end = nullptr;
    std::strtod(value.c_str(), &end);
    return end != nullptr && *end == '\0';
}

bool looks_like_header_value(const std::string & value) {
    if (value.empty() || value.size() > 128) return false;
    return !looks_numeric(value) && value.find_first_of("\r\n") == std::string::npos;
}

} // namespace

common_agent_table_header_mode classify_common_agent_table_headers(
        const std::vector<std::vector<std::string>> & rows,
        double & confidence,
        std::string & reason) {
    confidence = 0.0;
    reason.clear();
    if (rows.size() < 2 || rows.front().size() < 2) {
        reason = "table sample is too small to classify headers";
        return common_agent_table_header_mode::ambiguous;
    }
    const size_t width = rows.front().size();
    if (std::any_of(rows.begin(), rows.end(), [width](const auto & row) { return row.size() != width; })) {
        reason = "table is not rectangular";
        return common_agent_table_header_mode::ambiguous;
    }
    size_t first_row_text = 0;
    for (size_t index = 0; index < rows.front().size(); ++index) {
        if (looks_like_header_value(rows.front()[index]) ||
                (index == 0 && rows.front()[index].empty())) ++first_row_text;
    }
    size_t first_column_text = 0;
    for (size_t index = 1; index < rows.size(); ++index) if (looks_like_header_value(rows[index][0])) ++first_column_text;
    const double row_score = static_cast<double>(first_row_text) / width;
    const double column_score = static_cast<double>(first_column_text) / (rows.size() - 1);
    if (row_score >= 0.8 && column_score >= 0.8 && rows.front().front().empty()) {
        confidence = 0.6;
        reason = "first row and first column both look like labels";
        return common_agent_table_header_mode::both;
    }
    if (row_score >= 0.8) {
        confidence = row_score;
        reason = "first row is text-like while sampled data follows column shape";
        return common_agent_table_header_mode::first_row;
    }
    if (column_score >= 0.8) {
        confidence = column_score;
        reason = "first column is text-like while sampled data follows row shape";
        return common_agent_table_header_mode::first_column;
    }
    confidence = 0.5;
    reason = "header orientation is not distinguishable from the bounded sample";
    return common_agent_table_header_mode::ambiguous;
}

bool validate_common_agent_dataset_ref(
        const common_agent_dataset_ref & ref,
        std::string & error) {
    if (ref.uri.empty() || ref.name.empty()) {
        error = "dataset reference requires uri and name";
        return false;
    }
    if (ref.uri.rfind("dataset://", 0) != 0) {
        error = "dataset reference URI must use the canonical dataset:// scheme";
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
    } else if (tool_name == "statistics.describe" || tool_name == "statistics.outliers") {
        if (arguments.contains("group_by") && arguments["group_by"].is_string()) {
            arguments["group_by"] = nlohmann::ordered_json::array({arguments["group_by"]});
        }
        if (!arguments.contains("columns") && arguments.contains("column")) {
            if (!arguments["column"].is_string() || arguments["column"].get<std::string>().empty()) {
                error = "statistics.describe column must be a non-empty string";
                return false;
            }
            arguments["columns"] = nlohmann::ordered_json::array({arguments["column"]});
            arguments.erase("column");
        }
    } else if (tool_name == "statistics.value_counts") {
        if (!arguments.contains("column") || !arguments["column"].is_string() || arguments["column"].get<std::string>().empty()) {
            error = "statistics.value_counts requires a non-empty column";
            return false;
        }
    } else if (tool_name == "data.aggregate") {
        if (arguments.contains("group_by") && arguments["group_by"].is_string()) {
            arguments["group_by"] = nlohmann::ordered_json::array({arguments["group_by"]});
        }
        if (!arguments.contains("measures") && arguments.contains("select") && arguments["select"].is_string()) {
            nlohmann::ordered_json measure;
            if (parse_aggregate_expression(arguments["select"].get<std::string>(), measure)) {
                arguments["measures"] = nlohmann::ordered_json::array({std::move(measure)});
                arguments.erase("select");
            }
        }
        if (!arguments.contains("measures")) {
            nlohmann::ordered_json measures = nlohmann::ordered_json::array();
            for (const auto & function : {"count", "sum", "avg", "min", "max"}) {
                if (!arguments.contains(function)) continue;
                const auto & value = arguments[function];
                nlohmann::ordered_json measure = {{"function", function}};
                if (std::string(function) != "count") {
                    if (!value.is_string() || value.get<std::string>().empty()) {
                        error = std::string("data.aggregate ") + function + " must name a column";
                        return false;
                    }
                    measure["column"] = value;
                } else {
                    measure["column"] = value.is_string() ? value : "*";
                }
                measures.push_back(std::move(measure));
                arguments.erase(function);
            }
            if (!measures.empty()) arguments["measures"] = std::move(measures);
        }
    } else if (tool_name == "data.join" && arguments.contains("on") && arguments["on"].is_string()) {
        if (!arguments.contains("left") || !arguments.contains("right")) {
            error = "data.join shorthand requires left and right datasets";
            return false;
        }
        const auto column = arguments["on"];
        arguments["on"] = nlohmann::ordered_json::array({
            nlohmann::ordered_json{{"left", column}, {"right", column}}});
    } else if (tool_name == "dataset.sample" && !arguments.contains("rows") && arguments.contains("limit")) {
        arguments["rows"] = arguments["limit"];
        arguments.erase("limit");
    }
    error.clear();
    return true;
}
