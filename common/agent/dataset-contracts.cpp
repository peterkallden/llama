#include "dataset-contracts.h"

#include <limits>
#include <iomanip>
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
            operator_name == "not_null" ? "not_null" : operator_name == "contains" ? "contains" : nullptr;
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
        std::string & error);

bool parse_membership_list(
        const std::string & text,
        nlohmann::ordered_json & values,
        std::string & error) {
    using json = nlohmann::ordered_json;
    const auto trimmed = trim_copy(text);
    if (trimmed.size() < 2 || trimmed.front() != '[' || trimmed.back() != ']') {
        error = "dataset condition in value must be a bounded non-empty array";
        return false;
    }

    const auto parsed = json::parse(trimmed, nullptr, false);
    if (!parsed.is_discarded()) {
        if (!parsed.is_array() || parsed.empty() || parsed.size() > 64) {
            error = "dataset condition in value must be a bounded non-empty array";
            return false;
        }
        for (const auto & item : parsed) {
            if (item.is_array() || item.is_object()) {
                error = "dataset condition in values must be scalars";
                return false;
            }
        }
        values = parsed;
        return true;
    }

    // The condition DSL also permits bare identifiers in a membership list:
    // `region in [north, east]`. Quoted strings and JSON scalar values retain
    // their normal types; nested structures are deliberately not supported.
    values = json::array();
    const std::string body = trimmed.substr(1, trimmed.size() - 2);
    size_t start = 0;
    bool quoted = false, escaped = false;
    for (size_t i = 0; i <= body.size(); ++i) {
        const char ch = i < body.size() ? body[i] : ',';
        if (quoted) {
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') quoted = false;
            continue;
        }
        if (i < body.size() && ch == '"') { quoted = true; continue; }
        if (i < body.size() && (ch == '[' || ch == ']' || ch == '{' || ch == '}')) {
            error = "dataset condition in values must be scalars";
            return false;
        }
        if (ch != ',') continue;
        const auto token = trim_copy(body.substr(start, i - start));
        if (token.empty()) {
            error = "dataset condition in value must be a bounded non-empty array";
            return false;
        }
        auto item = json::parse(token, nullptr, false);
        if (item.is_discarded()) {
            if (token.find_first_not_of(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-") != std::string::npos) {
                error = "dataset condition in list contains an invalid scalar";
                return false;
            }
            item = token;
        }
        if (item.is_array() || item.is_object()) {
            error = "dataset condition in values must be scalars";
            return false;
        }
        values.push_back(std::move(item));
        if (values.size() > 64) {
            error = "dataset condition in value must be a bounded non-empty array";
            return false;
        }
        start = i + 1;
    }
    if (quoted) {
        error = "dataset condition in list contains an unterminated string";
        return false;
    }
    return !values.empty();
}

// Preserve the store's conjunction-only contract. A same-field equality
// disjunction is represented exactly by its existing membership predicate.
bool parse_condition_expression(
        const std::string & source,
        nlohmann::ordered_json & output,
        std::string & error,
        size_t depth) {
    using json = nlohmann::ordered_json;
    const auto first = source.find_first_not_of(" \t\r\n");
    const auto last = source.find_last_not_of(" \t\r\n");
    if (first == std::string::npos || source.size() > 2048 || depth >= 12) {
        error = "dataset condition expression is empty or exceeds its size/depth limit";
        return false;
    }
    const std::string text = source.substr(first, last - first + 1);
    std::vector<std::pair<size_t, size_t>> conjunctions, disjunctions;
    bool quoted = false, escaped = false, outer_group = text.front() == '(';
    size_t parentheses = 0, brackets = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (quoted) {
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') quoted = false;
            continue;
        }
        if (ch == '"') { quoted = true; continue; }
        if (ch == '[') { ++brackets; continue; }
        if (ch == ']') {
            if (!brackets) { error = "dataset condition has unmatched brackets"; return false; }
            --brackets;
            continue;
        }
        if (brackets) continue;
        if (ch == '(') { ++parentheses; continue; }
        if (ch == ')') {
            if (!parentheses) { error = "dataset condition has unmatched parentheses"; return false; }
            if (--parentheses == 0 && i + 1 != text.size()) outer_group = false;
            continue;
        }
        if (parentheses) continue;
        if (ch == '&') { conjunctions.emplace_back(i, 1); continue; }
        const auto keyword = [&](const std::string & word) {
            if (i && !std::isspace(static_cast<unsigned char>(text[i - 1]))) return false;
            if (i + word.size() > text.size()) return false;
            if (i + word.size() < text.size() &&
                    !std::isspace(static_cast<unsigned char>(text[i + word.size()]))) return false;
            for (size_t j = 0; j < word.size(); ++j) {
                if (std::tolower(static_cast<unsigned char>(text[i + j])) != word[j]) return false;
            }
            return true;
        };
        if (keyword("or")) disjunctions.emplace_back(i, 2);
        else if (keyword("and")) conjunctions.emplace_back(i, 3);
    }
    if (quoted || parentheses || brackets) {
        error = "dataset condition contains an unterminated string or group";
        return false;
    }
    if (outer_group) return parse_condition_expression(text.substr(1, text.size() - 2), output, error, depth + 1);
    // Conventional precedence: AND binds more tightly than OR. Expressions
    // that cannot be represented by the store are rejected, never regrouped.
    const bool disjunction = !disjunctions.empty();
    const auto & separators = disjunction ? disjunctions : conjunctions;
    if (!separators.empty()) {
        output = json::array();
        size_t begin = 0;
        for (size_t i = 0; i <= separators.size(); ++i) {
            const size_t end = i < separators.size() ? separators[i].first : text.size();
            json children;
            if (!parse_condition_expression(text.substr(begin, end - begin), children, error, depth + 1)) return false;
            if (disjunction) {
                if (children.size() != 1 ||
                        (children[0]["operator"] != "=" && children[0]["operator"] != "in")) {
                    error = "dataset OR supports only equality alternatives for the same field";
                    return false;
                }
                const auto & child = children[0];
                if (output.empty()) output.push_back({{"field", child["field"]}, {"operator", "in"}, {"value", json::array()}});
                if (output[0]["field"] != child["field"]) {
                    error = "dataset OR alternatives must refer to the same field";
                    return false;
                }
                const json values = child["operator"] == "in" ? child["value"] : json::array({child["value"]});
                for (const auto & value : values) {
                    auto & members = output[0]["value"];
                    if (std::find(members.begin(), members.end(), value) == members.end()) members.push_back(value);
                    if (members.size() > 64) { error = "dataset OR has too many alternatives"; return false; }
                }
            } else {
                for (auto & child : children) output.push_back(std::move(child));
                if (output.size() > 16) { error = "dataset condition list is too large"; return false; }
            }
            if (i < separators.size()) begin = end + separators[i].second;
        }
        return true;
    }
    std::istringstream stream(text);
    std::string field, op, operand;
    if (!(stream >> field >> std::quoted(op)) || field.empty() || field.size() > 128) {
        error = "dataset condition expression requires field and operator";
        return false;
    }
    const auto canonical = op == "eq" || op == "-eq" ? "=" :
        op == "neq" || op == "-neq" ? "!=" : op == "gt" || op == "-gt" ? ">" :
        op == "gte" || op == "-gte" ? ">=" : op == "lt" || op == "-lt" ? "<" :
        op == "lte" || op == "-lte" ? "<=" : op;
    json condition = {{"field", field}, {"operator", canonical}};
    std::getline(stream, operand);
    const auto value_start = operand.find_first_not_of(" \t\r\n");
    if (canonical == "is_null" || canonical == "not_null") {
        if (value_start != std::string::npos) { error = "dataset null check does not accept a value"; return false; }
    } else {
        if (value_start == std::string::npos) { error = "dataset condition expression requires a value"; return false; }
        operand = operand.substr(value_start, operand.find_last_not_of(" \t\r\n") - value_start + 1);
        auto value = json::parse(operand, nullptr, false);
        if (canonical == "in" && operand.front() == '[') {
            if (!parse_membership_list(operand, value, error)) return false;
        } else if (value.is_discarded()) {
            if (operand.front() == '"' || operand.front() == '[' || operand.front() == '{') {
                error = "dataset condition contains an invalid quoted or structured value";
                return false;
            }
            value = operand;
        }
        condition["value"] = std::move(value);
    }
    return normalize_condition_list(json::array({condition}), output, error);
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
                    canonical != "<" && canonical != "<=" && canonical != "in" && canonical != "is_null" &&
                    canonical != "not_null" && canonical != "contains") {
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
        return parse_condition_expression(input.get<std::string>(), output, error, 0);
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

bool common_agent_dataset_parse_condition_expression(
        const std::string & expression,
        nlohmann::ordered_json & conditions,
        std::string & error) {
    nlohmann::ordered_json parsed;
    if (!parse_condition_expression(expression, parsed, error, 0)) return false;
    conditions = std::move(parsed);
    error.clear();
    return true;
}

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

const char * common_agent_dataset_storage_mode_name(common_agent_dataset_storage_mode mode) {
    return mode == common_agent_dataset_storage_mode::virtual_ ? "virtual" : "materialized";
}

const char * common_agent_dataset_count_kind_name(common_agent_dataset_count_kind kind) {
    switch (kind) {
        case common_agent_dataset_count_kind::exact: return "exact";
        case common_agent_dataset_count_kind::estimated: return "estimated";
        case common_agent_dataset_count_kind::unknown: return "unknown";
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
        {"storage_mode", common_agent_dataset_storage_mode_name(ref.storage_mode)},
        {"count_kind", common_agent_dataset_count_kind_name(ref.count_kind)},
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
    const auto storage_mode = value.value("storage_mode", std::string("materialized"));
    if (storage_mode == "materialized") ref.storage_mode = common_agent_dataset_storage_mode::materialized;
    else if (storage_mode == "virtual") ref.storage_mode = common_agent_dataset_storage_mode::virtual_;
    else { error = "dataset storage_mode is unsupported"; return false; }
    const auto count_kind = value.value("count_kind", std::string("exact"));
    if (count_kind == "exact") ref.count_kind = common_agent_dataset_count_kind::exact;
    else if (count_kind == "estimated") ref.count_kind = common_agent_dataset_count_kind::estimated;
    else if (count_kind == "unknown") ref.count_kind = common_agent_dataset_count_kind::unknown;
    else { error = "dataset count_kind is unsupported"; return false; }
    ref.source_provider = value.value("source_provider", std::string{});
    ref.source_operation = value.value("source_operation", std::string{});
    ref.source_request_json = value.value("source_request_json", std::string{});
    ref.retrieved_at = value.value("retrieved_at", int64_t(0));
    ref.content_hash = value.value("content_hash", std::string{});
    return validate_common_agent_dataset_ref(ref, error);
}

std::string common_agent_dataset_uri_scope_component(const std::string & value) {
    std::string component;
    for (const unsigned char character : value) {
        if (std::isalnum(character) || character == '-' || character == '_') {
            component += static_cast<char>(character);
        } else {
            component += '-';
        }
    }
    return component.empty() ? "turn" : component;
}

bool common_agent_dataset_uri_is_current_turn(
        const std::string & uri,
        const std::string & turn_id) {
    if (turn_id.empty()) return false;
    const std::string prefix = "dataset://agent/turn/";
    if (uri.rfind(prefix, 0) != 0) return false;
    const auto component_start = prefix.size();
    const auto component_end = uri.find('/', component_start);
    if (component_end == std::string::npos) return false;
    return uri.substr(component_start, component_end - component_start) ==
        common_agent_dataset_uri_scope_component(turn_id);
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
    if (ref.storage_mode == common_agent_dataset_storage_mode::materialized && ref.source_resource_uri.empty()) {
        error = "dataset reference requires source resource provenance";
        return false;
    }
    if (ref.storage_mode == common_agent_dataset_storage_mode::virtual_ &&
            (ref.source_provider.empty() || ref.source_operation.empty() || ref.source_request_json.empty())) {
        error = "virtual dataset reference requires a provider, operation and host query provenance";
        return false;
    }
    if (ref.count_kind != common_agent_dataset_count_kind::exact && ref.row_count != 0) {
        error = "virtual or estimated dataset counts must not be represented as exact row_count values";
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
    if (descriptor.ref.storage_mode == common_agent_dataset_storage_mode::materialized &&
            descriptor.import_processor_id.empty()) {
        error = "dataset descriptor requires import processor provenance";
        return false;
    }
    if (descriptor.ref.storage_mode == common_agent_dataset_storage_mode::virtual_ &&
            (descriptor.capabilities.empty() || descriptor.read_context_id.empty())) {
        error = "virtual dataset descriptor requires capabilities and a host read context";
        return false;
    }
    error.clear();
    return true;
}

common_agent_dataset_inventory_resolution common_agent_resolve_dataset_inventory_reference(
        const std::vector<common_agent_dataset_descriptor> & datasets,
        const std::string & reference,
        std::string & resolved,
        std::string & error) {
    resolved.clear();
    error.clear();
    const std::string prefix = "$datasets.datasets[";
    std::string digits;
    bool one_based = false;
    if (reference.rfind(prefix, 0) == 0) {
        const auto close = reference.find(']', prefix.size());
        if (close == std::string::npos ||
                (close + 1 != reference.size() && reference.substr(close + 1) != ".dataset")) {
            error = "plan.binding.invalid_syntax: expected '$datasets.datasets[index].dataset'";
            return common_agent_dataset_inventory_resolution::invalid;
        }
        digits = reference.substr(prefix.size(), close - prefix.size());
    } else if (reference.size() >= 2 && reference.front() == 'd' &&
            std::all_of(reference.begin() + 1, reference.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        digits = reference.substr(1);
        one_based = true;
    } else {
        return common_agent_dataset_inventory_resolution::not_applicable;
    }
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        error = "plan.binding.invalid_syntax: dataset inventory index must be numeric";
        return common_agent_dataset_inventory_resolution::invalid;
    }
    size_t index = 0;
    for (const unsigned char ch : digits) {
        const size_t digit = ch - '0';
        if (index > std::numeric_limits<size_t>::max() / 10 ||
                (index == std::numeric_limits<size_t>::max() / 10 && digit > std::numeric_limits<size_t>::max() % 10)) {
            error = "plan.binding.invalid_syntax: dataset inventory index overflows";
            return common_agent_dataset_inventory_resolution::invalid;
        }
        index = index * 10 + digit;
    }
    if (one_based && index > 0) --index;
    else if (one_based) index = datasets.size();
    if (index >= datasets.size() || datasets[index].ref.uri.empty()) {
        error = "plan.binding.unknown_dataset: dataset handle is unavailable in the current host snapshot";
        return common_agent_dataset_inventory_resolution::invalid;
    }
    resolved = datasets[index].ref.uri;
    return common_agent_dataset_inventory_resolution::resolved;
}

bool normalize_common_agent_dataset_tool_arguments(
        const std::string & tool_name,
        nlohmann::ordered_json & arguments,
        std::string & error) {
    if (tool_name == "dataset.inspect" || tool_name == "dataset.schema" ||
            tool_name == "dataset.sample" || tool_name == "dataset.validate") {
        size_t source_count = 0;
        for (const char * field : {"dataset", "resource", "path"}) {
            if (arguments.contains(field)) ++source_count;
        }
        if (source_count != 1) {
            error = tool_name + " requires exactly one of dataset, resource or path";
            return false;
        }
    }
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
        if (arguments.contains("columns") && arguments["columns"].is_string()) {
            arguments["columns"] = nlohmann::ordered_json::array({arguments["columns"]});
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

bool common_agent_resolve_numeric_columns(
        const common_agent_dataset_descriptor & descriptor,
        const nlohmann::ordered_json & requested,
        nlohmann::ordered_json & selected,
        std::string & error) {
    selected = nlohmann::ordered_json::array();
    error.clear();
    if (!requested.is_array() || requested.size() > 32) {
        error = "numeric columns must be an array with at most 32 fields";
        return false;
    }
    for (const auto & name : requested) {
        if (!name.is_string() || name.get<std::string>().empty()) {
            error = "numeric columns must be non-empty strings";
            return false;
        }
        selected.push_back(name);
    }
    if (!selected.empty()) return true;
    for (const auto & column : descriptor.columns) {
        if (column.type != common_agent_dataset_column_type::integer &&
                column.type != common_agent_dataset_column_type::decimal) continue;
        selected.push_back(column.name);
        if (selected.size() == 32) break;
    }
    if (selected.empty()) {
        error = "no schema-declared numeric columns; select explicit columns or inspect the schema";
        return false;
    }
    return true;
}
