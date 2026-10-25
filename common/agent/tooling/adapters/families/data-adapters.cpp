#include "agent/tooling/adapters/families/data-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

namespace {

using json = common_adapter_json;

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    char buffer[1024];
    input.read(buffer, sizeof(buffer));
    return input.good() || input.eof()
        ? std::find(buffer, buffer + input.gcount(), '\0') == buffer + input.gcount()
        : false;
}

bool repository_path(const std::string & root, const std::string & relative,
        std::filesystem::path & output, std::string & error) {
    if (root.empty()) {
        error = "repository tools require a runtime repository root";
        return false;
    }
    std::error_code fs_error;
    const auto base = std::filesystem::weakly_canonical(root, fs_error);
    if (fs_error) {
        error = "repository root could not be resolved";
        return false;
    }
    const auto requested = relative.empty()
        ? base
        : std::filesystem::weakly_canonical(base / relative, fs_error);
    if (fs_error) {
        error = "repository path could not be resolved";
        return false;
    }
    const auto base_text = base.generic_string();
    const auto requested_text = requested.generic_string();
    if (requested_text != base_text && requested_text.rfind(base_text + "/", 0) != 0) {
        error = "repository path escapes the runtime root";
        return false;
    }
    output = requested;
    return true;
}

std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(std::move(field));
    return fields;
}

bool dataset_file(const common_native_tool_bindings & bindings,
        const std::string & relative, std::filesystem::path & path, std::string & error) {
    if (!repository_path(bindings.repository_root, relative, path, error)) return false;
    const auto extension = lower_copy(path.extension().string());
    if (extension != ".csv" && extension != ".json" && extension != ".parquet") {
        error = "dataset format is not supported by the host-native foundation tool";
        return false;
    }
    if (extension != ".parquet" && !text_file(path)) {
        error = "dataset path is not a readable regular text file";
        return false;
    }
    return true;
}

struct prepared_dataset_rule {
    std::string type;
    std::vector<size_t> columns;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::set<std::string> allowed_values;
};

bool finite_number(const json & value, double & out) {
    if (!value.is_number()) return false;
    out = value.get<double>();
    return std::isfinite(out);
}

bool parse_cell_number(const std::string & value, double & out) {
    if (value.empty()) return false;
    char * end = nullptr;
    out = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || !std::isfinite(out)) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    return *end == '\0';
}

bool prepare_dataset_validation_rules(
        const json & rules,
        const std::vector<std::string> & columns,
        std::vector<prepared_dataset_rule> & prepared,
        std::string & error) {
    if (!rules.is_array() || rules.empty() || rules.size() > 32) {
        error = "dataset.validate requires 1 to 32 rules";
        return false;
    }
    prepared.clear();
    for (const auto & rule : rules) {
        if (!rule.is_object() || !rule.contains("type") || !rule["type"].is_string()) {
            error = "each dataset.validate rule requires a string type";
            return false;
        }
        const auto type = rule["type"].get<std::string>();
        prepared_dataset_rule item;
        item.type = type;
        if (type == "unique_combination") {
            if (rule.size() != 2 || !rule.contains("columns") ||
                    !rule["columns"].is_array() || rule["columns"].size() < 2 ||
                    rule["columns"].size() > 16) {
                error = "unique_combination requires only a columns list with 2 to 16 columns";
                return false;
            }
            std::set<size_t> seen_columns;
            for (const auto & name : rule["columns"]) {
                if (!name.is_string() || name.get<std::string>().empty() ||
                        name.get<std::string>().size() > 128) {
                    error = "unique_combination column names must be non-empty strings of at most 128 characters";
                    return false;
                }
                const auto found = std::find(columns.begin(), columns.end(), name.get<std::string>());
                if (found == columns.end()) {
                    error = "dataset.validate column does not exist in the dataset schema: " + name.get<std::string>();
                    return false;
                }
                const auto index = static_cast<size_t>(std::distance(columns.begin(), found));
                if (!seen_columns.insert(index).second) {
                    error = "unique_combination columns must be distinct";
                    return false;
                }
                item.columns.push_back(index);
            }
            prepared.push_back(std::move(item));
            continue;
        }
        if (type != "not_null" && type != "unique" &&
                type != "numeric_range" && type != "allowed_values") {
            error = "unsupported dataset.validate rule type";
            return false;
        }
        if (!rule.contains("column") || !rule["column"].is_string()) {
            error = "dataset.validate rule requires a column";
            return false;
        }
        if ((type == "not_null" || type == "unique") && rule.size() != 2) {
            error = "not_null and unique rules accept only type and column";
            return false;
        }
        if (type == "numeric_range") {
            if ((!rule.contains("min") && !rule.contains("max")) ||
                    rule.size() != 2 + size_t(rule.contains("min")) + size_t(rule.contains("max"))) {
                error = "numeric_range requires min, max, or both, and no other fields";
                return false;
            }
            double bound = 0.0;
            if (rule.contains("min")) {
                if (!finite_number(rule["min"], bound)) { error = "numeric_range min must be finite"; return false; }
                item.minimum = bound;
            }
            if (rule.contains("max")) {
                if (!finite_number(rule["max"], bound)) { error = "numeric_range max must be finite"; return false; }
                item.maximum = bound;
            }
            if (item.minimum && item.maximum && *item.minimum > *item.maximum) {
                error = "numeric_range min must not exceed max";
                return false;
            }
        } else if (type == "allowed_values") {
            if (rule.size() != 3 || !rule.contains("values") ||
                    !rule["values"].is_array() || rule["values"].empty() ||
                    rule["values"].size() > 64) {
                error = "allowed_values requires a non-empty values list of at most 64 strings";
                return false;
            }
            for (const auto & value : rule["values"]) {
                if (!value.is_string() || value.get<std::string>().size() > 256 ||
                        !item.allowed_values.insert(value.get<std::string>()).second) {
                    error = "allowed_values entries must be unique strings of at most 256 characters";
                    return false;
                }
            }
        }
        const auto column_name = rule["column"].get<std::string>();
        if (column_name.empty() || column_name.size() > 128) {
            error = "dataset.validate column must be a non-empty name of at most 128 characters";
            return false;
        }
        const auto column = std::find(columns.begin(), columns.end(), column_name);
        if (column == columns.end()) {
            error = "dataset.validate column does not exist in the dataset schema: " + column_name;
            return false;
        }
        item.columns.push_back(static_cast<size_t>(std::distance(columns.begin(), column)));
        prepared.push_back(std::move(item));
    }
    return true;
}

bool validate_dataset_descriptor_scope(
        const common_native_tool_bindings & bindings,
        const common_agent_dataset_descriptor & descriptor,
        common_tool_execution_result & failure);

common_tool_execution_result evaluate_dataset_validation_rows(
        const std::vector<std::string> & columns,
        const std::vector<std::vector<std::string>> & rows,
        const json & rules) {
    std::vector<prepared_dataset_rule> prepared;
    std::string error;
    if (!prepare_dataset_validation_rules(rules, columns, prepared, error)) {
        return common_adapter_validation_failure(
            "tool.dataset.validate.invalid_rule", std::move(error));
    }
    json violations = json::array();
    for (size_t rule_index = 0; rule_index < prepared.size(); ++rule_index) {
        const auto & rule = prepared[rule_index];
        size_t count = 0;
        std::set<std::string> seen;
        for (const auto & row : rows) {
            if (rule.type == "not_null" || rule.type == "unique") {
                const auto value = rule.columns[0] < row.size() ? row[rule.columns[0]] : std::string();
                if ((rule.type == "not_null" && value.empty()) ||
                        (rule.type == "unique" && !seen.insert(value).second)) ++count;
            } else if (rule.type == "numeric_range") {
                const auto value = rule.columns[0] < row.size() ? row[rule.columns[0]] : std::string();
                double number = 0.0;
                if (!parse_cell_number(value, number) ||
                        (rule.minimum && number < *rule.minimum) ||
                        (rule.maximum && number > *rule.maximum)) ++count;
            } else if (rule.type == "allowed_values") {
                const auto value = rule.columns[0] < row.size() ? row[rule.columns[0]] : std::string();
                if (!rule.allowed_values.count(value)) ++count;
            } else if (rule.type == "unique_combination") {
                std::string key;
                for (const auto column : rule.columns) {
                    const auto value = column < row.size() ? row[column] : std::string();
                    key += std::to_string(value.size()) + ":" + value + ";";
                }
                if (!seen.insert(std::move(key)).second) ++count;
            }
        }
        if (count > 0) {
            std::string label = rule.type + ":";
            for (size_t i = 0; i < rule.columns.size(); ++i) {
                if (i) label += ",";
                label += columns[rule.columns[i]];
            }
            violations.push_back({
                {"rule", std::move(label)},
                {"count", count},
            });
        }
    }
    return common_adapter_success_json({
        {"valid", violations.empty()},
        {"violations", std::move(violations)},
        {"rows_checked", rows.size()},
    });
}

common_tool_execution_result validate_csv_stream(
        std::istream & input,
        const json & rules) {
    std::string line;
    if (!std::getline(input, line)) {
        return common_adapter_validation_failure(
            "tool.dataset.validate.empty_file", "dataset.validate requires a CSV header row");
    }
    const auto columns = split_csv(line);
    std::vector<prepared_dataset_rule> prepared;
    std::string error;
    if (!prepare_dataset_validation_rules(rules, columns, prepared, error)) {
        return common_adapter_validation_failure(
            "tool.dataset.validate.invalid_rule", std::move(error));
    }
    std::vector<std::vector<std::string>> rows;
    rows.reserve(4096);
    while (rows.size() < 100000 && std::getline(input, line)) {
        rows.push_back(split_csv(line));
    }
    if (rows.size() == 100000 && std::getline(input, line)) {
        return common_adapter_failure(
            "tool.dataset.validate.row_limit", common_tool_failure_class::limit, false,
            "Dataset validation exceeded the 100000 row limit.",
            "dataset.validate stopped before reaching end of CSV");
    }
    return evaluate_dataset_validation_rows(columns, rows, rules);
}

common_tool_execution_result validate_dataset_reference(
        const common_native_tool_bindings & bindings,
        const std::string & dataset_uri,
        const json & rules) {
    if (bindings.data_store == nullptr) {
        return common_adapter_execution_failure(
            "tool.dataset.backend_unavailable", "dataset backend is unavailable",
            "The dataset backend is unavailable.");
    }
    common_agent_dataset_descriptor descriptor;
    std::string error;
    if (!bindings.data_store->get_dataset_descriptor(dataset_uri, descriptor, error)) {
        return common_adapter_not_found_failure(
            "tool.dataset.validate.unavailable", std::move(error),
            "The dataset is unavailable.");
    }
    if (descriptor.import_processor_id != "csv-resource-import-v1" ||
            descriptor.ref.source_representation != "csv:dataset") {
        return common_adapter_validation_failure(
            "tool.dataset.validate.unsupported_format",
            "dataset.validate currently supports only CSV-imported datasets");
    }
    common_tool_execution_result scope_failure;
    if (!validate_dataset_descriptor_scope(bindings, descriptor, scope_failure)) {
        return scope_failure;
    }
    json query = {
        {"dataset", dataset_uri},
        {"limit", 100000},
        {"max_scan_rows", 100000},
        {"max_result_rows", 100000},
    };
    std::string result_json;
    if (!bindings.data_store->execute("data.query", query.dump(), result_json, error)) {
        return common_adapter_execution_failure(
            "tool.dataset.validate.query_failed", std::move(error),
            "Dataset rows could not be read for validation.");
    }
    const auto result = json::parse(result_json, nullptr, false);
    if (!result.is_object() || !result.contains("rows") || !result["rows"].is_array()) {
        return common_adapter_execution_failure(
            "tool.dataset.validate.invalid_query_result",
            "dataset backend returned an invalid query result",
            "Dataset rows could not be validated.");
    }
    if (result.value("scan_truncated", false) || result.value("result_truncated", false) ||
            result["rows"].size() >= 100000) {
        return common_adapter_failure(
            "tool.dataset.validate.row_limit", common_tool_failure_class::limit, false,
            "Dataset validation exceeded the 100000 row limit.",
            "dataset.validate query result was truncated or reached the row limit");
    }
    std::vector<std::string> columns;
    if (descriptor.columns.empty() && result.contains("columns") && result["columns"].is_array()) {
        for (const auto & column : result["columns"]) {
            if (column.is_string()) columns.push_back(column.get<std::string>());
        }
    } else {
        for (const auto & column : descriptor.columns) columns.push_back(column.name);
    }
    std::vector<std::vector<std::string>> rows;
    rows.reserve(result["rows"].size());
    for (const auto & row : result["rows"]) {
        if (!row.is_object()) {
            return common_adapter_execution_failure(
                "tool.dataset.validate.invalid_query_result",
                "dataset query returned a non-object row",
                "Dataset rows could not be validated.");
        }
        std::vector<std::string> values;
        values.reserve(columns.size());
        for (const auto & column : columns) {
            if (!row.contains(column) || row[column].is_null()) values.emplace_back();
            else if (row[column].is_string()) values.push_back(row[column].get<std::string>());
            else values.push_back(row[column].dump());
        }
        rows.push_back(std::move(values));
    }
    return evaluate_dataset_validation_rows(columns, rows, rules);
}

bool is_current_turn_derived_dataset(
        const common_agent_dataset_descriptor & descriptor,
        const agent_resource_runtime & runtime) {
    if (descriptor.origin.kind != "derived" || runtime.turn_id.empty()) return false;
    return common_agent_dataset_uri_is_current_turn(
        descriptor.ref.uri, runtime.turn_id);
}

bool repository_source_is_available(
        const common_native_tool_bindings & bindings,
        const common_agent_dataset_descriptor & descriptor) {
    if (bindings.repository_root.empty() || descriptor.ref.source_representation.empty()) return false;
    const std::filesystem::path source(descriptor.ref.source_representation);
    if (!source.is_absolute()) return false;
    std::error_code fs_error;
    const auto base = std::filesystem::weakly_canonical(bindings.repository_root, fs_error);
    if (fs_error) return false;
    const auto requested = std::filesystem::weakly_canonical(source, fs_error);
    if (fs_error) return false;
    const auto base_text = base.generic_string();
    const auto requested_text = requested.generic_string();
    return (requested_text == base_text || requested_text.rfind(base_text + "/", 0) == 0) &&
        std::filesystem::is_regular_file(requested, fs_error) && !fs_error;
}

bool validate_dataset_descriptor_scope(
        const common_native_tool_bindings & bindings,
        const common_agent_dataset_descriptor & descriptor,
        common_tool_execution_result & failure) {
    std::string error;
    if (!validate_common_agent_dataset_descriptor(
            descriptor, common_agent_dataset_limits{}, error)) {
        failure = common_adapter_validation_failure(
            "tool.dataset.invalid_reference", std::move(error));
        return false;
    }
    if (descriptor.origin.kind == "derived") {
        if (is_current_turn_derived_dataset(descriptor, bindings.resource_runtime)) return true;
        failure = common_adapter_not_found_failure(
            "tool.dataset.out_of_scope", "derived dataset is not in the current turn",
            "The derived dataset is outside the current host turn scope.");
        return false;
    }
    // A descriptor registered in the active structured-data backend is already
    // host-owned input for this runtime.  Its source resource may be a durable
    // backend seed or an external ingestion record rather than an object in
    // the current-turn resource store, so requiring a second resource lookup
    // would incorrectly reject valid registered datasets.
    if (bindings.data_store != nullptr) return true;
    if (bindings.resource_runtime.store == nullptr) {
        return true;
    }

    agent_resource_descriptor source;
    const auto authority = make_agent_resource_read_authority(
        bindings.resource_runtime, std::time(nullptr));
    if (bindings.resource_runtime.store->stat(
            descriptor.ref.source_resource_uri, authority, source, error)) {
        return true;
    }
    if (repository_source_is_available(bindings, descriptor)) return true;

    failure = common_adapter_not_found_failure(
        "tool.dataset.out_of_scope", std::move(error),
        "The dataset source is unavailable in the current host scope.");
    return false;
}

bool validate_dataset_reference_scope(
        const common_native_tool_bindings & bindings,
        const std::string & dataset_uri,
        common_tool_execution_result & failure) {
    common_agent_dataset_descriptor descriptor;
    return common_authorize_dataset_reference(bindings, dataset_uri, descriptor, failure);
}

common_tool_execution_result execute_data_backend(
        const common_native_tool_bindings & bindings,
        const std::string & operation, const std::string & input,
        const std::string & result_schema = {}) {
    if (bindings.data_store == nullptr) {
        return common_adapter_execution_failure(
            "tool.data.backend_unavailable", "structured data backend is unavailable",
            "The configured data backend is unavailable.");
    }
    const auto arguments = json::parse(input, nullptr, false);
    if (arguments.is_object()) {
        // All structured-data operations use the same dataset authority. In
        // particular, joins have two inputs and must not get a weaker check
        // merely because their fields are named left/right.
        for (const char * field : {"dataset", "left", "right"}) {
            if (!arguments.contains(field) || !arguments[field].is_string()) continue;
            common_tool_execution_result failure;
            if (!validate_dataset_reference_scope(
                    bindings, arguments[field].get<std::string>(), failure)) {
                return failure;
            }
        }
    }
    json resolved_numeric_columns = arguments.is_object() ? arguments.value("columns", json::array()) : json::array();
    // Defaults may use declared schema types, never a guess from sampled rows.
    if (operation == "statistics.describe" && arguments.is_object() &&
            (!arguments.contains("columns") || arguments["columns"].empty())) {
        common_agent_dataset_descriptor descriptor;
        std::string schema_error;
        if (!bindings.data_store->get_dataset_descriptor(arguments.value("dataset", std::string{}),
                descriptor, schema_error)) {
            return common_adapter_not_found_failure("tool.statistics.schema_unavailable",
                schema_error, "The dataset schema is unavailable.");
        }
        if (!common_agent_resolve_numeric_columns(descriptor,
                arguments.value("columns", json::array()), resolved_numeric_columns, schema_error)) {
            return common_adapter_failure("tool.statistics.columns_unresolved",
                common_tool_failure_class::validation, true,
                "No schema-declared numeric columns are available; select explicit columns or inspect the schema.",
                schema_error);
        }
    }
    std::string result;
    std::string error;
    if (!bindings.data_store->execute_scoped(operation, input,
            bindings.tool_scope, bindings.allow_network, result, error)) {
        return common_adapter_execution_failure(
            "tool.data.backend_failed",
            error.empty() ? "structured data backend failed" : std::move(error),
            "The structured data operation failed.");
    }
    const auto parsed = json::parse(result, nullptr, false);
    if (!parsed.is_object() && !parsed.is_array()) {
        return common_adapter_execution_failure(
            "tool.data.invalid_backend_result",
            "structured data backend returned invalid JSON",
            "The structured data backend returned an invalid result.");
    }
    if (operation == "statistics.describe" &&
            (!parsed.is_object() || parsed.value("dataset", json()) != arguments["dataset"] ||
             parsed.value("selected_columns", json()) != resolved_numeric_columns)) {
        return common_adapter_execution_failure("tool.data.result_binding_mismatch",
            "statistics result does not retain the executed dataset and selected columns",
            "The result does not match the executed analysis selection.");
    }
    if (!result_schema.empty()) {
        std::string normalized;
        if (!common_schema_normalize_and_validate_object(result, result_schema, normalized, error)) {
            return common_adapter_execution_failure("tool.data.invalid_backend_result",
                error, "The structured data result does not satisfy its registered contract.");
        }
    }
    return common_adapter_success_json(parsed);
}

bool select_dataset_reference(const json & arguments, std::string & dataset, std::string & error) {
    const bool has_dataset = arguments.contains("dataset") && arguments["dataset"].is_string();
    const bool has_path = arguments.contains("path") && arguments["path"].is_string();
    if (has_dataset == has_path) {
        error = "dataset operation requires exactly one of dataset or path";
        return false;
    }
    dataset = has_dataset ? arguments["dataset"].get<std::string>() : std::string();
    return true;
}

common_tool_execution_result execute_dataset_descriptor_tool(
        const common_native_tool_bindings & bindings, const json & arguments,
        const char * operation) {
    std::string dataset;
    std::string error;
    if (!select_dataset_reference(arguments, dataset, error)) {
        return common_adapter_validation_failure("tool.dataset.invalid_reference", error);
    }
    if (!dataset.empty()) {
        if (!bindings.data_store) {
            return common_adapter_execution_failure(
                "tool.dataset.backend_unavailable",
                "dataset descriptor backend is unavailable",
                "The dataset backend is unavailable.");
        }
        common_agent_dataset_descriptor descriptor;
        if (!bindings.data_store->get_dataset_descriptor(dataset, descriptor, error)) {
            return common_adapter_not_found_failure(
                "tool.dataset.unavailable", error, "The dataset reference is unavailable.");
        }
        common_tool_execution_result scope_failure;
        if (!validate_dataset_descriptor_scope(bindings, descriptor, scope_failure)) {
            return scope_failure;
        }
        if (std::string(operation) == "inspect") {
            return common_adapter_success_json({
                {"dataset", descriptor.ref.uri}, {"name", descriptor.ref.name},
                {"rows", descriptor.ref.row_count}, {"columns", descriptor.ref.column_count},
                {"storage_mode", common_agent_dataset_storage_mode_name(descriptor.ref.storage_mode)},
                {"count_kind", common_agent_dataset_count_kind_name(descriptor.ref.count_kind)},
                {"capabilities", descriptor.capabilities},
                {"source", descriptor.ref.source_resource_uri},
                {"source_sheet", descriptor.source_sheet_name}, {"source_range", descriptor.source_range},
                {"import_processor", descriptor.import_processor_id},
                {"origin", {
                    {"kind", descriptor.origin.kind},
                    {"semantic_resource", descriptor.origin.source_representation_uri},
                    {"node_id", descriptor.origin.source_node_id},
                    {"table_index", descriptor.origin.table_index},
                    {"caption", descriptor.origin.caption},
                    {"header_mode", common_agent_table_header_mode_name(descriptor.origin.header_mode)},
                    {"header_confidence", descriptor.origin.header_confidence},
                }},
            });
        }
        json columns = json::array();
        for (const auto & column : descriptor.columns) {
            columns.push_back({
                {"name", column.name},
                {"type", common_agent_dataset_column_type_name(column.type)},
                {"nullable", column.nullable},
            });
        }
        return common_adapter_success_json({{"dataset", descriptor.ref.uri}, {"columns", columns}});
    }
    return common_adapter_validation_failure(
        "tool.dataset.legacy_path_required",
        "legacy path handling must be provided by the existing adapter branch");
}

} // namespace

bool common_authorize_dataset_reference(
        const common_native_tool_bindings & bindings,
        const std::string & dataset_uri,
        common_agent_dataset_descriptor & descriptor,
        common_tool_execution_result & failure) {
    if (dataset_uri.rfind("dataset://", 0) != 0) {
        failure = common_adapter_validation_failure(
            "tool.dataset.invalid_reference",
            "dataset references must use the canonical dataset:// scheme");
        return false;
    }
    if (bindings.data_store == nullptr) {
        failure = common_adapter_execution_failure(
            "tool.dataset.backend_unavailable", "structured data backend is unavailable",
            "The dataset backend is unavailable.");
        return false;
    }
    std::string error;
    if (!bindings.data_store->get_dataset_descriptor(dataset_uri, descriptor, error)) {
        failure = common_adapter_not_found_failure(
            "tool.dataset.unavailable", std::move(error),
            "The dataset reference is unavailable in the current host scope.");
        return false;
    }
    return validate_dataset_descriptor_scope(bindings, descriptor, failure);
}

bool common_try_register_data_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const bool is_data_tool =
        definition.executor_id == "builtin.dataset.validate" ||
        definition.executor_id == "builtin.dataset.list" ||
        definition.executor_id == "builtin.dataset.select" ||
        definition.executor_id == "builtin.dataset.inspect" ||
        definition.executor_id == "builtin.dataset.schema" ||
        definition.executor_id == "builtin.dataset.sample" ||
        definition.executor_id == "builtin.data.query" ||
        definition.executor_id == "builtin.data.filter" ||
        definition.executor_id == "builtin.data.aggregate" ||
        definition.executor_id == "builtin.data.join" ||
        definition.executor_id == "builtin.data.transform" ||
        definition.executor_id == "builtin.statistics.describe" ||
        definition.executor_id == "builtin.statistics.outliers" ||
        definition.executor_id == "builtin.statistics.value_counts";
    if (!is_data_tool) return false;

    if (definition.executor_id == "builtin.dataset.validate" &&
            (!bindings.repository_root.empty() || bindings.data_store != nullptr)) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error) ||
                    !arguments.contains("rules") || !arguments["rules"].is_array()) {
                return common_adapter_validation_failure(
                    "tool.dataset.validate.invalid_arguments", "dataset.validate requires a source and rules");
            }
            const auto source_count = static_cast<size_t>(arguments.contains("dataset")) +
                static_cast<size_t>(arguments.contains("resource")) +
                static_cast<size_t>(arguments.contains("path"));
            if (source_count != 1) {
                return common_adapter_validation_failure(
                    "tool.dataset.validate.invalid_arguments",
                    "dataset.validate requires exactly one of dataset, resource or path");
            }

            if (arguments.contains("dataset")) {
                if (!arguments["dataset"].is_string()) {
                    return common_adapter_validation_failure(
                        "tool.dataset.validate.invalid_arguments", "dataset must be a dataset:// reference");
                }
                common_tool_execution_result scope_failure;
                if (!validate_dataset_reference_scope(
                        bindings, arguments["dataset"].get<std::string>(), scope_failure)) {
                    return scope_failure;
                }
                return validate_dataset_reference(
                    bindings, arguments["dataset"].get<std::string>(), arguments["rules"]);
            }

            if (arguments.contains("resource")) {
                if (!arguments["resource"].is_string()) {
                    return common_adapter_validation_failure(
                        "tool.dataset.validate.invalid_arguments", "resource must be a host-owned resource reference");
                }
                if (!bindings.dataset_from_resource) {
                    return common_adapter_not_found_failure(
                        "tool.dataset.resource_unavailable", "resource-to-dataset materialization is unavailable",
                        "This resource cannot be validated as a dataset in the current runtime.");
                }
                const auto materialized = bindings.dataset_from_resource(
                    arguments["resource"].get<std::string>(), "inspect");
                if (!materialized.ok) return materialized;
                const auto materialized_json = json::parse(materialized.output, nullptr, false);
                if (!materialized_json.is_object() || !materialized_json.contains("dataset") ||
                        !materialized_json["dataset"].is_string()) {
                    return common_adapter_execution_failure(
                        "tool.dataset.materialization_invalid", "resource materializer returned no dataset reference",
                        "The resource could not be validated as a dataset.");
                }
                return validate_dataset_reference(
                    bindings, materialized_json["dataset"].get<std::string>(), arguments["rules"]);
            }

            if (!arguments["path"].is_string()) {
                return common_adapter_validation_failure(
                    "tool.dataset.validate.invalid_arguments", "path must be repository-relative");
            }
            std::filesystem::path path;
            if (!dataset_file(bindings, arguments["path"].get<std::string>(), path, error)) {
                return common_adapter_not_found_failure(
                    "tool.dataset.validate.unavailable", std::move(error), "Dataset is unavailable.");
            }
            std::error_code fs_error;
            if (!std::filesystem::is_regular_file(path, fs_error) || fs_error) {
                return common_adapter_not_found_failure(
                    "tool.dataset.validate.unavailable", "dataset path is not a regular file",
                    "Dataset is unavailable.");
            }
            if (lower_copy(path.extension().string()) != ".csv") {
                return common_adapter_validation_failure(
                    "tool.dataset.validate.unsupported_format", "dataset.validate currently supports CSV");
            }
            std::ifstream file(path);
            if (!file) {
                return common_adapter_not_found_failure(
                    "tool.dataset.validate.unavailable", "dataset file could not be opened",
                    "Dataset is unavailable.");
            }
            return validate_csv_stream(file, arguments["rules"]);
        }, error);
    } else if (definition.executor_id == "builtin.data.query" ||
            definition.executor_id == "builtin.data.filter" ||
            definition.executor_id == "builtin.data.aggregate" ||
            definition.executor_id == "builtin.data.join" ||
            definition.executor_id == "builtin.data.transform" ||
            definition.executor_id == "builtin.statistics.describe" ||
            definition.executor_id == "builtin.statistics.outliers" ||
            definition.executor_id == "builtin.statistics.value_counts") {
        const auto operation = definition.name;
        const auto result_schema = definition.executor_id == "builtin.statistics.describe"
            ? definition.result_schema_json : std::string{};
        installed = common_adapter_register_definition(definition, registry, [bindings, operation, result_schema](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) {
                return common_adapter_validation_failure("tool.data.invalid_arguments", std::move(error));
            }
            return execute_data_backend(bindings, operation, arguments.dump(), result_schema);
        }, error);
    } else if (definition.executor_id == "builtin.dataset.list" && bindings.data_store != nullptr) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) {
                return common_adapter_validation_failure("tool.dataset.list.invalid_arguments", std::move(error));
            }
            std::vector<common_agent_dataset_descriptor> descriptors;
            if (!bindings.data_store->list_dataset_descriptors(descriptors, error)) {
                return common_adapter_execution_failure("tool.dataset.list.backend_unavailable", std::move(error), "The dataset registry is unavailable.");
            }
            const size_t limit = std::min<size_t>(arguments.value("max_results", 128), 256);
            json datasets = json::array();
            json names = json::array();
            for (size_t index = 0; index < descriptors.size() && index < limit; ++index) {
                common_tool_execution_result scope_failure;
                if (!validate_dataset_descriptor_scope(
                        bindings, descriptors[index], scope_failure)) continue;
                datasets.push_back(descriptors[index].ref.uri);
                names.push_back(descriptors[index].ref.name);
            }
            return common_adapter_success_json({
                {"datasets", datasets}, {"names", names}, {"truncated", descriptors.size() > limit}});
        }, error);
    } else if (definition.executor_id == "builtin.dataset.list" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) return common_adapter_validation_failure("tool.dataset.list.invalid_arguments", std::move(error));
            std::filesystem::path root;
            if (!repository_path(bindings.repository_root, arguments.value("path", std::string("datasets")), root, error) || !std::filesystem::is_directory(root)) return common_adapter_not_found_failure("tool.dataset.list.path_not_found", "dataset list directory was not found", "Dataset directory was not found.");
            const int limit = arguments.value("max_results", 128);
            json datasets = json::array();
            for (const auto & entry : std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied)) {
                if (datasets.size() >= static_cast<size_t>(limit)) break;
                if (!entry.is_regular_file()) continue;
                const auto ext = lower_copy(entry.path().extension().string());
                if (ext == ".csv" || ext == ".json" || ext == ".parquet") datasets.push_back({{"path", std::filesystem::relative(entry.path(), bindings.repository_root).generic_string()}, {"format", ext.substr(1)}, {"size_bytes", entry.file_size()}});
            }
            return common_adapter_success_json({{"datasets", datasets}, {"truncated", datasets.size() >= static_cast<size_t>(limit)}});
        }, error);
    } else if (definition.executor_id == "builtin.dataset.select" && bindings.data_store != nullptr) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error) ||
                    !arguments.contains("name") || !arguments["name"].is_string()) {
                return common_adapter_validation_failure("tool.dataset.select.invalid_arguments", "dataset.select requires name");
            }
            common_agent_dataset_descriptor descriptor;
            if (!bindings.data_store->find_dataset_by_name(arguments["name"].get<std::string>(), descriptor, error)) {
                bool repairable = false;
                if (error == "dataset name was not found") {
                    std::vector<common_agent_dataset_descriptor> candidates;
                    std::string list_error;
                    if (bindings.data_store->list_dataset_descriptors(candidates, list_error) && !candidates.empty()) {
                        std::string choices;
                        for (auto candidate = candidates.begin(); candidate != candidates.end(); ++candidate) {
                            common_tool_execution_result scope_failure;
                            if (!validate_dataset_descriptor_scope(bindings, *candidate, scope_failure)) continue;
                            if (candidate->ref.name.empty() || candidate->ref.uri.empty()) continue;
                            if (!choices.empty()) choices += ", ";
                            choices += candidate->ref.name + " (" + candidate->ref.uri + ")";
                        }
                        if (!choices.empty()) {
                            error += "; choose one of: " + choices;
                            repairable = true;
                        }
                    }
                }
                // The lookup is safe to repeat after correcting the name. Mark
                // this specific failure repairable so reflection receives the
                // scope-filtered choices above; other not-found failures stay
                // non-retryable.
                return common_adapter_failure("tool.dataset.select.not_found",
                    common_tool_failure_class::not_found, repairable,
                    "The named dataset was not found.", std::move(error));
            }
            common_tool_execution_result scope_failure;
            if (!validate_dataset_descriptor_scope(bindings, descriptor, scope_failure)) {
                return scope_failure;
            }
            return common_adapter_success_json({
                {"dataset", descriptor.ref.uri}, {"name", descriptor.ref.name}});
        }, error);
    } else if (definition.executor_id == "builtin.dataset.inspect" && (!bindings.repository_root.empty() || bindings.data_store != nullptr)) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) return common_adapter_validation_failure("tool.dataset.inspect.invalid_arguments", std::move(error));
            if (arguments.contains("resource")) {
                if (!bindings.dataset_from_resource) return common_adapter_not_found_failure("tool.dataset.resource_unavailable", "resource-to-dataset materialization is unavailable", "This resource cannot be inspected as a dataset in the current runtime.");
                return bindings.dataset_from_resource(arguments["resource"].get<std::string>(), "inspect");
            }
            if (arguments.contains("dataset")) return execute_dataset_descriptor_tool(bindings, arguments, "inspect");
            if (!arguments.contains("path") || !arguments["path"].is_string()) return common_adapter_validation_failure("tool.dataset.inspect.invalid_arguments", "dataset.inspect requires dataset or path");
            std::filesystem::path path;
            if (!dataset_file(bindings, arguments["path"].get<std::string>(), path, error)) return common_adapter_not_found_failure("tool.dataset.inspect.unavailable", std::move(error), "Dataset is unavailable.");
            return common_adapter_success_json({{"path", std::filesystem::relative(path, bindings.repository_root).generic_string()}, {"format", lower_copy(path.extension().string()).substr(1)}, {"size_bytes", std::filesystem::file_size(path)}});
        }, error);
    } else if (definition.executor_id == "builtin.dataset.schema" && (!bindings.repository_root.empty() || bindings.data_store != nullptr)) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) return common_adapter_validation_failure("tool.dataset.schema.invalid_arguments", std::move(error));
            if (arguments.contains("resource")) {
                if (!bindings.dataset_from_resource) return common_adapter_not_found_failure("tool.dataset.resource_unavailable", "resource-to-dataset materialization is unavailable", "This resource cannot be inspected as a dataset in the current runtime.");
                return bindings.dataset_from_resource(arguments["resource"].get<std::string>(), "schema");
            }
            if (arguments.contains("dataset")) return execute_dataset_descriptor_tool(bindings, arguments, "schema");
            if (!arguments.contains("path") || !arguments["path"].is_string()) return common_adapter_validation_failure("tool.dataset.schema.invalid_arguments", "dataset.schema requires dataset or path");
            std::filesystem::path path;
            if (!dataset_file(bindings, arguments["path"].get<std::string>(), path, error)) return common_adapter_not_found_failure("tool.dataset.schema.unavailable", std::move(error), "Dataset is unavailable.");
            if (lower_copy(path.extension().string()) != ".csv") return common_adapter_validation_failure("tool.dataset.schema.unsupported_format", "dataset.schema currently supports CSV");
            std::ifstream file(path);
            std::string line;
            if (!std::getline(file, line)) return common_adapter_success_json({{"columns", json::array()}});
            json columns = json::array();
            for (const auto & name : split_csv(line)) columns.push_back({{"name", name}, {"type", "string"}, {"nullable", true}});
            return common_adapter_success_json({{"columns", columns}});
        }, error);
    } else if (definition.executor_id == "builtin.dataset.sample" && (!bindings.repository_root.empty() || bindings.data_store != nullptr)) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) return common_adapter_validation_failure("tool.dataset.sample.invalid_arguments", std::move(error));
            if (arguments.contains("resource")) {
                if (!bindings.dataset_from_resource) return common_adapter_not_found_failure("tool.dataset.resource_unavailable", "resource-to-dataset materialization is unavailable", "This resource cannot be inspected as a dataset in the current runtime.");
                const auto materialized = bindings.dataset_from_resource(arguments["resource"].get<std::string>(), "inspect");
                if (!materialized.ok) return materialized;
                const auto descriptor = json::parse(materialized.output, nullptr, false);
                if (!descriptor.is_object() || !descriptor.contains("dataset") || !descriptor["dataset"].is_string()) {
                    return common_adapter_execution_failure("tool.dataset.materialization_invalid",
                        "resource materializer returned no dataset reference", "The resource could not be sampled as a dataset.");
                }
                const size_t rows = std::min<size_t>(arguments.value("rows", 20), 100);
                auto sampled = execute_data_backend(bindings, "data.query", json({
                    {"dataset", descriptor["dataset"]}, {"limit", rows},
                    {"max_result_rows", rows}, {"max_scan_rows", 10000}}).dump());
                sampled.dataset_refs = materialized.dataset_refs;
                return sampled;
            }
            if (arguments.contains("dataset")) {
                const size_t rows = std::min<size_t>(arguments.value("rows", 20), 100);
                return execute_data_backend(bindings, "data.query", json({{"dataset", arguments["dataset"]}, {"limit", rows}, {"max_result_rows", rows}, {"max_scan_rows", 10000}}).dump());
            }
            if (!arguments.contains("path") || !arguments["path"].is_string()) return common_adapter_validation_failure("tool.dataset.sample.invalid_arguments", "dataset.sample requires dataset or path");
            std::filesystem::path path;
            if (!dataset_file(bindings, arguments["path"].get<std::string>(), path, error)) return common_adapter_not_found_failure("tool.dataset.sample.unavailable", std::move(error), "Dataset is unavailable.");
            if (lower_copy(path.extension().string()) != ".csv") return common_adapter_validation_failure("tool.dataset.sample.unsupported_format", "dataset.sample currently supports CSV");
            const int limit = arguments.value("rows", 20);
            std::ifstream file(path);
            std::string line;
            json rows = json::array();
            json columns = json::array();
            if (std::getline(file, line)) for (const auto & name : split_csv(line)) columns.push_back(name);
            while (rows.size() < static_cast<size_t>(limit) && std::getline(file, line)) {
                const auto fields = split_csv(line);
                json row = json::object();
                for (size_t i = 0; i < fields.size() && i < columns.size(); ++i) row[columns[i].get<std::string>()] = fields[i];
                rows.push_back(std::move(row));
            }
            return common_adapter_success_json({{"columns", columns}, {"rows", rows}});
        }, error);
    }
    return true;
}
