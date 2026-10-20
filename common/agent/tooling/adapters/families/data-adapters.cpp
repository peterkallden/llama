#include "agent/tooling/adapters/families/data-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "hash/hash.h"

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

bool parse_investigation_time(const json & value, int64_t & out) {
    if (value.is_number_integer()) { out = value.get<int64_t>(); return true; }
    if (value.is_number_float()) { out = static_cast<int64_t>(value.get<double>()); return true; }
    if (!value.is_string()) return false;
    const std::string text = value.get<std::string>();
    char * end = nullptr;
    const auto integer = std::strtoll(text.c_str(), &end, 10);
    if (end != text.c_str() && *end == '\0') { out = integer; return true; }
    if (text.size() < 20) return false;
    std::tm parsed{};
    std::istringstream input(text.substr(0, 19));
    input >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%S");
    if (input.fail()) return false;
    const time_t seconds = timegm(&parsed);
    if (seconds < 0) return false;
    int64_t millis = static_cast<int64_t>(seconds) * 1000;
    size_t cursor = 19;
    if (cursor < text.size() && text[cursor] == '.') {
        ++cursor;
        int fraction = 0, digits = 0;
        while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor]))) {
            if (digits < 3) fraction = fraction * 10 + (text[cursor] - '0');
            ++digits; ++cursor;
        }
        while (digits < 3) { fraction *= 10; ++digits; }
        millis += fraction;
    }
    if (cursor < text.size() && text[cursor] == 'Z') ++cursor;
    else if (cursor + 6 == text.size() && (text[cursor] == '+' || text[cursor] == '-') && text[cursor + 3] == ':') {
        const int hours = std::atoi(text.substr(cursor + 1, 2).c_str());
        const int minutes = std::atoi(text.substr(cursor + 4, 2).c_str());
        if (hours > 23 || minutes > 59) return false;
        const int64_t offset_ms = static_cast<int64_t>(hours * 60 + minutes) * 60 * 1000;
        millis += text[cursor] == '+' ? -offset_ms : offset_ms;
        cursor = text.size();
    }
    if (cursor != text.size()) return false;
    out = millis;
    return true;
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
    if (dataset_uri.rfind("dataset://", 0) != 0) {
        failure = common_adapter_validation_failure(
            "tool.dataset.invalid_reference",
            "dataset references must use the canonical dataset:// scheme");
        return false;
    }
    if (bindings.data_store == nullptr) {
        failure = common_adapter_execution_failure(
            "tool.dataset.backend_unavailable", "dataset backend is unavailable",
            "The dataset backend is unavailable.");
        return false;
    }
    common_agent_dataset_descriptor descriptor;
    std::string error;
    if (!bindings.data_store->get_dataset_descriptor(dataset_uri, descriptor, error)) {
        failure = common_adapter_not_found_failure(
            "tool.dataset.unavailable", std::move(error),
            "The dataset reference is unavailable in the current host scope.");
        return false;
    }
    return validate_dataset_descriptor_scope(bindings, descriptor, failure);
}

common_tool_execution_result execute_data_backend(
        const common_native_tool_bindings & bindings,
        const std::string & operation, const std::string & input) {
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
    return common_adapter_success_json(parsed);
}

common_tool_execution_result execute_investigation_tool(
        const common_native_tool_bindings & bindings, const std::string & operation,
        const json & arguments) {
    if (!bindings.data_store) return common_adapter_execution_failure(
        "tool.investigation.store_unavailable", "investigation persistence is unavailable",
        "Investigation storage is unavailable.");
    std::string error;
    if (operation == "investigation.case.create") {
        std::vector<common_agent_dataset_descriptor> datasets;
        if (!bindings.data_store->list_dataset_descriptors(datasets, error)) {
            return common_adapter_execution_failure("tool.investigation.sources_unavailable",
                std::move(error), "Configured investigation sources are unavailable.");
        }
        const auto source_id = arguments.value("source_id", std::string());
        const bool source_exists = std::any_of(datasets.begin(), datasets.end(), [&](const auto & descriptor) {
            return descriptor.ref.storage_mode == common_agent_dataset_storage_mode::virtual_ &&
                descriptor.ref.source_provider == source_id;
        });
        if (!source_exists) return common_adapter_not_found_failure(
            "tool.investigation.source_unavailable", "case source is not configured",
            "Choose a source available to the host.");
        common_investigation_case_revision value;
        static std::atomic<uint64_t> sequence{1};
        value.case_id = "case-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(sequence.fetch_add(1));
        value.title = arguments.value("title", std::string());
        value.question = arguments.value("question", std::string());
        value.source_id = source_id;
        value.window_start_ms = arguments.value("window_start_ms", int64_t(0));
        value.window_end_ms = arguments.value("window_end_ms", int64_t(0));
        value.scope = bindings.tool_scope;
        value.revision = 1;
        common_investigation_case_revision published;
        const auto revision_uri = "investigation-case://" + value.case_id + "/revision/1";
        if (!bindings.data_store->publish_investigation_case(value, 0, revision_uri, published, error)) {
            return common_adapter_execution_failure("tool.investigation.case_create_failed",
                std::move(error), "The investigation case could not be created.");
        }
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"scope", {{"namespace_id", published.scope.namespace_id},
                {"project_id", published.scope.project_id}, {"session_id", published.scope.session_id}}}});
    }
    const auto case_id = arguments.value("case_id", std::string());
    common_investigation_case_revision value;
    if (!bindings.data_store->get_investigation_case(case_id, bindings.tool_scope, value, error)) {
        return common_adapter_not_found_failure("tool.investigation.case_unavailable",
            std::move(error), "The case is unavailable in the current scope.");
    }
    if (operation == "investigation.case.open") {
        return common_adapter_success_json({{"case", common_investigation_case_to_json(value)}});
    }
    if (operation == "investigation.report.compose") {
        if (arguments.value("expected_revision", uint64_t(0)) != value.revision) return
            common_adapter_execution_failure("tool.investigation.revision_conflict",
                "case revision changed before report composition", "Reload the case before composing a report.");
        if (!bindings.resource_runtime.store) return common_adapter_execution_failure(
            "tool.investigation.report_store_unavailable", "resource store is unavailable",
            "A durable report resource cannot be created.");
        const auto safe_text = [](std::string text) {
            for (auto & c : text) {
                if (c == '\n' || c == '\r' || c == '\t') c = ' ';
                else if (static_cast<unsigned char>(c) < 0x20) c = '?';
            }
            std::string out;
            out.reserve(text.size());
            for (const char c : text) {
                if (c == '`') out += "'";
                else if (c == '<') out += "&lt;";
                else if (c == '>') out += "&gt;";
                else out += c;
            }
            return out;
        };
        std::string markdown = "# Investigation report: " + safe_text(value.title) + "\n\n";
        markdown += "- Case: `" + safe_text(value.case_id) + "`\n- Frozen revision: `" + std::to_string(value.revision) + "`\n";
        markdown += "- Source: `" + safe_text(value.source_id) + "`\n- Time window (epoch ms): `" +
            std::to_string(value.window_start_ms) + "` to `" + std::to_string(value.window_end_ms) + "`\n\n";
        markdown += "## Investigation question\n\n" + safe_text(value.question) + "\n\n";
        markdown += "## Scope and coverage\n\n";
        markdown += value.gaps.empty() ? "No recorded collection or analysis gaps. This does not establish exhaustive source coverage.\n\n"
            : "Coverage is partial or qualified; see the listed gaps.\n\n";
        std::vector<const common_investigation_event *> ordered_events;
        ordered_events.reserve(value.events.size());
        for (const auto & event : value.events) ordered_events.push_back(&event);
        std::stable_sort(ordered_events.begin(), ordered_events.end(), [](const auto * a, const auto * b) {
            if (a->time.has_event_time != b->time.has_event_time) return a->time.has_event_time;
            if (!a->time.has_event_time) return a->id < b->id;
            if (a->time.event_time_ms != b->time.event_time_ms) return a->time.event_time_ms < b->time.event_time_ms;
            return a->id < b->id;
        });
        markdown += "## Timeline\n\n";
        if (ordered_events.empty()) markdown += "No source events are retained in this revision.\n\n";
        for (const auto * event : ordered_events) {
            markdown += "- " + (event->time.has_event_time ? std::to_string(event->time.event_time_ms) : std::string("time unknown")) +
                " — " + safe_text(event->event_type) + " (`" + safe_text(event->id) + "`; document `" +
                safe_text(event->source_document_id) + "`) [raw: " + safe_text(event->raw_resource_uri) + "]\n";
        }
        markdown += "\n## Relations\n\n";
        if (value.relations.empty()) markdown += "No relations are derived in this revision.\n\n";
        for (const auto & relation : value.relations) {
            markdown += "- `" + safe_text(relation.from_entity_id) + "` → `" + safe_text(relation.to_entity_id) +
                "` (type " + std::to_string(static_cast<int>(relation.kind)) + ", " + safe_text(relation.confidence_label) +
                ", rule `" + safe_text(relation.rule_id) + "`); events: ";
            for (size_t i = 0; i < relation.evidence_event_ids.size(); ++i) {
                if (i) markdown += ", ";
                markdown += "`" + safe_text(relation.evidence_event_ids[i]) + "`";
            }
            markdown += "\n";
        }
        markdown += "\n## Claims\n\n";
        if (value.claims.empty()) markdown += "No claims are recorded.\n\n";
        for (const auto & claim : value.claims) {
            const char * state = claim.state == common_investigation_claim_state::hypothesis ? "Hypothesis" :
                claim.state == common_investigation_claim_state::reviewed_conclusion ? "Reviewed conclusion" :
                claim.state == common_investigation_claim_state::correlation ? "Correlation" : "Observation";
            markdown += "### " + std::string(state) + ": " + safe_text(claim.text) + "\n\n";
            for (const auto & id : claim.supporting_event_ids) markdown += "- Supports: `" + safe_text(id) + "`\n";
            for (const auto & id : claim.contradicting_event_ids) markdown += "- Contradicts: `" + safe_text(id) + "`\n";
            for (const auto & limitation : claim.limitations) markdown += "- Limitation: " + safe_text(limitation) + "\n";
            markdown += "\n";
        }
        markdown += "## Gaps and limitations\n\n";
        if (value.gaps.empty()) markdown += "No explicit gaps are recorded.\n";
        for (const auto & gap : value.gaps) markdown += "- " + safe_text(gap) + "\n";
        markdown += "\n## Evidence references\n\n";
        for (const auto & event : value.events) {
            markdown += "- Event `" + safe_text(event.id) + "`: raw `" + safe_text(event.raw_resource_uri) +
                "`; normalized `" + safe_text(event.normalized_resource_uri) + "`\n";
        }
        if (markdown.size() > 2 * 1024 * 1024) return common_adapter_validation_failure(
            "tool.investigation.report_too_large", "frozen report exceeds the 2 MiB resource limit");
        const std::string fingerprint = "sha256:" + hash_sha256_hex(markdown.data(), markdown.size());
        agent_resource_put_request put;
        put.name = value.case_id + "-report-r" + std::to_string(value.revision) + ".md";
        put.description = "Frozen investigation report for case revision " + std::to_string(value.revision);
        put.mime_type = "text/markdown"; put.text = markdown;
        put.scope = bindings.tool_scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        put.source_provider = value.source_id; put.source_tool = "investigation.report.compose";
        apply_agent_resource_runtime(bindings.resource_runtime, put);
        put.metadata.purpose = "Frozen investigation report";
        put.metadata.content_summary = "Case " + value.case_id + " revision " + std::to_string(value.revision);
        put.metadata.source_provider = value.source_id;
        put.metadata.source_operation = "investigation.report.compose";
        put.metadata.source_request_json = json{{"case_id", value.case_id},
            {"case_revision", value.revision}, {"content_fingerprint", fingerprint}}.dump();
        put.metadata.content_hash = fingerprint;
        put.metadata.keywords = {"investigation", "report", value.case_id};
        put.metadata.limitations = value.gaps.empty() ?
            "No recorded gaps; this does not establish exhaustive source coverage." :
            [&]() { std::string text; for (const auto & gap : value.gaps) { if (!text.empty()) text += "; "; text += gap; } return text; }();
        agent_resource_descriptor report;
        if (!bindings.resource_runtime.store->put_text(put, report, error)) return common_adapter_execution_failure(
            "tool.investigation.report_store_failed", std::move(error), "The frozen report could not be persisted.");
        const bool complete = value.gaps.empty();
        return common_adapter_success_json({{"case_id", value.case_id}, {"case_revision", value.revision},
            {"report_uri", report.uri}, {"content_fingerprint", fingerprint}, {"complete", complete},
            {"limitations", value.gaps}});
    }
    if (operation == "investigation.collect") {
        if (arguments.value("expected_revision", uint64_t(0)) != value.revision) {
            return common_adapter_execution_failure("tool.investigation.revision_conflict",
                "case revision changed before collection", "Reload the case before collecting more evidence.");
        }
        if (!bindings.resource_runtime.store) return common_adapter_execution_failure(
            "tool.investigation.evidence_store_unavailable", "resource store is unavailable",
            "Evidence collection is unavailable because persistent resource storage is not configured.");
        common_agent_dataset_descriptor dataset;
        const auto dataset_uri = arguments.value("dataset", std::string());
        if (!bindings.data_store->get_dataset_descriptor(dataset_uri, dataset, error) ||
                dataset.ref.storage_mode != common_agent_dataset_storage_mode::virtual_ ||
                dataset.ref.source_provider != value.source_id) {
            return common_adapter_not_found_failure("tool.investigation.dataset_unavailable",
                error.empty() ? "dataset is not a configured virtual dataset for this case source" : std::move(error),
                "Choose a virtual dataset configured for this case source.");
        }
        const size_t limit = arguments.value("limit", size_t(200));
        if (limit == 0 || limit > 500) return common_adapter_validation_failure(
            "tool.investigation.collect_limit_invalid", "collection page size must be from 1 to 500");
        json selected = json::array();
        for (const auto & column : dataset.columns) selected.push_back(column.name);
        json predicates = json::array({{{"field", arguments.value("identifier_field", std::string())},
            {"operator", "="}, {"value", arguments.value("identifier_value", std::string())}}});
        const auto has_column = [&](const std::string & name) {
            return std::any_of(dataset.columns.begin(), dataset.columns.end(),
                [&](const auto & column) { return column.name == name; });
        };
        const auto first_column = [&](std::initializer_list<const char *> candidates) {
            for (const char * candidate : candidates) if (has_column(candidate)) return std::string(candidate);
            return std::string();
        };
        const auto time_field = first_column({"@timestamp", "event_time", "event.time"});
        if (!time_field.empty()) {
            predicates.push_back({{"field", time_field}, {"operator", ">="}, {"value", value.window_start_ms}});
            predicates.push_back({{"field", time_field}, {"operator", "<"}, {"value", value.window_end_ms}});
        }
        const json query = {{"dataset", dataset_uri}, {"select", selected}, {"where", predicates},
            {"limit", limit}, {"max_result_rows", limit}, {"max_scan_rows", limit}};
        std::string response;
        if (!bindings.data_store->execute_scoped("data.query", query.dump(), bindings.tool_scope,
                bindings.allow_network, response, error)) {
            return common_adapter_execution_failure("tool.investigation.collect_failed", std::move(error),
                "The configured source could not return this evidence page.");
        }
        const auto page = json::parse(response, nullptr, false);
        if (!page.is_object() || !page.contains("columns") || !page["columns"].is_array() ||
                !page.contains("rows") || !page["rows"].is_array()) return common_adapter_execution_failure(
            "tool.investigation.collect_invalid_result", "virtual dataset returned an invalid row page",
            "The source result could not be normalized.");
        std::vector<std::string> columns;
        for (const auto & column : page["columns"]) if (column.is_string()) columns.push_back(column.get<std::string>());
        std::vector<json> rows;
        for (const auto & row : page["rows"]) {
            if (!row.is_array() || row.size() != columns.size()) continue;
            json object = json::object();
            for (size_t i = 0; i < columns.size(); ++i) object[columns[i]] = row[i];
            rows.push_back(std::move(object));
        }
        const auto get_string = [](const json & row, const std::string & field) {
            if (field.empty() || !row.contains(field) || row[field].is_null()) return std::string();
            if (row[field].is_string()) return row[field].get<std::string>();
            return row[field].dump();
        };
        const auto get_alias = [&](const json & row, std::initializer_list<const char *> candidates) {
            for (const char * candidate : candidates) {
                const auto result = get_string(row, candidate);
                if (!result.empty()) return result;
            }
            return std::string();
        };
        std::map<std::string, common_investigation_entity> entities_by_id;
        std::unordered_set<std::string> existing_events;
        for (const auto & event : value.events) existing_events.insert(event.id);
        std::vector<common_investigation_event> collected_events;
        std::vector<std::string> limitations;
        if (!page.value("complete", false) || page.value("scan_truncated", false) ||
                page.value("result_truncated", false)) {
            limitations.push_back("source page is partial or truncated; case coverage does not represent a complete search");
        }
        const auto entity_id = [&](common_investigation_entity_kind kind, const std::string & raw,
                const std::string & host = std::string()) {
            if (raw.empty()) return std::string();
            const std::string kind_name = kind == common_investigation_entity_kind::user ? "user" :
                kind == common_investigation_entity_kind::host ? "host" :
                kind == common_investigation_entity_kind::process ? "process" :
                kind == common_investigation_entity_kind::ip ? "ip" :
                kind == common_investigation_entity_kind::file ? "file" :
                kind == common_investigation_entity_kind::hash ? "hash" :
                kind == common_investigation_entity_kind::service ? "service" :
                kind == common_investigation_entity_kind::trace ? "trace" : "message";
            std::string identity = value.source_id + ":" + kind_name + ":" + (host.empty() ? "" : host + ":") + raw;
            if (identity.size() > 480) identity = value.source_id + ":" + kind_name + ":sha256:" + hash_sha256_hex(identity.data(), identity.size());
            const std::string id = "entity:" + identity;
            if (!entities_by_id.count(id)) entities_by_id.emplace(id, common_investigation_entity{
                id, value.source_id, kind, raw, raw, host, 0, 0});
            return id;
        };
        json normalized_rows = json::array();
        for (const auto & row : rows) {
            const std::string source_document_id = get_alias(row, {"event.id", "event_id", "id"});
            const std::string serialized_row = row.dump();
            const std::string event_id = "event:" + value.source_id + ":" +
                (source_document_id.empty() ? hash_sha256_hex(serialized_row.data(), serialized_row.size()) : source_document_id);
            if (existing_events.count(event_id)) continue;
            const std::string event_type_value = get_alias(row, {"event.type", "event_type", "event.category"});
            const std::string event_type = event_type_value.empty() ? "unknown" : event_type_value;
            common_investigation_time event_time;
            const std::string timestamp_field = first_column({"@timestamp", "event_time", "event.time"});
            if (!timestamp_field.empty() && row.contains(timestamp_field))
                event_time.has_event_time = parse_investigation_time(row[timestamp_field], event_time.event_time_ms);
            if (!event_time.has_event_time) limitations.push_back("one or more source timestamps were unavailable or invalid; event time remains unknown");
            event_time.collected_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const std::string user_id = entity_id(common_investigation_entity_kind::user, get_alias(row, {"user.id", "user_id", "user.name"}));
            const std::string host_value = get_alias(row, {"host.id", "host_id", "host.name", "host"});
            const std::string host_id = entity_id(common_investigation_entity_kind::host, host_value);
            const std::string source_ip = entity_id(common_investigation_entity_kind::ip, get_alias(row, {"source.ip", "source_ip"}));
            const std::string destination_ip = entity_id(common_investigation_entity_kind::ip, get_alias(row, {"destination.ip", "destination_ip"}));
            const std::string process_value = get_alias(row, {"process.entity_id", "process_entity_id", "process.pid", "process_pid"});
            const std::string process_start = get_alias(row, {"process.start", "process_start", "process.start_time"});
            const std::string process_identity = process_value.empty() ? std::string() :
                process_start.empty() ? process_value + "@" + event_id : process_value + "@" + process_start;
            if (!process_value.empty() && process_start.empty()) limitations.push_back(
                "process start time was missing; the process identity is isolated to its source event");
            const std::string process_id = entity_id(common_investigation_entity_kind::process, process_identity, host_value);
            const std::string file_hash = get_alias(row, {"file.hash.sha256", "file_hash"});
            const std::string file_value = file_hash.empty()
                ? get_alias(row, {"file.path", "file_path"}) : file_hash;
            const std::string file_id = entity_id(file_hash.empty()
                ? common_investigation_entity_kind::file : common_investigation_entity_kind::hash, file_value);
            const std::string trace_id = entity_id(common_investigation_entity_kind::trace, get_alias(row, {"trace.id", "trace_id"}));
            const std::string message_id = entity_id(common_investigation_entity_kind::message, get_alias(row, {"message.id", "message_id"}));
            std::vector<std::string> refs;
            for (const auto & id : {user_id, host_id, source_ip, destination_ip, process_id, file_id, trace_id, message_id}) if (!id.empty()) refs.push_back(id);
            collected_events.push_back({event_id, value.source_id,
                source_document_id.empty() ? event_id : source_document_id, event_type, event_time,
                refs, {}, {}});
            existing_events.insert(event_id);
            normalized_rows.push_back({{"event_id", event_id}, {"event_type", event_type},
                {"event_time_ms", event_time.event_time_ms}, {"has_event_time", event_time.has_event_time},
                {"entity_ids", refs}});
        }
        if (collected_events.empty()) return common_adapter_success_json({{"case_id", value.case_id},
            {"revision", value.revision}, {"collected_events", 0}, {"complete", page.value("complete", false)},
            {"limitations", limitations}});
        json raw_page = {{"source_id", value.source_id}, {"dataset", dataset_uri},
            {"query", query}, {"result", page}, {"collected_at_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}};
        agent_resource_put_request raw_put;
        raw_put.name = value.case_id + "-source-page-r" + std::to_string(value.revision + 1) + ".json";
        raw_put.description = "Immutable raw search page collected for investigation case " + value.case_id;
        raw_put.mime_type = "application/json"; raw_put.text = raw_page.dump(2);
        raw_put.scope = bindings.tool_scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        raw_put.source_provider = value.source_id; raw_put.source_tool = "investigation.collect";
        apply_agent_resource_runtime(bindings.resource_runtime, raw_put);
        agent_resource_descriptor raw_descriptor;
        if (!bindings.resource_runtime.store->put_text(raw_put, raw_descriptor, error)) return common_adapter_execution_failure(
            "tool.investigation.raw_evidence_store_failed", std::move(error), "The raw search result could not be preserved.");
        json normalized = {{"schema_version", 1}, {"case_id", value.case_id}, {"case_revision", value.revision + 1},
            {"source_id", value.source_id}, {"events", normalized_rows}, {"limitations", limitations}};
        agent_resource_put_request normalized_put = raw_put;
        normalized_put.name = value.case_id + "-normalized-r" + std::to_string(value.revision + 1) + ".json";
        normalized_put.description = "Normalized investigation events and entity references";
        normalized_put.text = normalized.dump(2);
        agent_resource_descriptor normalized_descriptor;
        if (!bindings.resource_runtime.store->put_text(normalized_put, normalized_descriptor, error)) return common_adapter_execution_failure(
            "tool.investigation.normalized_evidence_store_failed", std::move(error), "The normalized evidence could not be preserved.");
        for (auto & event : collected_events) {
            event.raw_resource_uri = raw_descriptor.uri;
            event.normalized_resource_uri = normalized_descriptor.uri;
            value.events.push_back(event);
        }
        for (auto & item : entities_by_id) {
            if (std::none_of(value.entities.begin(), value.entities.end(), [&](const auto & old) { return old.id == item.first; }))
                value.entities.push_back(std::move(item.second));
        }
        value.gaps.insert(value.gaps.end(), limitations.begin(), limitations.end());
        std::vector<common_investigation_relation_rule> rules;
        for (const auto & event : collected_events) {
            if (event.event_type.find("auth") != std::string::npos || event.event_type.find("login") != std::string::npos)
                rules.push_back({"builtin-authentication-v1:" + hash_sha256_hex(event.event_type.data(), event.event_type.size()), event.event_type, common_investigation_entity_kind::user,
                    common_investigation_entity_kind::host, common_investigation_relation_kind::authenticated_to});
            if (event.event_type.find("network") != std::string::npos || event.event_type.find("connection") != std::string::npos)
                rules.push_back({"builtin-network-v1:" + hash_sha256_hex(event.event_type.data(), event.event_type.size()), event.event_type, common_investigation_entity_kind::ip,
                    common_investigation_entity_kind::ip, common_investigation_relation_kind::communicated_with});
            if (event.event_type.find("process") != std::string::npos)
                rules.push_back({"builtin-process-v1:" + hash_sha256_hex(event.event_type.data(), event.event_type.size()), event.event_type, common_investigation_entity_kind::process,
                    common_investigation_entity_kind::process, common_investigation_relation_kind::spawned});
            if (event.event_type.find("file") != std::string::npos)
                rules.push_back({"builtin-file-v1:" + hash_sha256_hex(event.event_type.data(), event.event_type.size()), event.event_type, common_investigation_entity_kind::process,
                    common_investigation_entity_kind::file, common_investigation_relation_kind::accessed});
            if (event.event_type.find("trace") != std::string::npos)
                rules.push_back({"builtin-trace-v1:" + hash_sha256_hex(event.event_type.data(), event.event_type.size()), event.event_type, common_investigation_entity_kind::trace,
                    common_investigation_entity_kind::message, common_investigation_relation_kind::correlated_with});
        }
        std::sort(rules.begin(), rules.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
        rules.erase(std::unique(rules.begin(), rules.end(), [](const auto & a, const auto & b) { return a.id == b.id && a.event_type == b.event_type; }), rules.end());
        std::vector<common_investigation_relation> relations;
        if (!common_investigation_build_relations(value.source_id, value.entities, value.events,
                rules, relations, limitations, error)) return common_adapter_validation_failure(
            "tool.investigation.normalization_failed", std::move(error));
        value.relations = std::move(relations);
        value.gaps.insert(value.gaps.end(), limitations.begin(), limitations.end());
        const uint64_t expected = value.revision++;
        common_investigation_case_revision published;
        const auto revision_uri = "investigation-case://" + value.case_id + "/revision/" + std::to_string(value.revision);
        if (!bindings.data_store->publish_investigation_case(value, expected, revision_uri, published, error))
            return common_adapter_execution_failure("tool.investigation.revision_conflict", std::move(error),
                "The case changed while evidence was being collected; reload and retry.");
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"collected_events", collected_events.size()}, {"complete", page.value("complete", false)},
            {"limitations", limitations}, {"raw_resource_uri", raw_descriptor.uri},
            {"normalized_resource_uri", normalized_descriptor.uri}});
    }
    if (operation == "investigation.analyze") {
        const auto analysis = arguments.value("analysis", std::string());
        common_investigation_analysis_budget budget;
        budget.max_expansions = arguments.value("max_expansions", budget.max_expansions);
        budget.max_path_length = arguments.value("max_path_length", budget.max_path_length);
        if (analysis == "graph_metrics") {
            common_investigation_graph_metrics metrics;
            if (!common_investigation_graph_analyze(value, budget, metrics, error)) {
                return common_adapter_validation_failure("tool.investigation.analysis_invalid", std::move(error));
            }
            json nodes = json::array();
            for (const auto & node : metrics.nodes) nodes.push_back({{"entity_id", node.entity_id},
                {"degree", node.degree}, {"component", node.component},
                {"betweenness", node.betweenness}});
            return common_adapter_success_json({{"case_revision", metrics.case_revision},
                {"analysis", analysis}, {"complete", metrics.complete},
                {"nodes", nodes}, {"explored_states", metrics.explored_states},
                {"betweenness_normalized", metrics.betweenness_normalized},
                {"optimality", metrics.complete ? "exact_for_case_revision" : "budget_limited"}});
        }
        const auto start = arguments.value("start_entity_id", std::string());
        const auto goal = arguments.value("goal_entity_id", std::string());
        if (start.empty() || goal.empty()) return common_adapter_validation_failure(
            "tool.investigation.analysis_endpoints_required", "path analysis requires start_entity_id and goal_entity_id");
        std::vector<std::string> relation_ids;
        bool complete = false;
        const bool ok = analysis == "shortest_path"
            ? common_investigation_graph_shortest_path(value, start, goal, budget, relation_ids, complete, error)
            : analysis == "temporal_path"
                ? common_investigation_temporal_path(value, start, goal,
                    arguments.value("not_before_ms", value.window_start_ms), budget, relation_ids, complete, error)
                : false;
        if (!ok) return common_adapter_validation_failure("tool.investigation.analysis_invalid",
            error.empty() ? "analysis must be graph_metrics, shortest_path, or temporal_path" : std::move(error));
        std::set<std::string> evidence_ids;
        for (const auto & relation_id : relation_ids) {
            const auto relation = std::find_if(value.relations.begin(), value.relations.end(),
                [&](const auto & item) { return item.id == relation_id; });
            if (relation != value.relations.end()) evidence_ids.insert(relation->evidence_event_ids.begin(),
                relation->evidence_event_ids.end());
        }
        return common_adapter_success_json({{"case_revision", value.revision},
            {"analysis", analysis}, {"complete", complete}, {"relation_ids", relation_ids},
            {"evidence_refs", std::vector<std::string>(evidence_ids.begin(), evidence_ids.end())},
            {"optimality", complete ? (relation_ids.empty() ? "no_path_in_case" : "shortest_hop_count") : "budget_limited"}});
    }
    if (operation == "investigation.graph.expand") {
        common_investigation_graph_query query;
        query.case_id = case_id;
        query.case_revision = arguments.value("case_revision", uint64_t(0));
        query.center_entity_id = arguments.value("center_entity_id", std::string());
        query.max_depth = arguments.value("max_depth", size_t(1));
        query.max_nodes = arguments.value("max_nodes", size_t(100));
        query.max_edges = arguments.value("max_edges", size_t(200));
        query.window_start_ms = arguments.value("window_start_ms", value.window_start_ms);
        query.window_end_ms = arguments.value("window_end_ms", value.window_end_ms);
        common_investigation_graph_result graph;
        if (!common_investigation_graph_expand(value, query, graph, error)) {
            return common_adapter_validation_failure("tool.investigation.graph_invalid", std::move(error));
        }
        json nodes = json::array(), edges = json::array();
        for (const auto & entity : graph.nodes) nodes.push_back({{"id", entity.id},
            {"type", static_cast<int>(entity.kind)}, {"label", entity.display_name.empty() ? entity.value : entity.display_name}});
        for (const auto & relation : graph.edges) edges.push_back({{"id", relation.id},
            {"source", relation.from_entity_id}, {"target", relation.to_entity_id},
            {"type", static_cast<int>(relation.kind)}, {"first_seen_ms", relation.first_seen_ms},
            {"last_seen_ms", relation.last_seen_ms}, {"event_count", relation.event_count},
            {"evidence_refs", relation.evidence_event_ids}, {"classification", relation.confidence_label}});
        return common_adapter_success_json({{"case_revision", graph.case_revision}, {"nodes", nodes},
            {"edges", edges}, {"truncated", graph.truncated}, {"limitations", graph.limitations}});
    }
    if (operation == "investigation.events") {
        const auto entity_id = arguments.value("entity_id", std::string());
        const auto relation_id = arguments.value("relation_id", std::string());
        if (entity_id.empty() == relation_id.empty()) return common_adapter_validation_failure(
            "tool.investigation.events_invalid", "events requires exactly one entity_id or relation_id");
        std::set<std::string> evidence_ids;
        if (!relation_id.empty()) {
            const auto relation = std::find_if(value.relations.begin(), value.relations.end(),
                [&](const auto & item) { return item.id == relation_id; });
            if (relation == value.relations.end()) return common_adapter_not_found_failure(
                "tool.investigation.relation_unavailable", "relation is not in this case",
                "The selected relation is unavailable in this case.");
            evidence_ids.insert(relation->evidence_event_ids.begin(), relation->evidence_event_ids.end());
        }
        json events = json::array();
        const size_t limit = arguments.value("limit", size_t(100));
        bool truncated = false;
        for (const auto & event : value.events) {
            if (!entity_id.empty() && std::find(event.entity_ids.begin(), event.entity_ids.end(), entity_id) == event.entity_ids.end()) continue;
            if (!relation_id.empty() && !evidence_ids.count(event.id)) continue;
            if (event.time.has_event_time && ((arguments.contains("window_start_ms") && event.time.event_time_ms < arguments["window_start_ms"].get<int64_t>()) ||
                    (arguments.contains("window_end_ms") && event.time.event_time_ms >= arguments["window_end_ms"].get<int64_t>()))) continue;
            if (events.size() >= limit) { truncated = true; break; }
            events.push_back({{"id", event.id}, {"source_document_id", event.source_document_id},
                {"event_type", event.event_type}, {"time", {{"event_time_ms", event.time.event_time_ms},
                    {"collected_time_ms", event.time.collected_time_ms}, {"uncertainty_ms", event.time.uncertainty_ms},
                    {"has_event_time", event.time.has_event_time}}}, {"raw_resource_uri", event.raw_resource_uri},
                {"normalized_resource_uri", event.normalized_resource_uri}, {"entity_ids", event.entity_ids}});
        }
        return common_adapter_success_json({{"events", events}, {"truncated", truncated},
            {"limitations", value.gaps}});
    }
    if (operation == "investigation.claim.hypothesize") {
        const auto claim_id = arguments.value("claim_id", std::string());
        if (std::any_of(value.claims.begin(), value.claims.end(),
                [&](const auto & claim) { return claim.id == claim_id; })) {
            return common_adapter_validation_failure("tool.investigation.claim_duplicate", "claim_id already exists");
        }
        common_investigation_claim claim;
        claim.id = claim_id;
        claim.text = arguments.value("text", std::string());
        claim.state = common_investigation_claim_state::hypothesis;
        claim.supporting_event_ids = arguments.value("supporting_event_ids", std::vector<std::string>{});
        claim.contradicting_event_ids = arguments.value("contradicting_event_ids", std::vector<std::string>{});
        claim.limitations = arguments.value("limitations", std::vector<std::string>{});
        value.claims.push_back(std::move(claim));
        const uint64_t expected = value.revision++;
        common_investigation_case_revision published;
        const auto revision_uri = "investigation-case://" + value.case_id + "/revision/" + std::to_string(value.revision);
        if (!bindings.data_store->publish_investigation_case(value, expected, revision_uri, published, error)) {
            return common_adapter_execution_failure("tool.investigation.revision_conflict",
                std::move(error), "The case changed; reload it before recording the hypothesis.");
        }
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"claim_state", "hypothesis"}});
    }
    return common_adapter_validation_failure("tool.investigation.operation_unknown", "unknown investigation operation");
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
        definition.executor_id == "builtin.investigation.case.create" ||
        definition.executor_id == "builtin.investigation.case.open" ||
        definition.executor_id == "builtin.investigation.report.compose" ||
        definition.executor_id == "builtin.investigation.collect" ||
        definition.executor_id == "builtin.investigation.graph.expand" ||
        definition.executor_id == "builtin.investigation.analyze" ||
        definition.executor_id == "builtin.investigation.events" ||
        definition.executor_id == "builtin.investigation.claim.hypothesize" ||
        definition.executor_id == "builtin.data.query" ||
        definition.executor_id == "builtin.data.filter" ||
        definition.executor_id == "builtin.data.aggregate" ||
        definition.executor_id == "builtin.data.join" ||
        definition.executor_id == "builtin.data.transform" ||
        definition.executor_id == "builtin.statistics.describe" ||
        definition.executor_id == "builtin.statistics.outliers" ||
        definition.executor_id == "builtin.statistics.value_counts";
    if (!is_data_tool) return false;

    if (definition.executor_id.rfind("builtin.investigation.", 0) == 0) {
        installed = common_adapter_register_definition(definition, registry,
            [bindings, operation = definition.name](const std::string & input) {
                std::string error;
                json arguments;
                if (!common_adapter_parse_object(input, arguments, error)) {
                    return common_adapter_validation_failure(
                        "tool.investigation.invalid_arguments", std::move(error));
                }
                return execute_investigation_tool(bindings, operation, arguments);
            }, error);
    } else if (definition.executor_id == "builtin.dataset.validate" &&
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
        installed = common_adapter_register_definition(definition, registry, [bindings, operation](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) {
                return common_adapter_validation_failure("tool.data.invalid_arguments", std::move(error));
            }
            return execute_data_backend(bindings, operation, arguments.dump());
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
                if (error == "dataset name was not found") {
                    std::vector<common_agent_dataset_descriptor> candidates;
                    std::string list_error;
                    if (bindings.data_store->list_dataset_descriptors(candidates, list_error) && !candidates.empty()) {
                        std::string choices;
                        for (const auto & candidate : candidates) {
                            if (candidate.ref.name.empty() || candidate.ref.uri.empty()) continue;
                            if (!choices.empty()) choices += ", ";
                            choices += candidate.ref.name + " (" + candidate.ref.uri + ")";
                        }
                        if (!choices.empty()) error += "; choose one of: " + choices;
                    }
                }
                return common_adapter_not_found_failure("tool.dataset.select.not_found", std::move(error), "The named dataset was not found.");
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
