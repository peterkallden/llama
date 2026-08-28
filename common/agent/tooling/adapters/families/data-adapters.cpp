#include "agent/tooling/adapters/families/data-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
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

std::string dataset_scope_component(const std::string & value) {
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

bool is_current_turn_derived_dataset(
        const common_agent_dataset_descriptor & descriptor,
        const agent_resource_runtime & runtime) {
    if (descriptor.origin.kind != "derived" || runtime.turn_id.empty()) return false;
    const std::string prefix = "dataset://agent/turn/";
    if (descriptor.ref.uri.rfind(prefix, 0) != 0) return false;
    const auto component_start = prefix.size();
    const auto component_end = descriptor.ref.uri.find('/', component_start);
    if (component_end == std::string::npos) return false;
    return descriptor.ref.uri.substr(component_start, component_end - component_start) ==
        dataset_scope_component(runtime.turn_id);
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
    if (arguments.is_object() && arguments.contains("dataset") &&
            arguments["dataset"].is_string()) {
        const std::string dataset_uri = arguments["dataset"].get<std::string>();
        if (dataset_uri.rfind("dataset://", 0) != 0) {
            return common_adapter_validation_failure(
                "tool.dataset.invalid_reference",
                "dataset references must use the canonical dataset:// scheme");
        }
        common_agent_dataset_descriptor descriptor;
        std::string lookup_error;
        if (!bindings.data_store->get_dataset_descriptor(dataset_uri, descriptor, lookup_error)) {
            return common_adapter_not_found_failure(
                "tool.dataset.unavailable", std::move(lookup_error),
                "The dataset reference is unavailable in the current host scope.");
        }
        common_agent_dataset_limits limits;
        if (!validate_common_agent_dataset_descriptor(descriptor, limits, lookup_error)) {
            return common_adapter_validation_failure(
                "tool.dataset.invalid_reference", std::move(lookup_error));
        }
        if (bindings.resource_runtime.store != nullptr &&
                !is_current_turn_derived_dataset(descriptor, bindings.resource_runtime)) {
            agent_resource_descriptor source;
            const auto authority = make_agent_resource_read_authority(
                bindings.resource_runtime, std::time(nullptr));
            if (!bindings.resource_runtime.store->stat(
                    descriptor.ref.source_resource_uri, authority, source, lookup_error)) {
                return common_adapter_not_found_failure(
                    "tool.dataset.out_of_scope", std::move(lookup_error),
                    "The dataset source is unavailable in the current host scope.");
            }
        }
    }
    std::string result;
    std::string error;
    if (!bindings.data_store->execute(operation, input, result, error)) {
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
        if (std::string(operation) == "inspect") {
            return common_adapter_success_json({
                {"dataset", descriptor.ref.uri}, {"name", descriptor.ref.name},
                {"rows", descriptor.ref.row_count}, {"columns", descriptor.ref.column_count},
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
        definition.executor_id == "builtin.data.query" ||
        definition.executor_id == "builtin.data.filter" ||
        definition.executor_id == "builtin.data.aggregate" ||
        definition.executor_id == "builtin.data.join" ||
        definition.executor_id == "builtin.data.transform" ||
        definition.executor_id == "builtin.statistics.describe" ||
        definition.executor_id == "builtin.statistics.outliers" ||
        definition.executor_id == "builtin.statistics.value_counts";
    if (!is_data_tool) return false;

    if (definition.executor_id == "builtin.dataset.validate" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error) ||
                    !arguments.contains("dataset") || !arguments["dataset"].is_string() ||
                    !arguments.contains("rules") || !arguments["rules"].is_array()) {
                return common_adapter_validation_failure(
                    "tool.dataset.validate.invalid_arguments", "dataset.validate requires dataset and rules");
            }
            std::filesystem::path path;
            if (!dataset_file(bindings, arguments["dataset"].get<std::string>(), path, error)) {
                return common_adapter_not_found_failure(
                    "tool.dataset.validate.unavailable", std::move(error), "Dataset is unavailable.");
            }
            if (lower_copy(path.extension().string()) != ".csv") {
                return common_adapter_validation_failure(
                    "tool.dataset.validate.unsupported_format", "dataset.validate currently supports CSV");
            }
            std::ifstream file(path);
            std::string line;
            if (!std::getline(file, line)) return common_adapter_success_json({{"valid", true}, {"violations", json::array()}});
            const auto columns = split_csv(line);
            std::vector<std::string> rows;
            while (rows.size() < 100000 && std::getline(file, line)) rows.push_back(line);
            json violations = json::array();
            for (const auto & rule : arguments["rules"]) {
                if (!rule.is_object() || !rule.contains("type") || !rule["type"].is_string() ||
                        !rule.contains("column") || !rule["column"].is_string()) continue;
                const auto type = rule["type"].get<std::string>();
                const auto it = std::find(columns.begin(), columns.end(), rule["column"].get<std::string>());
                if (it == columns.end()) {
                    violations.push_back({{"rule", type + ":" + rule["column"].get<std::string>()}, {"count", rows.size()}, {"message", "column not found"}});
                    continue;
                }
                const auto index = static_cast<size_t>(std::distance(columns.begin(), it));
                size_t count = 0;
                std::set<std::string> values;
                for (const auto & row_text : rows) {
                    const auto fields = split_csv(row_text);
                    const auto value = index < fields.size() ? fields[index] : std::string();
                    if ((type == "not_null" && value.empty()) || (type == "unique" && !values.insert(value).second)) ++count;
                }
                if (count > 0) violations.push_back({{"rule", type + ":" + rule["column"].get<std::string>()}, {"count", count}});
            }
            return common_adapter_success_json({{"valid", violations.empty()}, {"violations", violations}});
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
                return bindings.dataset_from_resource(arguments["resource"].get<std::string>(), "sample");
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
