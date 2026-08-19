#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/adapters/families/data-adapters.h"
#include "agent/tooling/adapters/families/diagnostics-adapters.h"
#include "agent/tooling/adapters/families/document-adapters.h"
#include "agent/tooling/adapters/families/memory-adapters.h"
#include "agent/tooling/adapters/families/repository-adapters.h"
#include "agent/tooling/adapters/families/resource-adapters.h"

#include "agent/tooling/contracts/tool-result-contracts.h"
#include "base64.hpp"
#include "http.h"
#include "memory/memory-retrieval.h"
#include "memory/memory-tool-service.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <sstream>

#ifdef LLAMA_AGENT_WEB_USE_WINHTTP
#include <windows.h>
#include <winhttp.h>
#endif

using json = nlohmann::ordered_json;

namespace {

class calculator_parser {
public:
    explicit calculator_parser(const std::string & text) : text(text) {}

    bool parse(double & value, std::string & error) {
        value = expression(error);
        skip_space();
        if (!error.empty()) return false;
        if (pos != text.size() || !std::isfinite(value)) { error = "invalid bounded arithmetic expression"; return false; }
        return true;
    }

private:
    double expression(std::string & error) {
        double value = term(error);
        while (error.empty()) {
            skip_space();
            if (take('+')) value += term(error);
            else if (take('-')) value -= term(error);
            else break;
        }
        return value;
    }
    double term(std::string & error) {
        double value = factor(error);
        while (error.empty()) {
            skip_space();
            if (take('*')) value *= factor(error);
            else if (take('/')) {
                const double divisor = factor(error);
                if (error.empty() && divisor == 0.0) error = "division by zero";
                else value /= divisor;
            } else break;
        }
        return value;
    }
    double factor(std::string & error) {
        skip_space();
        if (take('+')) return factor(error);
        if (take('-')) return -factor(error);
        if (take('(')) {
            const double value = expression(error);
            skip_space();
            if (error.empty() && !take(')')) error = "missing closing parenthesis";
            return value;
        }
        const size_t begin = pos;
        bool digit = false;
        while (pos < text.size() && (std::isdigit((unsigned char) text[pos]) || text[pos] == '.')) { digit = digit || std::isdigit((unsigned char) text[pos]); ++pos; }
        if (!digit || pos - begin > 64) { error = "expected a number"; return 0.0; }
        try { return std::stod(text.substr(begin, pos - begin)); }
        catch (...) { error = "invalid number"; return 0.0; }
    }
    void skip_space() { while (pos < text.size() && std::isspace((unsigned char) text[pos])) ++pos; }
    bool take(char c) { if (pos < text.size() && text[pos] == c) { ++pos; return true; } return false; }
    const std::string & text;
    size_t pos = 0;
};

bool parse_object(const std::string & arguments_json, json & arguments, std::string & error) {
    arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object()) { error = "tool arguments must be a JSON object"; return false; }
    return true;
}

common_tool_execution_result tool_success_json(const json & value) {
    return common_tool_execution_result::success(value.dump());
}

common_tool_execution_result tool_success_text(std::string value) {
    return common_tool_execution_result::success(std::move(value));
}

common_tool_execution_result tool_execution_failure(
    std::string code,
    std::string raw_diagnostic,
    std::string safe_summary);

common_tool_execution_result execute_data_backend(
        const common_native_tool_bindings & bindings,
        const std::string & operation,
        const std::string & input) {
    if (bindings.data_store == nullptr) {
        return tool_execution_failure(
            "tool.data.backend_unavailable",
            "structured data backend is unavailable",
            "The configured data backend is unavailable.");
    }
    std::string result;
    std::string error;
    if (!bindings.data_store->execute(operation, input, result, error)) {
        return tool_execution_failure(
            "tool.data.backend_failed",
            error.empty() ? "structured data backend failed" : std::move(error),
            "The structured data operation failed.");
    }
    const auto parsed = json::parse(result, nullptr, false);
    if (!parsed.is_object() && !parsed.is_array()) {
        return tool_execution_failure(
            "tool.data.invalid_backend_result",
            "structured data backend returned invalid JSON",
            "The structured data backend returned an invalid result.");
    }
    return tool_success_json(parsed);
}

common_runtime_resource_metadata make_tool_resource_metadata(
        std::string purpose,
        std::string content_summary,
        std::string usage_hint,
        std::string limitations,
        std::vector<std::string> keywords = {},
        std::vector<std::string> entities = {}) {
    common_runtime_resource_metadata metadata;
    metadata.purpose = std::move(purpose);
    metadata.content_summary = std::move(content_summary);
    metadata.usage_hint = std::move(usage_hint);
    metadata.limitations = std::move(limitations);
    metadata.keywords = std::move(keywords);
    metadata.entities = std::move(entities);
    return metadata;
}

bool persist_tool_resource(
        const common_native_tool_bindings & bindings,
        const std::string & name,
        const std::string & description,
        const std::string & mime_type,
        const std::string & text,
        const std::string & source_tool,
        const common_runtime_resource_metadata & metadata,
        common_runtime_resource_ref & resource,
        std::string & error,
        common_runtime_resource_lineage lineage = {}) {
    if (bindings.resource_runtime.store == nullptr) {
        error = "resource store is unavailable";
        return false;
    }

    agent_resource_put_request request;
    request.name = name;
    request.description = description;
    request.mime_type = mime_type;
    request.text = text;
    request.scope = common_runtime_resource_scope::turn;
    request.source_provider = "native";
    request.source_tool = source_tool;
    request.metadata = metadata;
    request.lineage = std::move(lineage);
    apply_agent_resource_runtime(bindings.resource_runtime, request);

    agent_resource_descriptor descriptor;
    if (!bindings.resource_runtime.store->put_text(request, descriptor, error)) {
        return false;
    }

    resource = descriptor;
    error.clear();
    return true;
}

agent_resource_read_authority make_resource_read_authority(
        const common_native_tool_bindings & bindings) {
    return make_agent_resource_read_authority(bindings.resource_runtime, std::time(nullptr));
}

common_tool_execution_result tool_failure(std::string code, common_tool_failure_class failure_class, bool retryable,
        std::string safe_summary, std::string raw_diagnostic) {
    return common_tool_execution_result::failure(std::move(code), failure_class, retryable, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_validation_failure(std::string code, std::string raw_diagnostic, std::string safe_summary = "Tool arguments are invalid.") {
    return tool_failure(std::move(code), common_tool_failure_class::validation, false, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_execution_failure(std::string code, std::string raw_diagnostic, std::string safe_summary) {
    return tool_failure(std::move(code), common_tool_failure_class::execution, false, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_not_found_failure(std::string code, std::string raw_diagnostic, std::string safe_summary) {
    return tool_failure(std::move(code), common_tool_failure_class::not_found, false, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_network_failure(std::string code, std::string raw_diagnostic, std::string safe_summary, bool retryable = true) {
    return tool_failure(std::move(code), common_tool_failure_class::network, retryable, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_limit_failure(std::string code, std::string raw_diagnostic, std::string safe_summary) {
    return tool_failure(std::move(code), common_tool_failure_class::limit, false, std::move(safe_summary), std::move(raw_diagnostic));
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

bool register_definition(const common_tool_definition & definition, common_tool_registry & registry,
        std::function<common_tool_execution_result(const std::string &)> handler, std::string & error, bool read_only = true, bool policy_gated = false) {
    common_registered_tool tool;
    tool.name = definition.name;
    tool.version = definition.version;
    tool.executor_id = definition.executor_id;
    tool.arguments_schema = definition.input_schema_json;
    tool.read_only = read_only;
    tool.policy_gated = policy_gated;
    tool.handler = std::move(handler);
    return registry.register_tool(std::move(tool), error);
}

bool repository_path(const std::string & root, const std::string & relative, std::filesystem::path & out, std::string & error) {
    if (root.empty()) { error = "repository tools require a runtime repository root"; return false; }
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
    if (requested_text != base_text && requested_text.rfind(base_text + "/", 0) != 0) { error = "repository path escapes the runtime root"; return false; }
    out = requested; return true;
}

bool text_file(const std::filesystem::path & path);
std::string lower_copy(std::string value);

common_tool_execution_result fallback_diagnostics_symbol(
        const common_native_tool_bindings & bindings,
        const json & arguments,
        bool references) {
    if (bindings.repository_root.empty()) {
        return tool_execution_failure("tool.diagnostics.repository_unavailable", "diagnostics requires a repository root", "Diagnostics are unavailable without a host-owned repository.");
    }
    const auto symbol = arguments.value("symbol", std::string{});
    const int limit = arguments.value("max_results", references ? 128 : 64);
    if (symbol.empty() || symbol.size() > 256 || limit < 1 || limit > (references ? 128 : 64)) {
        return tool_validation_failure(references ? "tool.diagnostics.references.invalid_arguments" : "tool.diagnostics.symbol.invalid_arguments", "diagnostics symbol arguments are out of bounds");
    }
    std::filesystem::path root;
    std::string path_error;
    const auto path_hint = arguments.value(references ? "definition_path" : "path_hint", std::string{});
    if (!repository_path(bindings.repository_root, path_hint, root, path_error)) return tool_validation_failure("tool.diagnostics.invalid_path", std::move(path_error));
    if (std::filesystem::is_regular_file(root)) {
        // Keep the path-hint contract useful for a single file as well.
    } else if (!std::filesystem::is_directory(root)) {
        return tool_not_found_failure("tool.diagnostics.path_not_found", "diagnostics path was not found", "The diagnostics path was not found.");
    }
    json matches = json::array();
    auto scan = [&](const std::filesystem::path & file) {
        std::error_code file_error;
        if (matches.size() >= static_cast<size_t>(limit) || !std::filesystem::is_regular_file(file) || std::filesystem::file_size(file, file_error) > 512 * 1024 || file_error || !text_file(file)) return;
        std::ifstream stream(file); std::string line;
        for (int number = 1; std::getline(stream, line) && matches.size() < static_cast<size_t>(limit); ++number) {
            size_t offset = line.find(symbol);
            while (offset != std::string::npos && matches.size() < static_cast<size_t>(limit)) {
                json item = {{"path", std::filesystem::relative(file, bindings.repository_root).generic_string()}, {"line", number}, {"column", offset + 1}, {"symbol", symbol}, {"kind", references ? "reference" : "text_match"}};
                if (!references) item["kind"] = (line.find("class " + symbol) != std::string::npos || line.find("struct " + symbol) != std::string::npos || line.find("enum " + symbol) != std::string::npos) ? "definition" : "text_match";
                matches.push_back(item);
                offset = line.find(symbol, offset + symbol.size());
            }
        }
    };
    if (std::filesystem::is_regular_file(root)) scan(root);
    else for (auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied); it != std::filesystem::recursive_directory_iterator() && matches.size() < static_cast<size_t>(limit); ++it) scan(it->path());
    return tool_success_json({{"backend", "text-fallback"}, {"semantic", false}, {"symbol", symbol}, {references ? "references" : "definitions", matches}, {"count", matches.size()}, {"truncated", matches.size() >= static_cast<size_t>(limit)}});
}

std::string normalize_failure_message(std::string value) {
    value = std::regex_replace(value, std::regex(R"([A-Za-z]:[\\/][^ :]+)"), "<path>");
    value = std::regex_replace(value, std::regex(R"([^\s:]+\.(?:c|cc|cpp|cxx|h|hh|hpp):\d+(?::\d+)?)"), "<location>");
    value = std::regex_replace(value, std::regex(R"((expected\s+)\d+)"), "$1<value>");
    value = std::regex_replace(value, std::regex(R"(0x[0-9A-Fa-f]+|\b\d{4,}\b)"), "<number>");
    value = std::regex_replace(value, std::regex(R"(\s+)"), " ");
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

std::string classify_failure(const std::string & line) {
    const auto lower = lower_copy(line);
    if (lower.find("timeout") != std::string::npos || lower.find("timed out") != std::string::npos) return "timeout";
    if (lower.find("segmentation fault") != std::string::npos || lower.find("access violation") != std::string::npos || lower.find("crash") != std::string::npos) return "crash";
    if (lower.find("linker") != std::string::npos || lower.find("undefined reference") != std::string::npos || lower.find("unresolved external") != std::string::npos) return "link_failure";
    if (lower.find("compile") != std::string::npos || lower.find("compiler") != std::string::npos) return "compile_failure";
    if (lower.find("permission") != std::string::npos || lower.find("access denied") != std::string::npos) return "permission";
    if (lower.find("network") != std::string::npos || lower.find("connection") != std::string::npos) return "network";
    if (lower.find("assert") != std::string::npos || lower.find("expect") != std::string::npos) return "assertion_failure";
    return "test_failure";
}

bool make_sandbox_request(
        const std::string & tool_name,
        const std::string & input,
        common_agent_sandbox_request & request,
        std::string & error) {
    json arguments;
    if (!parse_object(input, arguments, error) || !arguments.contains("target") || !arguments["target"].is_string()) {
        if (error.empty()) error = tool_name + " requires a target";
        return false;
    }
    const auto target = arguments["target"].get<std::string>();
    if (target.empty() || target.size() > 256) {
        error = tool_name + " target is out of bounds";
        return false;
    }
    request = {};
    const bool is_test = tool_name == "development.test";
    request.operation_id = tool_name + "/" + target;
    request.execution_class = "developer-build";
    request.command.program = is_test ? "agent.development.test" : "agent.development.build";
    request.command.arguments.push_back(target);
    if (arguments.contains("configuration")) {
        if (!arguments["configuration"].is_string()) { error = tool_name + " configuration must be a string"; return false; }
        request.command.arguments.push_back(arguments["configuration"].get<std::string>());
    }
    if (is_test && arguments.contains("filter")) {
        if (!arguments["filter"].is_string()) { error = "development.test filter must be a string"; return false; }
        request.command.arguments.push_back(arguments["filter"].get<std::string>());
    }
    request.limits.timeout_ms = arguments.value("timeout_ms", 120000u);
    request.limits.cpu_count = 1;
    request.limits.max_output_bytes = 65536;
    request.network = common_agent_sandbox_network_scope::none;
    request.filesystem = common_agent_sandbox_filesystem_scope::workspace_write;
    request.artifacts.collect = true;
    if (arguments.contains("resource_refs")) {
        if (!arguments["resource_refs"].is_array() || arguments["resource_refs"].size() > 32) {
            error = tool_name + " resource_refs must be an array with at most 32 entries";
            return false;
        }
        for (const auto & value : arguments["resource_refs"]) {
            if (!value.is_string() || value.get<std::string>().empty()) {
                error = tool_name + " resource_refs entries must be non-empty strings";
                return false;
            }
            common_runtime_resource_ref resource;
            resource.uri = value.get<std::string>();
            request.workspace.input_resources.push_back(std::move(resource));
        }
    }
    error.clear();
    return true;
}

bool text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    char buffer[1024]; input.read(buffer, sizeof(buffer));
    return input.good() || input.eof() ? std::find(buffer, buffer + input.gcount(), '\0') == buffer + input.gcount() : false;
}

std::string workspace_content_token(const std::string & content) {
    return "host:" + std::to_string(std::hash<std::string>{}(content));
}

std::string lower_copy(std::string value);

std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
            else quoted = !quoted;
        } else if (c == ',' && !quoted) {
            fields.push_back(field); field.clear();
        } else field += c;
    }
    fields.push_back(std::move(field));
    return fields;
}

bool dataset_file(const common_native_tool_bindings & bindings, const std::string & relative,
        std::filesystem::path & path, std::string & error) {
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

bool git_read(const std::string & root, const std::string & arguments, std::string & output, std::string & error) {
    if (root.find_first_of("\"&|;<>`") != std::string::npos) { error = "repository root cannot be represented safely for Git"; return false; }
    const std::string command = "git -C \"" + root + "\" " + arguments + " 2>&1";
#ifdef _WIN32
    FILE * process = _popen(command.c_str(), "r");
#else
    FILE * process = popen(command.c_str(), "r");
#endif
    if (!process) { error = "unable to launch Git"; return false; }
    char buffer[512]; output.clear(); while (fgets(buffer, sizeof(buffer), process) && output.size() < 16384) output += buffer;
#ifdef _WIN32
    const int status = _pclose(process);
#else
    const int status = pclose(process);
#endif
    if (status != 0) { error = output.empty() ? "Git command failed" : output; return false; }
    return true;
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return value;
}

std::string trim_copy(const std::string & value) {
    size_t begin = 0, end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char) value[end - 1])) --end;
    return value.substr(begin, end - begin);
}

std::string csv_escape(const json & value) {
    if (value.is_null()) return {};
    std::string text = value.is_string() ? value.get<std::string>() : value.dump();
    const bool quote = text.find_first_of(",\"\r\n") != std::string::npos;
    if (!quote) return text;
    std::string escaped = "\"";
    for (const char c : text) {
        if (c == '\"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += '"';
    return escaped;
}

common_tool_execution_result export_dataset_csv(
        const common_native_tool_bindings & bindings,
        const json & arguments) {
    if (!bindings.data_store) {
        return tool_execution_failure(
            "tool.artifact.export.backend_unavailable",
            "dataset export requires a structured data backend",
            "The dataset backend is unavailable.");
    }
    if (!arguments.contains("source_dataset") || !arguments["source_dataset"].is_string() ||
            arguments["source_dataset"].get<std::string>().empty()) {
        return tool_validation_failure(
            "tool.artifact.export.invalid_dataset",
            "source_dataset must be a non-empty dataset URI");
    }
    const auto format = lower_copy(arguments.value("format", std::string("csv")));
    if (format != "csv") {
        return tool_validation_failure(
            "tool.artifact.export.unsupported_format",
            "dataset artifact export currently supports only csv");
    }
    const auto name = arguments.value("name", std::string("dataset-export.csv"));
    const int max_rows = arguments.value("max_rows", 10000);
    if (name.empty() || name.size() > 256 || max_rows < 1 || max_rows > 10000) {
        return tool_validation_failure(
            "tool.artifact.export.out_of_bounds",
            "dataset CSV export name or max_rows is out of bounds");
    }

    const auto dataset_uri = arguments["source_dataset"].get<std::string>();
    common_agent_dataset_descriptor descriptor;
    std::string error;
    if (!bindings.data_store->get_dataset_descriptor(dataset_uri, descriptor, error)) {
        return tool_not_found_failure(
            "tool.artifact.export.dataset_unavailable", std::move(error),
            "The source dataset is unavailable.");
    }
    if (descriptor.ref.row_count > static_cast<size_t>(max_rows)) {
        return tool_limit_failure(
            "tool.artifact.export.row_limit",
            "dataset contains more rows than the bounded CSV export limit",
            "The dataset is too large for this bounded export.");
    }

    json query = {
        {"dataset", dataset_uri}, {"limit", max_rows},
        {"max_scan_rows", max_rows}, {"max_result_rows", max_rows}};
    const auto queried = execute_data_backend(bindings, "data.query", query.dump());
    if (!queried.ok) return queried;
    const auto result = json::parse(queried.output, nullptr, false);
    if (!result.is_object() || result.value("scan_truncated", false) ||
            result.value("result_truncated", false) || !result.contains("rows") ||
            !result["rows"].is_array()) {
        return tool_limit_failure(
            "tool.artifact.export.incomplete_dataset",
            "dataset query was incomplete or did not return rows",
            "The dataset could not be exported completely.");
    }

    std::vector<std::string> columns;
    for (const auto & column : descriptor.columns) columns.push_back(column.name);
    if (columns.empty() && result.contains("columns") && result["columns"].is_array()) {
        for (const auto & column : result["columns"]) if (column.is_string()) columns.push_back(column.get<std::string>());
    }
    if (columns.size() > 512) {
        return tool_limit_failure(
            "tool.artifact.export.column_limit",
            "dataset contains more than 512 columns",
            "The dataset has too many columns for this export.");
    }
    std::string csv;
    for (size_t index = 0; index < columns.size(); ++index) {
        if (index) csv += ',';
        csv += csv_escape(columns[index]);
    }
    csv += '\n';
    for (const auto & row : result["rows"]) {
        for (size_t index = 0; index < columns.size(); ++index) {
            if (index) csv += ',';
            csv += row.is_object() ? csv_escape(row.value(columns[index], json())) : std::string();
        }
        csv += '\n';
        if (csv.size() > 65536) {
            return tool_limit_failure(
                "tool.artifact.export.byte_limit",
                "dataset CSV export exceeded the bounded artifact size",
                "The dataset is too large for this bounded export.");
        }
    }

    common_runtime_resource_ref resource;
    common_runtime_resource_lineage lineage;
    lineage.parent_uri = dataset_uri;
    lineage.chunk_count = 1;
    lineage.byte_length = csv.size();
    lineage.derivation = "artifact.export:dataset-csv";
    if (!persist_tool_resource(
            bindings, name, "CSV artifact exported from a dataset", "text/csv", csv,
            "artifact.export",
            make_tool_resource_metadata(
                "Provide a bounded CSV artifact derived from a structured dataset.",
                "CSV export of dataset " + dataset_uri + ".",
                "Use the artifact resource for download or bounded resource reads.",
                "The export is bounded by row, column and byte limits; the source dataset remains authoritative."),
            resource, error, std::move(lineage))) {
        return tool_execution_failure("tool.artifact.export.store_failed", std::move(error), "The CSV artifact could not be stored.");
    }
    return common_tool_execution_result::success(
        json({{"resource", resource.uri}, {"name", resource.name}, {"mime_type", resource.mime_type},
              {"source_dataset", dataset_uri}, {"rows", result["rows"].size()}, {"columns", columns.size()}}).dump(),
        "Exported a bounded CSV artifact from the dataset.", {std::move(resource)});
}

bool select_dataset_reference(const json & arguments, std::string & dataset, std::string & error) {
    const bool has_dataset = arguments.contains("dataset") && arguments["dataset"].is_string();
    const bool has_path = arguments.contains("path") && arguments["path"].is_string();
    if (has_dataset == has_path) { error = "dataset operation requires exactly one of dataset or path"; return false; }
    dataset = has_dataset ? arguments["dataset"].get<std::string>() : std::string();
    return true;
}

common_tool_execution_result execute_dataset_descriptor_tool(
        const common_native_tool_bindings & bindings,
        const json & arguments,
        const char * operation) {
    std::string dataset, error;
    if (!select_dataset_reference(arguments, dataset, error)) return tool_validation_failure("tool.dataset.invalid_reference", error);
    if (!dataset.empty()) {
        if (!bindings.data_store) return tool_execution_failure("tool.dataset.backend_unavailable", "dataset descriptor backend is unavailable", "The dataset backend is unavailable.");
        common_agent_dataset_descriptor descriptor;
        if (!bindings.data_store->get_dataset_descriptor(dataset, descriptor, error)) return tool_not_found_failure("tool.dataset.unavailable", error, "The dataset reference is unavailable.");
        if (std::string(operation) == "inspect") return tool_success_json({
            {"dataset", descriptor.ref.uri}, {"name", descriptor.ref.name}, {"rows", descriptor.ref.row_count},
            {"columns", descriptor.ref.column_count}, {"source", descriptor.ref.source_resource_uri},
            {"source_sheet", descriptor.source_sheet_name}, {"source_range", descriptor.source_range},
            {"import_processor", descriptor.import_processor_id},
            {"origin", {{"kind", descriptor.origin.kind}, {"semantic_resource", descriptor.origin.source_representation_uri},
                         {"node_id", descriptor.origin.source_node_id}, {"table_index", descriptor.origin.table_index},
                         {"caption", descriptor.origin.caption},
                         {"header_mode", common_agent_table_header_mode_name(descriptor.origin.header_mode)},
                         {"header_confidence", descriptor.origin.header_confidence}}}});
        json columns = json::array();
        for (const auto & column : descriptor.columns) columns.push_back({
            {"name", column.name}, {"type", common_agent_dataset_column_type_name(column.type)}, {"nullable", column.nullable}});
        return tool_success_json({{"dataset", descriptor.ref.uri}, {"columns", columns}});
    }
    return tool_validation_failure("tool.dataset.legacy_path_required", "legacy dataset path handling must be provided by the existing adapter branch");
}

bool agent_resource_has_text_representation(const agent_resource_descriptor & descriptor) {
    return common_resource_media_type_is_text_like(descriptor.mime_type);
}

std::vector<std::string> agent_resource_available_representations(
        const agent_resource_descriptor & descriptor) {
    std::vector<std::string> representations{"bytes"};
    if (agent_resource_has_text_representation(descriptor)) {
        representations.push_back("text");
    }
    return representations;
}

std::string collapse_ws(const std::string & value) {
    std::string out;
    bool spaced = false;
    for (unsigned char c : value) {
        if (std::isspace(c)) {
            if (!spaced && !out.empty()) out.push_back(' ');
            spaced = true;
        } else {
            out.push_back((char) c);
            spaced = false;
        }
    }
    return trim_copy(out);
}

void replace_all(std::string & value, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string html_decode(std::string value) {
    replace_all(value, "&amp;", "&");
    replace_all(value, "&lt;", "<");
    replace_all(value, "&gt;", ">");
    replace_all(value, "&quot;", "\"");
    replace_all(value, "&#39;", "'");
    replace_all(value, "&nbsp;", " ");
    return value;
}

std::string html_to_text(std::string html) {
    html = std::regex_replace(html, std::regex(R"(<script[\s\S]*?</script>)", std::regex::icase), " ");
    html = std::regex_replace(html, std::regex(R"(<style[\s\S]*?</style>)", std::regex::icase), " ");
    html = std::regex_replace(html, std::regex(R"(<[^>]+>)"), " ");
    return collapse_ws(html_decode(html));
}

std::string html_title(const std::string & html) {
    std::smatch match;
    if (!std::regex_search(html, match, std::regex(R"(<title[^>]*>([\s\S]*?)</title>)", std::regex::icase))) return {};
    return collapse_ws(html_to_text(match[1].str()));
}

json trim_search_results_for_inline(
        const json & results,
        size_t inline_limit) {
    if (!results.is_array() || results.size() <= inline_limit) {
        return results;
    }

    json trimmed = json::array();
    for (size_t i = 0; i < inline_limit; ++i) {
        trimmed.push_back(results.at(i));
    }
    return trimmed;
}

std::string bounded_text_preview(
        const std::string & text,
        size_t max_chars) {
    if (text.size() <= max_chars) {
        return text;
    }
    return text.substr(0, max_chars);
}

std::string url_encode(const std::string & value) {
    std::ostringstream out;
    out << std::hex << std::uppercase;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out << c;
        else if (c == ' ') out << '+';
        else out << '%' << std::setw(2) << std::setfill('0') << (int) c;
    }
    return out.str();
}

std::string url_decode(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '+' ) out.push_back(' ');
        else if (value[i] == '%' && i + 2 < value.size()) {
            const auto hex = value.substr(i + 1, 2);
            char * end = nullptr;
            const long code = std::strtol(hex.c_str(), &end, 16);
            if (end == hex.c_str() + 2) {
                out.push_back((char) code);
                i += 2;
            } else out.push_back(value[i]);
        } else out.push_back(value[i]);
    }
    return out;
}

bool private_ipv4(const std::string & host) {
    int a = -1, b = -1, c = -1, d = -1;
    char tail = 0;
    if (std::sscanf(host.c_str(), "%d.%d.%d.%d%c", &a, &b, &c, &d, &tail) != 4) return false;
    if (a == 10 || a == 127) return true;
    if (a == 192 && b == 168) return true;
    if (a == 172 && b >= 16 && b <= 31) return true;
    if (a == 169 && b == 254) return true;
    if (a == 0) return true;
    return false;
}

bool public_https_url(const common_http_url & parts, std::string & error) {
    if (parts.scheme != "https") { error = "only HTTPS URLs are allowed"; return false; }
    if (!parts.user.empty() || !parts.password.empty()) { error = "credentialed URLs are not allowed"; return false; }
    const auto host = lower_copy(parts.host);
    if (host.empty()) { error = "URL host is empty"; return false; }
    if (host == "localhost" || host == "::1" || host.find(".local") != std::string::npos) { error = "private or loopback hosts are not allowed"; return false; }
    if (private_ipv4(host)) { error = "private or loopback hosts are not allowed"; return false; }
    if (host.find(':') != std::string::npos) {
        if (host.rfind("fc", 0) == 0 || host.rfind("fd", 0) == 0 || host.rfind("fe80", 0) == 0 || host == "::1") {
            error = "private or loopback hosts are not allowed"; return false;
        }
    }
    return true;
}

#ifdef LLAMA_AGENT_WEB_USE_WINHTTP

std::wstring utf8_to_wide(const std::string & value) {
    if (value.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int) value.size(), nullptr, 0);
    if (count <= 0) return {};
    std::wstring wide((size_t) count, L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int) value.size(), wide.data(), count) != count) return {};
    return wide;
}

std::string winhttp_error(DWORD code) {
    LPSTR text = nullptr;
    const DWORD length = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &text, 0, nullptr);
    std::string result = length && text ? trim_copy(text) : "WinHTTP error " + std::to_string(code);
    if (text) LocalFree(text);
    return result;
}

bool winhttp_query_string(HINTERNET request, DWORD query, std::string & value, std::string & error) {
    DWORD bytes = 0;
    WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes, WINHTTP_NO_HEADER_INDEX);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes < sizeof(wchar_t)) {
        error = "WinHTTP header query failed: " + winhttp_error(GetLastError());
        return false;
    }
    std::wstring wide(bytes / sizeof(wchar_t), L'\0');
    if (!WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, wide.data(), &bytes, WINHTTP_NO_HEADER_INDEX)) {
        error = "WinHTTP header query failed: " + winhttp_error(GetLastError());
        return false;
    }
    const int output_size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (output_size <= 0) { error = "WinHTTP header conversion failed"; return false; }
    value.resize((size_t) output_size);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, value.data(), output_size, nullptr, nullptr);
    value.pop_back();
    return true;
}

bool winhttp_fetch_text(const std::string & url, const common_http_url & parts, size_t max_bytes,
        std::string & body, int & status, std::string & content_type, std::string & error, bool & truncated) {
    const auto host = utf8_to_wide(parts.host);
    const auto path = utf8_to_wide(parts.path);
    if (host.empty() || path.empty()) { error = "URL is not valid UTF-8"; return false; }

    HINTERNET session = WinHttpOpen(L"llama-agent/1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) { error = "WinHTTP session failed: " + winhttp_error(GetLastError()); return false; }
    const auto close_session = [&] { WinHttpCloseHandle(session); };
    WinHttpSetTimeouts(session, 10000, 10000, 10000, 10000);

    HINTERNET connection = WinHttpConnect(session, host.c_str(), (INTERNET_PORT) parts.port, 0);
    if (!connection) { error = "WinHTTP connection failed: " + winhttp_error(GetLastError()); close_session(); return false; }
    const auto close_connection = [&] { WinHttpCloseHandle(connection); close_session(); };

    HINTERNET request = WinHttpOpenRequest(connection, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    if (!request) { error = "WinHTTP request creation failed: " + winhttp_error(GetLastError()); close_connection(); return false; }
    const auto close_request = [&] { WinHttpCloseHandle(request); close_connection(); };
    const wchar_t headers[] = L"Accept: text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8\r\n";
    if (!WinHttpSendRequest(request, headers, (DWORD) -1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) || !WinHttpReceiveResponse(request, nullptr)) {
        error = "WinHTTP request failed: " + winhttp_error(GetLastError()); close_request(); return false;
    }

    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
            &status_code, &status_size, WINHTTP_NO_HEADER_INDEX)) {
        error = "WinHTTP status query failed: " + winhttp_error(GetLastError()); close_request(); return false;
    }
    status = (int) status_code;
    if (status < 200 || status >= 300) { error = "HTTP status " + std::to_string(status); close_request(); return false; }
    std::string ignored;
    if (!winhttp_query_string(request, WINHTTP_QUERY_CONTENT_TYPE, content_type, ignored)) content_type.clear();

    body.clear();
    body.reserve(std::min<size_t>(max_bytes, 65536));
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request, &available)) { error = "WinHTTP read failed: " + winhttp_error(GetLastError()); close_request(); return false; }
        if (available == 0) break;
        const size_t remaining = max_bytes > body.size() ? max_bytes - body.size() : 0;
        if (remaining == 0) { truncated = true; break; }
        const DWORD to_read = (DWORD) std::min<size_t>(available, std::min<size_t>(remaining, 8192));
        std::string chunk(to_read, '\0');
        DWORD received = 0;
        if (!WinHttpReadData(request, chunk.data(), to_read, &received)) { error = "WinHTTP read failed: " + winhttp_error(GetLastError()); close_request(); return false; }
        body.append(chunk.data(), received);
        if (received < available || body.size() >= max_bytes) { truncated = true; break; }
    }
    close_request();
    return true;
}

#endif

bool http_fetch_text(const std::string & url, size_t max_bytes, json & response, std::string & error, std::string * raw_body = nullptr) {
    try {
#ifdef LLAMA_AGENT_WEB_USE_WINHTTP
        const auto parts = common_http_parse_url(url);
        if (!public_https_url(parts, error)) return false;
        std::string body;
        bool truncated = false;
        int status = 0;
        std::string content_type;
        if (!winhttp_fetch_text(url, parts, max_bytes, body, status, content_type, error, truncated)) return false;
#else
        auto [cli, parts] = common_http_client(url);
        if (!public_https_url(parts, error)) return false;
        cli.set_connection_timeout(10);
        cli.set_read_timeout(10);
        cli.set_write_timeout(10);
        cli.set_follow_location(true);
        cli.set_default_headers({{"User-Agent", "llama-agent/1"}, {"Accept", "text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8"}});

        std::string body;
        body.reserve(std::min<size_t>(max_bytes, 65536));
        bool truncated = false;
        int status = 0;
        std::string content_type;
        auto res = cli.Get(parts.path,
            [&](const httplib::Response & r) {
                status = r.status;
                if (r.has_header("Content-Type")) content_type = r.get_header_value("Content-Type");
                return r.status >= 200 && r.status < 300;
            },
            [&](const char * data, size_t len) {
                const size_t remaining = max_bytes > body.size() ? max_bytes - body.size() : 0;
                const size_t take = std::min(len, remaining);
                body.append(data, take);
                if (take < len || body.size() >= max_bytes) {
                    truncated = true;
                    return false;
                }
                return true;
            },
            nullptr);
        if (!res) { error = "HTTP request failed: " + std::string(httplib::to_string(res.error())); return false; }
#endif
        if (raw_body) *raw_body = body;
        response = {
            {"url", url},
            {"final_url", url},
            {"status", status},
            {"content_type", content_type},
            {"title", html_title(body)},
            {"text", html_to_text(body)},
            {"truncated", truncated},
        };
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool parse_search_results(const std::string & html, int limit, json & results) {
    results = json::array();
    std::regex anchor(R"ddg(<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>)ddg", std::regex::icase);
    std::set<std::string> seen;
    for (std::sregex_iterator it(html.begin(), html.end(), anchor), end; it != end && results.size() < (size_t) limit; ++it) {
        std::string href = html_decode((*it)[1].str());
        std::string title = collapse_ws(html_to_text((*it)[2].str()));
        if (title.empty()) continue;
        const auto lower_href = lower_copy(href);
        if (lower_href.find("duckduckgo.com/l/?") != std::string::npos) {
            const auto pos = href.find("uddg=");
            if (pos != std::string::npos) {
                auto encoded = href.substr(pos + 5);
                const auto amp = encoded.find('&');
                if (amp != std::string::npos) encoded = encoded.substr(0, amp);
                href = url_decode(encoded);
            }
        }
        if (lower_copy(href).rfind("https://", 0) != 0) continue;
        if (!seen.insert(href).second) continue;
        results.push_back({{"title", title}, {"url", href}, {"snippet", ""}, {"source", "duckduckgo-lite"}});
    }
    return !results.empty();
}

} // namespace

bool common_register_native_tool_adapters(const common_tool_catalog & catalog, const std::string & profile_id,
        const common_native_tool_bindings & bindings, common_tool_registry & registry,
        common_tool_adapter_result & result, std::string & error) {
    result = {};
    const auto definitions = catalog.load_profile(profile_id, error);
    if (!error.empty()) return false;
    for (const auto & definition : definitions) {
        bool installed = false;
        const bool handled_by_memory_family = common_try_register_memory_tool_adapter(
                definition, bindings, registry, installed, error);
        if (handled_by_memory_family) {
            // The memory family owns its bindings and policy details.
        } else if (common_try_register_data_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The data family owns dataset, data and statistics adapters.
        } else if (common_try_register_document_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The document family owns document table projection adapters.
        } else if (common_try_register_repository_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The repository family owns repository/workspace filesystem and Git adapters.
        } else if (common_try_register_diagnostics_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The diagnostics family owns compiler, symbol and test-analysis adapters.
        } else if (common_try_register_resource_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The resource family owns resource inspection, reads and representation materialization.
        } else if (definition.executor_id == "builtin.calculator") {
            installed = register_definition(definition, registry, [](const std::string & input) {
                std::string err;
                json arguments;
                if (!parse_object(input, arguments, err) || !arguments.contains("expression") || !arguments["expression"].is_string()) {
                    if (err.empty()) err = "calculator requires an expression";
                    return tool_validation_failure("tool.calculator.invalid_expression", std::move(err), "Calculator requires a valid expression.");
                }
                const auto expression = arguments["expression"].get<std::string>();
                if (expression.size() > 256) return tool_limit_failure("tool.calculator.expression_too_large", "calculator expression exceeds limit", "Calculator expression exceeds the allowed size.");
                double value = 0.0;
                calculator_parser parser(expression);
                if (!parser.parse(value, err)) return tool_validation_failure("tool.calculator.invalid_expression", std::move(err), "Calculator expression could not be parsed.");
                return tool_success_json({{"value", value}});
            }, error);
        } else if (definition.executor_id == "builtin.time_now") {
            installed = register_definition(definition, registry, [](const std::string & input) {
                std::string err;
                json arguments;
                if (!parse_object(input, arguments, err)) return tool_validation_failure("tool.time_now.invalid_arguments", std::move(err));
                const auto timezone = arguments.value("timezone", std::string("UTC"));
                if (timezone != "UTC") return tool_validation_failure("tool.time_now.unsupported_timezone", "time_now currently supports only UTC", "Only UTC is currently supported.");
                return tool_success_json({{"timezone", "UTC"}, {"time", utc_now()}});
            }, error);
        } else if (definition.executor_id == "builtin.artifact.export" && bindings.resource_runtime.store != nullptr) {
            installed = register_definition(definition, registry, [bindings](const std::string & input) {
                std::string err; json arguments;
                if (!parse_object(input, arguments, err)) return tool_validation_failure("tool.artifact.export.invalid_arguments", std::move(err));
                if (arguments.contains("source_dataset")) return export_dataset_csv(bindings, arguments);
                if (!arguments.contains("name") || !arguments["name"].is_string() || !arguments.contains("content") || !arguments["content"].is_string()) return tool_validation_failure("tool.artifact.export.invalid_arguments", "artifact.export requires name and content, or source_dataset");
                common_runtime_resource_ref resource;
                const auto mime = arguments.value("mime_type", std::string("text/plain"));
                if (!persist_tool_resource(bindings, arguments["name"].get<std::string>(), "Exported tool artifact", mime, arguments["content"].get<std::string>(), "artifact.export", make_tool_resource_metadata("artifact", "Exported tool result", "Read as an artifact resource", "Bounded host-owned export"), resource, err)) return tool_execution_failure("tool.artifact.export.failed", std::move(err), "Artifact export failed.");
                return tool_success_json({{"resource", resource.uri}, {"name", resource.name}, {"mime_type", resource.mime_type}});
            }, error);
        } else if ((definition.executor_id == "sandbox.development.build" || definition.executor_id == "sandbox.development.test") && bindings.sandbox_execute) {
            installed = register_definition(definition, registry, [bindings, tool_name = definition.name](const std::string & input) {
                common_agent_sandbox_request request;
                std::string err;
                if (!make_sandbox_request(tool_name, input, request, err)) {
                    return tool_validation_failure("tool." + tool_name + ".invalid_arguments", std::move(err), "Sandbox execution arguments are invalid.");
                }
                return bindings.sandbox_execute(std::move(request));
            }, error, false, true);
        } else if (definition.executor_id == "builtin.web_search") {
            if (bindings.web_search) {
                installed = register_definition(definition, registry, bindings.web_search, error);
            } else {
                installed = register_definition(definition, registry, [bindings](const std::string & input) {
                    std::string err;
                    json arguments;
                    if (!parse_object(input, arguments, err) || !arguments.contains("query") || !arguments["query"].is_string()) {
                        if (err.empty()) err = "web_search requires a query";
                        return tool_validation_failure("tool.web_search.invalid_query", std::move(err), "Web search requires a valid query.");
                    }
                    const auto query = trim_copy(arguments["query"].get<std::string>());
                    const int limit = arguments.value("limit", 5);
                    const auto site = trim_copy(arguments.value("site", std::string{}));
                    if (query.empty() || query.size() > 512 || limit < 1 || limit > 8 || site.size() > 256) {
                        return tool_validation_failure("tool.web_search.out_of_bounds", "web_search arguments are out of bounds", "Web search arguments are out of bounds.");
                    }
                    const auto search_query = site.empty() ? query : query + " site:" + site;
                    json fetched;
                    std::string raw_html;
                    if (!http_fetch_text("https://lite.duckduckgo.com/lite/?q=" + url_encode(search_query), 128 * 1024, fetched, err, &raw_html)) {
                        return tool_network_failure("tool.web_search.fetch_failed", std::move(err), "Web search request failed.");
                    }
                    json results;
                    if (!parse_search_results(raw_html, limit, results)) {
                        return tool_success_json(common_tool_web_search_result_to_json({
                            json::array(),
                            "duckduckgo-lite",
                            std::nullopt,
                            std::nullopt,
                        }));
                    }

                    common_tool_web_search_result payload{
                        results,
                        "duckduckgo-lite",
                        std::nullopt,
                        std::nullopt,
                    };

                    const std::string full_payload = common_tool_web_search_result_to_json(payload).dump();
                    const bool should_externalize =
                        bindings.resource_runtime.store != nullptr &&
                        (full_payload.size() > 2048 || results.size() > 3);
                    if (!should_externalize) {
                        return common_tool_execution_result::success(
                            common_tool_web_search_result_to_json(payload).dump(),
                            "Web search returned " + std::to_string(results.size()) + " candidate(s).");
                    }

                    std::vector<std::string> keywords;
                    keywords.push_back(query);
                    if (!site.empty()) {
                        keywords.push_back(site);
                    }

                    common_runtime_resource_ref resource_ref;
                    if (!persist_tool_resource(
                            bindings,
                            "web-search-results.json",
                            "Full web search result set for the current turn.",
                            "application/json",
                            full_payload,
                            "web_search",
                            make_tool_resource_metadata(
                                "Preserve the full bounded web search candidate set outside the inline model context.",
                                "DuckDuckGo Lite search candidates for query \"" + query + "\".",
                                "Use this resource when the planner or a later tool step needs the full candidate set rather than the inline top hits.",
                                "Search results are unverified provider candidates with short snippets; they are not fetched page contents.",
                                std::move(keywords)),
                            resource_ref,
                            err)) {
                        return tool_execution_failure("tool.web_search.resource_store_failed", std::move(err), "Web search results could not be materialized as a host resource.");
                    }

                    payload.results = trim_search_results_for_inline(results, 3);
                    payload.truncated = results.size() > 3;
                    payload.total_results = results.size();

                    return common_tool_execution_result::success(
                        common_tool_web_search_result_to_json(payload).dump(),
                        "Web search returned " + std::to_string(results.size()) + " candidate(s); the full result set was stored as a turn resource.",
                        {std::move(resource_ref)});
                }, error);
            }
        } else if (definition.executor_id == "builtin.web_fetch") {
            if (bindings.web_fetch) {
                installed = register_definition(definition, registry, bindings.web_fetch, error);
            } else {
                installed = register_definition(definition, registry, [bindings](const std::string & input) {
                    std::string err;
                    json arguments;
                    if (!parse_object(input, arguments, err) || !arguments.contains("url") || !arguments["url"].is_string()) {
                        if (err.empty()) err = "web_fetch requires a url";
                        return tool_validation_failure("tool.web_fetch.invalid_url", std::move(err), "Web fetch requires a valid URL.");
                    }
                    const auto url = trim_copy(arguments["url"].get<std::string>());
                    const auto extract = arguments.value("extract", std::string("text"));
                    const int max_bytes = arguments.value("max_bytes", 64000);
                    if (url.size() < 9 || url.size() > 2048 || extract != "text" || max_bytes < 1 || max_bytes > 500000) {
                        return tool_validation_failure("tool.web_fetch.out_of_bounds", "web_fetch arguments are out of bounds", "Web fetch arguments are out of bounds.");
                    }
                    json fetched;
                    if (!http_fetch_text(url, (size_t) max_bytes, fetched, err)) {
                        return tool_network_failure("tool.web_fetch.request_failed", std::move(err), "Web fetch request failed.");
                    }

                    common_tool_web_fetch_result fetch_result{
                        fetched.value("url", url),
                        fetched.value("final_url", url),
                        fetched.value("status", 0),
                        fetched.value("content_type", std::string{}),
                        fetched.value("title", std::string{}),
                        fetched.value("text", std::string{}),
                        fetched.value("truncated", false),
                    };
                    const std::string full_payload = common_tool_web_fetch_result_to_json(fetch_result).dump();
                    const std::string title = trim_copy(fetch_result.title);
                    const std::string text = fetch_result.text;
                    const bool truncated = fetch_result.truncated;
                    const bool should_externalize =
                        bindings.resource_runtime.store != nullptr &&
                        (full_payload.size() > 4096 || text.size() > 2048 || truncated);
                    if (!should_externalize) {
                        return common_tool_execution_result::success(
                            common_tool_web_fetch_result_to_json(fetch_result).dump(),
                            title.empty()
                                ? "Fetched bounded page text from " + url + "."
                                : "Fetched bounded page text for \"" + title + "\".");
                    }

                    std::vector<std::string> keywords;
                    keywords.push_back(url);
                    if (!title.empty()) {
                        keywords.push_back(title);
                    }

                    common_runtime_resource_ref resource_ref;
                    if (!persist_tool_resource(
                            bindings,
                            "web-fetch-result.json",
                            "Full bounded web fetch payload for the current turn.",
                            "application/json",
                            full_payload,
                            "web_fetch",
                            make_tool_resource_metadata(
                                "Preserve the full bounded web fetch result outside the inline model context.",
                                title.empty()
                                    ? "Fetched bounded page text from " + url + "."
                                    : "Fetched bounded page text for \"" + title + "\".",
                                "Use this resource when a later step needs the full fetched text or metadata rather than the inline excerpt.",
                                truncated
                                    ? "The fetched body was truncated by the configured byte limit and the text field contains extracted text only."
                                    : "The text field contains extracted text only; raw HTML is not preserved in this native payload.",
                                std::move(keywords)),
                            resource_ref,
                            err)) {
                        return tool_execution_failure("tool.web_fetch.resource_store_failed", std::move(err), "Web fetch result could not be materialized as a host resource.");
                    }

                    common_tool_web_fetch_inline_result payload{
                        fetch_result.url,
                        fetch_result.final_url,
                        fetch_result.status,
                        fetch_result.content_type,
                        fetch_result.title,
                        bounded_text_preview(text, 1024),
                        text.size(),
                        truncated,
                    };

                    return common_tool_execution_result::success(
                        common_tool_web_fetch_inline_result_to_json(payload).dump(),
                        title.empty()
                            ? "Fetched bounded page text from " + url + "; the full payload was stored as a turn resource."
                            : "Fetched bounded page text for \"" + title + "\"; the full payload was stored as a turn resource.",
                        {std::move(resource_ref)});
                }, error);
            }
        } else if (definition.executor_id == "builtin.plan_get" && bindings.plan_store && bindings.plan_id != nullptr && !bindings.plan_id->empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input) {
                std::string err;
                json arguments;
                if (!parse_object(input, arguments, err)) return tool_validation_failure("tool.plan_get.invalid_arguments", std::move(err));
                const auto plan = bindings.plan_store->get(*bindings.plan_id, err);
                if (!err.empty()) return tool_execution_failure("tool.plan_get.load_failed", std::move(err), "Plan could not be loaded.");
                if (!plan) return tool_not_found_failure("tool.plan_get.unavailable", "bound plan is unavailable", "Bound plan is unavailable.");
                common_tool_plan_get_payload response;
                response.plan_id = plan->id;
                response.version = plan->version;
                response.goal = plan->goal;
                response.active_step = plan->active_step_id;
                response.next_action = plan->next_action;
                for (const auto & step : plan->steps) {
                    response.steps.push_back({
                        step.id,
                        step.title,
                        step.objective,
                        (int) step.status,
                        step.selected_tool,
                    });
                }
                if (arguments.value("include_history", false)) {
                    const auto history = bindings.plan_store->history(*bindings.plan_id, err);
                    if (!err.empty()) return tool_execution_failure("tool.plan_get.history_failed", std::move(err), "Plan history could not be loaded.");
                    response.history_count = history.size();
                }
                return tool_success_text(common_tool_plan_get_result_to_json(response).dump());
            }, error);
        }
        if (!error.empty()) return false;
        if (installed) result.registered.push_back(definition.name);
        else result.unavailable.push_back(definition.name);
    }
    error.clear();
    return true;
}
