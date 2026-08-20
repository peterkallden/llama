#include "agent/tooling/adapters/families/repository-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

using json = common_adapter_json;

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
    if (fs_error) { error = "repository root could not be resolved"; return false; }
    const auto requested = relative.empty()
        ? base : std::filesystem::weakly_canonical(base / relative, fs_error);
    if (fs_error) { error = "repository path could not be resolved"; return false; }
    const auto base_text = base.generic_string();
    const auto requested_text = requested.generic_string();
    if (requested_text != base_text && requested_text.rfind(base_text + "/", 0) != 0) {
        error = "repository path escapes the runtime root";
        return false;
    }
    output = requested;
    return true;
}

std::string workspace_content_token(const std::string & content) {
    return "host:" + std::to_string(std::hash<std::string>{}(content));
}

bool git_read(const std::string & root, const std::string & arguments,
        std::string & output, std::string & error) {
    if (root.find_first_of("\"&|;<>`") != std::string::npos) {
        error = "repository root cannot be represented safely for Git";
        return false;
    }
    const std::string command = "git -C \"" + root + "\" " + arguments + " 2>&1";
#ifdef _WIN32
    FILE * process = _popen(command.c_str(), "r");
#else
    FILE * process = popen(command.c_str(), "r");
#endif
    if (!process) { error = "unable to launch Git"; return false; }
    char buffer[512];
    output.clear();
    while (fgets(buffer, sizeof(buffer), process) && output.size() < 16384) output += buffer;
#ifdef _WIN32
    const int status = _pclose(process);
#else
    const int status = pclose(process);
#endif
    if (status != 0) { error = output.empty() ? "Git command failed" : output; return false; }
    return true;
}

} // namespace

bool common_try_register_repository_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const bool is_repository_tool =
        definition.executor_id == "builtin.repository.list" ||
        definition.executor_id == "builtin.workspace.list" ||
        definition.executor_id == "builtin.repository.read" ||
        definition.executor_id == "builtin.workspace.read" ||
        definition.executor_id == "builtin.repository.search" ||
        definition.executor_id == "builtin.workspace.search" ||
        definition.executor_id == "builtin.workspace.patch" ||
        definition.executor_id == "builtin.repository.diff" ||
        definition.executor_id == "builtin.repository.log" ||
        definition.executor_id == "builtin.repository.status" ||
        definition.executor_id == "builtin.repository.changed_files";
    if (!is_repository_tool) return false;

    if ((definition.executor_id == "builtin.repository.list" || definition.executor_id == "builtin.workspace.list") && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) return common_adapter_validation_failure("tool.repository.list.invalid_arguments", std::move(error));
            std::filesystem::path path;
            if (!repository_path(bindings.repository_root, arguments.value("path", std::string{}), path, error)) return common_adapter_validation_failure("tool.repository.list.invalid_path", std::move(error), "Repository path is outside the allowed root.");
            const int depth = arguments.value("depth", 1);
            if (depth < 0 || depth > 3) return common_adapter_validation_failure("tool.repository.list.invalid_depth", "repository.list path or depth is invalid", "Repository list depth is out of bounds.");
            if (!std::filesystem::is_directory(path)) return common_adapter_not_found_failure("tool.repository.list.path_not_found", "repository.list path or depth is invalid", "Repository directory was not found.");
            json entries = json::array();
            for (auto it = std::filesystem::recursive_directory_iterator(path, std::filesystem::directory_options::skip_permission_denied); it != std::filesystem::recursive_directory_iterator() && entries.size() < 128; ++it) {
                if (it.depth() >= depth && it->is_directory()) it.disable_recursion_pending();
                entries.push_back({{"path", std::filesystem::relative(it->path(), bindings.repository_root).generic_string()}, {"directory", it->is_directory()}});
            }
            return common_adapter_success_json({{"entries", entries}});
        }, error);
    } else if ((definition.executor_id == "builtin.repository.read" || definition.executor_id == "builtin.workspace.read") && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.contains("path") || !arguments["path"].is_string()) {
                if (error.empty()) error = "repository.read requires a path";
                return common_adapter_validation_failure("tool.repository.read.invalid_path", std::move(error), "Repository read requires a valid path.");
            }
            std::filesystem::path path;
            if (!repository_path(bindings.repository_root, arguments["path"].get<std::string>(), path, error)) return common_adapter_validation_failure("tool.repository.read.path_escapes_root", std::move(error), "Repository path is outside the allowed root.");
            const int start = arguments.value("start_line", 1), end = arguments.value("end_line", start + 199);
            if (start < 1 || end < start || end - start > 399) return common_adapter_validation_failure("tool.repository.read.invalid_range", "repository.read range or file is invalid", "Requested line range is invalid.");
            if (!std::filesystem::is_regular_file(path)) return common_adapter_not_found_failure("tool.repository.read.file_not_found", "repository.read range or file is invalid", "Repository file was not found.");
            if (!text_file(path)) return common_adapter_validation_failure("tool.repository.read.not_text", "repository.read range or file is invalid", "Repository file is not a readable text file.");
            std::ifstream file(path); std::string line; json lines = json::array();
            for (int number = 1; std::getline(file, line); ++number) if (number >= start && number <= end) lines.push_back({{"line", number}, {"text", line}});
            return common_adapter_success_json({{"path", std::filesystem::relative(path, bindings.repository_root).generic_string()}, {"lines", lines}});
        }, error);
    } else if ((definition.executor_id == "builtin.repository.search" || definition.executor_id == "builtin.workspace.search") && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.contains("query") || !arguments["query"].is_string()) {
                if (error.empty()) error = "repository.search requires a query";
                return common_adapter_validation_failure("tool.repository.search.invalid_query", std::move(error), "Repository search requires a valid query.");
            }
            const auto query = arguments["query"].get<std::string>();
            const int limit = arguments.value("max_results", 16);
            if (query.empty() || query.size() > 256 || limit < 1 || limit > 32) return common_adapter_validation_failure("tool.repository.search.out_of_bounds", "repository.search arguments are out of bounds", "Repository search arguments are out of bounds.");
            std::filesystem::path root;
            if (!repository_path(bindings.repository_root, arguments.value("path", std::string{}), root, error)) return common_adapter_validation_failure("tool.repository.search.invalid_path", std::move(error), "Repository path is outside the allowed root.");
            json matches = json::array();
            for (auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied); it != std::filesystem::recursive_directory_iterator() && matches.size() < static_cast<size_t>(limit); ++it) {
                if (!it->is_regular_file() || it->file_size() > 512 * 1024 || !text_file(it->path())) continue;
                std::ifstream file(it->path()); std::string line;
                for (int number = 1; std::getline(file, line) && matches.size() < static_cast<size_t>(limit); ++number) if (line.find(query) != std::string::npos) matches.push_back({{"path", std::filesystem::relative(it->path(), bindings.repository_root).generic_string()}, {"line", number}, {"preview", line.substr(0, 512)}});
            }
            return common_adapter_success_json({{"matches", matches}});
        }, error);
    } else if (definition.executor_id == "builtin.workspace.patch" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.contains("path") || !arguments["path"].is_string() || !arguments.contains("operations") || !arguments["operations"].is_array()) return common_adapter_validation_failure("tool.workspace.patch.invalid_arguments", "workspace.patch requires path and operations");
            std::filesystem::path path;
            if (!repository_path(bindings.repository_root, arguments["path"].get<std::string>(), path, error)) return common_adapter_validation_failure("tool.workspace.patch.invalid_path", std::move(error));
            std::string content;
            if (std::filesystem::exists(path)) {
                if (!std::filesystem::is_regular_file(path) || !text_file(path)) return common_adapter_validation_failure("tool.workspace.patch.not_text", "workspace.patch target is not a text file");
                std::ifstream file(path); content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            }
            if (arguments.contains("expected_hash") && arguments["expected_hash"].is_string() && arguments["expected_hash"].get<std::string>() != workspace_content_token(content)) return common_adapter_failure("tool.workspace.patch.conflict", common_tool_failure_class::policy, false, "Workspace content changed since it was read.", "expected_hash does not match current workspace content");
            std::vector<std::string> lines; std::istringstream stream(content); std::string line;
            while (std::getline(stream, line)) lines.push_back(line);
            for (const auto & operation : arguments["operations"]) {
                if (!operation.is_object() || !operation.contains("type") || !operation["type"].is_string()) return common_adapter_validation_failure("tool.workspace.patch.invalid_operation", "workspace.patch operation is invalid");
                const auto type = operation["type"].get<std::string>(); const int start = operation.value("start_line", 1), end = operation.value("end_line", start); const auto replacement = operation.value("content", std::string());
                if (start < 1 || end < start || static_cast<size_t>(end) > lines.size() + 1 || (type != "create_file" && static_cast<size_t>(start) > lines.size())) return common_adapter_validation_failure("tool.workspace.patch.invalid_range", "workspace.patch line range is invalid");
                if (type == "create_file") { if (std::filesystem::exists(path)) return common_adapter_validation_failure("tool.workspace.patch.file_exists", "workspace.patch create_file target already exists"); lines = {replacement}; }
                else if (type == "replace_range") lines.erase(lines.begin() + start - 1, lines.begin() + std::min<int>(end, static_cast<int>(lines.size()))), lines.insert(lines.begin() + start - 1, replacement);
                else if (type == "insert_before") lines.insert(lines.begin() + start - 1, replacement);
                else if (type == "insert_after") lines.insert(lines.begin() + std::min<int>(end, static_cast<int>(lines.size())), replacement);
                else if (type == "delete_range") lines.erase(lines.begin() + start - 1, lines.begin() + std::min<int>(end, static_cast<int>(lines.size())));
                else return common_adapter_validation_failure("tool.workspace.patch.unsupported_operation", "workspace.patch operation type is unsupported");
            }
            std::ofstream file(path, std::ios::trunc); if (!file) return common_adapter_execution_failure("tool.workspace.patch.write_failed", "workspace.patch could not open target for writing", "Workspace patch could not be written.");
            for (size_t i = 0; i < lines.size(); ++i) { if (i > 0) file << '\n'; file << lines[i]; }
            if (!file) return common_adapter_execution_failure("tool.workspace.patch.write_failed", "workspace.patch write failed", "Workspace patch could not be written.");
            std::ifstream updated(path); const std::string updated_content((std::istreambuf_iterator<char>(updated)), std::istreambuf_iterator<char>());
            return common_adapter_success_json({{"path", std::filesystem::relative(path, bindings.repository_root).generic_string()}, {"content_hash", workspace_content_token(updated_content)}, {"changed", true}});
        }, error, false, true);
    } else if (definition.executor_id == "builtin.repository.diff" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.empty()) { if (error.empty()) error = "repository.diff takes no arguments"; return common_adapter_validation_failure("tool.repository.diff.invalid_arguments", std::move(error), "Repository diff does not take arguments."); }
            std::string diff; if (!git_read(bindings.repository_root, "diff --no-ext-diff --stat", diff, error)) return common_adapter_execution_failure("tool.repository.diff.git_failed", std::move(error), "Git diff could not be read.");
            return common_adapter_success_json({{"summary", diff}});
        }, error);
    } else if (definition.executor_id == "builtin.repository.log" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments; if (!common_adapter_parse_object(input, arguments, error)) return common_adapter_validation_failure("tool.repository.log.invalid_arguments", std::move(error));
            const int limit = arguments.value("limit", 8); if (limit < 1 || limit > 20) return common_adapter_validation_failure("tool.repository.log.invalid_limit", "repository.log limit is out of bounds", "Repository log limit is out of bounds.");
            std::string log; if (!git_read(bindings.repository_root, "log --no-ext-diff --max-count=" + std::to_string(limit) + " --pretty=format:%h%x09%s", log, error)) return common_adapter_execution_failure("tool.repository.log.git_failed", std::move(error), "Git log could not be read.");
            return common_adapter_success_json({{"commits", log}});
        }, error);
    } else if (definition.executor_id == "builtin.repository.status" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.empty()) { if (error.empty()) error = "repository.status takes no arguments"; return common_adapter_validation_failure("tool.repository.status.invalid_arguments", std::move(error), "Repository status does not take arguments."); }
            std::string status; if (!git_read(bindings.repository_root, "status --short --branch", status, error)) return common_adapter_execution_failure("tool.repository.status.git_failed", std::move(error), "Git status could not be read.");
            return common_adapter_success_json({{"status", status}});
        }, error);
    } else if (definition.executor_id == "builtin.repository.changed_files" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.empty()) { if (error.empty()) error = "repository.changed_files takes no arguments"; return common_adapter_validation_failure("tool.repository.changed_files.invalid_arguments", std::move(error), "Changed files does not take arguments."); }
            std::string status; if (!git_read(bindings.repository_root, "status --short --untracked-files=all", status, error)) return common_adapter_execution_failure("tool.repository.changed_files.git_failed", std::move(error), "Changed files could not be read.");
            json files = json::array(); std::istringstream lines(status); std::string line;
            while (std::getline(lines, line) && files.size() < 512) { if (line.size() < 3) continue; files.push_back({{"status", line.substr(0, 2)}, {"path", line.substr(3)}}); }
            return common_adapter_success_json({{"files", files}});
        }, error);
    }
    return true;
}
