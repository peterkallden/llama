#include "agent/tooling/adapters/families/repository-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "subproc.h"
#include "hash/hash.h"
#include "agent/sandbox/source-snapshot.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

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

std::string legacy_workspace_content_token(const std::string & content) {
    return "host:" + std::to_string(std::hash<std::string>{}(content));
}

bool write_file_atomically(const std::filesystem::path & path, const std::string & content) {
    static std::atomic<uint64_t> temp_sequence{0};
    auto temporary = path;
    temporary += ".agent-tmp-" + std::to_string(temp_sequence.fetch_add(1, std::memory_order_relaxed));
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) return false;
        file.write(content.data(), static_cast<std::streamsize>(content.size()));
        file.flush();
        if (!file) {
            file.close();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
    }
    std::error_code status_error;
    const auto old_status = std::filesystem::status(path, status_error);
    if (!status_error && std::filesystem::exists(old_status)) {
        std::filesystem::permissions(temporary, old_status.permissions(),
            std::filesystem::perm_options::replace, status_error);
        if (status_error) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
    }
    bool replaced = false;
#ifdef _WIN32
    replaced = MoveFileExW(temporary.c_str(), path.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code rename_error;
    std::filesystem::rename(temporary, path, rename_error);
    replaced = !rename_error;
#endif
    if (!replaced) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    return true;
}

bool git_read(const std::string & root, const std::vector<std::string> & git_arguments,
        std::string & output, std::string & error, bool * truncated = nullptr) {
    if (!common_subproc::is_supported()) {
        error = "structured subprocess support is unavailable";
        return false;
    }
    std::vector<std::string> command = {"git", "-C", root, "--no-pager"};
    command.insert(command.end(), git_arguments.begin(), git_arguments.end());
    common_subproc process;
    if (!process.create(command, subprocess_option_combined_stdout_stderr |
            subprocess_option_enable_async | subprocess_option_enable_async_no_wait |
            subprocess_option_inherit_environment | subprocess_option_search_user_path)) {
        error = "unable to launch Git";
        return false;
    }
    output.clear();
    if (truncated) *truncated = false;
    char buffer[4096];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (process.alive()) {
        const unsigned count = process.read_stdout(buffer, sizeof(buffer));
        if (count != 0) {
            const size_t retained = std::min<size_t>(count, 256 * 1024 - output.size());
            output.append(buffer, retained);
            if (truncated && retained < count) *truncated = true;
        } else if (std::chrono::steady_clock::now() >= deadline) {
            process.terminate();
            process.join();
            error = "Git command timed out";
            return false;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    while (const unsigned count = process.read_stdout(buffer, sizeof(buffer))) {
        const size_t retained = std::min<size_t>(count, 256 * 1024 - output.size());
        output.append(buffer, retained);
        if (truncated && retained < count) *truncated = true;
    }
    const int status = process.join();
    if (status != 0) { error = output.empty() ? "Git command failed" : output; return false; }
    return true;
}

std::mutex & workspace_write_mutex(const std::string & root) {
    static std::mutex registry_mutex;
    static std::map<std::string, std::shared_ptr<std::mutex>> locks;
    std::error_code canonical_error;
    const auto canonical_root = std::filesystem::weakly_canonical(root, canonical_error);
    const std::string key = canonical_error ? std::filesystem::path(root).lexically_normal().generic_string()
        : canonical_root.generic_string();
    std::lock_guard<std::mutex> guard(registry_mutex);
    auto & lock = locks[key];
    if (!lock) lock = std::make_shared<std::mutex>();
    return *lock;
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
        definition.executor_id == "builtin.repository.snapshot" ||
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
        installed = common_adapter_register_definition(definition, registry, [bindings, max_bytes = definition.max_result_bytes](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.contains("path") || !arguments["path"].is_string()) {
                if (error.empty()) error = "repository.read requires a path";
                return common_adapter_validation_failure("tool.repository.read.invalid_path", std::move(error), "Repository read requires a valid path.");
            }
            std::filesystem::path path;
            if (!repository_path(bindings.repository_root, arguments["path"].get<std::string>(), path, error)) return common_adapter_validation_failure("tool.repository.read.path_escapes_root", std::move(error), "Repository path is outside the allowed root.");
            if (!std::filesystem::is_regular_file(path)) return common_adapter_not_found_failure(
                "tool.repository.read.file_not_found", "repository.read target is not a regular file",
                "Repository file was not found.");
            std::error_code size_error;
            const auto file_size = std::filesystem::file_size(path, size_error);
            if (size_error || file_size > 64 * 1024 * 1024) return common_adapter_limit_failure(
                "tool.repository.read.file_limit", "repository.read file exceeds the 64 MiB hash limit",
                "The selected file is too large to hash safely in one operation.");
            const int start = arguments.value("start_line", 1), end = arguments.value("end_line", start + 199);
            if (start < 1 || end < start || end - start > 399) return common_adapter_validation_failure("tool.repository.read.invalid_range", "repository.read range or file is invalid", "Requested line range is invalid.");
            if (!text_file(path)) return common_adapter_validation_failure("tool.repository.read.not_text", "repository.read range or file is invalid", "Repository file is not a readable text file.");
            std::ifstream file(path, std::ios::binary); std::string line; json lines = json::array();
            for (int number = 1; std::getline(file, line); ++number) if (number >= start && number <= end) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                lines.push_back({{"line", number}, {"text", line}});
            }
            if (file.bad()) return common_adapter_execution_failure(
                "tool.repository.read.read_failed", "repository.read could not finish reading the file",
                "The selected file could not be read completely.");
            std::ifstream bytes_file(path, std::ios::binary);
            const std::string bytes((std::istreambuf_iterator<char>(bytes_file)), std::istreambuf_iterator<char>());
            if (bytes_file.bad()) return common_adapter_execution_failure(
                "tool.repository.read.hash_failed", "repository.read could not hash the complete file",
                "The selected file could not be read completely.");
            const json output = {{"path", std::filesystem::relative(path, bindings.repository_root).generic_string()},
                {"lines", lines}, {"sha256", "sha256:" + hash_sha256_hex(bytes.data(), bytes.size())}};
            if (output.dump().size() > max_bytes) return common_adapter_limit_failure(
                "tool.repository.read.output_limit", "repository.read line output exceeds the response limit",
                "Reduce the requested line range or read a smaller portion of the file.");
            return common_adapter_success_json(output);
        }, error);
    } else if (definition.executor_id == "builtin.repository.snapshot" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.empty()) {
                if (error.empty()) error = "repository.snapshot takes no arguments";
                return common_adapter_validation_failure("tool.repository.snapshot.invalid_arguments", std::move(error));
            }
            if (bindings.resource_runtime.store == nullptr) return common_adapter_failure(
                "tool.repository.snapshot.resource_store_unavailable", common_tool_failure_class::execution,
                false, "A resource store is required to retain a source manifest.",
                "repository.snapshot has no host resource store binding");
            auto & write_mutex = workspace_write_mutex(bindings.repository_root);
            std::lock_guard<std::mutex> workspace_guard(write_mutex);
            std::string revision;
            if (!git_read(bindings.repository_root, {"rev-parse", "--verify", "HEAD"}, revision, error))
                return common_adapter_execution_failure("tool.repository.snapshot.git_failed", std::move(error), "The Git revision could not be read.");
            if (!revision.empty() && revision.back() == '\n') revision.pop_back();
            bool tracked_truncated = false;
            bool untracked_truncated = false;
            std::string tracked_bytes, untracked_bytes;
            if (!git_read(bindings.repository_root, {"ls-files", "--cached", "-z"}, tracked_bytes, error, &tracked_truncated) ||
                    !git_read(bindings.repository_root, {"ls-files", "--others", "--exclude-standard", "-z"}, untracked_bytes, error, &untracked_truncated))
                return common_adapter_execution_failure("tool.repository.snapshot.files_failed", std::move(error), "The repository file list could not be read.");
            bool status_truncated = false;
            std::string status;
            if (!git_read(bindings.repository_root, {"status", "--short", "--branch"}, status, error, &status_truncated))
                return common_adapter_execution_failure("tool.repository.snapshot.status_failed", std::move(error), "The repository status could not be read.");
            std::vector<std::pair<std::string, std::string>> paths;
            const auto append_paths = [&](const std::string & bytes, const std::string & kind) {
                size_t begin = 0;
                while (begin < bytes.size() && paths.size() < 8192) {
                    const size_t end = bytes.find('\0', begin);
                    if (end == std::string::npos) break;
                    if (end > begin) paths.emplace_back(bytes.substr(begin, end - begin), kind);
                    begin = end + 1;
                }
                return begin == bytes.size();
            };
            bool complete = !tracked_truncated && !untracked_truncated && !status_truncated;
            complete = append_paths(tracked_bytes, "tracked") && complete;
            complete = append_paths(untracked_bytes, "untracked") && complete;
            json files = json::array();
            json unavailable = json::array();
            json deleted_paths = json::array();
            size_t total_bytes = 0;
            for (const auto & entry : paths) {
                if (files.size() >= 8192) { complete = false; break; }
                std::filesystem::path file_path;
                std::error_code size_error;
                // Inspect the lexical path first: repository_path canonicalizes
                // symlinks and would otherwise hide their original file type.
                const auto lexical_path = std::filesystem::path(bindings.repository_root) / entry.first;
                const auto lexical_status = std::filesystem::symlink_status(lexical_path, size_error);
                if (entry.second == "tracked" &&
                        (size_error == std::errc::no_such_file_or_directory ||
                         (!size_error && lexical_status.type() == std::filesystem::file_type::not_found))) {
                    deleted_paths.push_back(entry.first);
                    continue;
                }
                if (size_error || std::filesystem::is_symlink(lexical_status) ||
                        !repository_path(bindings.repository_root, entry.first, file_path, error)) {
                    complete = false;
                    if (unavailable.size() < 32) unavailable.push_back(entry.first);
                    continue;
                }
                const auto file_status = std::filesystem::symlink_status(file_path, size_error);
                if (size_error || std::filesystem::is_symlink(file_status) || !std::filesystem::is_regular_file(file_status)) {
                    complete = false;
                    if (unavailable.size() < 32) unavailable.push_back(entry.first);
                    continue;
                }
                const auto file_size = std::filesystem::file_size(file_path, size_error);
                if (size_error || file_size > 64 * 1024 * 1024 || total_bytes + file_size > 256 * 1024 * 1024) {
                    complete = false;
                    if (unavailable.size() < 32) unavailable.push_back(entry.first);
                    continue;
                }
                std::ifstream file(file_path, std::ios::binary);
                std::string content(static_cast<size_t>(file_size), '\0');
                if (!file) {
                    complete = false;
                    if (unavailable.size() < 32) unavailable.push_back(entry.first);
                    continue;
                }
                if (file_size != 0) file.read(content.data(), static_cast<std::streamsize>(file_size));
                if (file.bad() || static_cast<uintmax_t>(file.gcount()) != file_size ||
                        file.peek() != std::char_traits<char>::eof()) {
                    complete = false;
                    if (unavailable.size() < 32) unavailable.push_back(entry.first);
                    continue;
                }
                total_bytes += content.size();
                const auto file_permissions = std::filesystem::status(file_path, size_error).permissions();
                if (size_error) {
                    complete = false;
                    if (unavailable.size() < 32) unavailable.push_back(entry.first);
                    continue;
                }
                const auto permissions = file_permissions;
                const auto content_hash = "sha256:" + hash_sha256_hex(content.data(), content.size());
                const std::string file_identity = entry.first + "\n" + content_hash;
                agent_resource_put_request file_put;
                file_put.name = "code-source-file-" + hash_sha256_hex(file_identity.data(), file_identity.size()).substr(0, 24);
                file_put.description = "Immutable source file captured for a repository snapshot.";
                file_put.mime_type = "application/vnd.llama-agent.code-source-file";
                file_put.bytes = content;
                file_put.bytes_are_authoritative = true;
                file_put.scope = bindings.tool_scope.project_id.empty()
                    ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
                file_put.source_provider = "native";
                file_put.source_tool = "repository.snapshot";
                file_put.created_at = static_cast<int64_t>(std::time(nullptr));
                file_put.metadata.purpose = "code_source_file";
                file_put.metadata.content_summary = "Exact bytes of a source file in a bounded repository snapshot.";
                file_put.metadata.content_hash = content_hash;
                file_put.lineage.derivation = "code-source-snapshot-file:v1";
                apply_agent_resource_runtime(bindings.resource_runtime, file_put);
                agent_resource_descriptor file_descriptor;
                if (!bindings.resource_runtime.store->put_bytes(file_put, file_descriptor, error) || file_descriptor.uri.empty()) {
                    complete = false;
                    if (unavailable.size() < 32) unavailable.push_back(entry.first);
                    continue;
                }
                files.push_back({{"path", entry.first}, {"kind", entry.second},
                    {"bytes", content.size()}, {"sha256", content_hash},
                    {"permissions", static_cast<uint32_t>(permissions) & 0777u}, {"resource_uri", file_descriptor.uri}});
            }
            const auto status_hash = "sha256:" + hash_sha256_hex(status.data(), status.size());
            const json manifest = {{"schema", "code-source-snapshot.v1"},
                {"source_revision", revision}, {"git_status", status}, {"git_status_sha256", status_hash},
                {"files", files}, {"file_count", files.size()}, {"total_bytes", total_bytes},
                {"complete", complete}, {"unavailable_paths", unavailable}, {"deleted_paths", deleted_paths},
                {"limits", {{"max_files", 8192}, {"max_file_bytes", 64 * 1024 * 1024},
                    {"max_total_bytes", 256 * 1024 * 1024}}}};
            const std::string serialized = manifest.dump(2);
            agent_resource_put_request put;
            put.name = "code-source-manifest-" + hash_sha256_hex(serialized.data(), serialized.size()).substr(0, 20) + ".json";
            put.description = "Bounded, immutable source snapshot for the current repository workspace.";
            put.mime_type = "application/vnd.llama-agent.code-source-snapshot+json";
            put.text = serialized;
            put.scope = bindings.tool_scope.project_id.empty()
                ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
            put.source_provider = "native";
            put.source_tool = "repository.snapshot";
            put.created_at = static_cast<int64_t>(std::time(nullptr));
            put.metadata.purpose = "code_source_manifest";
            put.metadata.content_summary = "Git revision, workspace status and immutable file-resource references for a bounded source snapshot.";
            put.metadata.content_hash = "sha256:" + hash_sha256_hex(serialized.data(), serialized.size());
            put.lineage.derivation = "code-source-snapshot:v1";
            apply_agent_resource_runtime(bindings.resource_runtime, put);
            agent_resource_descriptor descriptor;
            if (!bindings.resource_runtime.store->put_text(put, descriptor, error) || descriptor.uri.empty())
                return common_adapter_execution_failure("tool.repository.snapshot.store_failed", std::move(error), "The source manifest could not be retained.");
            return common_adapter_success_json({{"schema", "code-source-snapshot.v1"},
                {"source_revision", revision}, {"complete", complete}, {"file_count", files.size()},
                {"total_bytes", total_bytes}, {"resource_uri", descriptor.uri},
                {"sha256", descriptor.sha256}});
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
            std::filesystem::path repository_root;
            if (!repository_path(bindings.repository_root, {}, repository_root, error)) return common_adapter_validation_failure("tool.repository.search.invalid_root", std::move(error), "Repository root could not be resolved.");
            std::error_code fs_error;
            const bool path_exists = std::filesystem::exists(root, fs_error);
            if (fs_error) return common_adapter_failure("tool.repository.search.path_error", common_tool_failure_class::execution, false, "Repository search path could not be inspected.", fs_error.message());
            if (!path_exists) return common_adapter_not_found_failure("tool.repository.search.path_not_found", "repository.search path was not found", "Repository search path was not found.");
            const bool is_file = std::filesystem::is_regular_file(root, fs_error);
            if (fs_error) return common_adapter_failure("tool.repository.search.path_error", common_tool_failure_class::execution, false, "Repository search path could not be inspected.", fs_error.message());
            const bool is_directory = !is_file && std::filesystem::is_directory(root, fs_error);
            if (fs_error) return common_adapter_failure("tool.repository.search.path_error", common_tool_failure_class::execution, false, "Repository search path could not be inspected.", fs_error.message());
            if (!is_file && !is_directory) return common_adapter_validation_failure("tool.repository.search.invalid_path_type", "repository.search path must be a file or directory", "Repository search path must be a text file or directory.");
            json matches = json::array();
            const auto scan_file = [&](const std::filesystem::path & candidate) {
                std::error_code candidate_error;
                if (!std::filesystem::is_regular_file(candidate, candidate_error) || candidate_error) return;
                const auto size = std::filesystem::file_size(candidate, candidate_error);
                if (candidate_error || size > 512 * 1024 || !text_file(candidate)) return;
                std::ifstream file(candidate);
                std::string line;
                for (int number = 1; std::getline(file, line) && matches.size() < static_cast<size_t>(limit); ++number) {
                    if (line.find(query) == std::string::npos) continue;
                    matches.push_back({{"path", candidate.lexically_relative(repository_root).generic_string()},
                        {"line", number}, {"preview", line.substr(0, 512)}});
                }
            };
            if (is_file) {
                scan_file(root);
            } else {
                std::filesystem::recursive_directory_iterator it(
                    root, std::filesystem::directory_options::skip_permission_denied, fs_error);
                const std::filesystem::recursive_directory_iterator end;
                if (fs_error) return common_adapter_failure("tool.repository.search.directory_error", common_tool_failure_class::execution, false, "Repository search could not open the requested directory.", fs_error.message());
                while (it != end && matches.size() < static_cast<size_t>(limit)) {
                    scan_file(it->path());
                    it.increment(fs_error);
                    if (fs_error) return common_adapter_failure("tool.repository.search.directory_error", common_tool_failure_class::execution, false, "Repository search could not continue through the requested directory.", fs_error.message());
                }
            }
            return common_adapter_success_json({{"matches", matches}});
        }, error);
    } else if (definition.executor_id == "builtin.workspace.patch" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.contains("path") || !arguments["path"].is_string() || !arguments.contains("operations") || !arguments["operations"].is_array()) return common_adapter_validation_failure("tool.workspace.patch.invalid_arguments", "workspace.patch requires path and operations");
            auto & write_mutex = workspace_write_mutex(bindings.repository_root);
            std::lock_guard<std::mutex> write_guard(write_mutex);
            std::filesystem::path path;
            if (!repository_path(bindings.repository_root, arguments["path"].get<std::string>(), path, error)) return common_adapter_validation_failure("tool.workspace.patch.invalid_path", std::move(error));
            std::string content;
            if (std::filesystem::exists(path)) {
                if (!std::filesystem::is_regular_file(path) || !text_file(path)) return common_adapter_validation_failure("tool.workspace.patch.not_text", "workspace.patch target is not a text file");
            std::ifstream file(path, std::ios::binary); content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            }
            if (arguments.contains("expected_hash")) {
                if (!arguments["expected_hash"].is_string()) return common_adapter_validation_failure("tool.workspace.patch.invalid_hash", "workspace.patch expected_hash must be a string");
                const auto expected = arguments["expected_hash"].get<std::string>();
                const auto stable = "sha256:" + hash_sha256_hex(content.data(), content.size());
                if (expected != stable && expected != legacy_workspace_content_token(content)) return common_adapter_failure("tool.workspace.patch.conflict", common_tool_failure_class::policy, false, "Workspace content changed since it was read.", "expected_hash does not match current workspace content");
            }
            std::string source_snapshot_ref;
            if (arguments.contains("source_snapshot_ref")) {
                if (!arguments["source_snapshot_ref"].is_string()) return common_adapter_validation_failure(
                    "tool.workspace.patch.invalid_snapshot_ref", "source_snapshot_ref must be a resource URI");
                source_snapshot_ref = arguments["source_snapshot_ref"].get<std::string>();
                if (source_snapshot_ref.rfind("resource://", 0) != 0 || bindings.resource_runtime.store == nullptr) {
                    return common_adapter_validation_failure("tool.workspace.patch.invalid_snapshot_ref",
                        "source_snapshot_ref must resolve through the host resource store");
                }
                const auto source_path = std::filesystem::relative(path, bindings.repository_root).generic_string();
                const auto authority = make_agent_resource_read_authority(bindings.resource_runtime);
                agent_resource_descriptor snapshot_descriptor;
                std::string snapshot_json;
                if (!bindings.resource_runtime.store->stat(source_snapshot_ref, authority, snapshot_descriptor, error) ||
                        snapshot_descriptor.mime_type != "application/vnd.llama-agent.code-source-snapshot+json" ||
                        !bindings.resource_runtime.store->read_text(source_snapshot_ref, authority, 16 * 1024 * 1024, snapshot_json, error)) {
                    return common_adapter_failure("tool.workspace.patch.snapshot_unavailable", common_tool_failure_class::policy,
                        false, "The source snapshot is unavailable or has the wrong type.", error);
                }
                common_agent_source_snapshot snapshot;
                if (!common_agent_parse_source_snapshot(snapshot_json, snapshot, error)) {
                    return common_adapter_validation_failure("tool.workspace.patch.snapshot_incomplete", error);
                }
                const auto source_file = std::find_if(snapshot.files.begin(), snapshot.files.end(),
                    [&source_path](const common_agent_source_snapshot_file & file) { return file.path == source_path; });
                if (source_file == snapshot.files.end() || source_file->sha256 !=
                            "sha256:" + hash_sha256_hex(content.data(), content.size())) {
                    return common_adapter_failure("tool.workspace.patch.snapshot_stale", common_tool_failure_class::policy,
                        false, "The workspace file no longer matches the supplied source snapshot.",
                        "source snapshot file hash does not match the current exact file bytes");
                }
            }
            const bool trailing_newline = !content.empty() && content.back() == '\n';
            const std::string line_ending = content.find("\r\n") != std::string::npos ? "\r\n" : "\n";
            std::vector<std::string> lines;
            size_t line_begin = 0;
            while (line_begin < content.size()) {
                const size_t line_end = content.find('\n', line_begin);
                const size_t end = line_end == std::string::npos ? content.size() : line_end;
                size_t text_end = end;
                if (text_end > line_begin && content[text_end - 1] == '\r') --text_end;
                lines.push_back(content.substr(line_begin, text_end - line_begin));
                if (line_end == std::string::npos) break;
                line_begin = line_end + 1;
            }
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
            std::string updated_content;
            for (size_t i = 0; i < lines.size(); ++i) { if (i > 0) updated_content += line_ending; updated_content += lines[i]; }
            if (trailing_newline && !updated_content.empty() && updated_content.back() != '\n') updated_content += line_ending;
            const auto before_hash = "sha256:" + hash_sha256_hex(content.data(), content.size());
            const auto after_hash = "sha256:" + hash_sha256_hex(updated_content.data(), updated_content.size());
            const bool changed = content != updated_content;
            if (changed && !write_file_atomically(path, updated_content)) return common_adapter_execution_failure("tool.workspace.patch.write_failed", "workspace.patch could not atomically replace the target", "Workspace patch could not be written.");
            json patch_receipt = {{"schema", "workspace-change-receipt.v1"},
                {"path", std::filesystem::relative(path, bindings.repository_root).generic_string()},
                {"source_snapshot_ref", source_snapshot_ref.empty() ? json(nullptr) : json(source_snapshot_ref)},
                {"expected_hash", arguments.value("expected_hash", std::string{})},
                {"before_sha256", before_hash}, {"after_sha256", after_hash},
                {"changed", changed}, {"status", changed ? "applied" : "unchanged"},
                {"operations", arguments["operations"]}, {"created_at", static_cast<int64_t>(std::time(nullptr))}};
            common_runtime_resource_ref receipt_ref;
            if (bindings.resource_runtime.store != nullptr) {
                const auto receipt_content = patch_receipt.dump(2);
                agent_resource_put_request receipt_put;
                receipt_put.name = "workspace-change-" + hash_sha256_hex(receipt_content.data(), receipt_content.size()).substr(0, 20) + ".json";
                receipt_put.description = "Immutable receipt for a guarded workspace patch.";
                receipt_put.mime_type = "application/vnd.llama-agent.workspace-change-receipt+json";
                receipt_put.text = receipt_content;
                receipt_put.scope = bindings.tool_scope.project_id.empty()
                    ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
                receipt_put.source_provider = "native";
                receipt_put.source_tool = "workspace.patch";
                receipt_put.created_at = static_cast<int64_t>(std::time(nullptr));
                receipt_put.metadata.purpose = "workspace_change_receipt";
                receipt_put.metadata.content_summary = "Exact before/after hashes, optional source snapshot binding and accepted line operations.";
                receipt_put.metadata.content_hash = "sha256:" + hash_sha256_hex(receipt_content.data(), receipt_content.size());
                receipt_put.lineage.derivation = "workspace-change-receipt:v1";
                apply_agent_resource_runtime(bindings.resource_runtime, receipt_put);
                agent_resource_descriptor receipt_descriptor;
                if (!bindings.resource_runtime.store->put_text(receipt_put, receipt_descriptor, error) || receipt_descriptor.uri.empty()) {
                    if (changed) return common_adapter_execution_failure("tool.workspace.patch.receipt_failed",
                        "workspace.patch was applied but its durable change receipt could not be retained: " + error,
                        "Workspace patch applied, but its receipt could not be saved.");
                    return common_adapter_execution_failure("tool.workspace.patch.receipt_failed",
                        "workspace.patch receipt could not be retained: " + error,
                        "Workspace patch result could not be saved.");
                }
                receipt_ref = receipt_descriptor;
                patch_receipt["receipt_uri"] = receipt_ref.uri;
            }
            json output = {{"path", std::filesystem::relative(path, bindings.repository_root).generic_string()},
                {"content_hash", after_hash}, {"before_sha256", before_hash}, {"after_sha256", after_hash},
                {"sha256", after_hash}, {"changed", changed}, {"status", changed ? "applied" : "unchanged"},
                {"receipt_uri", receipt_ref.uri}, {"source_snapshot_ref", source_snapshot_ref.empty() ? json(nullptr) : json(source_snapshot_ref)}};
            auto result = common_adapter_success_json(output);
            if (!receipt_ref.uri.empty()) result.resource_refs.push_back(receipt_ref);
            return result;
        }, error, false, true);
    } else if (definition.executor_id == "builtin.repository.diff" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) ||
                    arguments.size() > 3 ||
                    (arguments.contains("max_bytes") && !arguments["max_bytes"].is_number_integer()) ||
                    (arguments.contains("path") && (!arguments["path"].is_string() || arguments["path"].get_ref<const std::string &>().empty() || arguments["path"].get_ref<const std::string &>().size() > 512 || arguments["path"].get_ref<const std::string &>().find('\0') != std::string::npos)) ||
                    (arguments.contains("include_untracked") && !arguments["include_untracked"].is_boolean()))
                return common_adapter_validation_failure("tool.repository.diff.invalid_arguments", "repository.diff accepts path, include_untracked and max_bytes");
            for (auto it = arguments.begin(); it != arguments.end(); ++it)
                if (it.key() != "path" && it.key() != "max_bytes" && it.key() != "include_untracked")
                    return common_adapter_validation_failure("tool.repository.diff.invalid_arguments", "repository.diff contains an unknown argument");
            const int max_bytes = arguments.value("max_bytes", 32768);
            if (max_bytes < 1 || max_bytes > 65536) return common_adapter_validation_failure("tool.repository.diff.invalid_limit", "repository.diff max_bytes is out of bounds");
            const std::string path = arguments.value("path", std::string{});
            std::filesystem::path resolved;
            if ((!path.empty() && std::filesystem::path(path).is_absolute()) || !repository_path(bindings.repository_root, path, resolved, error))
                return common_adapter_validation_failure("tool.repository.diff.invalid_path", error.empty() ? "path must be repository-relative" : error);
            const bool include_untracked = arguments.value("include_untracked", false);
            std::vector<std::string> command = {"diff", "HEAD", "--no-ext-diff", "--no-color", "--unified=3", "--"};
            if (!path.empty()) command.push_back(":(literal)" + path);
            std::string diff; bool truncated = false;
            if (!git_read(bindings.repository_root, command, diff, error, &truncated)) return common_adapter_execution_failure("tool.repository.diff.git_failed", std::move(error), "Git diff could not be read.");
            size_t omitted = 0;
            if (include_untracked) {
                std::vector<std::string> list = {"ls-files", "--others", "--exclude-standard", "-z", "--"};
                if (!path.empty()) list.push_back(":(literal)" + path);
                std::string names; bool names_truncated = false;
                if (!git_read(bindings.repository_root, list, names, error, &names_truncated)) return common_adapter_execution_failure("tool.repository.diff.git_failed", std::move(error), "Untracked files could not be listed.");
                truncated = truncated || names_truncated;
                for (size_t pos = 0; pos < names.size();) {
                    const auto end = names.find('\0', pos);
                    if (end == std::string::npos) { truncated = true; break; }
                    const auto name = names.substr(pos, end - pos); pos = end + 1;
                    std::filesystem::path file_path;
                    std::error_code ec;
                    if (!repository_path(bindings.repository_root, name, file_path, error) ||
                            !std::filesystem::is_regular_file(std::filesystem::symlink_status(std::filesystem::path(bindings.repository_root) / name, ec)) || ec) { ++omitted; continue; }
                    if (diff.size() >= static_cast<size_t>(max_bytes)) { truncated = true; break; }
                    std::ifstream file(file_path, std::ios::binary);
                    std::string content(static_cast<size_t>(max_bytes) + 1, '\0');
                    if (!file) { ++omitted; continue; }
                    file.read(&content[0], content.size()); content.resize(static_cast<size_t>(file.gcount()));
                    if (file.bad()) { ++omitted; continue; }
                    const bool partial_file = content.size() > static_cast<size_t>(max_bytes);
                    bool binary = content.find('\0') != std::string::npos;
                    if (partial_file && !binary) {
                        for (int tail = 0; tail < 4; ++tail) {
                            try { (void) json(content).dump(); break; }
                            catch (const json::exception &) { if (!content.empty()) content.pop_back(); }
                        }
                    }
                    try { (void) json(content).dump(); (void) json(name).dump(); } catch (const json::exception &) { binary = true; }
                    const std::string safe_name = json(name).dump(-1, ' ', false, json::error_handler_t::replace);
                    diff += "\n--- untracked " + safe_name + " ---\n";
                    if (binary) { diff += "Binary or non-UTF-8 file omitted\n"; ++omitted; continue; }
                    if (partial_file) { diff += "Untracked file preview (incomplete):\n"; truncated = true; }
                    else {
                        const auto lines = std::count(content.begin(), content.end(), '\n') +
                            (!content.empty() && content.back() != '\n' ? 1 : 0);
                        diff += "--- /dev/null\n+++ " + safe_name + "\n@@ -0,0 +1," + std::to_string(lines) + " @@\n";
                    }
                    for (size_t offset = 0; offset < content.size() && diff.size() <= static_cast<size_t>(max_bytes);) {
                        const auto newline = content.find('\n', offset);
                        const auto end = newline == std::string::npos ? content.size() : newline + 1;
                        diff += "+" + content.substr(offset, end - offset);
                        if (newline == std::string::npos) diff += "\n\\ No newline at end of file\n";
                        offset = end;
                    }
                }
            }
            if (diff.size() > static_cast<size_t>(max_bytes)) { diff.resize(max_bytes); truncated = true; }
            // A byte limit may split a UTF-8 sequence. Keep the response valid JSON.
            while (!diff.empty()) { try { (void) json(diff).dump(); break; } catch (const json::exception &) { diff.pop_back(); truncated = true; } }
            json result = {{"summary", diff}, {"truncated", truncated}, {"base", "HEAD"},
                {"path", path.empty() ? json(nullptr) : json(path)}, {"include_untracked", include_untracked}, {"omitted_files", omitted}};
            constexpr size_t output_limit = 65536;
            if (result.dump().size() > output_limit) {
                size_t low = 0, high = diff.size();
                result["truncated"] = true;
                while (low < high) {
                    const size_t mid = low + (high - low + 1) / 2;
                    auto prefix = diff.substr(0, mid);
                    while (!prefix.empty()) { try { (void) json(prefix).dump(); break; } catch (const json::exception &) { prefix.pop_back(); } }
                    result["summary"] = prefix;
                    if (result.dump().size() <= output_limit) low = mid; else high = mid - 1;
                }
                auto prefix = diff.substr(0, low);
                while (!prefix.empty()) { try { (void) json(prefix).dump(); break; } catch (const json::exception &) { prefix.pop_back(); } }
                result["summary"] = prefix;
            }
            return common_adapter_success_json(result);
        }, error);
    } else if (definition.executor_id == "builtin.repository.log" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments; if (!common_adapter_parse_object(input, arguments, error)) return common_adapter_validation_failure("tool.repository.log.invalid_arguments", std::move(error));
            const int limit = arguments.value("limit", 8); if (limit < 1 || limit > 20) return common_adapter_validation_failure("tool.repository.log.invalid_limit", "repository.log limit is out of bounds", "Repository log limit is out of bounds.");
            std::string log; if (!git_read(bindings.repository_root, {"log", "--no-ext-diff", "--max-count=" + std::to_string(limit), "--pretty=format:%H%x09%s"}, log, error)) return common_adapter_execution_failure("tool.repository.log.git_failed", std::move(error), "Git log could not be read.");
            return common_adapter_success_json({{"commits", log}});
        }, error);
    } else if (definition.executor_id == "builtin.repository.status" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.empty()) { if (error.empty()) error = "repository.status takes no arguments"; return common_adapter_validation_failure("tool.repository.status.invalid_arguments", std::move(error), "Repository status does not take arguments."); }
            std::string status; if (!git_read(bindings.repository_root, {"status", "--short", "--branch"}, status, error)) return common_adapter_execution_failure("tool.repository.status.git_failed", std::move(error), "Git status could not be read.");
            return common_adapter_success_json({{"status", status}});
        }, error);
    } else if (definition.executor_id == "builtin.repository.changed_files" && !bindings.repository_root.empty()) {
        installed = common_adapter_register_definition(definition, registry, [bindings, max_bytes = definition.max_result_bytes](const std::string & input) {
            std::string error; json arguments;
            if (!common_adapter_parse_object(input, arguments, error) || !arguments.empty()) { if (error.empty()) error = "repository.changed_files takes no arguments"; return common_adapter_validation_failure("tool.repository.changed_files.invalid_arguments", std::move(error), "Changed files does not take arguments."); }
            bool truncated = false;
            std::string status; if (!git_read(bindings.repository_root, {"status", "--short", "--untracked-files=all"}, status, error, &truncated)) return common_adapter_execution_failure("tool.repository.changed_files.git_failed", std::move(error), "Changed files could not be read.");
            if (truncated) {
                const auto last_line = status.rfind('\n');
                status.resize(last_line == std::string::npos ? 0 : last_line + 1);
            }
            json files = json::array(); std::istringstream lines(status); std::string line;
            size_t serialized_bytes = 64; // Envelope and truncation marker.
            while (std::getline(lines, line) && files.size() < 512) {
                if (line.size() < 3) continue;
                json file = {{"status", line.substr(0, 2)}, {"path", line.substr(3)}};
                const size_t bytes = file.dump().size() + 1;
                if (serialized_bytes + bytes > max_bytes) { truncated = true; break; }
                serialized_bytes += bytes;
                files.push_back(std::move(file));
            }
            truncated = truncated || !lines.eof();
            return common_adapter_success_json({{"files", files}, {"truncated", truncated}});
        }, error);
    }
    return true;
}
