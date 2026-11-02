#include "agent-clangd-provider.h"
#include "agent-clangd-projection.h"

#include <algorithm>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>
#include <chrono>

namespace {
using json = agent_clangd_json;

common_tool_execution_result provider_error(const std::string & code, const std::string & detail) {
    if (detail == "clangd request timed out") {
        return common_tool_execution_result::failure("tool.code.clangd_timeout",
            common_tool_failure_class::timeout, true, "Clangd semantic code analysis timed out.", detail);
    }
    return common_tool_execution_result::failure(code, common_tool_failure_class::execution, false, "Clangd semantic code analysis is unavailable.", detail);
}

std::string language_id(const std::filesystem::path & path) {
    const auto extension = path.extension().string();
    if (extension == ".c") return "c";
    if (extension == ".h") return "cpp";
    if (extension == ".m") return "objective-c";
    if (extension == ".mm") return "objective-cpp";
    return "cpp";
}

size_t utf16_units(const std::string & utf8, size_t byte_count) {
    size_t units = 0;
    for (size_t i = 0; i < byte_count;) {
        const auto lead = static_cast<unsigned char>(utf8[i]);
        size_t width = lead < 0x80 ? 1 : (lead & 0xe0) == 0xc0 ? 2 : (lead & 0xf0) == 0xe0 ? 3 : (lead & 0xf8) == 0xf0 ? 4 : 1;
        if (i + width > byte_count) width = 1;
        units += width == 4 ? 2 : 1;
        i += width;
    }
    return units;
}

bool valid_arguments(const json & args, int limit) {
    if (!args.is_object() || !args.contains("symbol") || !args["symbol"].is_string() ||
            args["symbol"].get_ref<const std::string &>().empty() ||
            args["symbol"].get_ref<const std::string &>().size() > 256) return false;
    for (const auto * key : {"definition_path", "path_hint"})
        if (args.contains(key) && (!args[key].is_string() ||
                args[key].get_ref<const std::string &>().size() > 512 ||
                args[key].get_ref<const std::string &>().find('\0') != std::string::npos)) return false;
    for (const auto * key : {"definition_line", "definition_column", "line", "column", "max_results"})
        if (args.contains(key) && (!args[key].is_number_integer() || args[key] < 1 ||
                args[key] > (std::string(key) == "max_results" ? limit : 1000000))) return false;
    return !args.contains("direction") || (args["direction"].is_string() &&
        (args["direction"] == "callers" || args["direction"] == "callees" || args["direction"] == "both"));
}

bool result_array(const json & response, json & result) {
    if (!response.is_object() || !response.contains("result")) return false;
    result = response["result"].is_null() ? json::array() : response["result"];
    return result.is_array();
}

json coverage(bool truncated, bool incomplete) {
    json limitations = json::array();
    limitations.push_back("Project-wide clangd index coverage is unknown; results may omit symbols and relationships.");
    if (incomplete) limitations.push_back("Some provider entries were invalid or outside the authorized repository.");
    if (truncated) limitations.push_back("The configured result limit was reached.");
    return {{"complete", false}, {"truncated", truncated}, {"limitations", limitations}};
}

bool valid_compile_database(const std::string & contents) {
    const auto entries = json::parse(contents, nullptr, false);
    if (!entries.is_array() || entries.size() > 100000) return false;
    for (const auto & entry : entries) {
        if (!entry.is_object() || !entry.contains("directory") || !entry["directory"].is_string() ||
                entry["directory"].get_ref<const std::string &>().empty() ||
                entry["directory"].get_ref<const std::string &>().find('\0') != std::string::npos ||
                !entry.contains("file") || !entry["file"].is_string() ||
                entry["file"].get_ref<const std::string &>().empty() ||
                entry["file"].get_ref<const std::string &>().find('\0') != std::string::npos ||
                (!entry.contains("command") && !entry.contains("arguments")) ||
                (entry.contains("command") && (!entry["command"].is_string() ||
                    entry["command"].get_ref<const std::string &>().empty() ||
                    entry["command"].get_ref<const std::string &>().find('\0') != std::string::npos)) ||
                (entry.contains("arguments") && (!entry["arguments"].is_array() || entry["arguments"].empty()))) return false;
        if (entry.contains("arguments")) {
            for (const auto & argument : entry["arguments"])
                if (!argument.is_string() || argument.get_ref<const std::string &>().find('\0') != std::string::npos) return false;
        }
    }
    return true;
}

std::filesystem::path database_path(const std::string & value) {
    auto path = std::filesystem::path(value);
    if (!value.empty() && path.extension() != ".json") path /= "compile_commands.json";
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(path, ec);
    return ec ? path.lexically_normal() : canonical;
}

}

agent_clangd_code_provider::agent_clangd_code_provider(agent_clangd_session_config config)
    : file_watch_service_(config.file_watch_service ? std::move(config.file_watch_service)
          : std::make_shared<agent_project_file_watch_service>()),
      repository_root_(config.repository_root), compile_commands_dir_(config.compile_commands_dir),
      compile_commands_source_(config.compile_commands_source),
      compile_commands_destination_(config.compile_commands_destination) {
    if (!compile_commands_destination_.empty()) {
        auto destination = std::filesystem::path(compile_commands_destination_);
        if (destination.extension() == ".json") destination = destination.parent_path();
        compile_commands_dir_ = destination.string();
        config.compile_commands_dir = compile_commands_dir_;
    } else if (!compile_commands_dir_.empty() && compile_commands_dir_ != "auto") {
        auto database = std::filesystem::path(compile_commands_dir_);
        if (database.extension() == ".json") {
            compile_commands_dir_ = database.parent_path().string();
            config.compile_commands_dir = compile_commands_dir_;
        }
    }
    session_ = std::make_unique<agent_clangd_session>(std::move(config));
}

agent_clangd_code_provider::~agent_clangd_code_provider() {
    if (watch_subscription_id_) file_watch_service_->unsubscribe(watch_subscription_id_);
}

agent_clangd_json agent_clangd_code_provider::status_snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string watch_error;
    const bool watching = ensure_watcher(watch_error);
    auto status = session_->status_snapshot();
    const auto latest = watching ? file_watch_service_->latest_generation(watch_subscription_id_) : 0;
    const auto delivered = watching ? file_watch_service_->delivered_generation(watch_subscription_id_) : 0;
    status["project_index_requested"] = project_index_started_.load() && status.value("session", std::string{}) == "initialized";
    status["file_watcher"] = {{"active", watching},
        {"backend", watching ? file_watch_service_->backend(watch_subscription_id_) : "unavailable"},
        {"needs_reconciliation", watcher_reconciliation_required_.load() ||
            (watch_subscription_id_ && file_watch_service_->needs_reconciliation(watch_subscription_id_))},
        {"latest_change_generation", latest},
        {"delivered_change_generation", delivered},
        {"delivery_pending", latest > delivered},
        {"error", !watcher_error_.empty() ? json(watcher_error_) : watching ? json(nullptr) : json(watch_error)}};
    status["compile_commands_source"] = compile_commands_source_.empty() ? json(nullptr) : json(compile_commands_source_);
    status["compile_commands_destination"] = compile_commands_destination_.empty()
        ? (compile_commands_dir_.empty() ? json(nullptr) : json(compile_commands_dir_)) : json(compile_commands_destination_);
    status["index_coverage"] = "unknown";
    status["compile_commands_may_be_stale"] = compile_commands_may_be_stale_.load();
    status["compile_commands_sync_error"] = compile_commands_error_.empty() ? json(nullptr) : json(compile_commands_error_);
    return status;
}

bool agent_clangd_code_provider::ensure_watcher(std::string & error) {
    if (watch_subscription_id_) { error.clear(); return true; }
    agent_project_file_watch_config config;
    config.root = repository_root_;
    if (!compile_commands_source_.empty()) config.explicit_files.push_back(database_path(compile_commands_source_).string());
    if (!compile_commands_destination_.empty()) config.explicit_files.push_back(database_path(compile_commands_destination_).string());
    else if (!compile_commands_dir_.empty() && compile_commands_dir_ != "auto") {
        auto path = std::filesystem::path(compile_commands_dir_);
        if (path.extension() != ".json") path /= "compile_commands.json";
        config.explicit_files.push_back(path.string());
    }
    config.poll_interval = std::chrono::milliseconds(2000);
    if (!file_watch_service_->subscribe(config, [this](const auto & changes) { return on_file_changes(changes); },
            watch_subscription_id_, error)) return false;
    watcher_reconciliation_required_ = false;
    return true;
}

bool agent_clangd_code_provider::on_file_changes(const std::vector<agent_project_file_change> & changes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (changes.empty()) return true;
    latest_change_generation_ = changes.back().generation;
    bool rescan = false;
    bool database_event = false;
    for (const auto & change : changes)
        if (change.kind == agent_project_file_change_kind::rescan_required) rescan = true;
        else {
            const auto filename = std::filesystem::path(change.path).filename().string();
            if (filename == "CMakeLists.txt" || filename == "Makefile" || filename == "makefile" ||
                    filename == "meson.build" || filename == "configure.ac" ||
                    std::filesystem::path(change.path).extension() == ".cmake" ||
                    std::filesystem::path(change.path).extension() == ".mk") compile_commands_may_be_stale_ = true;
            const auto matches_database = [&](const std::string & configured) {
                if (configured.empty()) return false;
                const auto expected = database_path(configured);
                return database_path(change.path) == expected ||
                    (!change.previous_path.empty() && database_path(change.previous_path) == expected);
            };
            if (matches_database(compile_commands_source_) || matches_database(compile_commands_destination_))
                database_event = true;
        }
    std::string error;
    if (rescan) {
        watcher_reconciliation_required_ = true;
        session_->stop_for_refresh();
        project_index_started_ = false;
        return false;
    } else if (!session_->apply_file_changes(changes, error)) {
        watcher_error_ = std::move(error);
        watcher_reconciliation_required_ = true;
        session_->stop_for_refresh();
        project_index_started_ = false;
        return false;
    }
    if (database_event && !rescan) {
        bool changed = false;
        if (!update_compile_commands(changed, error)) {
            compile_commands_error_ = error;
            compile_commands_may_be_stale_ = true;
        } else if (changed) {
            compile_commands_error_.clear();
            compile_commands_may_be_stale_ = false;
            session_->stop_for_refresh();
            project_index_started_ = false;
        }
    }
    delivered_change_generation_ = latest_change_generation_.load();
    return true;
}

bool agent_clangd_code_provider::update_compile_commands(bool & changed, std::string & error) {
    changed = false;
    if (compile_commands_source_.empty()) return true;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path source = fs::absolute(compile_commands_source_, ec);
    if (ec) { error = "configured compilation database source could not be resolved"; return false; }
    if (fs::is_directory(source, ec) && !ec) source /= "compile_commands.json";
    const auto bytes = fs::file_size(source, ec);
    if (ec || bytes > 16 * 1024 * 1024) { error = "configured compilation database source is unavailable or exceeds 16 MiB"; return false; }
    std::ifstream input(source, std::ios::binary);
    if (!input) { error = "configured compilation database source could not be opened"; return false; }
    std::string contents(static_cast<size_t>(bytes), '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (input.bad() || static_cast<size_t>(input.gcount()) != contents.size()) {
        error = "configured compilation database changed while being read";
        return false;
    }
    if (!valid_compile_database(contents)) { error = "configured compilation database is invalid or exceeds its entry limit"; return false; }

    fs::path destination = compile_commands_destination_.empty()
        ? fs::path(compile_commands_dir_) : fs::path(compile_commands_destination_);
    if (destination.empty() || destination == "auto") destination = repository_root_;
    destination = fs::absolute(destination, ec);
    if (ec) { error = "configured compilation database destination could not be resolved"; return false; }
    if (destination.extension() != ".json") destination /= "compile_commands.json";
    std::error_code source_canonical_error, destination_canonical_error;
    const auto canonical_source = fs::weakly_canonical(source, source_canonical_error);
    const auto canonical_destination = fs::weakly_canonical(destination, destination_canonical_error);
    if (!source_canonical_error && !destination_canonical_error && canonical_source == canonical_destination) {
        error.clear();
        return true;
    }
    std::ifstream existing(destination, std::ios::binary);
    if (existing) {
        std::string prior((std::istreambuf_iterator<char>(existing)), std::istreambuf_iterator<char>());
        if (prior == contents) { error.clear(); return true; }
    }
    fs::create_directories(destination.parent_path(), ec);
    if (ec) { error = "compilation database destination directory could not be created"; return false; }
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    auto temporary = destination;
    temporary += ".tmp." + std::to_string(suffix);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        output.flush();
        if (!output) { fs::remove(temporary, ec); error = "temporary compilation database could not be written"; return false; }
    }
    fs::rename(temporary, destination, ec);
    if (ec) {
        fs::remove(temporary, ec);
        error = "validated compilation database could not be published atomically";
        return false;
    }
    changed = true;
    error.clear();
    return true;
}

bool agent_clangd_code_provider::synchronize_compile_commands(std::string & error) {
    bool changed = false;
    if (!update_compile_commands(changed, error)) {
        compile_commands_error_ = error;
        compile_commands_may_be_stale_ = true;
        const auto cache_dir = compile_commands_dir_.empty() || compile_commands_dir_ == "auto"
            ? repository_root_ : compile_commands_dir_;
        auto cached = std::filesystem::path(cache_dir) / "compile_commands.json";
        std::error_code ec;
        const auto size = std::filesystem::file_size(cached, ec);
        if (ec || size > 16 * 1024 * 1024) return false;
        std::ifstream input(cached, std::ios::binary);
        if (!input) return false;
        std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (input.bad() || contents.size() > 16 * 1024 * 1024 || !valid_compile_database(contents)) return false;
        error.clear();
        return true;
    }
    compile_commands_error_.clear();
    if (changed) {
        if (session_->initialized()) session_->stop_for_refresh();
        project_index_started_ = false;
    }
    return true;
}

bool agent_clangd_code_provider::wait_for_changes(std::unique_lock<std::mutex> & lock) {
    const auto id = watch_subscription_id_;
    lock.unlock(); // The consumer callback needs the provider lock to apply changes.
    const bool ready = file_watch_service_->wait_for_delivery(id, std::chrono::milliseconds(1000));
    lock.lock();
    return ready && !file_watch_service_->needs_reconciliation(id) &&
        file_watch_service_->delivered_generation(id) >= file_watch_service_->latest_generation(id);
}

common_tool_execution_result agent_clangd_code_provider::refresh() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string error;
    if (!ensure_watcher(error)) return provider_error("tool.code.refresh.unavailable", error);
    bool database_changed = false;
    if (!update_compile_commands(database_changed, error)) {
        compile_commands_error_ = error;
        compile_commands_may_be_stale_ = true;
        return provider_error("tool.code.refresh.database_invalid", error);
    }
    compile_commands_error_.clear();
    if (database_changed) compile_commands_may_be_stale_ = false;
    session_->stop_for_refresh();
    project_index_started_ = false;
    if (!file_watch_service_->reconcile(watch_subscription_id_, error)) return provider_error("tool.code.refresh.unavailable", error);
    watcher_reconciliation_required_ = false;
    watcher_error_.clear();
    const bool reconciliation_needed = file_watch_service_->needs_reconciliation(watch_subscription_id_);
    const auto output = json{{"refreshed", true}, {"clangd_restart_on_next_query", true},
        {"compile_commands_changed", database_changed}, {"index_coverage", "unknown"},
        {"reconciliation_scheduled", true},
        {"watcher_backend", file_watch_service_->backend(watch_subscription_id_)},
        {"reconciliation_required", reconciliation_needed},
        {"message", reconciliation_needed
            ? "Clangd will reinitialize; the watcher remains degraded and requires reconciliation."
            : "File reconciliation was scheduled; clangd will reinitialize on the next semantic query."}};
    return common_tool_execution_result::success(output.dump());
}

bool agent_clangd_code_provider::start_project_index(std::string & error) {
    if (repository_root_.empty()) { error = "clangd project indexing requires a repository root"; return false; }
    if (!ensure_watcher(error)) { watcher_error_ = error; return false; }
    if (!synchronize_compile_commands(error)) { watcher_error_ = error; return false; }
    if (project_index_started_ && session_->initialized()) return true;
    project_index_started_ = false;
    std::error_code ec;
    auto directory = compile_commands_dir_.empty() || compile_commands_dir_ == "auto"
        ? std::filesystem::path(repository_root_) : std::filesystem::path(compile_commands_dir_);
    auto database = std::filesystem::absolute(directory / "compile_commands.json", ec);
    if (ec) return true;
    const auto size = std::filesystem::file_size(database, ec);
    if (ec || size > 16 * 1024 * 1024) return true;
    std::ifstream input(database, std::ios::binary);
    if (!input) return true;
    std::string contents(16 * 1024 * 1024 + 1, '\0');
    input.read(&contents[0], contents.size());
    contents.resize(static_cast<size_t>(input.gcount()));
    if (input.bad() || contents.size() > 16 * 1024 * 1024) return true;
    const auto entries = json::parse(contents, nullptr, false);
    if (!entries.is_array()) return true;
    for (const auto & entry : entries) {
        if (!entry.is_object() || !entry.contains("directory") || !entry["directory"].is_string() ||
                !entry.contains("file") || !entry["file"].is_string()) continue;
        std::filesystem::path source;
        std::string location_error;
        const auto path = std::filesystem::path(entry["directory"].get<std::string>()) /
            std::filesystem::path(entry["file"].get<std::string>());
        if (!agent_clangd_repository_path(repository_root_, path, source, location_error)) continue;
        const auto bytes = std::filesystem::file_size(source, ec);
        if (ec || bytes > 4 * 1024 * 1024) { ec.clear(); continue; }
        std::ifstream file(source, std::ios::binary);
        if (!file) continue;
        std::string text(4 * 1024 * 1024 + 1, '\0');
        file.read(&text[0], text.size());
        text.resize(static_cast<size_t>(file.gcount()));
        if (file.bad() || text.size() > 4 * 1024 * 1024) continue;
        // Opening one authorized translation unit makes clangd discover its
        // compilation database and schedule its own background index.
        if (session_->open_document(agent_clangd_file_uri(source), language_id(source), text, location_error))
            project_index_started_ = true;
        return true;
    }
    return true;
}

common_tool_execution_result agent_clangd_code_provider::symbol(const std::string & arguments_json) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::string watch_error;
    if (!ensure_watcher(watch_error)) return provider_error("tool.diagnostics.symbol.provider_unavailable", watch_error);
    if (!wait_for_changes(lock)) return common_tool_execution_result::failure("tool.code.changes_pending", common_tool_failure_class::execution, true,
        "File changes are still being applied or require code.refresh; retry after reconciliation.");
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!valid_arguments(arguments, 64)) return common_tool_execution_result::failure("tool.diagnostics.symbol.invalid_arguments", common_tool_failure_class::validation, false, "Symbol arguments are invalid.");
    if (!synchronize_compile_commands(watch_error)) return provider_error("tool.diagnostics.symbol.provider_unavailable", watch_error);
    return workspace_symbol(arguments);
}

common_tool_execution_result agent_clangd_code_provider::workspace_symbol(const json & arguments) {
    json response;
    std::string error;
    std::filesystem::path hint;
    if (arguments.contains("path_hint") &&
            !agent_clangd_repository_path(repository_root_, arguments["path_hint"].get<std::string>(), hint, error))
        return common_tool_execution_result::failure("tool.diagnostics.symbol.invalid_arguments",
            common_tool_failure_class::validation, false, "Symbol path is outside the repository.", error);
    if (hint.empty() && !start_project_index(error))
        return provider_error("tool.diagnostics.symbol.provider_unavailable", error);
    std::error_code fs_error;
    if (!hint.empty() && std::filesystem::is_regular_file(hint, fs_error) && !fs_error) {
        constexpr size_t source_limit = 4 * 1024 * 1024;
        std::ifstream input(hint, std::ios::binary);
        if (!input) return provider_error("tool.diagnostics.symbol.provider_unavailable", "could not open the hinted source file");
        std::string source(source_limit + 1, '\0');
        input.read(&source[0], source.size());
        source.resize(static_cast<size_t>(input.gcount()));
        if (input.bad() || source.size() > source_limit)
            return provider_error("tool.diagnostics.symbol.provider_unavailable", "hinted source read failed or exceeded its byte limit");
        const auto uri = agent_clangd_file_uri(hint);
        if (!session_->open_document(uri, language_id(hint), source, error))
            return provider_error("tool.diagnostics.symbol.provider_unavailable", error);
        json parsed;
        if (!session_->request("textDocument/documentSymbol", {{"textDocument", {{"uri", uri}}}}, parsed, error))
            return provider_error("tool.diagnostics.symbol.provider_unavailable", error);
    }
    if (!session_->request("workspace/symbol", {{"query", arguments["symbol"]}}, response, error)) return provider_error("tool.diagnostics.symbol.provider_unavailable", error);
    json definitions = json::array(), result;
    if (!result_array(response, result)) return provider_error("tool.diagnostics.symbol.invalid_result", "clangd returned an invalid symbol result");
    const auto limit = std::min<size_t>(arguments.value("max_results", 64), 64);
    bool incomplete = false;
    for (const auto & item : result) {
        if (definitions.size() >= limit) break;
        json definition;
        if (!item.is_object() || !item.contains("location") ||
                !agent_clangd_project_location(item["location"], repository_root_, definition, error)) {
            incomplete = true; continue;
        }
        if (!hint.empty()) {
            std::filesystem::path candidate;
            if (!agent_clangd_repository_path(repository_root_, definition["path"].get<std::string>(), candidate, error)) {
                incomplete = true; continue;
            }
            const auto relative = candidate.lexically_relative(hint);
            if (relative.is_absolute() || (!relative.empty() && *relative.begin() == "..")) continue;
        }
        if (item.contains("name") && item["name"].is_string()) definition["name"] = item["name"];
        if (item.contains("kind") && item["kind"].is_number_integer()) definition["kind"] = item["kind"];
        definitions.push_back(std::move(definition));
    }
    auto output = coverage(result.size() > definitions.size() && definitions.size() >= limit, incomplete);
    output.update({{"backend", "clangd"}, {"semantic", true}, {"symbol", arguments["symbol"]},
        {"definitions", definitions}, {"count", definitions.size()}});
    return common_tool_execution_result::success(output.dump());
}

bool agent_clangd_code_provider::resolve_location(const json & arguments, source_location & location, std::string & error) {
    if (repository_root_.empty()) { error = "semantic source requires a host repository"; return false; }
    std::filesystem::path requested;
    int line = 0;
    int column = 1;
    if (arguments.contains("definition_path") && arguments.contains("definition_line")) {
        requested = std::filesystem::u8path(arguments["definition_path"].get<std::string>());
        line = arguments["definition_line"].get<int>();
        column = arguments.value("definition_column", 1);
    } else if (arguments.contains("path_hint") && arguments.contains("line")) {
        requested = std::filesystem::u8path(arguments["path_hint"].get<std::string>());
        line = arguments["line"].get<int>();
        column = arguments.value("column", 1);
    } else {
        error = "semantic diagnostics request requires a source location";
        return false;
    }

    std::filesystem::path candidate;
    if (!agent_clangd_repository_path(repository_root_, requested, candidate, error)) return false;
    std::error_code path_error;
    if (!std::filesystem::is_regular_file(candidate, path_error) || path_error) {
        error = "semantic source must be a file inside the configured repository"; return false;
    }
    if (line < 1 || column < 1) {
        error = "semantic diagnostics source position is invalid";
        return false;
    }
    std::error_code size_error;
    if (std::filesystem::file_size(candidate, size_error) > 4 * 1024 * 1024 || size_error) {
        error = "semantic diagnostics source is unavailable or exceeds the 4 MiB limit";
        return false;
    }
    constexpr size_t source_limit = 4 * 1024 * 1024;
    std::ifstream source(candidate, std::ios::binary);
    if (!source) { error = "source could not be opened"; return false; }
    std::string text(source_limit + 1, '\0');
    source.read(&text[0], static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<size_t>(source.gcount()));
    if (source.bad() || text.size() > source_limit) { error = "source read failed or exceeded its byte limit"; return false; }
    std::istringstream lines(text);
    std::string source_line;
    bool line_found = false;
    for (int current = 1; current <= line && std::getline(lines, source_line); ++current) {
        if (current != line) continue;
        line_found = true;
        const auto symbol = arguments.value("symbol", std::string{});
        auto symbol_offset = source_line.find(symbol);
        if (symbol.empty() || symbol_offset == std::string::npos) {
            error = "symbol does not occur on the requested source line; choose a matching path and line";
            return false;
        }
        const bool explicit_column = arguments.contains("definition_path") && arguments.contains("definition_line")
            ? arguments.contains("definition_column") : arguments.contains("column");
        if (explicit_column) {
            while (symbol_offset != std::string::npos && utf16_units(source_line, symbol_offset) != static_cast<size_t>(column - 1))
                symbol_offset = source_line.find(symbol, symbol_offset + symbol.size());
            if (symbol_offset == std::string::npos) { error = "symbol does not match the explicit UTF-16 source column"; return false; }
        } else if (source_line.find(symbol, symbol_offset + symbol.size()) != std::string::npos) {
            error = "symbol occurs more than once on the source line; provide an explicit column"; return false;
        }
        column = static_cast<int>(utf16_units(source_line, symbol_offset)) + 1;
    }
    if (!line_found) {
        error = "requested source line is unavailable";
        return false;
    }
    location = {candidate, agent_clangd_file_uri(candidate), std::move(text), line - 1, column - 1};
    return true;
}

bool agent_clangd_code_provider::open_location(const source_location & location, std::string & error) {
    return session_->open_document(location.uri, language_id(location.path), location.text, error);
}

common_tool_execution_result agent_clangd_code_provider::location_query(const json & arguments, const std::string & method) {
    source_location location;
    std::string error;
    if (!resolve_location(arguments, location, error)) return common_tool_execution_result::failure("tool.diagnostics.semantic.invalid_arguments", common_tool_failure_class::validation, false, "A source location is required for this semantic query.", error);
    if (!open_location(location, error)) return provider_error("tool.diagnostics.semantic.provider_unavailable", error);
    json response;
    auto params = json{{"textDocument", {{"uri", location.uri}}}, {"position", {{"line", location.line}, {"character", location.column}}}};
    if (method == "textDocument/references") params["context"] = {{"includeDeclaration", true}};
    if (!session_->request(method, params, response, error)) return provider_error("tool.diagnostics.semantic.provider_unavailable", error);
    json entries = json::array(), result;
    if (!result_array(response, result)) return provider_error("tool.diagnostics.semantic.invalid_result", "clangd returned an invalid reference result");
    const auto limit = std::min<size_t>(arguments.value("max_results", 128), 128);
    bool incomplete = false;
    for (const auto & item : result) {
        if (entries.size() >= limit) break;
        json entry;
        if (!agent_clangd_project_location(item, repository_root_, entry, error)) { incomplete = true; continue; }
        entries.push_back(std::move(entry));
    }
    auto output = coverage(result.size() > entries.size() && entries.size() >= limit, incomplete);
    output.update({{"backend", "clangd"}, {"semantic", true}, {"results", entries}, {"count", entries.size()}});
    return common_tool_execution_result::success(output.dump());
}

common_tool_execution_result agent_clangd_code_provider::references(const std::string & arguments_json) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::string watch_error;
    if (!ensure_watcher(watch_error)) return provider_error("tool.diagnostics.references.provider_unavailable", watch_error);
    if (!wait_for_changes(lock)) return common_tool_execution_result::failure("tool.code.changes_pending", common_tool_failure_class::execution, true,
        "File changes are still being applied or require code.refresh; retry after reconciliation.");
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!valid_arguments(arguments, 128)) return common_tool_execution_result::failure("tool.diagnostics.references.invalid_arguments", common_tool_failure_class::validation, false, "Reference arguments are invalid.");
    if (!synchronize_compile_commands(watch_error)) return provider_error("tool.diagnostics.references.provider_unavailable", watch_error);
    return location_query(arguments, "textDocument/references");
}

common_tool_execution_result agent_clangd_code_provider::call_hierarchy(const std::string & arguments_json) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::string watch_error;
    if (!ensure_watcher(watch_error)) return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", watch_error);
    if (!wait_for_changes(lock)) return common_tool_execution_result::failure("tool.code.changes_pending", common_tool_failure_class::execution, true,
        "File changes are still being applied or require code.refresh; retry after reconciliation.");
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!valid_arguments(arguments, 128)) return common_tool_execution_result::failure("tool.diagnostics.call_hierarchy.invalid_arguments", common_tool_failure_class::validation, false, "Call hierarchy arguments are invalid.");
    if (!synchronize_compile_commands(watch_error)) return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", watch_error);
    source_location location;
    std::string error;
    if (!resolve_location(arguments, location, error)) return common_tool_execution_result::failure("tool.diagnostics.call_hierarchy.invalid_arguments", common_tool_failure_class::validation, false, "A source location is required for call hierarchy.", error);
    if (!open_location(location, error)) return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", error);
    json prepared;
    if (!session_->request("textDocument/prepareCallHierarchy", {{"textDocument", {{"uri", location.uri}}}, {"position", {{"line", location.line}, {"character", location.column}}}}, prepared, error)) return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", error);
    json items;
    if (!result_array(prepared, items)) return provider_error("tool.diagnostics.call_hierarchy.invalid_result", "clangd returned an invalid call hierarchy selection");
    if (items.size() > 1) return provider_error("tool.diagnostics.call_hierarchy.ambiguous", "source position resolved to multiple call hierarchy items");
    const auto direction = arguments.value("direction", std::string("both"));
    json callers = json::array(), callees = json::array();
    const auto limit = std::min<size_t>(arguments.value("max_results", 128), 128);
    bool truncated = false, incomplete = false;
    if (!items.empty()) {
        const auto & item = items[0];
        json selected;
        if (!agent_clangd_project_location(item, repository_root_, selected, error))
            return provider_error("tool.diagnostics.call_hierarchy.invalid_result", error);
        auto collect = [&](const char * method, json & output, bool incoming) {
            json response, result;
            if (!session_->request(method, {{"item", item}}, response, error)) return false;
            if (!result_array(response, result)) { error = "clangd returned an invalid call hierarchy result"; return false; }
            for (const auto & entry : result) {
                if (callers.size() + callees.size() >= limit) { truncated = true; break; }
                json normalized;
                if (!agent_clangd_project_call(entry, item, incoming, repository_root_, normalized, error)) {
                    incomplete = true; continue;
                }
                output.push_back(std::move(normalized));
            }
            return true;
        };
        if ((direction == "callers" || direction == "both") && !collect("callHierarchy/incomingCalls", callers, true))
            return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", error);
        if ((direction == "callees" || direction == "both") && !collect("callHierarchy/outgoingCalls", callees, false))
            return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", error);
    }
    auto output = coverage(truncated, incomplete);
    output.update({{"backend", "clangd"}, {"semantic", true}, {"callers", callers}, {"callees", callees},
        {"count", callers.size() + callees.size()}});
    return common_tool_execution_result::success(output.dump());
}
