#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <vector>
#include <utility>

// Shared source identity contract; resource authorization and filesystem effects
// remain responsibilities of the calling host adapters.
struct common_agent_source_snapshot_file {
    std::string path;
    std::string resource_uri;
    std::string sha256;
    uint64_t bytes = 0;
    uint32_t permissions = 0;
};

struct common_agent_source_snapshot {
    static constexpr size_t max_files = 8192;
    static constexpr uint64_t max_file_bytes = 64 * 1024 * 1024;
    static constexpr uint64_t max_total_bytes = 256 * 1024 * 1024;
    std::vector<common_agent_source_snapshot_file> files;
};

inline bool common_agent_parse_source_snapshot(
        const std::string & serialized, common_agent_source_snapshot & output, std::string & error) {
    output = {};
    const auto value = nlohmann::json::parse(serialized, nullptr, false);
    const auto fail = [&error](const char * message) { error = message; return false; };
    if (!value.is_object() || !value.contains("schema") || !value["schema"].is_string() ||
            value["schema"] != "code-source-snapshot.v1" || !value.contains("complete") ||
            !value["complete"].is_boolean() || value["complete"] != true ||
            !value.contains("files") || !value["files"].is_array() ||
            value["files"].size() > common_agent_source_snapshot::max_files) {
        return fail("source snapshot is malformed, incomplete or exceeds its file limit");
    }
    common_agent_source_snapshot parsed;
    std::set<std::string> paths;
    uint64_t total_bytes = 0;
    for (const auto & entry : value["files"]) {
        if (!entry.is_object() || !entry.contains("path") || !entry["path"].is_string() ||
                !entry.contains("resource_uri") || !entry["resource_uri"].is_string() ||
                !entry.contains("sha256") || !entry["sha256"].is_string() ||
                !entry.contains("bytes") || !entry["bytes"].is_number_unsigned() ||
                !entry.contains("permissions") || !entry["permissions"].is_number_unsigned()) {
            return fail("source snapshot contains an invalid file entry");
        }
        common_agent_source_snapshot_file file;
        file.path = entry["path"].get<std::string>();
        file.resource_uri = entry["resource_uri"].get<std::string>();
        file.sha256 = entry["sha256"].get<std::string>();
        file.bytes = entry["bytes"].get<uint64_t>();
        const auto mode = entry["permissions"].get<uint64_t>();
        const std::filesystem::path relative(file.path);
        if (file.path.empty() || file.path.find('\0') != std::string::npos || relative.is_absolute() ||
                relative.has_root_name() || file.path.find('\\') != std::string::npos ||
                relative.filename().empty() || relative.generic_string() != file.path ||
                !paths.insert(file.path).second) {
            return fail("source snapshot contains an unsafe or duplicate path");
        }
        for (const auto & component : relative) {
            if (component == "." || component == "..") return fail("source snapshot path escapes the staged source root");
        }
        if (file.resource_uri.rfind("resource://", 0) != 0 || file.sha256.size() != 71 ||
                file.sha256.rfind("sha256:", 0) != 0 ||
                file.sha256.find_first_not_of("0123456789abcdef", 7) != std::string::npos || mode > 0777u ||
                file.bytes > common_agent_source_snapshot::max_file_bytes ||
                total_bytes > common_agent_source_snapshot::max_total_bytes - file.bytes) {
            return fail("source snapshot file identity, mode or size is invalid");
        }
        file.permissions = static_cast<uint32_t>(mode);
        total_bytes += file.bytes;
        parsed.files.push_back(std::move(file));
    }
    for (const auto & path : paths) {
        auto parent = std::filesystem::path(path).parent_path();
        while (!parent.empty()) {
            if (paths.count(parent.generic_string())) return fail("source snapshot has conflicting file and directory paths");
            parent = parent.parent_path();
        }
    }
    if ((value.contains("file_count") && (!value["file_count"].is_number_unsigned() ||
                value["file_count"].get<uint64_t>() != parsed.files.size())) ||
            (value.contains("total_bytes") && (!value["total_bytes"].is_number_unsigned() ||
                value["total_bytes"].get<uint64_t>() != total_bytes))) {
        return fail("source snapshot counts do not match its files");
    }
    output = std::move(parsed);
    error.clear();
    return true;
}
