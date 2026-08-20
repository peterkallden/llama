#pragma once

#include "workspace-contract.h"

#include <filesystem>
#include <fstream>

class common_agent_workspace_manager {
public:
    explicit common_agent_workspace_manager(common_agent_workspace_roots roots)
        : roots(std::move(roots)) {}

    bool create_operation(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            common_agent_workspace_operation & operation,
            std::string & error) const {
        if (!validate_common_agent_workspace_context(context, error)) return false;
        if (operation_id.empty()) {
            error = "workspace operation requires an operation id";
            return false;
        }
        if (roots.workspace_root.empty()) {
            error = "workspace root is required";
            return false;
        }

        const auto workspace_name = safe_name(context.workspace_id);
        const auto operation_name = safe_name(operation_id);
        if (workspace_name.empty() || operation_name.empty()) {
            error = "workspace identity cannot be represented as a directory name";
            return false;
        }

        const std::filesystem::path workspace_root = std::filesystem::path(roots.workspace_root);
        const std::filesystem::path operation_root = workspace_root / workspace_name / operation_name;
        const std::filesystem::path artifact_root = roots.artifact_root.empty()
            ? operation_root / "artifacts"
            : std::filesystem::path(roots.artifact_root) / workspace_name / operation_name;
        std::error_code fs_error;
        std::filesystem::create_directories(operation_root / "source", fs_error);
        if (fs_error) return fail("workspace source directory could not be created", fs_error, error);
        std::filesystem::create_directories(operation_root / "writable", fs_error);
        if (fs_error) return fail("workspace writable directory could not be created", fs_error, error);
        std::filesystem::create_directories(artifact_root, fs_error);
        if (fs_error) return fail("workspace artifact directory could not be created", fs_error, error);

        operation = {};
        operation.context = context;
        operation.operation_id = operation_id;
        operation.operation_root = operation_root.string();
        operation.source_path = (operation_root / "source").string();
        operation.writable_path = (operation_root / "writable").string();
        operation.artifact_path = artifact_root.string();
        error.clear();
        return true;
    }

    bool materialize_text_resource(
            const common_agent_workspace_operation & operation,
            const common_runtime_resource_ref & resource,
            agent_resource_store & store,
            const agent_resource_read_authority & authority,
            const std::string & file_name,
            size_t max_bytes,
            std::string & output_path,
            std::string & error) const {
        if (operation.source_path.empty() || resource.uri.empty() || file_name.empty()) {
            error = "workspace resource materialization requires an operation, resource and file name";
            return false;
        }
        const auto safe_file_name = safe_name(file_name);
        if (safe_file_name.empty()) {
            error = "workspace resource file name is invalid";
            return false;
        }
        std::string text;
        if (!store.read_text(resource.uri, authority, max_bytes, text, error)) return false;
        const std::filesystem::path path = std::filesystem::path(operation.source_path) / safe_file_name;
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "workspace resource file could not be opened for writing";
            return false;
        }
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!output) {
            error = "workspace resource file could not be written";
            return false;
        }
        output_path = path.string();
        error.clear();
        return true;
    }

    bool materialize_resource(
            const common_agent_workspace_operation & operation,
            const common_runtime_resource_ref & resource,
            agent_resource_store & store,
            const agent_resource_read_authority & authority,
            const std::string & file_name,
            size_t max_bytes,
            std::string & output_path,
            std::string & error) const {
        if (operation.source_path.empty() || resource.uri.empty() || file_name.empty()) {
            error = "workspace resource materialization requires an operation, resource and file name";
            return false;
        }
        const auto safe_file_name = safe_name(file_name);
        if (safe_file_name.empty()) {
            error = "workspace resource file name is invalid";
            return false;
        }
        std::string bytes;
        if (!store.read_bytes(resource.uri, authority, max_bytes, bytes, error)) return false;
        const std::filesystem::path path = std::filesystem::path(operation.source_path) / safe_file_name;
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "workspace resource file could not be opened for writing";
            return false;
        }
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            error = "workspace resource file could not be written";
            return false;
        }
        output_path = path.string();
        error.clear();
        return true;
    }

private:
    static std::string safe_name(const std::string & value) {
        std::string result;
        for (const char character : value) {
            const bool safe =
                (character >= 'a' && character <= 'z') ||
                (character >= 'A' && character <= 'Z') ||
                (character >= '0' && character <= '9') ||
                character == '-' || character == '_' || character == '.';
            result += safe ? character : '_';
        }
        while (!result.empty() && result.front() == '.') result.front() = '_';
        return result;
    }

    static bool fail(const char * message, const std::error_code & fs_error, std::string & error) {
        error = std::string(message) + ": " + fs_error.message();
        return false;
    }

    common_agent_workspace_roots roots;
};
