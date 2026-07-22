#pragma once

#include "workspace-contract.h"

#include <filesystem>

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
